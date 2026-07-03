//! Block-level inference driver.
//!
//! Loads `.feat` files produced by Stage 01, runs the `PointNet` forward pass on
//! each block in parallel (Rayon), and returns a map from block ID to a
//! `BlockInferenceResult` that holds a 2-D spatial index and per-point ASPRS
//! labels.  The downstream output writer uses this map to substitute the
//! classification field while streaming the original LAS/LAZ file.

#![allow(clippy::manual_is_multiple_of)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetClassifier;
use crate::preprocessing::{
    validate_block_filename, BlockManifest, BlockMeta, FEAT_MAGIC, FEAT_VERSION,
    MAX_FEAT_PAYLOAD_BYTES, N_EIGEN_FEATURES_PER_RADIUS, N_SCALAR_FEATURES, RAYON_MIN_CHUNK,
};

// ─────────────────────────────────────────────────────────────────────────────
// BlockInferenceResult
// ─────────────────────────────────────────────────────────────────────────────

/// Inference results for a single spatial block.
///
/// Holds a 2-D k-d tree over reconstructed `(x, y)` coordinates for each
/// sampled point, with the ASPRS classification label as the tree payload.
/// The output writer queries this by nearest-neighbour to assign a label to
/// every original `LiDAR` point — O(log N) per query instead of O(N).
pub struct BlockInferenceResult {
    /// 2-D spatial index: key = `[x, y]` (projection units), value = ASPRS label.
    tree: KdTree<f64, u8, [f64; 2]>,
    /// Number of sampled points inserted into the tree (used for empty-check).
    n_points: usize,
}

impl BlockInferenceResult {
    /// Build a `BlockInferenceResult` from parallel coordinate + label slices.
    ///
    /// `xs`, `ys`, and `labels` must all have the same length.
    ///
    /// # Errors
    /// Returns an error if inserting a point into the k-d tree fails (dimension
    /// mismatch — unreachable in practice given the fixed `[f64; 2]` key type).
    pub fn from_points(xs: &[f64], ys: &[f64], labels: &[u8]) -> Result<Self> {
        debug_assert_eq!(xs.len(), ys.len());
        debug_assert_eq!(xs.len(), labels.len());
        let mut tree: KdTree<f64, u8, [f64; 2]> = KdTree::with_capacity(2, xs.len());
        for ((x, y), &label) in xs.iter().zip(ys.iter()).zip(labels.iter()) {
            tree.add([*x, *y], label)
                .map_err(|e| ClassifierError::Pipeline(format!("kd-tree insert error: {e}")))?;
        }
        Ok(Self {
            tree,
            n_points: xs.len(),
        })
    }

    /// Find the ASPRS label of the sampled point nearest to `(qx, qy)`.
    ///
    /// Uses a 2-D kd-tree for O(log N) lookup.  Falls back to ASPRS
    /// `Unassigned` (1) when the tree is empty.
    #[must_use]
    pub fn nearest_label(&self, qx: f64, qy: f64) -> u8 {
        if self.n_points == 0 {
            return 1u8; // ASPRS Unassigned fallback
        }
        self.tree
            .nearest(&[qx, qy], 1, &squared_euclidean)
            .ok()
            .and_then(|v| v.into_iter().next())
            .map_or(1u8, |(_, &label)| label)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WBFT header parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed header from a `.feat` file.
struct FeatHeader {
    n_points: usize,
    n_features: usize,
    origin_x: f64,
    origin_y: f64,
}

/// Parse and validate the 37-byte WBFT file header, returning the header
/// fields and leaving the reader positioned at the start of the data payload.
fn read_feat_header<R: Read>(r: &mut R, path_hint: &str) -> Result<FeatHeader> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| ClassifierError::Pipeline(format!("{path_hint}: header read error: {e}")))?;
    if &magic != FEAT_MAGIC {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: bad magic {magic:?} (expected {FEAT_MAGIC:?})"
        )));
    }

    let version = read_u8(r)?;
    if version != FEAT_VERSION {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: unsupported .feat version {version}"
        )));
    }

    let n_points = read_u32_le(r)? as usize;
    let n_features = read_u32_le(r)? as usize;
    let _block_id = read_u64_le(r)?;
    let origin_x = read_f64_le(r)?;
    let origin_y = read_f64_le(r)?;

    // Multi-scale-aware validation (Stage 06): n_features must be
    // N_SCALAR_FEATURES + N_EIGEN_FEATURES_PER_RADIUS × n_radii, where n_radii ≥ 1.
    // This accepts both the legacy 12-feature single-scale format and any
    // multi-scale format produced by Stage 06.
    if n_features < N_SCALAR_FEATURES
        || (n_features - N_SCALAR_FEATURES) % N_EIGEN_FEATURES_PER_RADIUS != 0
        || (n_features - N_SCALAR_FEATURES) / N_EIGEN_FEATURES_PER_RADIUS == 0
    {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: n_features={n_features} is not a valid multi-scale feature count \
             (expected {N_SCALAR_FEATURES} + {N_EIGEN_FEATURES_PER_RADIUS}×N for N≥1)"
        )));
    }

    Ok(FeatHeader {
        n_points,
        n_features,
        origin_x,
        origin_y,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run `PointNet` inference on all blocks in `manifest` and return a map from
/// block ID to `BlockInferenceResult`.
///
/// - Feature files are resolved relative to `feat_dir`.
/// - Inference is parallelised at the block level via Rayon.
/// - `model` is shared read-only via `Arc`.
/// - Results are collected lock-free: each worker returns an owned
///   `Result<(u64, BlockInferenceResult)>`, which are drained into a plain
///   `HashMap` sequentially after the parallel phase completes.
///
/// # Errors
/// Returns an error if any block's `.feat` file cannot be read, parsed, or
/// processed by the model.
pub fn run_inference(
    manifest: &BlockManifest,
    model: &Arc<PointNetClassifier>,
    feat_dir: &Path,
) -> Result<HashMap<u64, BlockInferenceResult>> {
    // ── Parallel phase — no locks, each worker owns its Result ────────────
    let block_results: Vec<Result<(u64, BlockInferenceResult)>> = manifest
        .blocks
        .par_iter()
        .with_min_len(RAYON_MIN_CHUNK)
        .map(|meta| {
            let result = process_block(meta, model, feat_dir, manifest.block_size)?;
            Ok((meta.id, result))
        })
        .collect();

    // ── Sequential drain — propagate first error, build HashMap ──────────
    let mut map = HashMap::with_capacity(manifest.blocks.len());
    for item in block_results {
        let (id, result) = item?;
        map.insert(id, result);
    }
    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-block processing
// ─────────────────────────────────────────────────────────────────────────────

fn process_block(
    meta: &BlockMeta,
    model: &PointNetClassifier,
    feat_dir: &Path,
    block_size: f64,
) -> Result<BlockInferenceResult> {
    // Stage 20 (Security Hardening): reject manifest file names that could
    // escape `feat_dir` via path traversal before joining.
    validate_block_filename(&meta.file)?;
    let feat_path = feat_dir.join(&meta.file);

    // ── Load .feat file ───────────────────────────────────────────────────
    let f = File::open(&feat_path).map_err(|e| {
        ClassifierError::Pipeline(format!(
            "cannot open feat file '{}': {e}",
            feat_path.display()
        ))
    })?;
    let mut r = BufReader::new(f);

    let header = read_feat_header(&mut r, &feat_path.to_string_lossy())?;
    let n = header.n_points;

    // Read f32 data payload: n_points × n_features.
    // Stage 20 (Security Hardening): use checked arithmetic and enforce an
    // upper bound *before* allocating, so a corrupted or maliciously-crafted
    // header (e.g. n_points ≈ u32::MAX) cannot drive a multi-gigabyte
    // allocation attempt.
    let n_floats = n.checked_mul(header.n_features).ok_or_else(|| {
        ClassifierError::Pipeline(format!(
            "'{}': n_points × n_features overflows usize (n_points={n}, n_features={})",
            feat_path.display(),
            header.n_features
        ))
    })?;
    let payload_bytes = n_floats.checked_mul(4).ok_or_else(|| {
        ClassifierError::Pipeline(format!(
            "'{}': data payload size overflows usize",
            feat_path.display()
        ))
    })?;
    if payload_bytes > MAX_FEAT_PAYLOAD_BYTES {
        return Err(ClassifierError::Pipeline(format!(
            "'{}': data payload of {payload_bytes} bytes exceeds the {MAX_FEAT_PAYLOAD_BYTES}-byte \
             safety cap (n_points={n}, n_features={}); refusing to allocate",
            feat_path.display(),
            header.n_features
        )));
    }

    let mut raw = vec![0u8; payload_bytes];
    r.read_exact(&mut raw).map_err(|e| {
        ClassifierError::Pipeline(format!(
            "'{}': data payload read error: {e}",
            feat_path.display()
        ))
    })?;

    // Stage 21 (Performance): zero-copy byte→f32 reinterpretation via
    // `bytemuck::try_cast_slice` instead of a manual per-element
    // `chunks_exact(4).map(from_le_bytes)` loop. `try_cast_slice` (rather
    // than the panicking `cast_slice`) preserves the project-wide
    // no-panics rule: on the (practically unreachable, since a `Vec<u8>`'s
    // heap allocation is always suitably aligned for `f32` on every
    // supported platform) misalignment case, fall back to the original
    // manual conversion instead of erroring or panicking.
    let floats: Vec<f32> = match bytemuck::try_cast_slice::<u8, f32>(&raw) {
        Ok(slice) => slice.to_vec(),
        Err(_) => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    };

    let features = Array2::from_shape_vec((n, header.n_features), floats)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    // ── Reconstruct approximate (x, y) from x_norm, y_norm columns ───────
    // block_size is stored in the manifest, not in the .feat header.
    // We approximate it from the origin and use feature cols 0 (x_norm) and
    // 1 (y_norm): x ≈ x_norm * block_size + origin_x.
    // The manifest carries block_size at the manifest level.
    // Since process_block doesn't have access to manifest.block_size directly
    // (it only receives BlockMeta), the caller must pass block_size.
    // To keep the API clean, we embed block_size in BlockMeta via the manifest.
    // Here we just use origin_x/y from the .feat header (same as meta.origin_x/y).
    //
    // NOTE: block_size is passed from BlockManifest (manifest-level field).
    // It is needed to reconstruct x/y from x_norm/y_norm.

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x_norm = f64::from(features[[i, 0]]);
        let y_norm = f64::from(features[[i, 1]]);
        xs.push(x_norm * block_size + header.origin_x);
        ys.push(y_norm * block_size + header.origin_y);
    }

    // ── Run PointNet forward pass ─────────────────────────────────────────
    let labels = model.classify(features)?;

    // ── Build 2-D k-d tree for O(log N) nearest-label lookup ─────────────
    BlockInferenceResult::from_points(&xs, &ys, &labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive I/O helpers
// ─────────────────────────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64_le<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // DoD #16 — nearest_label: exact hit and epsilon-offset
    #[test]
    fn test_nearest_label_exact_and_near() {
        let xs = vec![0.0f64, 10.0, 20.0];
        let ys = vec![0.0f64, 10.0, 20.0];
        let labels = vec![2u8, 5u8, 6u8];
        let result = BlockInferenceResult::from_points(&xs, &ys, &labels)
            .expect("kd-tree build must succeed");

        // Exact coordinates → should return the matching label
        assert_eq!(result.nearest_label(0.0, 0.0), 2u8);
        assert_eq!(result.nearest_label(10.0, 10.0), 5u8);
        assert_eq!(result.nearest_label(20.0, 20.0), 6u8);

        // ε-offset from point 1 → still nearest to point 1
        assert_eq!(result.nearest_label(10.001, 10.001), 5u8);
    }
}
