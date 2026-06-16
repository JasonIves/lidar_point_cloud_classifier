//! Block-level inference driver.
//!
//! Loads `.feat` files produced by Stage 01, runs the `PointNet` forward pass on
//! each block in parallel (Rayon), and returns a map from block ID to a
//! `BlockInferenceResult` that holds a 2-D spatial index and per-point ASPRS
//! labels.  The downstream output writer uses this map to substitute the
//! classification field while streaming the original LAS/LAZ file.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetClassifier;
use crate::preprocessing::{BlockManifest, BlockMeta, FEAT_MAGIC, FEAT_VERSION, N_FEATURES};
use crate::preprocessing::RAYON_MIN_CHUNK;

// ─────────────────────────────────────────────────────────────────────────────
// BlockInferenceResult
// ─────────────────────────────────────────────────────────────────────────────

/// Inference results for a single spatial block.
///
/// Holds the reconstructed (x, y) coordinates for each sampled point and the
/// ASPRS classification label inferred by the model.  The output writer queries
/// this by nearest-neighbour to assign a label to each original `LiDAR` point.
pub struct BlockInferenceResult {
    /// Reconstructed X coordinates for each sampled point (projection units).
    pub xs: Vec<f64>,
    /// Reconstructed Y coordinates for each sampled point (projection units).
    pub ys: Vec<f64>,
    /// ASPRS classification code (u8) for each sampled point.
    pub labels: Vec<u8>,
}

impl BlockInferenceResult {
    /// Find the ASPRS label of the sampled point nearest to `(qx, qy)`.
    ///
    /// Uses a linear scan.
    #[must_use]
    pub fn nearest_label(&self, qx: f64, qy: f64) -> u8 {
        debug_assert_eq!(self.xs.len(), self.labels.len());
        let mut best_dist_sq = f64::INFINITY;
        let mut best_label = 1u8; // ASPRS Unassigned as fallback
        for (i, (&px, &py)) in self.xs.iter().zip(self.ys.iter()).enumerate() {
            let dx = px - qx;
            let dy = py - qy;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_label = self.labels[i];
            }
        }
        best_label
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
    r.read_exact(&mut magic).map_err(|e| {
        ClassifierError::Pipeline(format!("{path_hint}: header read error: {e}"))
    })?;
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

    let n_points   = read_u32_le(r)? as usize;
    let n_features = read_u32_le(r)? as usize;
    let _block_id  = read_u64_le(r)?;
    let origin_x   = read_f64_le(r)?;
    let origin_y   = read_f64_le(r)?;

    if n_features != N_FEATURES {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: n_features={n_features} != N_FEATURES={N_FEATURES}"
        )));
    }

    Ok(FeatHeader { n_points, n_features, origin_x, origin_y })
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
///
/// # Errors
/// Returns an error if any block's `.feat` file cannot be read, parsed, or
/// processed by the model.
///
/// # Panics
/// Panics if the internal Mutex becomes poisoned (indicates a prior panic in a
/// Rayon worker thread).
pub fn run_inference(
    manifest: &BlockManifest,
    model: &Arc<PointNetClassifier>,
    feat_dir: &Path,
) -> Result<HashMap<u64, BlockInferenceResult>> {
    let results: Arc<Mutex<HashMap<u64, BlockInferenceResult>>> =
        Arc::new(Mutex::new(HashMap::with_capacity(manifest.blocks.len())));

    let errors: Arc<Mutex<Vec<ClassifierError>>> = Arc::new(Mutex::new(Vec::new()));

    manifest
        .blocks
        .par_iter()
        .with_min_len(RAYON_MIN_CHUNK)
        .for_each(|meta| {
            match process_block(meta, model, feat_dir, manifest.block_size) {
                Ok(result) => {
                    results.lock().unwrap().insert(meta.id, result);
                }
                Err(e) => {
                    errors.lock().unwrap().push(e);
                }
            }
        });

    // Surface the first error if any blocks failed.
    let mut errs = errors.lock().unwrap();
    if !errs.is_empty() {
        return Err(errs.remove(0));
    }

    Arc::try_unwrap(results)
        .map_err(|_| ClassifierError::Pipeline("inference result Arc still shared".into()))?
        .into_inner()
        .map_err(|e| ClassifierError::Pipeline(format!("mutex poison: {e}")))
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

    // Read f32 data payload: n_points × n_features
    let n_floats = n * header.n_features;
    let mut raw = vec![0u8; n_floats * 4];
    r.read_exact(&mut raw).map_err(|e| {
        ClassifierError::Pipeline(format!(
            "'{}': data payload read error: {e}",
            feat_path.display()
        ))
    })?;

    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

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

    Ok(BlockInferenceResult { xs, ys, labels })
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
        let result = BlockInferenceResult {
            xs: vec![0.0, 10.0, 20.0],
            ys: vec![0.0, 10.0, 20.0],
            labels: vec![2u8, 5u8, 6u8],
        };

        // Exact coordinates → should return exact label
        assert_eq!(result.nearest_label(0.0, 0.0), 2u8);
        assert_eq!(result.nearest_label(10.0, 10.0), 5u8);
        assert_eq!(result.nearest_label(20.0, 20.0), 6u8);

        // ε-offset from point 1 → still nearest to point 1
        assert_eq!(result.nearest_label(10.001, 10.001), 5u8);

        // Midpoint between point 0 and 1: distance to both is equal,
        // tie goes to whichever is encountered first (point 0)
        // At (5.0, 5.0): d²(p0)=50, d²(p1)=50, d²(p2)=450 → first seen wins = p0
        assert_eq!(result.nearest_label(5.0, 5.0), 2u8);
    }
}
