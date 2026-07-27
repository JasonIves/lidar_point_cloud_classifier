//! Block-level inference driver.
//!
//! Loads `.feat` files produced by Stage 01, runs the `PointNet` forward pass on
//! each block in parallel (Rayon), and returns a map from block ID to a
//! `BlockInferenceResult` that holds a 2-D spatial index and per-point softmax
//! **probability vectors** (model class-index space).  Downstream consumers —
//! the output writer (`output::las_writer`) and fused evaluation
//! (`cli::evaluate_cmd`) — combine these per-block votes via the cross-block
//! fusion rules in [`crate::model::fusion`] (Stage 44) instead of inheriting a
//! single hard-argmax label per block.

// f64 → f32 narrowing casts in the softmax path are intentional (probabilities
// are stored as f32; the exponentials/sums are accumulated in f64 for
// numerical safety and narrowed once, per element).
#![allow(clippy::manual_is_multiple_of, clippy::cast_possible_truncation)]

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
    validate_block_filename, BlockManifest, BlockMeta, FEAT_MAGIC, FEAT_VERSION, FEAT_VERSION_V1,
    MAX_FEAT_PAYLOAD_BYTES, N_FEATURES, RAYON_MIN_CHUNK,
};

// ─────────────────────────────────────────────────────────────────────────────
// BlockInferenceResult
// ─────────────────────────────────────────────────────────────────────────────

/// Inference results for a single spatial block.
///
/// Holds a 2-D k-d tree over reconstructed `(x, y)` coordinates for each
/// sampled point; the tree payload is the row index into `probs`, a row-major
/// `[n_points × n_classes]` matrix of temperature-softmaxed class
/// probabilities **in model class-index space** (i.e. *not* yet mapped through
/// `label_map`).  Stage 44 (Prediction Fusion): consumers fuse the probability
/// rows of one or more blocks and only then argmax + map to ASPRS codes, so
/// confidence information survives cross-block reconciliation.
pub struct BlockInferenceResult {
    /// 2-D spatial index: key = `[x, y]` (projection units), value = row index
    /// into `probs`.
    tree: KdTree<f64, u32, [f64; 2]>,
    /// Row-major `[n_points × n_classes]` softmax probability matrix.
    probs: Vec<f32>,
    /// Number of sampled points inserted into the tree (used for empty-check).
    n_points: usize,
    /// Number of model classes per probability row.
    n_classes: usize,
}

impl BlockInferenceResult {
    /// Build a `BlockInferenceResult` from parallel coordinate slices and the
    /// raw `[N, n_classes]` logit matrix produced by
    /// [`PointNetClassifier::forward`].
    ///
    /// Each logit row is converted to a probability row via a max-subtracted,
    /// temperature-scaled softmax:
    /// `p_i = exp((z_i − z_max) / τ) / Σ_j exp((z_j − z_max) / τ)`.
    /// At `temperature = 1.0` this is ordinary softmax.  Because softmax is
    /// monotone, a block voting alone produces exactly the same argmax as the
    /// legacy hard-argmax path.
    ///
    /// `xs`, `ys`, and `logits.rows()` must all have the same length.
    ///
    /// # Errors
    /// Returns an error if `temperature` is non-finite or non-positive, if the
    /// row count exceeds `u32::MAX`, or if inserting a point into the k-d tree
    /// fails (dimension mismatch — unreachable in practice given the fixed
    /// `[f64; 2]` key type).
    pub fn from_logits(
        xs: &[f64],
        ys: &[f64],
        logits: &Array2<f32>,
        temperature: f64,
    ) -> Result<Self> {
        debug_assert_eq!(xs.len(), ys.len());
        debug_assert_eq!(xs.len(), logits.nrows());

        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ClassifierError::Pipeline(format!(
                "from_logits: temperature must be finite and > 0.0, got {temperature}"
            )));
        }
        if xs.len() > u32::MAX as usize {
            return Err(ClassifierError::Pipeline(format!(
                "from_logits: {} rows exceed the u32 row-index capacity",
                xs.len()
            )));
        }

        let n = logits.nrows();
        let n_classes = logits.ncols();

        // ── Temperature softmax per row (f64 accumulation, f32 storage) ────
        let mut probs = vec![0.0f32; n * n_classes];
        for i in 0..n {
            let row = logits.row(i);
            let out = &mut probs[i * n_classes..(i + 1) * n_classes];
            softmax_row_into(&row, temperature, out);
        }

        // ── 2-D k-d tree: (x, y) → row index ──────────────────────────────
        let mut tree: KdTree<f64, u32, [f64; 2]> = KdTree::with_capacity(2, xs.len());
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            tree.add([*x, *y], i as u32)
                .map_err(|e| ClassifierError::Pipeline(format!("kd-tree insert error: {e}")))?;
        }

        Ok(Self {
            tree,
            probs,
            n_points: n,
            n_classes,
        })
    }

    /// Number of model classes per probability row.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Find the sampled point nearest to `(qx, qy)` and return its squared
    /// 2-D distance and its softmax probability row (model class-index
    /// space).
    ///
    /// Returns `None` when the tree is empty (degenerate zero-row block);
    /// callers treat `None` as "no vote" and preserve/fallback accordingly.
    #[must_use]
    pub fn nearest_vote(&self, qx: f64, qy: f64) -> Option<(f64, &[f32])> {
        if self.n_points == 0 {
            return None;
        }
        let (dist, &row_idx) = self
            .tree
            .nearest(&[qx, qy], 1, &squared_euclidean)
            .ok()?
            .into_iter()
            .next()?;
        let start = row_idx as usize * self.n_classes;
        Some((dist, &self.probs[start..start + self.n_classes]))
    }
}

/// Temperature-scaled, max-subtracted softmax of one logit row into `out`.
///
/// All exponentials and the normalising sum are accumulated in `f64` and
/// narrowed to `f32` once per element.  A degenerate (non-finite or zero)
/// normaliser leaves `out` as the raw exponentiated values rather than
/// dividing by zero — unreachable for finite model logits, but the no-panics /
/// no-NaN rule is kept unconditional.
fn softmax_row_into(row: &ndarray::ArrayView1<f32>, temperature: f64, out: &mut [f32]) {
    debug_assert_eq!(row.len(), out.len());
    if row.is_empty() {
        return;
    }
    let mut max = f64::NEG_INFINITY;
    for &v in row {
        max = max.max(f64::from(v));
    }
    let mut sum = 0.0f64;
    for (k, &v) in row.into_iter().enumerate() {
        let e = ((f64::from(v) - max) / temperature).exp();
        out[k] = e as f32;
        sum += e;
    }
    if sum.is_finite() && sum > 0.0 {
        for v in out.iter_mut() {
            *v = (f64::from(*v) / sum) as f32;
        }
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
    /// Trailing rows that are halo (overlap-margin) samples (Stage 45).
    /// `0` for v1 files (all-core) and for v2 files with halo disabled.
    #[allow(dead_code)] // Reserved for halo-aware consumers (fused-eval band masks).
    n_halo: usize,
}

/// Parse and validate a WBFT file header — v1 (37 bytes) or v2 (41 bytes,
/// Stage 45 with the trailing `n_halo` field) — returning the header fields
/// and leaving the reader positioned at the start of the data payload.
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
    if version != FEAT_VERSION && version != FEAT_VERSION_V1 {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: unsupported .feat version {version} (supported: {FEAT_VERSION_V1}, {FEAT_VERSION})"
        )));
    }

    let n_points = read_u32_le(r)? as usize;
    let n_features = read_u32_le(r)? as usize;
    let _block_id = read_u64_le(r)?;
    let origin_x = read_f64_le(r)?;
    let origin_y = read_f64_le(r)?;

    // v2 (Stage 45) carries a trailing n_halo field; v1 implies all-core.
    let n_halo = if version == FEAT_VERSION {
        read_u32_le(r)? as usize
    } else {
        0
    };

    // Fixed-width validation (Stage 30, Step 5e+5f+5g): n_features must equal
    // the fixed N_FEATURES constant (7 scalar + 10 pre-pass eigenvalue features).
    // The prior multi-scale (7 + 5×N) format no longer exists.
    if n_features != N_FEATURES {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: n_features={n_features} does not match expected fixed-width \
             feature count N_FEATURES={N_FEATURES}"
        )));
    }

    // Corruption guard (Stage 45): the halo row count can never consume the
    // whole payload — a header claiming n_halo ≥ n_points is corrupt.
    if n_halo >= n_points {
        return Err(ClassifierError::Pipeline(format!(
            "{path_hint}: n_halo={n_halo} must be less than n_points={n_points}"
        )));
    }

    Ok(FeatHeader {
        n_points,
        n_features,
        origin_x,
        origin_y,
        n_halo,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate reconstruction (shared by process_block and fused evaluation)
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct approximate absolute `(x, y)` coordinates for every row of a
/// block feature matrix from the block-relative `x_norm` / `y_norm` columns
/// (feature indices 0 and 1): `x ≈ x_norm · block_size + origin_x`.
///
/// Shared by the per-block inference path (`process_block`) and by
/// `evaluate --fused-eval` (Stage 44), which must place labeled points on the
/// same grid to run cross-block fusion.
pub(crate) fn reconstruct_xy(
    features: &Array2<f32>,
    origin_x: f64,
    origin_y: f64,
    block_size: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = features.nrows();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x_norm = f64::from(features[[i, 0]]);
        let y_norm = f64::from(features[[i, 1]]);
        xs.push(x_norm * block_size + origin_x);
        ys.push(y_norm * block_size + origin_y);
    }
    (xs, ys)
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
/// - `temperature` is the softmax temperature applied per block before any
///   downstream fusion (`1.0` = ordinary softmax).
/// - Results are collected lock-free: each worker returns an owned
///   `Result<(u64, BlockInferenceResult)>`, which are drained into a plain
///   `HashMap` sequentially after the parallel phase completes.
///
/// # Errors
/// Returns an error if any block's `.feat` file cannot be read, parsed, or
/// processed by the model, or if `temperature` is invalid.
pub fn run_inference(
    manifest: &BlockManifest,
    model: &Arc<PointNetClassifier>,
    feat_dir: &Path,
    temperature: f64,
) -> Result<HashMap<u64, BlockInferenceResult>> {
    // ── Parallel phase — no locks, each worker owns its Result ────────────
    let block_results: Vec<Result<(u64, BlockInferenceResult)>> = manifest
        .blocks
        .par_iter()
        .with_min_len(RAYON_MIN_CHUNK)
        .map(|meta| {
            let result = process_block(meta, model, feat_dir, manifest.block_size, temperature)?;
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
    temperature: f64,
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
    let (xs, ys) = reconstruct_xy(&features, header.origin_x, header.origin_y, block_size);

    // ── Run PointNet forward pass (logits — argmax deferred to fusion) ────
    let logits = model.forward(features)?;

    // ── Build 2-D k-d tree of probability rows for O(log N) vote lookup ───
    BlockInferenceResult::from_logits(&xs, &ys, &logits, temperature)
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

// Exact-hit k-d tree queries return a squared distance of exactly 0.0, so
// strict float equality is intentional and safe in these tests.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Known 3-class logits whose reference softmax (τ = 1) is
    /// ≈ [0.090031, 0.244728, 0.665241].
    fn reference_logits() -> Array2<f32> {
        arr2(&[[1.0f32, 2.0, 3.0]])
    }

    // DoD #3 — softmax correctness at τ = 1.0 (reference values + unit sum).
    #[test]
    fn test_from_logits_softmax_reference_values() {
        let xs = vec![0.0f64];
        let ys = vec![0.0f64];
        let res = BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), 1.0)
            .expect("from_logits must succeed");
        let (d2, row) = res.nearest_vote(0.0, 0.0).expect("vote must exist");
        assert_eq!(d2, 0.0);
        let expected = [0.090_030_57_f32, 0.244_728_46, 0.665_240_94];
        for (p, e) in row.iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-5, "prob {p} vs expected {e}");
        }
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "row must sum to 1, got {sum}");
    }

    // DoD #3 — temperature: τ < 1 sharpens, τ > 1 flattens relative to τ = 1.
    #[test]
    fn test_softmax_temperature_sharpens_and_flattens() {
        let xs = vec![0.0f64];
        let ys = vec![0.0f64];
        let sharp = BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), 0.5)
            .expect("sharp ok");
        let flat =
            BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), 4.0).expect("flat ok");
        let unit =
            BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), 1.0).expect("unit ok");

        let max_sharp = sharp
            .nearest_vote(0.0, 0.0)
            .map(|(_, r)| r.iter().copied().fold(0.0f32, f32::max))
            .expect("vote");
        let max_unit = unit
            .nearest_vote(0.0, 0.0)
            .map(|(_, r)| r.iter().copied().fold(0.0f32, f32::max))
            .expect("vote");
        let max_flat = flat
            .nearest_vote(0.0, 0.0)
            .map(|(_, r)| r.iter().copied().fold(0.0f32, f32::max))
            .expect("vote");

        assert!(max_sharp > max_unit, "τ=0.5 must sharpen");
        assert!(max_flat < max_unit, "τ=4.0 must flatten");
    }

    // DoD #3 — extreme-magnitude logits must not overflow (max-subtraction).
    #[test]
    fn test_softmax_extreme_logits_no_overflow() {
        let xs = vec![0.0f64];
        let ys = vec![0.0f64];
        let logits = arr2(&[[1000.0f32, 1001.0, 999.0]]);
        let res = BlockInferenceResult::from_logits(&xs, &ys, &logits, 1.0).expect("ok");
        let (_, row) = res.nearest_vote(0.0, 0.0).expect("vote");
        assert!(row.iter().all(|p| p.is_finite()));
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // DoD #3 — monotonicity: softmax argmax == logit argmax (legacy parity).
    #[test]
    fn test_softmax_argmax_matches_logit_argmax() {
        let logits = arr2(&[[0.1f32, 0.9, 0.0], [3.0, 1.0, 2.0], [1.0, 1.0, 1.0]]);
        let xs = vec![0.0f64, 10.0, 20.0];
        let ys = vec![0.0f64, 10.0, 20.0];
        let res = BlockInferenceResult::from_logits(&xs, &ys, &logits, 1.0).expect("ok");

        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let logit_argmax = logits
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            let (_, row) = res.nearest_vote(*x, *y).expect("vote");
            let prob_argmax = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            assert_eq!(logit_argmax, prob_argmax);
        }
    }

    // DoD — nearest_vote: exact hit, epsilon-offset, and returned distance.
    #[test]
    fn test_nearest_vote_exact_near_and_distance() {
        let logits = arr2(&[[0.1f32, 0.9], [0.8, 0.2], [0.3, 0.7]]);
        let xs = vec![0.0f64, 10.0, 20.0];
        let ys = vec![0.0f64, 10.0, 20.0];
        let res = BlockInferenceResult::from_logits(&xs, &ys, &logits, 1.0).expect("ok");

        // Exact hit on point 1 → row 0, squared distance 0.
        let (d2, row) = res.nearest_vote(0.0, 0.0).expect("vote");
        assert_eq!(d2, 0.0);
        assert!(row[1] > row[0]);

        // ε-offset from point 2 → still row 1; distance is squared.
        let (d2_eps, row_eps) = res.nearest_vote(10.001, 10.001).expect("vote");
        assert!(row_eps[0] > row_eps[1]);
        let expected_d2 = 2.0 * 0.001f64.powi(2);
        assert!((d2_eps - expected_d2).abs() < 1e-12);
    }

    // DoD — empty tree → None (no vote), never a panic.
    #[test]
    fn test_nearest_vote_empty_tree_returns_none() {
        let logits = Array2::<f32>::zeros((0, 3));
        let res = BlockInferenceResult::from_logits(&[], &[], &logits, 1.0).expect("ok");
        assert!(res.nearest_vote(1.0, 2.0).is_none());
    }

    // DoD — invalid temperatures are rejected (no panic).
    #[test]
    fn test_from_logits_rejects_bad_temperature() {
        let xs = vec![0.0f64];
        let ys = vec![0.0f64];
        assert!(BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), 0.0).is_err());
        assert!(BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), -1.0).is_err());
        assert!(
            BlockInferenceResult::from_logits(&xs, &ys, &reference_logits(), f64::NAN).is_err()
        );
    }
}
