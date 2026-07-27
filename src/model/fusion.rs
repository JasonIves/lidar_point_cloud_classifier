//! Cross-block prediction fusion (Stage 44 — Classify-Time Prediction Fusion).
//!
//! Reconciles the classifications of multiple blocks over the same location:
//! every original `LiDAR` point is labeled by fusing the per-block softmax
//! probability rows of **all** blocks whose inference footprint covers it,
//! weighted by
//!
//! ```text
//! vote_b(P) = t_b(P) · 1 / (d_b² + σ²)
//! ```
//!
//! where `t_b` is the *centrality* of point `P` with respect to block `b`'s
//! canonical rect (a distance-to-rect trapezoid: plateau 1 inside the rect,
//! linear ramp to 0 at `fusion_radius` beyond it — C⁰-continuous, so the blend
//! introduces no seam of its own) and `d_b²` is the squared distance from `P`
//! to `b`'s nearest sampled point.  The **proximity bandwidth** `σ` (default:
//! the characteristic inter-sample spacing, `block_size / √target_points`)
//! bounds the inverse-square term so a query point that coincides with a
//! block's own sample cannot dominate the blend by orders of magnitude —
//! without it, `d² = 0` self-hits would make every vote a home-block
//! dictatorship (the fused-eval blind spot found in real-data validation).
//! The final class is `argmax_c Σ_b vote_b · p_b[c]`, mapped through the
//! model `label_map` by the caller.
//!
//! Two call sites share this single implementation:
//! - `output::las_writer::write_classified` (streaming write of the
//!   classified output file), and
//! - `cli::evaluate_cmd` `--fused-eval` (validation metrics under the same
//!   deployed-decision rule).
//!
//! See `docs/stages/stage-44-classify-time-prediction-fusion.md`.

// Coordinate casts are deliberate: grid cell indices are computed via
// `f64::floor` then narrowed to `i64` (values are bounded by grid dimensions),
// and `i64` cell indices are widened back to `f64` for rect arithmetic
// (projection coordinates are far below 2^53, so no precision is lost).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::error::{ClassifierError, Result};
use crate::model::inference::BlockInferenceResult;
use crate::preprocessing::{block_id, BlockManifest};

/// Fallback proximity bandwidth² used only when a caller passes a degenerate
/// (non-finite or non-positive) `proximity_sigma` — defensive, unreachable
/// for the derived defaults (which are always positive).
const MIN_SIGMA_SQ: f64 = 1e-9;

/// Derive the default proximity bandwidth `σ`: the characteristic
/// inter-sample spacing of a block, `block_size / √target_points`.
///
/// At this bandwidth, a sample at distance 0 contributes at most ~2× the
/// weight of a sample one spacing away, keeping cross-block blends genuine
/// while still preferring nearby votes.
#[must_use]
pub fn default_proximity_sigma(block_size: f64, target_points: usize) -> f64 {
    block_size / (target_points.max(1) as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Fusion behaviour for one classify/evaluate run.
#[derive(Debug, Clone, Copy)]
pub struct FusionConfig {
    /// Voting reach beyond each block's canonical rect, in projection units.
    /// `0.0` disables fusion (legacy single-block behaviour).  Constrained to
    /// `≤ block_size / 2` by CLI validation, which guarantees at most 4
    /// candidate blocks per query point.
    pub radius: f64,
}

/// Grid geometry needed to map between projection coordinates and block
/// cells.  Mirrors the authoritative header-derived values stored in
/// [`BlockManifest`] (never re-derived from retained block origins — the
/// density filter may have dropped trailing cells).
#[derive(Debug, Clone, Copy)]
pub struct GridGeometry {
    /// South-west X origin of the block grid (projection units).
    pub x_min: f64,
    /// South-west Y origin of the block grid (projection units).
    pub y_min: f64,
    /// Cell edge length (projection units).
    pub block_size: f64,
    /// Number of grid columns (authoritative for `block_id` arithmetic).
    pub grid_cols: i64,
    /// Number of grid rows.
    pub grid_rows: i64,
}

impl GridGeometry {
    /// Build from a preprocessing manifest, rejecting manifests that predate
    /// the grid-geometry fields (they deserialize as `0`).
    ///
    /// # Errors
    /// Returns an error if `grid_cols` or `grid_rows` is zero (manifest too
    /// old — re-run preprocessing) or `block_size` is not positive.
    pub fn from_manifest(manifest: &BlockManifest) -> Result<Self> {
        if manifest.grid_cols == 0 || manifest.grid_rows == 0 {
            return Err(ClassifierError::Pipeline(
                "blocks.json is missing grid_cols/grid_rows — re-run preprocessing to \
                 regenerate it"
                    .to_string(),
            ));
        }
        if manifest.block_size <= 0.0 {
            return Err(ClassifierError::Pipeline(format!(
                "blocks.json has non-positive block_size ({})",
                manifest.block_size
            )));
        }
        Ok(Self {
            x_min: manifest.grid_x_min,
            y_min: manifest.grid_y_min,
            block_size: manifest.block_size,
            grid_cols: i64::from(manifest.grid_cols),
            grid_rows: i64::from(manifest.grid_rows),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Centrality weight (distance-to-rect trapezoid)
// ─────────────────────────────────────────────────────────────────────────────

/// Centrality of point `(x, y)` with respect to the canonical block rect
/// `[ox, ox+s] × [oy, oy+s]`, with voting reach `r`:
///
/// ```text
/// per axis:  gap  = max(ox − x, x − (ox + s), 0)     (0 inside the rect)
///            w_ax = clamp((r − gap) / r, 0, 1)
/// weight  t = w_x · w_y
/// ```
///
/// Properties:
/// - **1 everywhere inside the canonical rect** (plateau — the home block
///   always argues at full strength for its own territory);
/// - linear ramp to **0 at distance `r` beyond the rect**;
/// - C⁰-continuous (the blend introduces no seam of its own).
///
/// For `r ≤ 0` the function degenerates to a hard membership test
/// (1 inside, 0 outside) without dividing by zero.
#[must_use]
pub fn centrality_weight(x: f64, y: f64, ox: f64, oy: f64, s: f64, r: f64) -> f64 {
    let gap_x = (ox - x).max(x - (ox + s)).max(0.0);
    let gap_y = (oy - y).max(y - (oy + s)).max(0.0);
    if r <= 0.0 {
        return if gap_x <= 0.0 && gap_y <= 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let wx = ((r - gap_x) / r).clamp(0.0, 1.0);
    let wy = ((r - gap_y) / r).clamp(0.0, 1.0);
    wx * wy
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate enumeration
// ─────────────────────────────────────────────────────────────────────────────

/// Inclusive cell-index ranges `(col_lo, col_hi, row_lo, row_hi)` of blocks
/// whose `r`-expanded canonical rects may contain `(x, y)`, clamped to the
/// grid.  With `r ≤ s/2` the ranges span at most 2 cells per axis (≤ 4
/// candidates): span = `(s + 2r) / s ≤ 2`.
fn candidate_ranges(x: f64, y: f64, grid: &GridGeometry, r: f64) -> (i64, i64, i64, i64) {
    let s = grid.block_size;
    let col_lo = (((x - r - grid.x_min) / s).floor() as i64).max(0);
    let col_hi = (((x + r - grid.x_min) / s).floor() as i64).min(grid.grid_cols - 1);
    let row_lo = (((y - r - grid.y_min) / s).floor() as i64).max(0);
    let row_hi = (((y + r - grid.y_min) / s).floor() as i64).min(grid.grid_rows - 1);
    (col_lo, col_hi, row_lo, row_hi)
}

// ─────────────────────────────────────────────────────────────────────────────
// Argmax (ties → highest index, matching PointNetClassifier::classify)
// ─────────────────────────────────────────────────────────────────────────────

fn argmax_index<T: PartialOrd>(values: &[T]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused label decision
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the fused class for one original point at `(x, y)`.
///
/// Returns `Some(model_class_index)` when at least one block voted, or `None`
/// when no block's footprint covers the point (caller preserves the original
/// classification / falls back).
///
/// - **Fast path** (fusion disabled, or the point lies ≥ `radius` inside its
///   canonical block — no other block can have a nonzero vote): single
///   canonical-block lookup, argmax of its nearest sample's probability row.
/// - **Fusion path**: accumulate `t_b · p_b / (d_b² + σ²)` over all candidate
///   blocks into `acc` (a caller-owned scratch of length `n_classes`, reused
///   across points to avoid per-point allocation; fully overwritten on every
///   call).  The accumulator is intentionally left unnormalized — argmax is
///   scale-invariant.
///
/// `proximity_sigma` is the proximity bandwidth `σ` in projection units (see
/// [`default_proximity_sigma`]); a degenerate value (non-finite or ≤ 0)
/// falls back to the division guard rather than panicking.
pub fn fused_label<S: BuildHasher>(
    x: f64,
    y: f64,
    inference_map: &HashMap<u64, BlockInferenceResult, S>,
    grid: &GridGeometry,
    fusion_radius: f64,
    proximity_sigma: f64,
    acc: &mut [f64],
) -> Option<usize> {
    let sigma_sq = if proximity_sigma.is_finite() && proximity_sigma > 0.0 {
        proximity_sigma * proximity_sigma
    } else {
        MIN_SIGMA_SQ
    };
    let s = grid.block_size;
    let col = ((x - grid.x_min) / s).floor() as i64;
    let row = ((y - grid.y_min) / s).floor() as i64;

    // ── Fast path: fusion off, or deep inside the canonical block ─────────
    let home_x = grid.x_min + col as f64 * s;
    let home_y = grid.y_min + row as f64 * s;
    let x_inset = (x - home_x).min(home_x + s - x);
    let y_inset = (y - home_y).min(home_y + s - y);
    if fusion_radius <= 0.0 || x_inset.min(y_inset) >= fusion_radius {
        let result = inference_map.get(&block_id(row, col, grid.grid_cols))?;
        let (_d2, probs) = result.nearest_vote(x, y)?;
        return Some(argmax_index(probs));
    }

    // ── Fusion path: weighted soft voting across all covering blocks ──────
    let (col_lo, col_hi, row_lo, row_hi) = candidate_ranges(x, y, grid, fusion_radius);

    for v in acc.iter_mut() {
        *v = 0.0;
    }
    let mut any_votes = false;

    for rr in row_lo..=row_hi {
        for cc in col_lo..=col_hi {
            let Some(result) = inference_map.get(&block_id(rr, cc, grid.grid_cols)) else {
                continue;
            };
            let west = grid.x_min + cc as f64 * s;
            let south = grid.y_min + rr as f64 * s;
            let centrality = centrality_weight(x, y, west, south, s, fusion_radius);
            if centrality <= 0.0 {
                continue;
            }
            let Some((d2, probs)) = result.nearest_vote(x, y) else {
                continue;
            };
            let vote = centrality / (d2 + sigma_sq);
            for (a, p) in acc.iter_mut().zip(probs.iter()) {
                *a += vote * f64::from(*p);
            }
            any_votes = true;
        }
    }

    if any_votes {
        Some(argmax_index(acc))
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// Test assertions compare against exact, exactly-representable constants
// (0.0, 0.25, 0.5, 1.0, …) produced by closed-form weight arithmetic —
// strict float equality is intentional and safe here.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn grid_2x1() -> GridGeometry {
        GridGeometry {
            x_min: 0.0,
            y_min: 0.0,
            block_size: 50.0,
            grid_cols: 2,
            grid_rows: 1,
        }
    }

    /// Build a one-point inference result whose nearest sample is at `(x, y)`
    /// with the given raw logits (softmaxed internally at τ = 1).
    fn one_point_result(x: f64, y: f64, logits: &[f32]) -> BlockInferenceResult {
        let n_classes = logits.len();
        let mat = Array2::from_shape_vec((1, n_classes), logits.to_vec())
            .expect("shape must be valid in tests");
        BlockInferenceResult::from_logits(&[x], &[y], &mat, 1.0).expect("from_logits must succeed")
    }

    // ── DoD #4 — centrality weight behaviour ────────────────────────────────

    #[test]
    fn test_centrality_plateau_inside_and_at_edges() {
        // Deep interior → 1.0
        assert_eq!(centrality_weight(25.0, 25.0, 0.0, 0.0, 50.0, 10.0), 1.0);
        // Exactly on the canonical edge (still inside) → 1.0
        assert_eq!(centrality_weight(50.0, 25.0, 0.0, 0.0, 50.0, 10.0), 1.0);
    }

    #[test]
    fn test_centrality_ramps_linearly_outside() {
        // 5 units beyond the rect with r = 10 → (10 − 5) / 10 = 0.5
        assert_eq!(centrality_weight(55.0, 25.0, 0.0, 0.0, 50.0, 10.0), 0.5);
        // At exactly r beyond → 0
        assert_eq!(centrality_weight(60.0, 25.0, 0.0, 0.0, 50.0, 10.0), 0.0);
        // Beyond r → 0
        assert_eq!(centrality_weight(61.0, 25.0, 0.0, 0.0, 50.0, 10.0), 0.0);
    }

    #[test]
    fn test_centrality_corner_is_axis_product() {
        // 5 beyond on both axes → 0.5 · 0.5 = 0.25
        assert_eq!(centrality_weight(55.0, 55.0, 0.0, 0.0, 50.0, 10.0), 0.25);
    }

    #[test]
    fn test_centrality_continuity_at_rect_edge_and_ramp_toe() {
        // Continuity at the rect edge: inside limit == outside limit == 1
        let inside = centrality_weight(50.0 - 1e-9, 25.0, 0.0, 0.0, 50.0, 10.0);
        let outside = centrality_weight(50.0 + 1e-9, 25.0, 0.0, 0.0, 50.0, 10.0);
        assert!((inside - outside).abs() < 1e-6);
        // Continuity at the ramp toe (distance r): both sides ≈ 0
        let before = centrality_weight(60.0 - 1e-9, 25.0, 0.0, 0.0, 50.0, 10.0);
        let after = centrality_weight(60.0 + 1e-9, 25.0, 0.0, 0.0, 50.0, 10.0);
        assert!((before - after).abs() < 1e-6);
        assert_eq!(after, 0.0);
    }

    #[test]
    fn test_centrality_zero_radius_is_membership() {
        assert_eq!(centrality_weight(25.0, 25.0, 0.0, 0.0, 50.0, 0.0), 1.0);
        assert_eq!(centrality_weight(55.0, 25.0, 0.0, 0.0, 50.0, 0.0), 0.0);
    }

    // ── DoD #5 — candidacy enumeration ──────────────────────────────────────

    #[test]
    fn test_candidate_ranges_at_most_four_for_r_le_half_block() {
        // Arbitrary positions across the grid, r = s/2 (the CLI maximum).
        let grid = grid_2x1();
        for &x in &[0.0, 12.3, 25.0, 37.7, 49.999, 50.0, 62.4, 75.0, 99.9] {
            let (c_lo, c_hi, r_lo, r_hi) = candidate_ranges(x, 25.0, &grid, 25.0);
            assert!(c_hi - c_lo <= 1, "x={x}: cols {c_lo}..={c_hi}");
            assert!(r_hi - r_lo <= 1, "x={x}: rows {r_lo}..={r_hi}");
        }
    }

    #[test]
    fn test_candidate_ranges_hit_both_cells_at_seam_and_clamp_at_borders() {
        let grid = grid_2x1();
        // At the seam x=50 with r=10 → both columns are candidates.
        let (c_lo, c_hi, _, _) = candidate_ranges(50.0, 25.0, &grid, 10.0);
        assert_eq!((c_lo, c_hi), (0, 1));
        // Near the left border x=1 with r=10 → col_lo clamps to 0 (not −1).
        let (c_lo, c_hi, _, _) = candidate_ranges(1.0, 25.0, &grid, 10.0);
        assert_eq!((c_lo, c_hi), (0, 0));
        // Single-row grid: rows always clamp to 0..=0.
        let (_, _, r_lo, r_hi) = candidate_ranges(25.0, 25.0, &grid, 10.0);
        assert_eq!((r_lo, r_hi), (0, 0));
    }

    // ── DoD #2/#6/#7 — fused decisions ─────────────────────────────────────

    /// Two-block fixture: block A (col 0) has one sample at (49, 25) with
    /// weak class-0 logits; block B (col 1) has one sample at (51, 25) with
    /// strong class-1 logits.
    fn two_block_map() -> HashMap<u64, BlockInferenceResult> {
        let mut map = HashMap::new();
        map.insert(0u64, one_point_result(49.0, 25.0, &[0.51, 0.49])); // A: weak class 0
        map.insert(1u64, one_point_result(51.0, 25.0, &[0.05, 0.95])); // B: strong class 1
        map
    }

    #[test]
    fn test_fusion_off_matches_legacy_single_block() {
        let grid = grid_2x1();
        let map = two_block_map();
        let mut acc = vec![0.0f64; 2];
        // Just inside block A's side of the seam (x = 49.999 → col 0):
        // fusion disabled → canonical block A alone decides (weak class 0).
        let label = fused_label(49.999, 25.0, &map, &grid, 0.0, 1.0, &mut acc);
        assert_eq!(label, Some(0));
    }

    #[test]
    fn test_fused_blend_can_flip_weak_canonical_via_confident_neighbour() {
        let grid = grid_2x1();
        let map = two_block_map();
        let mut acc = vec![0.0f64; 2];
        // Same point, r = 10: both blocks vote at ≈ equal weight (both
        // samples are d² ≈ 1 away; B's centrality ≈ 0.9999) — B's confident
        // class-1 distribution dominates the blend and flips the label.
        let label = fused_label(49.999, 25.0, &map, &grid, 10.0, 1.0, &mut acc);
        assert_eq!(label, Some(1));
    }

    #[test]
    fn test_interior_point_unaffected_by_fusion() {
        let grid = grid_2x1();
        let map = two_block_map();
        let mut acc = vec![0.0f64; 2];
        // (25, 25) is 25 units from every edge of A — beyond r → fast path,
        // canonical A's weak class-0 row wins regardless of B.
        assert_eq!(
            fused_label(25.0, 25.0, &map, &grid, 10.0, 1.0, &mut acc),
            Some(0)
        );
    }

    #[test]
    fn test_missing_canonical_with_neighbour_in_range_still_labels() {
        let grid = grid_2x1();
        let mut map = HashMap::new();
        // Only block B exists (block A density-dropped).
        map.insert(1u64, one_point_result(51.0, 25.0, &[0.05, 0.95]));
        let mut acc = vec![0.0f64; 2];
        // (49, 25) is inside missing A but within r of B's rect → B votes.
        let label = fused_label(49.0, 25.0, &map, &grid, 10.0, 1.0, &mut acc);
        assert_eq!(label, Some(1));
    }

    #[test]
    fn test_missing_canonical_deep_interior_returns_none() {
        let grid = grid_2x1();
        let mut map = HashMap::new();
        map.insert(1u64, one_point_result(51.0, 25.0, &[0.05, 0.95]));
        let mut acc = vec![0.0f64; 2];
        // (25, 25) is deep inside the *missing* block A: no neighbour can
        // have a nonzero vote → None (caller preserves original class).
        assert_eq!(
            fused_label(25.0, 25.0, &map, &grid, 10.0, 1.0, &mut acc),
            None
        );
    }

    #[test]
    fn test_empty_map_returns_none() {
        let grid = grid_2x1();
        let map: HashMap<u64, BlockInferenceResult> = HashMap::new();
        let mut acc = vec![0.0f64; 2];
        assert_eq!(
            fused_label(49.0, 25.0, &map, &grid, 10.0, 1.0, &mut acc),
            None
        );
        assert_eq!(
            fused_label(49.0, 25.0, &map, &grid, 0.0, 1.0, &mut acc),
            None
        );
    }

    #[test]
    fn test_proximity_weight_demotes_distant_sample() {
        let grid = grid_2x1();
        let mut map = HashMap::new();
        // A: confident class 0, sample right next to the query.
        map.insert(0u64, one_point_result(49.5, 25.0, &[0.9, 0.1]));
        // B: confident class 1, but its only sample is ~48 units away.
        map.insert(1u64, one_point_result(99.0, 25.0, &[0.1, 0.9]));
        let mut acc = vec![0.0f64; 2];
        // Query at the seam: B is inside its plateau but 48.5² away — the
        // inverse-square proximity term must let A's close sample win.
        assert_eq!(
            fused_label(50.0, 25.0, &map, &grid, 10.0, 1.0, &mut acc),
            Some(0)
        );
    }

    // ── GridGeometry validation ─────────────────────────────────────────────

    #[test]
    fn test_grid_geometry_rejects_missing_grid_fields() {
        let mut manifest = crate::preprocessing::BlockManifest {
            source: "t.las".into(),
            block_size: 50.0,
            target_points: 4,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 1,
            crs_epsg: None,
            grid_cols: 0, // predates grid fields
            grid_rows: 0,
            grid_x_min: 0.0,
            grid_y_min: 0.0,
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap: 0.0,
            oversample_jitter: 0.0,
            z_norm_use_block_relative: false,
            halo_fraction: 0.0,
            blocks: vec![],
        };
        assert!(GridGeometry::from_manifest(&manifest).is_err());
        manifest.grid_cols = 2;
        manifest.grid_rows = 1;
        assert!(GridGeometry::from_manifest(&manifest).is_ok());
    }
}
