//! Local elevation residual outlier filter.
//!
//! ## Algorithm
//! This is a faithful in-process reimplementation of the algorithm from
//! `wbtools_oss::tools::LidarRemoveOutliersTool` (see
//! `whitebox_next_gen/crates/wbtools_oss/src/tools/lidar_processing/mod.rs`).
//!
//! For each point:
//! 1. Query all neighbours within `radius` in the XY plane using a 2-D k-d tree.
//! 2. Compute the mean or median Z of those neighbours (controlled by `use_median`).
//! 3. Compute the residual: `|point.z − neighbour_baseline_z|`.
//! 4. Remove the point if `residual >= elev_diff`.
//!
//! Points that have no neighbours are kept (residual = 0 by convention).
//!
//! ## Revert note (2026-06-17)
//! `wbtools_oss` and `wbcore` were removed from `Cargo.toml` to eliminate their
//! heavy transitive dependency tree (~25 extra crates, ~7-minute cold build).
//! This file replaces the only algorithm we used from those crates.
//!
//! **To revert to the original `LidarRemoveOutliersTool`:**
//! 1. Re-enable `wbcore` and `wbtools_oss` in `Cargo.toml`.
//! 2. Delete this file.
//! 3. Remove the `pub mod outlier_filter;` declaration from `preprocessing/mod.rs`.
//! 4. Restore `run_outlier_removal()` in `preprocessing/pipeline.rs` (see
//!    `docs/stages/stage-04-outlier-removal.md` for the original implementation).
//! 5. Revert Step 1b in `pipeline.rs::run_internal` to the temp-file path.

#![allow(clippy::cast_precision_loss)]

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use rayon::prelude::*;
use wblidar::PointRecord;

/// Apply local elevation residual outlier filter to a slice of points.
///
/// Returns a new `Vec<PointRecord>` containing only the points whose Z value
/// does not deviate from their neighbourhood baseline by `>= elev_diff`.
///
/// # Parameters
/// - `pts`        — input point set (the full raw block or file)
/// - `radius`     — XY search radius (projection units)
/// - `elev_diff`  — absolute elevation residual threshold; points with
///   `|z − baseline| >= elev_diff` are removed
/// - `use_median` — use neighbourhood median Z instead of mean
///
/// # Panics
/// Does not panic; degenerate inputs (empty slice, NaN coordinates) are
/// handled gracefully.
#[must_use]
pub fn apply(
    pts: &[PointRecord],
    radius: f64,
    elev_diff: f64,
    use_median: bool,
) -> Vec<PointRecord> {
    if pts.is_empty() {
        return Vec::new();
    }

    // Build a 2-D XY k-d tree once for the whole point set.
    let mut tree: KdTree<f64, usize, [f64; 2]> = KdTree::new(2);
    for (i, p) in pts.iter().enumerate() {
        // Skip NaN coordinates silently — consistent with wbtools_oss impl.
        if p.x.is_finite() && p.y.is_finite() {
            let _ = tree.add([p.x, p.y], i);
        }
    }

    let radius_sq = radius * radius;

    // Compute per-point elevation residuals in parallel.
    let residuals: Vec<f64> = pts
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let neighbours = tree
                .within(&[p.x, p.y], radius_sq, &squared_euclidean)
                .unwrap_or_default();

            let mut z_vals: Vec<f64> = neighbours
                .iter()
                .filter(|(_, &idx)| idx != i)
                .map(|(_, &idx)| pts[idx].z)
                .collect();

            if z_vals.is_empty() {
                // No neighbours → keep the point (residual = 0).
                return 0.0;
            }

            let baseline = if use_median {
                z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = z_vals.len();
                if n % 2 == 1 {
                    z_vals[n / 2]
                } else {
                    // Use f64 arithmetic to avoid integer overflow on midpoint.
                    let lo = z_vals[n / 2 - 1];
                    let hi = z_vals[n / 2];
                    lo + (hi - lo) / 2.0
                }
            } else {
                z_vals.iter().sum::<f64>() / z_vals.len() as f64
            };

            (p.z - baseline).abs()
        })
        .collect();

    // Retain only points below the threshold.
    pts.iter()
        .zip(residuals.iter())
        .filter(|(_, &r)| r < elev_diff)
        .map(|(p, _)| *p)
        .collect()
}
