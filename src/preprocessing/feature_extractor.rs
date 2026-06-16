#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
//! Per-point feature extraction: assembles the full 12-element feature vector
//! by combining scalar attributes (from `normalizer`) with eigenvalue-derived
//! structural features.
//!
//! ## Feature index layout
//! | Idx | Name            | Source           |
//! |-----|-----------------|------------------|
//! |  0  | `x_norm`        | normalizer       |
//! |  1  | `y_norm`        | normalizer       |
//! |  2  | `z_norm`        | normalizer       |
//! |  3  | `intensity_norm`| normalizer       |
//! |  4  | `return_ratio`  | normalizer       |
//! |  5  | `scan_angle_norm`| normalizer      |
//! |  6  | `hag`           | normalizer       |
//! |  7  | `linearity`     | eigenvalue       |
//! |  8  | `planarity`     | eigenvalue       |
//! |  9  | `sphericity`    | eigenvalue       |
//! | 10  | `omnivariance`  | eigenvalue       |
//! | 11  | `curvature`     | eigenvalue       |

use nalgebra::{Matrix3, linalg::SymmetricEigen};
use wblidar::PointRecord;

use crate::preprocessing::{spatial_index::BlockSpatialIndex, N_FEATURES};
use crate::preprocessing::normalizer::{compute_hag, normalise_scalar_features, DtmView};

/// Extract the full `[f32; N_FEATURES]` vector for every point in `pts`.
///
/// # Parameters
/// - `pts`          — sampled point records for this block
/// - `all_pts`      — *full unsampled* block points (used for k-d tree queries)
/// - `index`        — k-d tree built from `all_pts`
/// - `dtm`          — optional DTM view for HAG; `None` → block-min-z proxy
/// - `origin_x/y`   — block south-west corner
/// - `block_size`   — cell edge length
/// - `search_radius`— base neighbourhood radius
/// - `min_neighbors`— adaptive expansion threshold
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn extract_features(
    pts: &[PointRecord],
    all_pts: &[PointRecord],
    index: &BlockSpatialIndex,
    dtm: Option<&DtmView>,
    origin_x: f64,
    origin_y: f64,
    block_size: f64,
    search_radius: f64,
    min_neighbors: usize,
) -> Vec<[f32; N_FEATURES]> {
    // Step 1: HAG
    let hag_values = compute_hag(pts, dtm);

    // Step 2: Scalar features (indices 0–6)
    let scalar = normalise_scalar_features(pts, origin_x, origin_y, block_size, &hag_values);

    // Step 3: Eigenvalue features (indices 7–11) — one neighbourhood query per point
    let eigen: Vec<[f32; 5]> = pts
        .iter()
        .map(|pt| {
            let center = [pt.x, pt.y, pt.z];
            let neighbor_indices =
                index.adaptive_radius_search(center, search_radius, min_neighbors);
            eigenvalue_features(all_pts, &neighbor_indices)
        })
        .collect();

    // Step 4: Assemble into full feature rows
    scalar
        .into_iter()
        .zip(eigen)
        .map(|(s, e)| {
            [s[0], s[1], s[2], s[3], s[4], s[5], s[6],
             e[0], e[1], e[2], e[3], e[4]]
        })
        .collect()
}

/// Compute the 5 eigenvalue-derived structural features for a point given its
/// neighbourhood (identified by `indices` into `all_pts`).
///
/// If fewer than 3 neighbours are found (degenerate case), returns `[0.0; 5]`.
///
/// Features returned in order: linearity, planarity, sphericity, omnivariance, curvature.
fn eigenvalue_features(all_pts: &[PointRecord], indices: &[usize]) -> [f32; 5] {
    if indices.len() < 3 {
        return [0.0; 5];
    }

    // Build 3×3 covariance matrix from neighbour coordinates.
    let n = indices.len() as f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_z = 0.0_f64;

    for &i in indices {
        let p = &all_pts[i];
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.z;
    }
    let cx = sum_x / n;
    let cy = sum_y / n;
    let cz = sum_z / n;

    let mut cxx = 0.0_f64;
    let mut cxy = 0.0_f64;
    let mut cxz = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cyz = 0.0_f64;
    let mut czz = 0.0_f64;

    for &i in indices {
        let p = &all_pts[i];
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dz = p.z - cz;
        cxx += dx * dx;
        cxy += dx * dy;
        cxz += dx * dz;
        cyy += dy * dy;
        cyz += dy * dz;
        czz += dz * dz;
    }
    cxx /= n;
    cxy /= n;
    cxz /= n;
    cyy /= n;
    cyz /= n;
    czz /= n;

    let cov = Matrix3::new(
        cxx, cxy, cxz,
        cxy, cyy, cyz,
        cxz, cyz, czz,
    );

    let eig = SymmetricEigen::new(cov);
    // nalgebra returns eigenvalues in ascending order; we want λ1 ≥ λ2 ≥ λ3.
    let mut lambdas = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
    lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let l1 = lambdas[0].max(0.0);
    let l2 = lambdas[1].max(0.0);
    let l3 = lambdas[2].max(0.0);

    if l1 < 1e-12 {
        // All eigenvalues effectively zero → structureless neighbourhood
        return [0.0; 5];
    }

    let linearity   = ((l1 - l2) / l1).clamp(0.0, 1.0) as f32;
    let planarity   = ((l2 - l3) / l1).clamp(0.0, 1.0) as f32;
    let sphericity  = (l3 / l1).clamp(0.0, 1.0) as f32;
    let omnivariance = (l1 * l2 * l3).cbrt() as f32;
    let sum = l1 + l2 + l3;
    let curvature   = if sum < 1e-12 { 0.0 } else { (l3 / sum).clamp(0.0, 1.0) as f32 };

    [linearity, planarity, sphericity, omnivariance, curvature]
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64, z: f64) -> PointRecord {
        let mut p = PointRecord::default();
        p.x = x; p.y = y; p.z = z;
        p.intensity = 32767;
        p.return_number = 1;
        p.number_of_returns = 1;
        p
    }

    /// Hand-calculate expected eigenvalue features for a known 5-point cloud.
    ///
    /// Points lie perfectly in the XY plane at z=0 → λ3 = 0 exactly.
    /// Expected: sphericity = 0, curvature = 0, planarity > 0, linearity > 0.
    #[test]
    fn test_eigenvalue_features_planar_cloud() {
        let pts = vec![
            pt(-1.0, 0.0, 0.0),
            pt( 1.0, 0.0, 0.0),
            pt( 0.0,-1.0, 0.0),
            pt( 0.0, 1.0, 0.0),
            pt( 0.0, 0.0, 0.0), // centroid
        ];
        let indices: Vec<usize> = (0..pts.len()).collect();
        let feats = eigenvalue_features(&pts, &indices);

        // λ3 ≈ 0 → sphericity ≈ 0, curvature ≈ 0
        assert!(feats[2] < 0.01, "sphericity should be ~0 for planar cloud, got {}", feats[2]);
        assert!(feats[4] < 0.01, "curvature should be ~0 for planar cloud, got {}", feats[4]);
        // Planarity should be non-trivial (l2 > l3)
        assert!(feats[1] > 0.0, "planarity should be positive, got {}", feats[1]);
    }

    /// A perfectly linear cloud along the X axis.
    /// Expected: linearity ≈ 1, planarity ≈ 0, sphericity ≈ 0.
    #[test]
    fn test_eigenvalue_features_linear_cloud() {
        let pts: Vec<PointRecord> = (-5..=5)
            .map(|i| pt(i as f64, 0.0, 0.0))
            .collect();
        let indices: Vec<usize> = (0..pts.len()).collect();
        let feats = eigenvalue_features(&pts, &indices);

        assert!(feats[0] > 0.9, "linearity should be ~1 for linear cloud, got {}", feats[0]);
        assert!(feats[1] < 0.1, "planarity should be ~0 for linear cloud, got {}", feats[1]);
        assert!(feats[2] < 0.1, "sphericity should be ~0 for linear cloud, got {}", feats[2]);
    }

    /// Degenerate case: fewer than 3 neighbours → all-zero features.
    #[test]
    fn test_eigenvalue_features_degenerate() {
        let pts = vec![pt(0.0, 0.0, 0.0), pt(1.0, 0.0, 0.0)];
        let feats = eigenvalue_features(&pts, &[0, 1]);
        assert_eq!(feats, [0.0; 5]);
    }

    /// Full pipeline sanity: extract_features produces N_FEATURES values per point,
    /// all scalar features are within [0, 1].
    #[test]
    fn test_extract_features_output_shape_and_range() {
        let all_pts: Vec<PointRecord> = (0..20)
            .map(|i| {
                let mut p = pt(i as f64 * 2.0, i as f64 * 2.0, i as f64 * 0.5);
                p.intensity = (i * 3000) as u16;
                p.return_number = 1;
                p.number_of_returns = 2;
                p
            })
            .collect();

        let index = BlockSpatialIndex::build(&all_pts);
        let feats = extract_features(
            &all_pts,
            &all_pts,
            &index,
            None,
            0.0, 0.0,
            50.0,
            5.0,
            3,
        );

        assert_eq!(feats.len(), all_pts.len());
        for row in &feats {
            assert_eq!(row.len(), N_FEATURES);
            // Scalar features (0–6) and ratio features (7–9, 11) should be in [0,1]
            for &v in &row[0..7] {
                assert!((0.0..=1.0).contains(&v), "scalar feature out of range: {v}");
            }
            // omnivariance (index 10) can be > 1, just check it's non-negative
            assert!(row[10] >= 0.0, "omnivariance must be non-negative: {}", row[10]);
        }
    }
}
