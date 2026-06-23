#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
//! Per-point feature extraction: assembles the feature vector
//! by combining scalar attributes (from `normalizer`) with eigenvalue-derived
//! structural features at one or more search radii.
//!
//! ## Feature index layout (N search radii)
//! | Idx range | Name group      | Source                  |
//! |-----------|-----------------|-------------------------|
//! | 0–6       | scalar          | normalizer               |
//! | 7–11      | eigenvalue @ r₀ | covariance @ radius 0    |
//! | 12–16     | eigenvalue @ r₁ | covariance @ radius 1    |
//! | …         | …               | …                        |
//!
//! Each 5-element eigenvalue block: linearity, planarity, sphericity,
//! omnivariance, curvature.
//!
//! **Single-scale mode** (`search_radii.len() == 1`) uses adaptive radius
//! expansion (up to `radius × 4`) to handle sparse blocks robustly — identical
//! to the Stage 01 behaviour.
//!
//! **Multi-scale mode** (`search_radii.len() > 1`) uses a fixed radius per
//! scale.  Adaptive expansion is disabled so each scale remains faithful to
//! its intended neighbourhood size.  Degenerate cases (fewer than 3 neighbours)
//! fall back to `[0.0; 5]`.

use nalgebra::{Matrix3, linalg::SymmetricEigen};
use wblidar::PointRecord;

use crate::preprocessing::spatial_index::BlockSpatialIndex;
use crate::preprocessing::normalizer::{compute_hag, normalise_scalar_features, DtmView};

/// Extract the full feature vector for every point in `pts`.
///
/// Returns one `Vec<f32>` per sampled point.  The length of each row is
/// `7 + 5 × search_radii.len()`.  When `search_radii` has a single entry,
/// the output is identical to the Stage 01 12-feature baseline.
///
/// # Parameters
/// - `pts`          — sampled point records for this block
/// - `all_pts`      — *full unsampled* block points (used for k-d tree queries)
/// - `index`        — k-d tree built from `all_pts`
/// - `dtm`          — optional DTM view for HAG; `None` → block-min-z proxy
/// - `origin_x/y`   — block south-west corner
/// - `block_size`   — cell edge length
/// - `search_radii` — sorted ascending list of eigenvalue search radii
/// - `min_neighbors`— adaptive expansion threshold (single-scale mode only)
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
    search_radii: &[f64],
    min_neighbors: usize,
) -> Vec<Vec<f32>> {
    let multi_scale = search_radii.len() > 1;

    // Step 1: HAG
    let hag_values = compute_hag(pts, dtm);

    // Step 2: Scalar features (indices 0–6)
    let scalar = normalise_scalar_features(pts, origin_x, origin_y, block_size, &hag_values);

    // Step 3: Eigenvalue features per radius.
    // Single-scale: adaptive radius expansion (Stage 01 behaviour preserved).
    // Multi-scale:  fixed radius per scale to keep scales distinct.
    let n_radii = search_radii.len().max(1);
    let row_len = 7 + 5 * n_radii;

    scalar
        .into_iter()
        .zip(pts.iter())
        .map(|(s, pt)| {
            let center = [pt.x, pt.y, pt.z];
            let mut row = Vec::with_capacity(row_len);
            row.extend_from_slice(&s);  // indices 0–6

            for &radius in search_radii {
                let indices = if multi_scale {
                    // Fixed radius: preserve scale fidelity.
                    index.radius_search(center, radius)
                } else {
                    // Adaptive expansion: robustness on sparse blocks.
                    index.adaptive_radius_search(center, radius, min_neighbors)
                };
                let eig = eigenvalue_features(all_pts, &indices);
                row.extend_from_slice(&eig);
            }
            row
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
    // usize→f64: lossless for any plausible neighbourhood size (< 2^53).
    #[allow(clippy::cast_precision_loss)]
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

    // f64→f32 precision loss is intentional: all values are clamped to [0,1]
    // or are the cube-root of a product of small eigenvalues.
    #[allow(clippy::cast_precision_loss)]
    let linearity   = ((l1 - l2) / l1).clamp(0.0, 1.0) as f32;
    #[allow(clippy::cast_precision_loss)]
    let planarity   = ((l2 - l3) / l1).clamp(0.0, 1.0) as f32;
    #[allow(clippy::cast_precision_loss)]
    let sphericity  = (l3 / l1).clamp(0.0, 1.0) as f32;
    #[allow(clippy::cast_precision_loss)]
    let omnivariance = (l1 * l2 * l3).cbrt() as f32;
    let sum = l1 + l2 + l3;
    #[allow(clippy::cast_precision_loss)]
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

    /// Full pipeline sanity: single-scale extract_features produces N_FEATURES
    /// values per point, all scalar features are within [0, 1].
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
            &[5.0],   // single radius
            3,
        );

        assert_eq!(feats.len(), all_pts.len());
        for row in &feats {
            assert_eq!(row.len(), crate::preprocessing::N_FEATURES, "single-radius should produce 12 features");
            // Scalar features (0–6) should be in [0,1]
            for &v in &row[0..7] {
                assert!((0.0..=1.0).contains(&v), "scalar feature out of range: {v}");
            }
            // omnivariance (index 10) can be > 1, just check it's non-negative
            assert!(row[10] >= 0.0, "omnivariance must be non-negative: {}", row[10]);
        }
    }

    /// Multi-scale: 3 radii → 7 + 5×3 = 22 features per point.
    #[test]
    fn test_extract_features_multi_scale_width() {
        let all_pts: Vec<PointRecord> = (0..30)
            .map(|i| pt(i as f64 * 1.5, (i % 5) as f64, (i % 3) as f64 * 0.2))
            .collect();

        let index = BlockSpatialIndex::build(&all_pts);
        let feats = extract_features(
            &all_pts,
            &all_pts,
            &index,
            None,
            0.0, 0.0,
            50.0,
            &[1.0, 3.0, 6.0],  // 3 radii
            3,
        );

        let expected_n = crate::preprocessing::n_features_for_radii(3); // 22
        assert_eq!(feats.len(), all_pts.len());
        for row in &feats {
            assert_eq!(row.len(), expected_n, "3-radius should produce 22 features");
        }
    }

    /// Single-radius extract_features output matches the legacy N_FEATURES baseline.
    #[test]
    fn test_extract_features_single_radius_matches_legacy_width() {
        let all_pts: Vec<PointRecord> = (0..10)
            .map(|i| pt(i as f64, 0.0, 0.0))
            .collect();
        let index = BlockSpatialIndex::build(&all_pts);

        // Single radius in the Vec form
        let feats_new = extract_features(&all_pts, &all_pts, &index, None, 0.0, 0.0, 50.0, &[2.0], 3);
        // All rows should be exactly N_FEATURES wide
        for row in &feats_new {
            assert_eq!(row.len(), crate::preprocessing::N_FEATURES);
        }
    }
}
