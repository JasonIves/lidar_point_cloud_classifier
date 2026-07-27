#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
//! Per-point feature extraction: assembles the feature vector
//! by combining scalar attributes (from `normalizer`) with the 10
//! eigenvalue-derived structural features precomputed by the whole-file
//! `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass (Stage 30, Step
//! 5e+5f+5g).
//!
//! ## Feature index layout
//! | Idx range | Name group | Source                                   |
//! |-----------|------------|-------------------------------------------|
//! | 0–6       | scalar     | normalizer                                 |
//! | 7–16      | eigenvalue | `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass |
//!
//! The 10-element eigenvalue block (in order): `lambda1, lambda2, lambda3,
//! linearity, planarity, sphericity, omnivariance, eigentropy, slope,
//! residual`.
//!
//! Prior to Stage 30, this module computed 5 eigenvalue features locally per
//! point via an in-process k-d tree radius search, repeated once per
//! configured search radius (multi-scale). That local computation has been
//! entirely replaced: eigenvalue features are now supplied by the caller
//! (`pipeline.rs`), sourced from a single whole-file pre-pass table joined
//! by point index. This module's sole remaining responsibility is combining
//! those precomputed rows with the scalar features computed here.
//!
//! **Stage 31**: operates on the lean, project-local [`LitePoint`] rather
//! than `wblidar::PointRecord` — see `docs/stages/stage-31-lean-point-record.md`.

use crate::preprocessing::lite_point::LitePoint;
use crate::preprocessing::normalizer::{
    compute_hag, normalise_scalar_features, DtmView, HagNormalization, ZNormalization,
};

/// Extract the full feature vector for every point in `pts`.
///
/// Returns one `Vec<f32>` per point, each of length
/// `crate::preprocessing::N_FEATURES` (currently 17: 7 scalar + 10
/// eigenvalue).
///
/// # Parameters
/// - `pts`        — sampled point records for this block
/// - `eigen_rows` — precomputed 10-value eigenvalue feature rows, one per
///   entry in `pts`, in the same order (looked up by the caller from the
///   whole-file pre-pass table via each point's original stream index —
///   see `block_partitioner::Block::point_indices`).
/// - `dtm`        — optional DTM view for HAG; `None` → block-min-z proxy
/// - `origin_x/y` — block south-west corner
/// - `block_size` — cell edge length
/// - `hag_norm`   — HAG normalisation strategy (Stage 37); forwarded to
///   `normalise_scalar_features`. Default is a fixed absolute metre reference.
/// - `z_norm_strategy` — Z-elevation normalisation strategy (`z_norm` bug fix,
///   follow-up to Stage 37); forwarded to `normalise_scalar_features`.
///   Default is a whole-file absolute elevation range sourced from the
///   LAS/LAZ/COPC header (see `pipeline.rs`).
///
/// # Panics
/// This function does not panic, but callers must ensure
/// `eigen_rows.len() == pts.len()`; a length mismatch will cause rows to be
/// truncated or missing eigenvalue data would be zipped incorrectly (the
/// `zip` below silently stops at the shorter of the two iterators).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn extract_features(
    pts: &[LitePoint],
    eigen_rows: &[[f32; 10]],
    dtm: Option<&DtmView>,
    origin_x: f64,
    origin_y: f64,
    block_size: f64,
    hag_norm: HagNormalization,
    z_norm_strategy: ZNormalization,
) -> Vec<Vec<f32>> {
    // Step 1: HAG
    let hag_values = compute_hag(pts, dtm);

    // Step 2: Scalar features (indices 0–6)
    let scalar = normalise_scalar_features(
        pts,
        origin_x,
        origin_y,
        block_size,
        &hag_values,
        hag_norm,
        z_norm_strategy,
    );

    // Step 3: Combine scalar + precomputed eigenvalue rows (indices 7–16).
    scalar
        .into_iter()
        .zip(eigen_rows.iter())
        .map(|(s, eig)| {
            let mut row = Vec::with_capacity(crate::preprocessing::N_FEATURES);
            row.extend_from_slice(&s);
            row.extend_from_slice(eig);
            row
        })
        .collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64, z: f64) -> LitePoint {
        LitePoint {
            x,
            y,
            z,
            intensity: 32767,
            return_number: 1,
            number_of_returns: 1,
            ..LitePoint::default()
        }
    }

    /// Full pipeline sanity: `extract_features` produces `N_FEATURES` values
    /// per point, and scalar features are within `[0, 1]`.
    // Test fixture intensity generation from a small index range (0..20);
    // sign loss is irrelevant since `i` is always non-negative here.
    #[allow(clippy::cast_sign_loss)]
    #[test]
    fn test_extract_features_output_shape_and_range() {
        let pts: Vec<LitePoint> = (0..20)
            .map(|i| {
                let mut p = pt(f64::from(i) * 2.0, f64::from(i) * 2.0, f64::from(i) * 0.5);
                p.intensity = (i * 3000) as u16;

                p.return_number = 1;
                p.number_of_returns = 2;
                p
            })
            .collect();

        // Fabricate a plausible eigen row per point (not physically
        // meaningful, just exercising shape/range checks).
        let eigen_rows: Vec<[f32; 10]> = (0..pts.len())
            .map(|i| {
                let mut r = [0.0f32; 10];
                r[3] = 0.5; // linearity
                r[4] = 0.3; // planarity
                r[5] = 0.2; // sphericity
                r[6] = (i as f32) * 0.01; // omnivariance (non-negative)
                r
            })
            .collect();

        let feats = extract_features(
            &pts,
            &eigen_rows,
            None,
            0.0,
            0.0,
            50.0,
            HagNormalization::default(),
            ZNormalization::Global {
                z_min: 0.0,
                z_max: 10.0,
            },
        );

        assert_eq!(feats.len(), pts.len());

        for row in &feats {
            assert_eq!(
                row.len(),
                crate::preprocessing::N_FEATURES,
                "row should have N_FEATURES entries"
            );
            // Scalar features (0–6) should be in [0,1]
            for &v in &row[0..7] {
                assert!((0.0..=1.0).contains(&v), "scalar feature out of range: {v}");
            }
            // omnivariance (index 7 + 6 = 13) should be non-negative in our fixture
            assert!(
                row[13] >= 0.0,
                "omnivariance must be non-negative: {}",
                row[13]
            );
        }
    }

    /// Eigenvalue rows are passed through unchanged into the output row's
    /// tail (indices 7..17).
    #[test]
    fn test_extract_features_eigen_passthrough() {
        let pts: Vec<LitePoint> = (0..5).map(|i| pt(f64::from(i), 0.0, 0.0)).collect();
        let eigen_rows: Vec<[f32; 10]> = (0..pts.len())
            .map(|i| {
                let mut r = [0.0f32; 10];
                for (j, v) in r.iter_mut().enumerate() {
                    *v = (i * 10 + j) as f32;
                }
                r
            })
            .collect();

        let feats = extract_features(
            &pts,
            &eigen_rows,
            None,
            0.0,
            0.0,
            50.0,
            HagNormalization::default(),
            ZNormalization::Global {
                z_min: 0.0,
                z_max: 10.0,
            },
        );
        for (row, eig) in feats.iter().zip(eigen_rows.iter()) {
            assert_eq!(&row[7..17], eig.as_slice());
        }
    }
}
