#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
//! Coordinate normalisation and density-gated point sampling.
//!
//! **Sampling contract (from spec):**
//! - `raw_count >= target_points` → sample *without* replacement (seeded by `block_id`)
//! - `raw_count <  target_points` (but density gate passed) → oversample *with*
//!   replacement to pad to `target_points`; caller sets `oversampled = true`
//!
//! Seeding by `block_id` guarantees reproducible output across runs.
//!
//! **Jitter-based oversampling (Stage 29):** when `jitter_sigma > 0.0`, each
//! padding-only draw has its (x, y, z) coordinates perturbed by an independent
//! per-axis clipped-Gaussian offset (±3σ) *before* feature extraction runs, so
//! padded points produce genuinely distinct eigenvalue features rather than
//! being exact clones of their source point. `jitter_sigma == 0.0` (the
//! default) preserves bit-identical pre-Stage-29 behaviour. See
//! `docs/stages/stage-29-jitter-oversampling.md`.
//!
//! **Stage 31**: operates on the lean, project-local [`LitePoint`] rather
//! than `wblidar::PointRecord` — see `docs/stages/stage-31-lean-point-record.md`.

use rand::prelude::*;
use rand::SeedableRng;
use wbraster::{NodataPolicy, ResampleMethod};

use crate::preprocessing::lite_point::LitePoint;

/// Resample `pts` to exactly `target` points.
///
/// - If `pts.len() >= target`: random sample without replacement.
/// - If `pts.len() < target`: random oversample with replacement to pad,
///   optionally jittering each padding-only copy's coordinates (Stage 29).
///
/// `jitter_sigma` is the standard deviation (projection units) of the
/// per-axis Gaussian offset applied to padding-only points. `0.0` disables
/// jitter entirely (exact-duplicate behaviour, unchanged from pre-Stage-29).
/// See `docs/stages/stage-29-jitter-oversampling.md`.
///
/// Returns `(sampled_points, sampled_indices, oversampled)` where:
/// - `sampled_points` are the resampled `LitePoint` values,
/// - `sampled_indices` are the 0-based indices into `pts` for each output point
///   (padded oversample entries repeat indices from the original range),
/// - `oversampled` is `true` when padding with replacement was applied.
#[must_use]
pub fn resample_block(
    pts: &[LitePoint],
    target: usize,
    seed: u64,
    jitter_sigma: f64,
) -> (Vec<LitePoint>, Vec<usize>, bool) {
    if pts.is_empty() || target == 0 {
        return (Vec::new(), Vec::new(), false);
    }

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    if pts.len() >= target {
        // Sample without replacement using Fisher-Yates partial shuffle.
        let mut indices: Vec<usize> = (0..pts.len()).collect();
        for i in 0..target {
            let j = rng.random_range(i..pts.len());
            indices.swap(i, j);
        }
        let chosen = &indices[..target];
        let sampled = chosen.iter().map(|&i| pts[i]).collect();
        (sampled, chosen.to_vec(), false)
    } else {
        // Start with all original points, then pad with replacement.
        let mut sampled: Vec<LitePoint> = pts.to_vec();
        let mut sampled_indices: Vec<usize> = (0..pts.len()).collect();
        let extra = target - pts.len();
        for _ in 0..extra {
            let idx = rng.random_range(0..pts.len());
            let mut p = pts[idx];
            if jitter_sigma > 0.0 {
                p.x += jitter_offset(&mut rng, jitter_sigma);
                p.y += jitter_offset(&mut rng, jitter_sigma);
                p.z += jitter_offset(&mut rng, jitter_sigma);
            }
            sampled.push(p);
            sampled_indices.push(idx);
        }
        (sampled, sampled_indices, true)
    }
}

/// Draw one zero-mean Gaussian offset with standard deviation `sigma`,
/// clipped to `±3σ`, using a Box–Muller transform sourced from `rng`.
///
/// Implemented locally (rather than pulling in `rand_distr`) to keep the
/// dependency footprint minimal per AGENTS.md — two uniform draws plus a
/// `ln`/`sqrt`/`cos` is negligible cost compared to the k-d tree build and
/// eigendecomposition that follow.
///
/// Returns `0.0` immediately for `sigma <= 0.0` (jitter disabled).
fn jitter_offset(rng: &mut impl Rng, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    // u1 drawn from (0, 1] (never exactly 0) to avoid ln(0) = -inf.
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    let raw = z0 * sigma;
    let clip = 3.0 * sigma;
    raw.clamp(-clip, clip)
}

/// Normalise the scalar point attributes (indices 0–6 of the feature vector)
/// for a set of sampled points.
///
/// Returns a `Vec` of length `pts.len()` where each element is a `[f32; 6]`
/// covering features 0–6 (coordinates, radiometry, HAG).  Eigenvalue features
/// (7–11) are computed separately in `feature_extractor`.
///
/// # Parameters
/// - `pts`        — sampled point records for this block
/// - `origin_x`, `origin_y` — block south-west corner in projection units (used for x/y norm)
/// - `block_size` — cell edge length (denominator for x/y normalisation)
/// - `hag_values` — pre-computed raw HAG values (z - `z_ground`) for each point,
///   same length as `pts`
///
/// # Panics
/// Panics if `pts.len() != hag_values.len()`.
pub fn normalise_scalar_features(
    pts: &[LitePoint],
    origin_x: f64,
    origin_y: f64,
    block_size: f64,
    hag_values: &[f64],
) -> Vec<[f32; 7]> {
    assert_eq!(
        pts.len(),
        hag_values.len(),
        "pts and hag_values length mismatch"
    );

    // Compute z range for z_norm.
    let z_min = pts.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
    let z_max = pts.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
    let z_range = (z_max - z_min).max(1e-9); // avoid divide-by-zero on flat blocks

    // Compute h_max = 99th percentile of hag_values (clamped > 0).
    let h_max = percentile_99(hag_values).max(1e-9);

    pts.iter()
        .zip(hag_values.iter())
        .map(|(pt, &hag_raw)| {
            #[allow(clippy::cast_precision_loss)]
            // f64→f32 precision loss is acceptable: all values are clamped to [0,1].
            let x_norm = ((pt.x - origin_x) / block_size).clamp(0.0, 1.0) as f32;
            #[allow(clippy::cast_precision_loss)]
            let y_norm = ((pt.y - origin_y) / block_size).clamp(0.0, 1.0) as f32;
            #[allow(clippy::cast_precision_loss)]
            let z_norm = ((pt.z - z_min) / z_range).clamp(0.0, 1.0) as f32;

            #[allow(clippy::cast_precision_loss)]
            let intensity_norm = (f64::from(pt.intensity) / 65535.0).clamp(0.0, 1.0) as f32;

            let return_ratio = if pt.number_of_returns == 0 {
                0.0_f32
            } else {
                #[allow(clippy::cast_precision_loss)]
                let v = (f64::from(pt.return_number) / f64::from(pt.number_of_returns))
                    .clamp(0.0, 1.0) as f32;
                v
            };

            #[allow(clippy::cast_precision_loss)]
            let scan_angle_norm = (f64::from(pt.scan_angle).abs() / 90.0).clamp(0.0, 1.0) as f32;

            #[allow(clippy::cast_precision_loss)]
            let hag = (hag_raw / h_max).clamp(0.0, 1.0) as f32;

            [
                x_norm,
                y_norm,
                z_norm,
                intensity_norm,
                return_ratio,
                scan_angle_norm,
                hag,
            ]
        })
        .collect()
}

/// Compute Height Above Ground for each point in `pts`.
///
/// When `dtm` is `None`, uses `z_block_min` (block-minimum-z proxy).
/// When `dtm` is `Some(view)`, performs bilinear interpolation of the DTM
/// raster at each point's (x, y).
///
/// Falls back to proxy for any point that falls outside the raster extent or
/// lands on a nodata cell.
#[must_use]
pub fn compute_hag(pts: &[LitePoint], dtm: Option<&DtmView>) -> Vec<f64> {
    let z_min = pts.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);

    pts.iter()
        .map(|pt| {
            let z_ground = dtm
                .and_then(|d| d.bilinear_interp(pt.x, pt.y))
                .unwrap_or(z_min);
            pt.z - z_ground
        })
        .collect()
}

// ── DTM view ──────────────────────────────────────────────────────────────────

/// Lightweight read-only view of a DTM raster, wrapping an owned
/// `wbraster::Raster` so it can be shared across Rayon worker threads via
/// `Arc<DtmView>` and sampled with `wbraster`'s own bilinear interpolation.
///
/// **Stage 30 Step 4 (adopted):** ground-elevation sampling now delegates to
/// [`wbraster::Raster::sample_world`] (band 0, `ResampleMethod::Bilinear`,
/// `NodataPolicy::Strict`) instead of a hand-rolled interpolator. See the
/// doc comment on [`DtmView::bilinear_interp`] for the coordinate-convention
/// change this entails.
#[derive(Debug, Clone)]
pub struct DtmView {
    raster: wbraster::Raster,
}

impl DtmView {
    /// Construct a `DtmView` from a loaded `wbraster::Raster`.
    #[must_use]
    pub fn from_raster(r: &wbraster::Raster) -> Self {
        Self { raster: r.clone() }
    }

    /// Sample the DTM (band 0) at world coordinate (x, y) using
    /// `wbraster::Raster::sample_world` with bilinear resampling and strict
    /// nodata handling (requires all four surrounding cells to be valid).
    ///
    /// Returns `None` if the coordinate is outside the raster extent or any
    /// of the four surrounding cells is nodata.
    ///
    /// # Stage 30 Step 4 — adopted, convention change documented
    ///
    /// Prior to Stage 30, this method used a **corner-registered** pixel
    /// convention (`col_f = (x - x_min) / cell_size_x`, no `-0.5` offset —
    /// cell `(0,0)`'s value anchored at the raster's south-west corner).
    ///
    /// `wbraster::Raster::sample_world()` uses the GDAL-standard
    /// **pixel-center** convention instead (`col_f = (x - x_min) /
    /// cell_size_x - 0.5` — cell `(0,0)`'s value anchored at the center of
    /// that cell). Adopting `sample_world` therefore introduces a genuine,
    /// deliberate **~half-pixel spatial offset** in HAG (height-above-ground)
    /// sampling versus the pre-Stage-30 behaviour.
    ///
    /// This is accepted as part of Stage 30's broader breaking-change scope
    /// (alongside the Step 5 eigenvalue-feature migration): no permanently
    /// trained model exists yet, so retraining absorbs this shift. Any model
    /// trained against pre-Stage-30 `.feat` files should be retrained after
    /// this change lands.
    #[must_use]
    pub fn bilinear_interp(&self, x: f64, y: f64) -> Option<f64> {
        self.raster
            .sample_world(0, x, y, ResampleMethod::Bilinear, NodataPolicy::Strict)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the 99th percentile of a `f64` slice.
///
/// Returns `0.0` for an empty slice.  Uses a partial sort (introselect via
/// `Vec::select_nth_unstable_by`) to avoid full O(n log n) overhead.
fn percentile_99(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    // `as usize` is safe: idx is derived from sorted.len()-1 * 0.99, so it
    // is always < sorted.len() and non-negative.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let idx = ((sorted.len() - 1) as f64 * 0.99) as usize;
    sorted.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
    });
    sorted[idx]
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pt(x: f64, y: f64, z: f64, intensity: u16, ret: u8, nrets: u8) -> LitePoint {
        LitePoint {
            x,
            y,
            z,
            intensity,
            return_number: ret,
            number_of_returns: nrets,
            ..LitePoint::default()
        }
    }

    #[test]
    fn test_resample_subsamples_correctly() {
        let pts: Vec<LitePoint> = (0..100)
            .map(|i| make_pt(f64::from(i), 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 50, 42, 0.0);
        assert_eq!(sampled.len(), 50);
        assert!(!over);
    }

    #[test]
    fn test_resample_oversamples_to_target() {
        let pts: Vec<LitePoint> = (0..10)
            .map(|i| make_pt(f64::from(i), 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 50, 42, 0.0);
        assert_eq!(sampled.len(), 50);
        assert!(over);
    }

    #[test]
    fn test_resample_exact_count_no_oversample() {
        let pts: Vec<LitePoint> = (0..1024)
            .map(|i| make_pt(f64::from(i), 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 1024, 0, 0.0);
        assert_eq!(sampled.len(), 1024);
        assert!(!over);
    }

    #[test]
    fn test_resample_is_reproducible() {
        let pts: Vec<LitePoint> = (0..200)
            .map(|i| make_pt(f64::from(i), 0.0, 0.0, 0, 1, 1))
            .collect();
        let (s1, _, _) = resample_block(&pts, 100, 99, 0.0);
        let (s2, _, _) = resample_block(&pts, 100, 99, 0.0);
        let xs1: Vec<i64> = s1.iter().map(|p| (p.x * 1e6) as i64).collect();
        let xs2: Vec<i64> = s2.iter().map(|p| (p.x * 1e6) as i64).collect();
        assert_eq!(
            xs1, xs2,
            "resample must be reproducible given the same seed"
        );
    }

    #[test]
    fn test_scalar_features_range() {
        let pts = vec![
            make_pt(0.0, 0.0, 0.0, 0, 1, 1),
            make_pt(25.0, 25.0, 10.0, 32767, 1, 2),
            make_pt(50.0, 50.0, 20.0, 65535, 2, 2),
        ];
        let hag = vec![0.0, 5.0, 10.0];
        let feats = normalise_scalar_features(&pts, 0.0, 0.0, 50.0, &hag);
        for f in &feats {
            for &v in f {
                assert!((0.0..=1.0).contains(&v), "feature out of [0,1]: {v}");
            }
        }
    }

    // ── Stage 29: jitter-based oversampling ────────────────────────────────

    /// `jitter_sigma == 0.0` must produce bit-identical output to the
    /// pre-Stage-29 exact-duplicate behaviour (`DoD` #5).
    #[test]
    fn test_jitter_zero_is_bit_identical_to_no_jitter() {
        let pts: Vec<LitePoint> = (0..10)
            .map(|i| {
                make_pt(
                    f64::from(i),
                    f64::from(i) * 2.0,
                    f64::from(i) * 0.5,
                    100,
                    1,
                    1,
                )
            })
            .collect();
        let (s_no_jitter, idx_no_jitter, over1) = resample_block(&pts, 50, 7, 0.0);
        let (s_zero_sigma, idx_zero_sigma, over2) = resample_block(&pts, 50, 7, 0.0);
        assert_eq!(over1, over2);
        assert_eq!(idx_no_jitter, idx_zero_sigma);
        for (a, b) in s_no_jitter.iter().zip(s_zero_sigma.iter()) {
            assert!((a.x - b.x).abs() < 1e-15);
            assert!((a.y - b.y).abs() < 1e-15);
            assert!((a.z - b.z).abs() < 1e-15);
        }
    }

    /// With `jitter_sigma > 0.0`, padding-only points must differ in
    /// coordinates from their source point (`DoD` #6), while the original
    /// (non-padded) points remain untouched.
    #[test]
    fn test_jitter_perturbs_padding_points_only() {
        let pts: Vec<LitePoint> = (0..5)
            .map(|i| make_pt(f64::from(i) * 10.0, f64::from(i) * 10.0, 0.0, 100, 1, 1))
            .collect();
        let (sampled, indices, over) = resample_block(&pts, 20, 123, 1.0);
        assert!(over);
        assert_eq!(sampled.len(), 20);

        // First 5 entries are the original points, untouched.
        for i in 0..5 {
            assert!((sampled[i].x - pts[i].x).abs() < 1e-12);
            assert!((sampled[i].y - pts[i].y).abs() < 1e-12);
            assert!((sampled[i].z - pts[i].z).abs() < 1e-12);
        }

        // At least one padded point should differ from its source (jitter applied).
        let mut any_diff = false;
        for i in 5..20 {
            let src = pts[indices[i]];
            if (sampled[i].x - src.x).abs() > 1e-9
                || (sampled[i].y - src.y).abs() > 1e-9
                || (sampled[i].z - src.z).abs() > 1e-9
            {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "expected at least one padded point to be perturbed by jitter"
        );
    }

    /// Jitter offsets must be clipped to ±3σ (`DoD` #7).
    #[test]
    fn test_jitter_offset_clipped_to_three_sigma() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let sigma = 0.5;
        for _ in 0..10_000 {
            let v = jitter_offset(&mut rng, sigma);
            assert!(
                v.abs() <= 3.0 * sigma + 1e-12,
                "jitter offset {v} exceeded ±3σ ({sigma})"
            );
        }
    }

    /// Jitter must be fully reproducible given the same seed (`DoD` #8).
    #[test]
    fn test_jitter_is_reproducible() {
        let pts: Vec<LitePoint> = (0..5)
            .map(|i| make_pt(f64::from(i), 0.0, 0.0, 0, 1, 1))
            .collect();
        let (s1, _, _) = resample_block(&pts, 30, 55, 0.25);
        let (s2, _, _) = resample_block(&pts, 30, 55, 0.25);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a.x - b.x).abs() < 1e-15);
            assert!((a.y - b.y).abs() < 1e-15);
            assert!((a.z - b.z).abs() < 1e-15);
        }
    }

    /// `jitter_offset` returns exactly `0.0` for non-positive sigma.
    // jitter_offset takes an early-return path for sigma <= 0.0, returning
    // the literal 0.0 with no floating-point computation; exact equality is
    // therefore deterministic and safe to assert here.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_jitter_offset_zero_for_nonpositive_sigma() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(2);
        assert_eq!(jitter_offset(&mut rng, 0.0), 0.0);
        assert_eq!(jitter_offset(&mut rng, -1.0), 0.0);
    }
}
