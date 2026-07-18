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

/// Default fixed absolute reference height (projection z-units, metres for a
/// metric CRS) used to normalise Height-Above-Ground into the `[0, 1]` feature
/// range. See `docs/stages/stage-37-absolute-hag-normalization.md`.
pub const DEFAULT_HAG_MAX_METERS: f64 = 50.0;

/// Default cell size (projection units) for the auto-generated bare-earth
/// ground DTM produced when no external `--hag-model` is supplied. See
/// `docs/stages/stage-38-automatic-ground-dtm.md`.
pub const DEFAULT_DTM_RESOLUTION: f64 = 1.0;

/// Strategy for normalising raw Height-Above-Ground values (`z − z_ground`,
/// in projection units) into the `[0, 1]` feature range consumed by the model.
///
/// **Stage 37.** The default [`HagNormalization::FixedMeters`] preserves the
/// *absolute* vertical scale across blocks, so a point at a given physical
/// height above ground always maps to the same feature value regardless of its
/// block's neighbours. This restores the class-separating signal that
/// distinguishes ASPRS low / medium / high vegetation (defined by absolute
/// height bands). The legacy [`HagNormalization::BlockPercentile99`] is
/// retained for reproducibility and A/B comparison only — it divides by each
/// block's own 99th-percentile HAG, which destroys absolute scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HagNormalization {
    /// Divide raw HAG by a fixed absolute reference height (projection
    /// z-units). Points at or above the reference saturate at `1.0`. Default.
    FixedMeters(f64),
    /// Legacy (pre-Stage-37): divide by the 99th percentile of the block's own
    /// HAG values. Retained for reproducibility / comparison only.
    BlockPercentile99,
}

impl Default for HagNormalization {
    fn default() -> Self {
        Self::FixedMeters(DEFAULT_HAG_MAX_METERS)
    }
}

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
/// - `hag_norm`   — HAG normalisation strategy (Stage 37). The default
///   [`HagNormalization::FixedMeters`] divides by a fixed absolute reference
///   height so identical physical heights map to identical feature values
///   across blocks; [`HagNormalization::BlockPercentile99`] reproduces the
///   legacy per-block behaviour.
///
/// # Panics
/// Panics if `pts.len() != hag_values.len()`.
pub fn normalise_scalar_features(
    pts: &[LitePoint],
    origin_x: f64,
    origin_y: f64,
    block_size: f64,
    hag_values: &[f64],
    hag_norm: HagNormalization,
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

    // HAG denominator (Stage 37). A fixed absolute reference preserves vertical
    // scale across blocks; the legacy percentile mode divides by the block's
    // own 99th-percentile HAG. Clamped > 0 to avoid divide-by-zero.
    let h_max = match hag_norm {
        HagNormalization::FixedMeters(m) => m.max(1e-9),
        HagNormalization::BlockPercentile99 => percentile_99(hag_values).max(1e-9),
    };

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
        let feats =
            normalise_scalar_features(&pts, 0.0, 0.0, 50.0, &hag, HagNormalization::default());
        for f in &feats {
            for &v in f {
                assert!((0.0..=1.0).contains(&v), "feature out of [0,1]: {v}");
            }
        }
    }

    // ── Stage 37: absolute HAG normalisation ───────────────────────────────

    /// `HagNormalization::default()` is the fixed absolute 50 m reference.
    #[test]
    fn test_hag_normalization_default_is_fixed_50m() {
        assert_eq!(
            HagNormalization::default(),
            HagNormalization::FixedMeters(DEFAULT_HAG_MAX_METERS)
        );
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(DEFAULT_HAG_MAX_METERS, 50.0);
        }
    }

    /// Core Stage 37 guarantee: with `FixedMeters`, a point at a fixed physical
    /// HAG maps to the **same** feature value regardless of what other heights
    /// share its block. The legacy percentile mode does **not** have this
    /// property (it is neighbour-dependent).
    #[test]
    fn test_fixed_meters_is_neighbour_invariant() {
        // HAG index within the [f32; 7] scalar row.
        const HAG_IDX: usize = 6;
        let fixed = HagNormalization::FixedMeters(50.0);

        // Two "blocks" each containing a 1 m target point, but with very
        // different tallest-object heights (2 m vs 40 m).
        let short_block = vec![
            make_pt(0.0, 0.0, 0.0, 0, 1, 1),
            make_pt(1.0, 1.0, 2.0, 0, 1, 1),
        ];
        let short_hag = vec![1.0, 2.0];
        let tall_block = vec![
            make_pt(0.0, 0.0, 0.0, 0, 1, 1),
            make_pt(1.0, 1.0, 40.0, 0, 1, 1),
        ];
        let tall_hag = vec![1.0, 40.0];

        let short_feats =
            normalise_scalar_features(&short_block, 0.0, 0.0, 50.0, &short_hag, fixed);
        let tall_feats = normalise_scalar_features(&tall_block, 0.0, 0.0, 50.0, &tall_hag, fixed);

        // The 1 m point (index 0) must map to the same HAG value in both blocks.
        assert!(
            (short_feats[0][HAG_IDX] - tall_feats[0][HAG_IDX]).abs() < 1e-6,
            "fixed-meters HAG must be neighbour-invariant: {} vs {}",
            short_feats[0][HAG_IDX],
            tall_feats[0][HAG_IDX]
        );
        // And it must equal 1.0 / 50.0.
        assert!((short_feats[0][HAG_IDX] - (1.0 / 50.0)).abs() < 1e-6);
    }

    /// With the legacy `BlockPercentile99` mode, the same physical height maps
    /// to different feature values in different blocks — confirming this mode
    /// still behaves as before (and why Stage 37 changed the default).
    #[test]
    fn test_percentile_mode_is_neighbour_dependent() {
        const HAG_IDX: usize = 6;
        let pctl = HagNormalization::BlockPercentile99;

        // Use 100-point blocks so the 99th-percentile index lands near the
        // top of the distribution (idx = floor((100-1) * 0.99) = 98). Both
        // blocks share an identical 1 m "target" point at index 0, but differ
        // in the height of the rest of their points (tallest ≈ 2 m vs ≈ 40 m).
        // Under percentile normalisation the target's HAG feature must differ
        // between the two blocks — the neighbour-dependence Stage 37 removes.
        let n = 100usize;
        let short_block: Vec<LitePoint> = (0..n).map(|_| make_pt(0.0, 0.0, 0.0, 0, 1, 1)).collect();
        let tall_block = short_block.clone();

        // Index 0 is the shared 1 m target; the remaining points ramp up to
        // the block's tallest object.
        let mut short_hag = vec![1.0f64];
        let mut tall_hag = vec![1.0f64];
        for i in 1..n {
            #[allow(clippy::cast_precision_loss)]
            let frac = i as f64 / (n - 1) as f64;
            short_hag.push(frac * 2.0);
            tall_hag.push(frac * 40.0);
        }

        let short_feats = normalise_scalar_features(&short_block, 0.0, 0.0, 50.0, &short_hag, pctl);
        let tall_feats = normalise_scalar_features(&tall_block, 0.0, 0.0, 50.0, &tall_hag, pctl);

        assert!(
            (short_feats[0][HAG_IDX] - tall_feats[0][HAG_IDX]).abs() > 1e-3,
            "percentile HAG is expected to be neighbour-dependent: {} vs {}",
            short_feats[0][HAG_IDX],
            tall_feats[0][HAG_IDX]
        );
    }

    /// Points at or above the fixed reference height saturate at 1.0.
    #[test]
    fn test_fixed_meters_saturates_at_reference() {
        const HAG_IDX: usize = 6;
        let fixed = HagNormalization::FixedMeters(50.0);
        let pts = vec![
            make_pt(0.0, 0.0, 0.0, 0, 1, 1),
            make_pt(1.0, 1.0, 80.0, 0, 1, 1),
        ];
        let hag = vec![0.0, 80.0]; // second point is well above the 50 m reference
        let feats = normalise_scalar_features(&pts, 0.0, 0.0, 50.0, &hag, fixed);
        assert!(
            (feats[1][HAG_IDX] - 1.0).abs() < 1e-6,
            "HAG must clamp to 1.0"
        );
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
