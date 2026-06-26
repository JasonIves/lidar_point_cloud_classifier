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

use rand::prelude::*;
use rand::SeedableRng;
use wblidar::PointRecord;

/// Resample `pts` to exactly `target` points.
///
/// - If `pts.len() >= target`: random sample without replacement.
/// - If `pts.len() < target`: random oversample with replacement to pad.
///
/// Returns `(sampled_points, sampled_indices, oversampled)` where:
/// - `sampled_points` are the resampled `PointRecord` values,
/// - `sampled_indices` are the 0-based indices into `pts` for each output point
///   (padded oversample entries repeat indices from the original range),
/// - `oversampled` is `true` when padding with replacement was applied.
#[must_use]
pub fn resample_block(
    pts: &[PointRecord],
    target: usize,
    seed: u64,
) -> (Vec<PointRecord>, Vec<usize>, bool) {
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
        let mut sampled: Vec<PointRecord> = pts.to_vec();
        let mut sampled_indices: Vec<usize> = (0..pts.len()).collect();
        let extra = target - pts.len();
        for _ in 0..extra {
            let idx = rng.random_range(0..pts.len());
            sampled.push(pts[idx]);
            sampled_indices.push(idx);
        }
        (sampled, sampled_indices, true)
    }
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
    pts: &[PointRecord],
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
pub fn compute_hag(pts: &[PointRecord], dtm: Option<&DtmView>) -> Vec<f64> {
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

/// Lightweight read-only view of a DTM raster, extracted from `wbraster::Raster`
/// before entering parallel processing so it can be shared via `Arc`.
///
/// Stores the flattened band-0 data as `Vec<f64>` (top-down row order) and the
/// spatial transform needed to convert (x, y) → (row, col).
#[derive(Debug, Clone)]
pub struct DtmView {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    nodata: f64,
    x_min: f64,
    y_max: f64,
    cell_size_x: f64,
    cell_size_y: f64,
}

impl DtmView {
    /// Construct a `DtmView` from a loaded `wbraster::Raster`.
    #[must_use]
    pub fn from_raster(r: &wbraster::Raster) -> Self {
        Self {
            data: r.band_to_vec_f64(0),
            rows: r.rows,
            cols: r.cols,
            nodata: r.nodata,
            x_min: r.x_min,
            // `wbraster::Raster` stores y_min (south edge); top = y_min + rows * cell_size_y
            // `as f64` cast for raster geometry is lossless for any realistic
            // row count (usize fits in f64 without precision loss up to 2^53).
            #[allow(clippy::cast_precision_loss)]
            y_max: r.y_min + r.rows as f64 * r.cell_size_y,
            cell_size_x: r.cell_size_x,
            cell_size_y: r.cell_size_y,
        }
    }

    /// Bilinear interpolation at world coordinate (x, y).
    ///
    /// Returns `None` if the coordinate is outside the raster extent or all
    /// four surrounding cells are nodata.
    #[must_use]
    pub fn bilinear_interp(&self, x: f64, y: f64) -> Option<f64> {
        // Convert to fractional pixel coordinates (row 0 = north edge).
        let col_f = (x - self.x_min) / self.cell_size_x;
        let row_f = (self.y_max - y) / self.cell_size_y;

        // isize casts: col_f.floor() returns a finite value; the raster is
        // bounded to realistic extents so truncation to isize is safe.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let col0 = col_f.floor() as isize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let row0 = row_f.floor() as isize;
        let col1 = col0 + 1;
        let row1 = row0 + 1;

        // `as f64` casts for integer raster indices are lossless.
        #[allow(clippy::cast_precision_loss)]
        let tx = col_f - col0 as f64;
        #[allow(clippy::cast_precision_loss)]
        let ty = row_f - row0 as f64;

        let v00 = self.get(row0, col0)?;
        let v10 = self.get(row0, col1)?;
        let v01 = self.get(row1, col0)?;
        let v11 = self.get(row1, col1)?;

        Some(
            v00 * (1.0 - tx) * (1.0 - ty)
                + v10 * tx * (1.0 - ty)
                + v01 * (1.0 - tx) * ty
                + v11 * tx * ty,
        )
    }

    #[inline]
    fn get(&self, row: isize, col: isize) -> Option<f64> {
        if row < 0 || col < 0 || row >= self.rows as isize || col >= self.cols as isize {
            return None;
        }
        // isize-checked bounds: both indices are verified positive above.
        #[allow(clippy::cast_sign_loss)]
        let v = self.data[row as usize * self.cols + col as usize];
        if self.is_nodata(v) {
            None
        } else {
            Some(v)
        }
    }

    #[inline]
    fn is_nodata(&self, v: f64) -> bool {
        if self.nodata.is_nan() {
            v.is_nan()
        } else {
            (v - self.nodata).abs() < 1e-9
        }
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

    fn make_pt(x: f64, y: f64, z: f64, intensity: u16, ret: u8, nrets: u8) -> PointRecord {
        let mut pt = PointRecord::default();
        pt.x = x;
        pt.y = y;
        pt.z = z;
        pt.intensity = intensity;
        pt.return_number = ret;
        pt.number_of_returns = nrets;
        pt
    }

    #[test]
    fn test_resample_subsamples_correctly() {
        let pts: Vec<PointRecord> = (0..100)
            .map(|i| make_pt(i as f64, 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 50, 42);
        assert_eq!(sampled.len(), 50);
        assert!(!over);
    }

    #[test]
    fn test_resample_oversamples_to_target() {
        let pts: Vec<PointRecord> = (0..10)
            .map(|i| make_pt(i as f64, 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 50, 42);
        assert_eq!(sampled.len(), 50);
        assert!(over);
    }

    #[test]
    fn test_resample_exact_count_no_oversample() {
        let pts: Vec<PointRecord> = (0..1024)
            .map(|i| make_pt(i as f64, 0.0, 0.0, 0, 1, 1))
            .collect();
        let (sampled, _indices, over) = resample_block(&pts, 1024, 0);
        assert_eq!(sampled.len(), 1024);
        assert!(!over);
    }

    #[test]
    fn test_resample_is_reproducible() {
        let pts: Vec<PointRecord> = (0..200)
            .map(|i| make_pt(i as f64, 0.0, 0.0, 0, 1, 1))
            .collect();
        let (s1, _, _) = resample_block(&pts, 100, 99);
        let (s2, _, _) = resample_block(&pts, 100, 99);
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
            for &v in f.iter() {
                assert!((0.0..=1.0).contains(&v), "feature out of [0,1]: {v}");
            }
        }
    }
}
