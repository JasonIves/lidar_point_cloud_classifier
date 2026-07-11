//! Lean, project-local in-pipeline point representation (Stage 31).
//!
//! `wblidar::PointRecord` is a flat, `Copy`, `Option`-heavy struct (~330–400
//! bytes) designed as a general-purpose LAS/LAZ I/O record — it carries RGB,
//! NIR, thermal, GPS time, waveform packets, a fixed 192-byte extra-bytes
//! buffer, and computed normals, none of which this project's internal
//! pipeline (block partitioning, spatial indexing, feature extraction,
//! normalisation) ever reads.
//!
//! `LitePoint` carries only the fields the pipeline actually uses:
//! `x, y, z, intensity, return_number, number_of_returns, scan_angle,
//! classification`. Conversion from `wblidar::PointRecord` happens exactly
//! once, at streaming-ingest time in `pipeline.rs::stream_points()`, before
//! points reach `BlockPartitioner`. The full-fidelity `PointRecord` never
//! enters the block-partitioning/spatial-indexing/feature-extraction/
//! normalisation stages — only `LitePoint` does.
//!
//! Final classified output is unaffected: `src/output/las_writer.rs`
//! re-opens and re-streams the *original* input file directly (full
//! `PointRecord` fidelity) for the final write, entirely independent of
//! whatever point type flows through the internal pipeline. See
//! `docs/stages/stage-31-lean-point-record.md` for the full design rationale.

use wblidar::PointRecord;

/// Lean in-pipeline point representation (Stage 31).
///
/// See module-level docs for rationale and the enabling `las_writer.rs`
/// decoupling fact that makes this refactor safe.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LitePoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_angle: i16,
    pub classification: u8,
}

impl From<&PointRecord> for LitePoint {
    fn from(pt: &PointRecord) -> Self {
        Self {
            x: pt.x,
            y: pt.y,
            z: pt.z,
            intensity: pt.intensity,
            return_number: pt.return_number,
            number_of_returns: pt.number_of_returns,
            scan_angle: pt.scan_angle,
            classification: pt.classification,
        }
    }
}

impl From<PointRecord> for LitePoint {
    fn from(pt: PointRecord) -> Self {
        Self::from(&pt)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Stage 31 `DoD` #5: document the achieved memory-footprint reduction.
    #[test]
    fn test_lite_point_is_much_smaller_than_point_record() {
        let lite_size = std::mem::size_of::<LitePoint>();
        let full_size = std::mem::size_of::<PointRecord>();
        assert!(
            lite_size < full_size,
            "LitePoint ({lite_size} bytes) must be smaller than PointRecord ({full_size} bytes)"
        );
        // Expect at least a 5x reduction (design doc estimates ~10x).
        assert!(
            full_size >= lite_size * 5,
            "expected at least a 5x reduction: LitePoint={lite_size} bytes, \
             PointRecord={full_size} bytes"
        );
    }

    // Straight-copy field equality is exactly what this conversion must
    // guarantee (no arithmetic occurs), so strict f64 comparison is correct
    // and intentional here.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_conversion_preserves_used_fields() {
        let pt = PointRecord {
            x: 1.5,
            y: 2.5,
            z: 3.5,
            intensity: 12345,
            return_number: 2,
            number_of_returns: 3,
            scan_angle: -15,
            classification: 6,
            ..PointRecord::default()
        };

        let lite: LitePoint = (&pt).into();
        assert_eq!(lite.x, pt.x);
        assert_eq!(lite.y, pt.y);
        assert_eq!(lite.z, pt.z);
        assert_eq!(lite.intensity, pt.intensity);
        assert_eq!(lite.return_number, pt.return_number);
        assert_eq!(lite.number_of_returns, pt.number_of_returns);
        assert_eq!(lite.scan_angle, pt.scan_angle);
        assert_eq!(lite.classification, pt.classification);

        // Owned-value conversion produces the same result.
        let lite_owned: LitePoint = pt.into();
        assert_eq!(lite, lite_owned);
    }
}
