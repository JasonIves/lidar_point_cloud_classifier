//! Streaming classified LAS/LAZ writer.
//!
//! Streams the original `LiDAR` source file point-by-point, looks up the inferred
//! ASPRS classification label for each point via nearest-sampled-point search
//! within its 2-D block, and writes the (potentially updated) record to the
//! output file.
//!
//! All non-classification fields (XYZ, intensity, GPS time, RGB, scan angle,
//! return info, extra bytes, etc.) are preserved verbatim.
//!
//! VLRs, CRS metadata, scale/offset, and point-data record format are all
//! carried over from the source file by mirroring `infer_stream_writer_config_from_source`
//! from `wblidar::frontend` (private there; reproduced here from the same logic
//! using only public wblidar APIs).

use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasher;
use std::io::BufWriter;
use std::path::Path;

use wblidar::io::PointReader;
use wblidar::io::PointWriter;
use wblidar::las::reader::LasReader;
use wblidar::las::writer::{LasWriter, WriterConfig};
use wblidar::laz::writer::{LazWriter, LazWriterConfig};
use wblidar::LidarFormat;
use wblidar::PointRecord;

use crate::error::{ClassifierError, Result};
use crate::model::inference::BlockInferenceResult;
use crate::preprocessing::BlockManifest;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Stream `input_path` and write a classified copy to `output_path`.
///
/// For every original point:
/// - Compute its 2-D block from the manifest geometry.
/// - Look up the `BlockInferenceResult` for that block.
/// - If found, assign `classification = nearest_label(x, y)`.
/// - If not found (density-filtered block), preserve the original classification.
///
/// `output_path` extension determines output format (`.las` or `.laz`).
///
/// # Errors
/// Returns an error if the source file cannot be read, the output file cannot
/// be created or written, or format detection fails.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn write_classified<S: BuildHasher>(
    input_path: &Path,
    output_path: &Path,
    inference_map: &HashMap<u64, BlockInferenceResult, S>,
    manifest: &BlockManifest,
) -> Result<()> {
    // ── Open the source for streaming ─────────────────────────────────────
    let mut reader = open_reader(input_path)?;

    // ── Infer writer config from source header ─────────────────────────────
    let writer_cfg = infer_writer_config(input_path)?;

    // ── Open the output writer ─────────────────────────────────────────────
    let out_format = LidarFormat::detect(output_path).map_err(|e| {
        ClassifierError::Pipeline(format!(
            "cannot detect output format for '{}': {e}",
            output_path.display()
        ))
    })?;

    let mut writer = open_writer(output_path, out_format, writer_cfg)?;

    // ── Precompute block-grid geometry from the manifest ───────────────────
    // These values mirror Stage 01's BlockPartitioner.
    let x_min = manifest
        .blocks
        .iter()
        .map(|b| b.origin_x)
        .fold(f64::INFINITY, f64::min);
    let y_min = manifest
        .blocks
        .iter()
        .map(|b| b.origin_y)
        .fold(f64::INFINITY, f64::min);
    let x_max_approx = manifest
        .blocks
        .iter()
        .map(|b| b.origin_x)
        .fold(f64::NEG_INFINITY, f64::max)
        + manifest.block_size;

    let block_size = manifest.block_size;
    let grid_cols = ((x_max_approx - x_min) / block_size).ceil().max(1.0) as i64;

    // ── Stream + substitute + write ───────────────────────────────────────
    let mut pt = PointRecord::default();
    let mut points_written: u64 = 0;

    while reader.read_point(&mut pt).map_err(ClassifierError::Lidar)? {
        // Assign point to block using same formula as BlockPartitioner
        let col = ((pt.x - x_min) / block_size).floor() as i64;
        let row = ((pt.y - y_min) / block_size).floor() as i64;
        let block_id = (row * grid_cols + col) as u64;

        if let Some(result) = inference_map.get(&block_id) {
            pt.classification = result.nearest_label(pt.x, pt.y);
        }
        // If block not in map: preserve original pt.classification

        writer.write_point(&pt).map_err(ClassifierError::Lidar)?;
        points_written += 1;
    }

    writer.finish().map_err(ClassifierError::Lidar)?;

    eprintln!("[classify] points written: {points_written}");
    eprintln!("[classify] output: {}", output_path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Reader / writer helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Open any supported `LiDAR` format for streaming point reads.
fn open_reader(path: &Path) -> Result<Box<dyn PointReader>> {
    use std::io::BufReader;
    use wblidar::copc::reader::CopcReader;
    use wblidar::las::reader::LasReader;
    use wblidar::laz::reader::LazReader;

    let format = LidarFormat::detect(path).map_err(|e| {
        ClassifierError::UnsupportedFormat {
            path: format!("{}: {e}", path.display()),
        }
    })?;

    match format {
        LidarFormat::Las => {
            let r = LasReader::new(BufReader::new(File::open(path)?))
                .map_err(ClassifierError::Lidar)?;
            Ok(Box::new(r))
        }
        LidarFormat::Laz => {
            let r = LazReader::new(BufReader::new(File::open(path)?))
                .map_err(ClassifierError::Lidar)?;
            Ok(Box::new(r))
        }
        LidarFormat::Copc => {
            let r = CopcReader::open_path(path)
                .map_err(ClassifierError::Lidar)?;
            Ok(Box::new(r))
        }
        _ => Err(ClassifierError::UnsupportedFormat {
            path: path.display().to_string(),
        }),
    }
}

/// Mirror the logic of `wblidar::frontend::infer_stream_writer_config_from_source`
/// (private there) using only the public `LasReader` API.
fn infer_writer_config(input_path: &Path) -> Result<WriterConfig> {
    use std::io::BufReader;
    let reader = LasReader::new(BufReader::new(File::open(input_path)?))
        .map_err(ClassifierError::Lidar)?;
    let hdr = reader.header();
    let cfg = WriterConfig {
        point_data_format:     hdr.point_data_format,
        x_scale:               hdr.x_scale,
        y_scale:               hdr.y_scale,
        z_scale:               hdr.z_scale,
        x_offset:              hdr.x_offset,
        y_offset:              hdr.y_offset,
        z_offset:              hdr.z_offset,
        extra_bytes_per_point: hdr.extra_bytes_count,
        crs:                   reader.crs().cloned(),
        ..WriterConfig::default()
    };
    Ok(cfg)
}

/// Open a LAS or LAZ output writer appropriate for the output format.
fn open_writer(
    output_path: &Path,
    format: LidarFormat,
    cfg: WriterConfig,
) -> Result<Box<dyn PointWriter>> {
    match format {
        LidarFormat::Las => {
            let w = LasWriter::new(BufWriter::new(File::create(output_path)?), cfg)
                .map_err(ClassifierError::Lidar)?;
            Ok(Box::new(w))
        }
        LidarFormat::Laz => {
            let laz_cfg = LazWriterConfig {
                las: cfg,
                ..LazWriterConfig::default()
            };
            let w = LazWriter::new(BufWriter::new(File::create(output_path)?), laz_cfg)
                .map_err(ClassifierError::Lidar)?;
            Ok(Box::new(w))
        }
        _ => Err(ClassifierError::Pipeline(
            "output format must be .las or .laz".into(),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::inference::BlockInferenceResult;
    use crate::preprocessing::{BlockManifest, BlockMeta};
    use std::io::BufReader;
    use tempfile::NamedTempFile;
    use wblidar::las::reader::LasReader;
    use wblidar::las::writer::{LasWriter, WriterConfig};
    use wblidar::las::header::PointDataFormat;

    /// Write a minimal single-block manifest for testing.
    fn single_block_manifest(block_size: f64, origin_x: f64, origin_y: f64) -> BlockManifest {
        BlockManifest {
            source: "test.las".into(),
            block_size,
            target_points: 4,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 1,
            crs_epsg: None,
            blocks: vec![BlockMeta {
                id: 0,
                file: "block_00000.feat".into(),
                origin_x,
                origin_y,
                raw_point_count: 4,
                sampled_point_count: 4,
                oversampled: false,
            }],
        }
    }

    /// Write a tiny synthetic LAS file with `n` points.
    fn write_synthetic_las(path: &Path, points: &[PointRecord]) -> Result<()> {
        let mut cfg = WriterConfig::default();
        cfg.point_data_format = PointDataFormat::Pdrf6;
        let mut writer = LasWriter::new(
            BufWriter::new(File::create(path)?),
            cfg,
        ).map_err(ClassifierError::Lidar)?;
        for pt in points {
            writer.write_point(pt).map_err(ClassifierError::Lidar)?;
        }
        writer.finish().map_err(ClassifierError::Lidar)?;
        Ok(())
    }

    // DoD #15 — write_classified: classification substituted, other fields preserved
    #[test]
    fn test_write_classified_substitutes_classification() -> Result<()> {
        // Four original points at known coordinates, all with classification=0
        let orig_pts = vec![
            PointRecord { x: 1.0, y: 1.0, z: 10.0, intensity: 100, classification: 0, return_number: 1, number_of_returns: 1, ..PointRecord::default() },
            PointRecord { x: 2.0, y: 2.0, z: 11.0, intensity: 200, classification: 0, return_number: 1, number_of_returns: 1, ..PointRecord::default() },
            PointRecord { x: 3.0, y: 3.0, z: 12.0, intensity: 300, classification: 0, return_number: 1, number_of_returns: 1, ..PointRecord::default() },
            PointRecord { x: 4.0, y: 4.0, z: 13.0, intensity: 400, classification: 0, return_number: 1, number_of_returns: 1, ..PointRecord::default() },
        ];

        // Write synthetic LAS input
        let input_tmp = NamedTempFile::new().map_err(|e| ClassifierError::Io(e.into()))?;
        write_synthetic_las(input_tmp.path(), &orig_pts)?;

        // Build inference map: block 0 covers [0,50)×[0,50), sampled points at exact input coords
        let inference_result = BlockInferenceResult {
            xs:     vec![1.0, 2.0, 3.0, 4.0],
            ys:     vec![1.0, 2.0, 3.0, 4.0],
            labels: vec![2u8, 5u8, 6u8, 3u8], // Ground, Building, Water, LowVeg
        };
        let mut inference_map = HashMap::new();
        inference_map.insert(0u64, inference_result);

        let manifest = single_block_manifest(50.0, 0.0, 0.0);

        // Run write_classified
        let output_tmp = NamedTempFile::new().map_err(|e| ClassifierError::Io(e.into()))?;
        // Need a .las extension for format detection; rename tempfile
        let output_path = output_tmp.path().with_extension("las");
        write_classified(input_tmp.path(), &output_path, &inference_map, &manifest)?;

        // Read back and verify
        let mut reader = LasReader::new(BufReader::new(File::open(&output_path)?))
            .map_err(ClassifierError::Lidar)?;
        let mut out_pts = Vec::new();
        let mut pt = PointRecord::default();
        while reader.read_point(&mut pt).map_err(ClassifierError::Lidar)? {
            out_pts.push(pt);
        }

        assert_eq!(out_pts.len(), 4, "should have 4 output points");

        // Classifications should be substituted
        assert_eq!(out_pts[0].classification, 2, "point 0 should be Ground");
        assert_eq!(out_pts[1].classification, 5, "point 1 should be Building");
        assert_eq!(out_pts[2].classification, 6, "point 2 should be Water");
        assert_eq!(out_pts[3].classification, 3, "point 3 should be LowVeg");

        // Intensity must be preserved verbatim
        assert_eq!(out_pts[0].intensity, 100);
        assert_eq!(out_pts[1].intensity, 200);
        assert_eq!(out_pts[2].intensity, 300);
        assert_eq!(out_pts[3].intensity, 400);

        Ok(())
    }
}
