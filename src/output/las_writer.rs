#![allow(clippy::cast_lossless)]
//! Streaming classified LAS/LAZ writer.
//!
//! Streams the original `LiDAR` source file point-by-point, computes the
//! predicted class for each point via [`crate::model::fusion::fused_label`]
//! (Stage 44 — weighted soft voting across the block(s) whose inference
//! footprint covers the point), and writes the (potentially updated) record to
//! the output file.  With `FusionConfig::radius == 0.0` the decision rule is
//! exactly the legacy one: the nearest sampled point's argmax within the
//! point's canonical block.
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
use crate::model::fusion::{default_proximity_sigma, fused_label, FusionConfig, GridGeometry};
use crate::model::inference::BlockInferenceResult;
use crate::preprocessing::BlockManifest;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Stream `input_path` and write a classified copy to `output_path`.
///
/// For every original point, the predicted model class index is computed by
/// [`fused_label`] (Stage 44) and mapped to an ASPRS code via `label_map`:
/// - With `fusion.radius == 0.0` the decision is the legacy single-block one:
///   the argmax of the nearest sampled point's softmax row within the point's
///   canonical block.
/// - With a positive radius, blocks adjacent to the point also vote
///   (centrality × inverse-square-proximity weighted softmax rows), producing
///   smooth labels across block seams.
/// - If no block's footprint covers the point (density-filtered region), the
///   original classification is preserved.
///
/// `output_path` extension determines output format (`.las` or `.laz`).
///
/// # Errors
/// Returns an error if the source file cannot be read, the output file cannot
/// be created or written, the manifest lacks grid geometry, or format
/// detection fails.
pub fn write_classified<S: BuildHasher>(
    input_path: &Path,
    output_path: &Path,
    inference_map: &HashMap<u64, BlockInferenceResult, S>,
    manifest: &BlockManifest,
    label_map: &[u8],
    fusion: &FusionConfig,
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

    // ── Grid geometry from the manifest (header-derived, authoritative) ────
    let grid = GridGeometry::from_manifest(manifest)?;

    // Scratch accumulator reused across points (no per-point allocation).
    // Sized once from any block's class count; all blocks share the model's
    // class count by construction.
    let n_classes = inference_map
        .values()
        .next()
        .map_or(0, BlockInferenceResult::n_classes);
    let mut acc = vec![0.0f64; n_classes];

    // Proximity bandwidth σ: the characteristic inter-sample spacing of a
    // block (Stage 44 fused-eval blind-spot fix).  Bounds the inverse-square
    // proximity term so self-coincident samples cannot dominate a blend.
    let proximity_sigma = default_proximity_sigma(manifest.block_size, manifest.target_points);

    // ── Stream + fuse + write ─────────────────────────────────────────────
    let mut pt = PointRecord::default();
    let mut points_written: u64 = 0;

    while reader.read_point(&mut pt).map_err(ClassifierError::Lidar)? {
        if n_classes > 0 {
            if let Some(class_idx) = fused_label(
                pt.x,
                pt.y,
                inference_map,
                &grid,
                fusion.radius,
                proximity_sigma,
                &mut acc,
            ) {
                // Model class index → ASPRS code (fallback: Unassigned = 1,
                // matching the legacy `PointNetClassifier::classify` fallback).
                pt.classification = label_map.get(class_idx).copied().unwrap_or(1);
            }
            // No votes (density-filtered region): preserve original class.
        }

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

    let format = LidarFormat::detect(path).map_err(|e| ClassifierError::UnsupportedFormat {
        path: format!("{}: {e}", path.display()),
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
            let r = CopcReader::open_path(path).map_err(ClassifierError::Lidar)?;
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
    let reader =
        LasReader::new(BufReader::new(File::open(input_path)?)).map_err(ClassifierError::Lidar)?;
    let hdr = reader.header();
    let cfg = WriterConfig {
        point_data_format: hdr.point_data_format,
        x_scale: hdr.x_scale,
        y_scale: hdr.y_scale,
        z_scale: hdr.z_scale,
        x_offset: hdr.x_offset,
        y_offset: hdr.y_offset,
        z_offset: hdr.z_offset,
        extra_bytes_per_point: hdr.extra_bytes_count,
        crs: reader.crs().cloned(),
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
        _ => {
            // COPC write is not supported — the wblidar public API does not
            // expose a COPC writer.  This is a known deviation from
            // PROJECT_SPEC.md §"Default to input format", documented in
            // stage-02-modeling-layer.md §"Known constraints".
            Err(ClassifierError::Pipeline(
                "output format must be .las or .laz (COPC write not supported by wblidar)".into(),
            ))
        }
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
    use ndarray::Array2;
    use std::io::BufReader;
    use tempfile::NamedTempFile;
    use wblidar::las::header::PointDataFormat;
    use wblidar::las::reader::LasReader;
    use wblidar::las::writer::{LasWriter, WriterConfig};

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
            grid_cols: 1,
            grid_rows: 1,
            grid_x_min: origin_x,
            grid_y_min: origin_y,
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap: 0.0,
            oversample_jitter: 0.0,
            z_norm_use_block_relative: false,
            halo_fraction: 0.0,
            blocks: vec![BlockMeta {
                id: 0,
                file: "block_00000.feat".into(),
                origin_x,
                origin_y,
                raw_point_count: 4,
                sampled_point_count: 4,
                oversampled: false,
                n_halo: 0,
            }],
        }
    }

    /// Write a minimal two-block manifest for fusion testing:
    /// block 0 covers `[0,50) × [0,50)`, block 1 covers `[50,100) × [0,50)`.
    fn two_block_manifest() -> BlockManifest {
        let mut m = single_block_manifest(50.0, 0.0, 0.0);
        m.grid_cols = 2;
        m.blocks.push(BlockMeta {
            id: 1,
            file: "block_00001.feat".into(),
            origin_x: 50.0,
            origin_y: 0.0,
            raw_point_count: 4,
            sampled_point_count: 4,
            oversampled: false,
            n_halo: 0,
        });
        m
    }

    /// Build a one-sample-per-block inference map.
    ///
    /// `blocks` is `(block_id, sample_x, sample_y, raw_logits)`; logits are
    /// softmaxed internally (τ = 1).
    fn block_map(blocks: &[(u64, f64, f64, Vec<f32>)]) -> HashMap<u64, BlockInferenceResult> {
        let mut map = HashMap::new();
        for &(id, x, y, ref logits) in blocks {
            let n_classes = logits.len();
            let mat = Array2::from_shape_vec((1, n_classes), logits.clone())
                .expect("logit matrix shape must be valid");
            let result = BlockInferenceResult::from_logits(&[x], &[y], &mat, 1.0)
                .expect("from_logits must succeed");
            map.insert(id, result);
        }
        map
    }

    /// Read all points back from a written LAS file.
    fn read_points(path: &Path) -> Result<Vec<PointRecord>> {
        let mut reader =
            LasReader::new(BufReader::new(File::open(path)?)).map_err(ClassifierError::Lidar)?;
        let mut out = Vec::new();
        let mut pt = PointRecord::default();
        while reader.read_point(&mut pt).map_err(ClassifierError::Lidar)? {
            out.push(pt);
        }
        Ok(out)
    }

    fn pt_at(x: f64, y: f64, classification: u8) -> PointRecord {
        PointRecord {
            x,
            y,
            z: 10.0,
            intensity: 100,
            classification,
            return_number: 1,
            number_of_returns: 1,
            ..PointRecord::default()
        }
    }

    /// Write a tiny synthetic LAS file with `n` points.
    fn write_synthetic_las(path: &Path, points: &[PointRecord]) -> Result<()> {
        let cfg = WriterConfig {
            point_data_format: PointDataFormat::Pdrf6,
            ..Default::default()
        };
        let mut writer = LasWriter::new(BufWriter::new(File::create(path)?), cfg)
            .map_err(ClassifierError::Lidar)?;
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
            PointRecord {
                x: 1.0,
                y: 1.0,
                z: 10.0,
                intensity: 100,
                classification: 0,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            },
            PointRecord {
                x: 2.0,
                y: 2.0,
                z: 11.0,
                intensity: 200,
                classification: 0,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            },
            PointRecord {
                x: 3.0,
                y: 3.0,
                z: 12.0,
                intensity: 300,
                classification: 0,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            },
            PointRecord {
                x: 4.0,
                y: 4.0,
                z: 13.0,
                intensity: 400,
                classification: 0,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            },
        ];

        // Write synthetic LAS input
        let input_tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;
        write_synthetic_las(input_tmp.path(), &orig_pts)?;

        // Build inference map: block 0 covers [0,50)×[0,50), sampled points at
        // exact input coords; one-hot logits make each sample's argmax its own
        // index, then label_map translates to ASPRS codes.
        let mut logits = Array2::zeros((4, 4));
        for i in 0..4 {
            logits[[i, i]] = 9.0;
        }
        let inference_result = BlockInferenceResult::from_logits(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &logits,
            1.0,
        )
        .expect("from_logits must succeed");
        let mut inference_map = HashMap::new();
        inference_map.insert(0u64, inference_result);
        // Model indices 0..3 → Ground, Building, Water, LowVeg
        let label_map = [2u8, 5u8, 6u8, 3u8];

        let manifest = single_block_manifest(50.0, 0.0, 0.0);

        // Run write_classified
        let output_tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;
        // Need a .las extension for format detection; rename tempfile
        let output_path = output_tmp.path().with_extension("las");
        write_classified(
            input_tmp.path(),
            &output_path,
            &inference_map,
            &manifest,
            &label_map,
            &FusionConfig { radius: 0.0 },
        )?;

        // Read back and verify
        let out_pts = read_points(&output_path)?;

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

    // DoD #1/#2/#6 — fusion-off regression + seam blending end-to-end.
    //
    // Two blocks: block 0 ([0,50)²) has one weak class-0 sample at (49,25);
    // block 1 ([50,100)×[0,50)) has one strong class-1 sample at (51,25).
    // A point just inside block 0's side of the seam:
    //   - radius 0  → canonical block 0 alone decides → class 0;
    //   - radius 10 → both blocks vote, block 1's confident distribution
    //     dominates the blend → class 1 (the seam flip fusion is meant to
    //     produce when the neighbour is genuinely more confident).
    // A deep-interior point must be unaffected in both modes.
    #[test]
    fn test_write_classified_fusion_blends_seam_and_preserves_interior() -> Result<()> {
        let orig_pts = vec![
            pt_at(49.999, 25.0, 9), // seam point (original class must be overwritten)
            pt_at(25.0, 25.0, 9),   // deep interior of block 0
        ];
        let input_tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;
        write_synthetic_las(input_tmp.path(), &orig_pts)?;

        let inference_map = block_map(&[
            (0u64, 49.0, 25.0, vec![0.51, 0.49]), // weak class 0
            (1u64, 51.0, 25.0, vec![0.05, 0.95]), // strong class 1
        ]);
        let label_map = [2u8, 6u8]; // class 0 → Ground(2), class 1 → Building(6)
        let manifest = two_block_manifest();

        // ── Fusion disabled → legacy single-block behaviour ────────────────
        let out_off = NamedTempFile::new()
            .map_err(ClassifierError::Io)?
            .path()
            .with_extension("las");
        write_classified(
            input_tmp.path(),
            &out_off,
            &inference_map,
            &manifest,
            &label_map,
            &FusionConfig { radius: 0.0 },
        )?;
        let pts_off = read_points(&out_off)?;
        assert_eq!(pts_off[0].classification, 2, "seam point, fusion off");
        assert_eq!(pts_off[1].classification, 2, "interior point, fusion off");

        // ── Fusion enabled → seam blends, interior unchanged ───────────────
        let out_on = NamedTempFile::new()
            .map_err(ClassifierError::Io)?
            .path()
            .with_extension("las");
        write_classified(
            input_tmp.path(),
            &out_on,
            &inference_map,
            &manifest,
            &label_map,
            &FusionConfig { radius: 10.0 },
        )?;
        let pts_on = read_points(&out_on)?;
        assert_eq!(pts_on[0].classification, 6, "seam point, fusion on");
        assert_eq!(pts_on[1].classification, 2, "interior point, fusion on");

        Ok(())
    }

    // DoD #7 — points outside every block's footprint preserve their original
    // classification (density-filtered region), while a point within fusion
    // reach of a retained neighbour is still labeled by that neighbour.
    #[test]
    fn test_write_classified_preserves_when_no_votes() -> Result<()> {
        let orig_pts = vec![
            pt_at(25.0, 25.0, 9), // deep inside missing block 0 → preserved (9)
            pt_at(49.0, 25.0, 9), // within r of retained block 1 → labeled
        ];
        let input_tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;
        write_synthetic_las(input_tmp.path(), &orig_pts)?;

        // Only block 1 exists (block 0 was density-dropped).
        let inference_map = block_map(&[(1u64, 51.0, 25.0, vec![0.05, 0.95])]);
        let label_map = [2u8, 6u8];
        let manifest = two_block_manifest();

        let out = NamedTempFile::new()
            .map_err(ClassifierError::Io)?
            .path()
            .with_extension("las");
        write_classified(
            input_tmp.path(),
            &out,
            &inference_map,
            &manifest,
            &label_map,
            &FusionConfig { radius: 10.0 },
        )?;
        let pts = read_points(&out)?;
        assert_eq!(pts[0].classification, 9, "no votes → original preserved");
        assert_eq!(pts[1].classification, 6, "neighbour vote labels the point");

        Ok(())
    }
}
