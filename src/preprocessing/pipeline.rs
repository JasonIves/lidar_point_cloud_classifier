//! Streaming preprocessing orchestrator.
//!
//! `PreprocessingPipeline::run(config)` ties together all sub-modules:
//!
//! 1. Open the input `LiDAR` file and stream all points into `BlockPartitioner`.
//! 2. After EOF, finalise blocks and filter by density.
//! 3. Optionally load the DTM raster once and wrap it in `Arc<DtmView>`.
//! 4. Process each retained block in parallel via Rayon:
//!    - Build 3-D k-d tree from raw block points.
//!    - Resample to `target_points`.
//!    - Compute features via `feature_extractor`.
//!    - Serialise to `.feat` binary file.
//!    - Optionally emit debug `.csv`.
//! 5. Write `blocks.json` manifest.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use wblidar::PointRecord;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::{
    block_partitioner::BlockPartitioner,
    feature_extractor::extract_features,
    normalizer::{resample_block, DtmView},
    spatial_index::BlockSpatialIndex,
    PreprocessConfig, FEAT_MAGIC, FEAT_VERSION, N_FEATURES, RAYON_MIN_CHUNK,
};

/// Manifest emitted as `blocks.json` after a successful pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockManifest {
    pub source: String,
    pub block_size: f64,
    pub target_points: usize,
    pub min_density: f64,
    pub search_radius: f64,
    pub min_neighbors: usize,
    pub crs_epsg: Option<u32>,
    pub blocks: Vec<BlockMeta>,
}

/// Per-block metadata entry in `blocks.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMeta {
    pub id: u64,
    pub file: String,
    pub origin_x: f64,
    pub origin_y: f64,
    pub raw_point_count: usize,
    pub sampled_point_count: usize,
    pub oversampled: bool,
}

/// The main preprocessing pipeline.
pub struct PreprocessingPipeline;

impl PreprocessingPipeline {
    /// Run the full preprocessing pipeline according to `config`.
    ///
    /// Returns the manifest on success; any I/O or processing error is
    /// propagated as `ClassifierError`.
    ///
    /// # Errors
    /// Returns `ClassifierError` on any I/O, `LiDAR` parse, or serialisation failure.
    #[allow(clippy::too_many_lines)]
    pub fn run(config: &PreprocessConfig) -> Result<BlockManifest> {
        // ── 0. Set up Rayon thread pool ────────────────────────────────────
        if let Some(threads) = config.threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok(); // ignore error if global pool was already initialised
        }

        // ── 1. Ensure output directory exists ─────────────────────────────
        fs::create_dir_all(&config.output_dir)?;

        // ── 2. Open the LiDAR file and inspect the header ─────────────────
        let input_path = &config.input;
        let (x_min, y_min, x_max, y_max, total_points, crs_epsg) =
            inspect_lidar_header(input_path)?;

        eprintln!(
            "[preprocessing] input:  {}",
            input_path.display()
        );
        eprintln!(
            "[preprocessing] extent: ({x_min:.2}, {y_min:.2}) → ({x_max:.2}, {y_max:.2})"
        );
        eprintln!("[preprocessing] points: {total_points}");

        // ── 3. Stream all points into the block partitioner ───────────────
        let mut partitioner = BlockPartitioner::new(
            x_min,
            y_min,
            x_max,
            y_max,
            config.block_size,
            &config.output_dir,
        );
        stream_points(input_path, &mut partitioner)?;

        // ── 4. Finalise — merges spill files, returns complete blocks ──────
        let raw_blocks = partitioner.finalize()?;
        eprintln!("[preprocessing] raw blocks: {}", raw_blocks.len());

        // ── 5. Density gate ───────────────────────────────────────────────
        let cell_area = config.block_size * config.block_size;
        let retained: Vec<_> = raw_blocks
            .into_iter()
            .filter(|b| {
                #[allow(clippy::cast_precision_loss)]
                let density = b.points.len() as f64 / cell_area;
                density >= config.min_density
            })
            .collect();

        eprintln!(
            "[preprocessing] retained blocks (density ≥ {:.2}): {}",
            config.min_density,
            retained.len()
        );

        // ── 6. Optionally load the DTM raster ─────────────────────────────
        let dtm: Option<Arc<DtmView>> = config
            .hag_model
            .as_ref()
            .map(|path| -> Result<Arc<DtmView>> {
                let r = wbraster::Raster::read(path).map_err(|e| {
                    ClassifierError::Raster(format!(
                        "failed to load DTM '{}': {e}",
                        path.display()
                    ))
                })?;
                Ok(Arc::new(DtmView::from_raster(&r)))
            })
            .transpose()?;

        // ── 7. Per-block parallel processing ─────────────────────────────
        let block_results: Vec<Result<BlockMeta>> = retained
            .into_par_iter()
            .with_min_len(RAYON_MIN_CHUNK)
            .map(|block| {
                let dtm_ref = dtm.as_deref();

                // (a) Build k-d tree from full unsampled block
                let index = BlockSpatialIndex::build(&block.points);

                // (b) Density-gated sampling
                let (sampled, oversampled) =
                    resample_block(&block.points, config.target_points, block.id);

                let raw_count = block.points.len();
                let sampled_count = sampled.len();

                // (c–e) Extract all 12 features
                let features = extract_features(
                    &sampled,
                    &block.points,
                    &index,
                    dtm_ref,
                    block.origin_x,
                    block.origin_y,
                    config.block_size,
                    config.search_radius,
                    config.min_neighbors,
                );

                // (f) Serialise to .feat
                let feat_filename = format!("block_{:05}.feat", block.id);
                let feat_path = config.output_dir.join(&feat_filename);
                write_feat_file(
                    &feat_path,
                    block.id,
                    sampled_count,
                    block.origin_x,
                    block.origin_y,
                    &features,
                )?;

                // (g) Optional debug CSV
                if config.debug_csv {
                    let csv_path = config.output_dir.join(format!("block_{:05}.csv", block.id));
                    write_debug_csv(&csv_path, &features)?;
                }

                Ok(BlockMeta {
                    id: block.id,
                    file: feat_filename,
                    origin_x: block.origin_x,
                    origin_y: block.origin_y,
                    raw_point_count: raw_count,
                    sampled_point_count: sampled_count,
                    oversampled,
                })
            })
            .collect();

        // Collect results, propagating the first error.
        let mut block_metas = Vec::with_capacity(block_results.len());
        for r in block_results {
            block_metas.push(r?);
        }
        block_metas.sort_by_key(|m| m.id);

        eprintln!("[preprocessing] wrote {} .feat files", block_metas.len());

        // ── 8. Write blocks.json ──────────────────────────────────────────
        let manifest = BlockManifest {
            source: input_path.display().to_string(),
            block_size: config.block_size,
            target_points: config.target_points,
            min_density: config.min_density,
            search_radius: config.search_radius,
            min_neighbors: config.min_neighbors,
            crs_epsg,
            blocks: block_metas,
        };

        let manifest_path = config.output_dir.join("blocks.json");
        let manifest_file = File::create(&manifest_path)?;
        serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)?;
        eprintln!("[preprocessing] wrote {}", manifest_path.display());

        Ok(manifest)
    }
}

// ── LiDAR file helpers ────────────────────────────────────────────────────────

/// Read the bounding box, point count, and CRS EPSG from a `LiDAR` file header
/// without loading any points.
///
/// For both LAS and LAZ, the LAS header lives at the start of the file and is
/// read via `LasReader::new` (no points are decoded).  For COPC, the dedicated
/// `CopcReader::open_path` is used.
fn inspect_lidar_header(
    path: &Path,
) -> Result<(f64, f64, f64, f64, u64, Option<u32>)> {
    use std::fs::File;
    use std::io::BufReader;
    use wblidar::las::LasReader;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        // LAS and LAZ both carry a standard LAS header; open with LasReader to
        // inspect it without decoding any compressed point data.
        "las" | "laz" => {
            let f = File::open(path)?;
            let reader = LasReader::new(BufReader::new(f))?;
            let h = reader.header();
            let epsg = reader.crs().and_then(|c| c.epsg);
            Ok((
                h.min_x, h.min_y, h.max_x, h.max_y,
                h.point_count(),
                epsg,
            ))
        }
        "copc" => {
            // COPC files carry a standard LAS header; use LasReader for header
            // inspection (CRS/bounds) and CopcReader only for point streaming.
            let f = File::open(path)?;
            let reader = LasReader::new(BufReader::new(f))?;
            let h = reader.header();
            let epsg = reader.crs().and_then(|c| c.epsg);
            Ok((
                h.min_x, h.min_y, h.max_x, h.max_y,
                h.point_count(),
                epsg,
            ))
        }
        _ => Err(ClassifierError::UnsupportedFormat {
            path: path.display().to_string(),
        }),
    }
}

/// Stream all points from the input file, dispatching each to the partitioner.
fn stream_points(path: &Path, partitioner: &mut BlockPartitioner) -> Result<()> {
    use std::fs::File;
    use std::io::BufReader;
    use wblidar::io::PointReader;
    use wblidar::las::LasReader;
    use wblidar::laz::LazReader;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let mut pt = PointRecord::default();

    match ext.as_str() {
        "las" => {
            let f = File::open(path)?;
            let mut reader = LasReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(pt)?;
            }
        }
        "laz" => {
            let f = File::open(path)?;
            let mut reader = LazReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(pt)?;
            }
        }
        "copc" => {
            use wblidar::copc::CopcReader;
            let mut reader = CopcReader::open_path(path)?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(pt)?;
            }
        }
        _ => {
            return Err(ClassifierError::UnsupportedFormat {
                path: path.display().to_string(),
            });
        }
    }

    Ok(())
}

// ── Output serialisation helpers ──────────────────────────────────────────────

/// Write a `.feat` binary block file.
///
/// ## File layout
/// ```text
/// [header — 37 bytes]
///   magic:      4 bytes  = b"WBFT"
///   version:    u8       = 1
///   n_points:   u32 LE
///   n_features: u32 LE
///   block_id:   u64 LE
///   origin_x:   f64 LE
///   origin_y:   f64 LE
/// [data]
///   f32[n_points × n_features]  row-major, little-endian
/// ```
fn write_feat_file(
    path: &Path,
    block_id: u64,
    n_points: usize,
    origin_x: f64,
    origin_y: f64,
    features: &[[f32; N_FEATURES]],
) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(FEAT_MAGIC)?;
    w.write_all(&[FEAT_VERSION])?;
    #[allow(clippy::cast_possible_truncation)]
    w.write_all(&(n_points as u32).to_le_bytes())?;
    #[allow(clippy::cast_possible_truncation)]
    w.write_all(&(N_FEATURES as u32).to_le_bytes())?;
    w.write_all(&block_id.to_le_bytes())?;
    w.write_all(&origin_x.to_le_bytes())?;
    w.write_all(&origin_y.to_le_bytes())?;

    // Data — cast `&[[f32; N_FEATURES]]` to `&[u8]` via bytemuck
    let flat: &[f32] = bytemuck::cast_slice(features);
    let bytes: &[u8] = bytemuck::cast_slice(flat);
    w.write_all(bytes)?;
    w.flush()?;

    Ok(())
}

/// Write a debug CSV file with one row per point and 12 named columns.
fn write_debug_csv(path: &Path, features: &[[f32; N_FEATURES]]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "x_norm,y_norm,z_norm,intensity_norm,return_ratio,scan_angle_norm,\
         hag,linearity,planarity,sphericity,omnivariance,curvature"
    )?;

    for row in features {
        let line = row
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(w, "{line}")?;
    }

    w.flush()?;
    Ok(())
}
