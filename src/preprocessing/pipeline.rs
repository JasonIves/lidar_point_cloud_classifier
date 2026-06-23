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

#![allow(clippy::doc_markdown, clippy::missing_errors_doc)]

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use wblidar::PointRecord;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::{
    block_partitioner::{BlockPartitioner, BlockStub},
    feature_extractor::extract_features,
    normalizer::{resample_block, DtmView},
    spatial_index::BlockSpatialIndex,
    PreprocessConfig, FEAT_MAGIC, FEAT_VERSION, RAYON_MIN_CHUNK,
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
    /// Number of grid columns derived from the LiDAR header bounding box.
    /// This is the authoritative value for block-ID arithmetic (`row * grid_cols + col`).
    /// Never re-derive from retained block origins — the density filter may have
    /// removed trailing columns, which would produce a smaller (wrong) value.
    #[serde(default)]
    pub grid_cols: u32,
    /// Number of grid rows derived from the LiDAR header bounding box.
    #[serde(default)]
    pub grid_rows: u32,
    /// Header-derived south-west X origin — the same value passed to `BlockPartitioner`.
    #[serde(default)]
    pub grid_x_min: f64,
    /// Header-derived south-west Y origin — the same value passed to `BlockPartitioner`.
    #[serde(default)]
<<<<<<< HEAD
    pub grid_y_min: f64,    /// Search radii used for multi-scale eigenvalue feature extraction.
    /// Empty list means single-scale using `search_radius`.
    #[serde(default)]
    pub search_radii: Vec<f64>,    /// Whether the outlier removal pre-pass was applied before block partitioning.
    #[serde(default)]
    pub outlier_removal: bool,
    /// Neighbourhood radius used for outlier elevation residual calculation.
    #[serde(default = "default_outlier_radius")]
    pub outlier_radius: f64,
    /// Absolute elevation residual threshold used during outlier removal.
    #[serde(default = "default_outlier_elev_diff")]
    pub outlier_elev_diff: f64,
    /// Whether median (true) or mean (false) was used for the neighbourhood baseline.
    #[serde(default)]
    pub outlier_use_median: bool,
=======
    pub grid_y_min: f64,
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    pub blocks: Vec<BlockMeta>,
}

// Serde default helpers for outlier fields — used when deserialising older
// `blocks.json` files that pre-date Stage 04.
fn default_outlier_radius() -> f64 {
    2.0
}
fn default_outlier_elev_diff() -> f64 {
    50.0
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

/// Internal per-block processing result that also carries the sampling indices.
///
/// Used by the labeled-preprocessing pipeline to retrieve the `classification`
/// byte for each sampled point without a second LiDAR pass (Option A from spec).
#[derive(Debug)]
pub struct BlockProcessResult {
    /// Public metadata (serialised to `blocks.json`).
    pub meta: BlockMeta,
    /// 0-based indices into the raw per-block `Vec<PointRecord>` for each
    /// sampled output point row.  Length equals `meta.sampled_point_count`.
    pub sampled_indices: Vec<usize>,
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
    pub fn run(config: &PreprocessConfig) -> Result<BlockManifest> {
        let (manifest, _) = Self::run_internal(config, false)?;
        Ok(manifest)
    }

    /// Run the pipeline and also return per-block `sampled_indices`.
    ///
    /// Used by the labeled-preprocessing pipeline (Stage 03) to retrieve the
    /// `classification` byte for each sampled point without a second LiDAR pass.
    ///
    /// # Errors
    /// Same as [`run`].
    pub fn run_with_indices(
        config: &PreprocessConfig,
    ) -> Result<(BlockManifest, Vec<BlockProcessResult>)> {
        Self::run_internal(config, true)
    }

    #[allow(clippy::too_many_lines)]
    fn run_internal(
        config: &PreprocessConfig,
        capture_indices: bool,
    ) -> Result<(BlockManifest, Vec<BlockProcessResult>)> {
        // ── 0. Set up Rayon thread pool ────────────────────────────────────
        if let Some(threads) = config.threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok(); // ignore error if global pool was already initialised
        }

        // ── 1. Ensure output directory exists ─────────────────────────────
        fs::create_dir_all(&config.output_dir)?;

        // ── 1b. Optional outlier removal pre-pass ─────────────────────────────────
        // When enabled, all points are loaded into memory, filtered with the local
        // elevation residual algorithm (outlier_filter::apply), and fed to the
        // partitioner directly — no temp file needed.
        //
        // Header inspection always uses `config.input` (the original file) because
        // the bounding box and CRS are unchanged by outlier filtering.
        //
        // NOTE (2026-06-17): This previously called `LidarRemoveOutliersTool` via
        // wbtools_oss and wrote a temp .las file.  That crate was removed for build
        // speed; see src/preprocessing/outlier_filter.rs for the revert path.
        let input_path = &config.input;
        let outlier_filtered_pts: Option<Vec<PointRecord>> = if config.outlier_removal {
            eprintln!(
                "[preprocessing] outlier removal: radius={:.2}, elev_diff={:.2}, use_median={}",
                config.outlier_radius, config.outlier_elev_diff, config.outlier_use_median
            );
            let raw = load_all_points(input_path)?;
            let filtered = crate::preprocessing::outlier_filter::apply(
                &raw,
                config.outlier_radius,
                config.outlier_elev_diff,
                config.outlier_use_median,
            );
            eprintln!(
                "[preprocessing] outlier removal: {} → {} points ({} removed)",
                raw.len(), filtered.len(), raw.len().saturating_sub(filtered.len())
            );
            Some(filtered)
        } else {
            None
        };

        // ── 2. Open the LiDAR file and inspect the header ───────────────────────────
        let (x_min, y_min, x_max, y_max, total_points, crs_epsg) =
            inspect_lidar_header(input_path)?;

        // Compute grid geometry from the header bounding box.  These values
        // are the authoritative source for block-ID arithmetic throughout the
        // entire pipeline — store them in the manifest so no consumer ever
        // needs to re-derive them from retained block origins.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let grid_cols = (((x_max - x_min) / config.block_size).ceil() as u32).max(1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let grid_rows = (((y_max - y_min) / config.block_size).ceil() as u32).max(1);

        eprintln!(
            "[preprocessing] input:  {}",
            input_path.display()
        );
        eprintln!(
            "[preprocessing] extent: ({x_min:.2}, {y_min:.2}) → ({x_max:.2}, {y_max:.2})"
        );
        eprintln!("[preprocessing] grid: {grid_cols} cols × {grid_rows} rows");
        eprintln!("[preprocessing] points: {total_points}");

        // ── 3. Stream / feed points into the block partitioner ───────────────────
        // If outlier removal produced a filtered Vec, feed it directly.
        // Otherwise stream from the original file (no allocation).
        let mut partitioner = BlockPartitioner::new(
            x_min,
            y_min,
            x_max,
            y_max,
            config.block_size,
            &config.output_dir,
        );
        if let Some(ref filtered) = outlier_filtered_pts {
            for pt in filtered {
                partitioner.add_point(*pt)?;
            }
        } else {
            stream_points(input_path, &mut partitioner)?;
        }

        // ── 4. Finalise — flush in-memory cells to disk, return lightweight stubs ───
        // `finalize_stubs()` clears the heap before returning: all block data is
        // on disk as spill files.  Point data is loaded per-block on-demand in
        // the parallel Step 7 closure and dropped immediately after, bounding
        // peak memory to (thread_count × largest_block_size) rather than the
        // entire dataset.
        let stubs = partitioner.finalize_stubs()?;
        eprintln!("[preprocessing] raw blocks: {}", stubs.len());

        // ── 5. Density gate (on stub point counts — no data load needed) ──────
        let cell_area = config.block_size * config.block_size;
        let retained: Vec<BlockStub> = stubs
            .into_iter()
            .filter(|s| {
                #[allow(clippy::cast_precision_loss)]
                let density = s.point_count as f64 / cell_area;
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
        let block_results: Vec<Result<BlockProcessResult>> = retained
            .into_par_iter()
            .with_min_len(RAYON_MIN_CHUNK)
            .map(|stub| {
                let dtm_ref = dtm.as_deref();

                // Capture metadata before consuming the stub.
                let raw_count = stub.point_count;
                let block_id  = stub.id;
                let origin_x  = stub.origin_x;
                let origin_y  = stub.origin_y;

                // Load point data from spill files for this block only.
                // `block.points` is dropped at the end of this closure so
                // peak memory stays at (thread_count × largest_block_size).
                let block = stub.load()?;

                // (a) Build k-d tree from full unsampled block
                let index = BlockSpatialIndex::build(&block.points);

                // (b) Density-gated sampling — now returns indices too
                let (sampled, sampled_indices, oversampled) =
<<<<<<< HEAD
                    resample_block(&block.points, config.target_points, block_id);
=======
                    resample_block(&block.points, config.target_points, block.id);
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4

                let sampled_count = sampled.len();

                // (c–e) Extract all features (multi-scale or single-scale)
                let search_radii = config.search_radii_effective();
                let features = extract_features(
                    &sampled,
                    &block.points,
                    &index,
                    dtm_ref,
                    origin_x,
                    origin_y,
                    config.block_size,
                    &search_radii,
                    config.min_neighbors,
                );
                let n_features = features.first().map_or(0, Vec::len);

                // (f) Serialise to .feat
                let feat_filename = format!("block_{block_id:05}.feat");
                let feat_path = config.output_dir.join(&feat_filename);
                write_feat_file(
                    &feat_path,
                    block_id,
                    sampled_count,
                    n_features,
                    origin_x,
                    origin_y,
                    &features,
                )?;

                // (g) Optional debug CSV
                if config.debug_csv {
                    let csv_path = config.output_dir.join(format!("block_{block_id:05}.csv"));
                    write_debug_csv(&csv_path, &features, &search_radii)?;
                }

                let meta = BlockMeta {
<<<<<<< HEAD
                    id: block_id,
=======
                    id: block.id,
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
                    file: feat_filename,
                    origin_x,
                    origin_y,
                    raw_point_count: raw_count,
                    sampled_point_count: sampled_count,
                    oversampled,
                };

                // Only allocate the indices vec when caller wants them.
                let indices = if capture_indices { sampled_indices } else { Vec::new() };

                Ok(BlockProcessResult { meta, sampled_indices: indices })
            })
            .collect();

        // Collect results, propagating the first error.
        let mut process_results = Vec::with_capacity(block_results.len());
        for r in block_results {
            process_results.push(r?);
        }
        process_results.sort_by_key(|r| r.meta.id);

        let block_metas: Vec<BlockMeta> = process_results.iter().map(|r| r.meta.clone()).collect();

        eprintln!("[preprocessing] wrote {} .feat files", block_metas.len());

        // ── 8. Write blocks.json ──────────────────────────────────────────
        let manifest = BlockManifest {
            // Always record the original user-supplied path, not the temp file.
            source: input_path.display().to_string(),
            block_size: config.block_size,
            target_points: config.target_points,
            min_density: config.min_density,
            search_radius: config.search_radius,
            min_neighbors: config.min_neighbors,
            crs_epsg,
            grid_cols,
            grid_rows,
            grid_x_min: x_min,
            grid_y_min: y_min,
<<<<<<< HEAD
            search_radii: config.search_radii_effective(),
            outlier_removal: config.outlier_removal,
            outlier_radius: config.outlier_radius,
            outlier_elev_diff: config.outlier_elev_diff,
            outlier_use_median: config.outlier_use_median,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
            blocks: block_metas,
        };

        let manifest_path = config.output_dir.join("blocks.json");
        let manifest_file = File::create(&manifest_path)?;
        serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)?;
        eprintln!("[preprocessing] wrote {}", manifest_path.display());

        Ok((manifest, process_results))
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

// ── Outlier removal helper ────────────────────────────────────────────────────
// REMOVED (2026-06-17): run_outlier_removal() via wbtools_oss has been replaced
// by the in-memory outlier_filter::apply() call in run_internal() Step 1b.
// See src/preprocessing/outlier_filter.rs for the revert path.

// ── LiDAR load helper ────────────────────────────────────────────────────────

/// Load all points from a LiDAR file into a `Vec<PointRecord>`.
/// Used only when outlier removal is enabled; for the normal path
/// `stream_points` is used to avoid loading the full file into memory.
fn load_all_points(path: &Path) -> Result<Vec<PointRecord>> {
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
    let mut out = Vec::new();

    match ext.as_str() {
        "las" => {
            let f = File::open(path)?;
            let mut reader = LasReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                out.push(pt);
            }
        }
        "laz" => {
            let f = File::open(path)?;
            let mut reader = LazReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                out.push(pt);
            }
        }
        "copc" => {
            use wblidar::copc::CopcReader;
            let mut reader = CopcReader::open_path(path)?;
            while reader.read_point(&mut pt)? {
                out.push(pt);
            }
        }
        _ => {
            return Err(ClassifierError::UnsupportedFormat {
                path: path.display().to_string(),
            });
        }
    }

    Ok(out)
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
///   n_features: u32 LE  (7 + 5 × n_radii)
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
    n_features: usize,
    origin_x: f64,
    origin_y: f64,
    features: &[Vec<f32>],
) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header
    w.write_all(FEAT_MAGIC)?;
    w.write_all(&[FEAT_VERSION])?;
    #[allow(clippy::cast_possible_truncation)]
    w.write_all(&(n_points as u32).to_le_bytes())?;
    #[allow(clippy::cast_possible_truncation)]
    w.write_all(&(n_features as u32).to_le_bytes())?;
    w.write_all(&block_id.to_le_bytes())?;
    w.write_all(&origin_x.to_le_bytes())?;
    w.write_all(&origin_y.to_le_bytes())?;

    // Data — write each row as raw f32 bytes
    for row in features {
        let bytes: &[u8] = bytemuck::cast_slice(row.as_slice());
        w.write_all(bytes)?;
    }
    w.flush()?;

    Ok(())
}

/// Write a debug CSV file with dynamic column names derived from `search_radii`.
fn write_debug_csv(path: &Path, features: &[Vec<f32>], search_radii: &[f64]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Build dynamic header.
    let mut cols = vec![
        "x_norm", "y_norm", "z_norm", "intensity_norm",
        "return_ratio", "scan_angle_norm", "hag",
    ];
    let eigen_names = ["linearity", "planarity", "sphericity", "omnivariance", "curvature"];
    let mut extra: Vec<String> = Vec::new();
    for r in search_radii {
        for name in &eigen_names {
            extra.push(format!("{name}_r{r:.2}"));
        }
    }
    let extra_refs: Vec<&str> = extra.iter().map(String::as_str).collect();
    cols.extend_from_slice(&extra_refs);
    writeln!(w, "{}", cols.join(","))?;

    for row in features {
        let line = row.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(",");
        writeln!(w, "{line}")?;
    }

    w.flush()?;
    Ok(())
}
