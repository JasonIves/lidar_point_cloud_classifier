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
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use wblidar::PointRecord;

use std::collections::HashMap;

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
    pub grid_y_min: f64,
    /// Search radii used for multi-scale eigenvalue feature extraction.
    /// Empty list means single-scale using `search_radius`.
    #[serde(default)]
    pub search_radii: Vec<f64>,
    /// Whether the outlier removal pre-pass was applied before block partitioning.
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
    /// Overlap radius (projection units) used for border-point augmentation.
    /// `0.0` means no overlap (default, backward-compatible).
    #[serde(default)]
    pub block_overlap: f64,
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
                raw.len(),
                filtered.len(),
                raw.len().saturating_sub(filtered.len())
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

        eprintln!("[preprocessing] input:  {}", input_path.display());
        eprintln!("[preprocessing] extent: ({x_min:.2}, {y_min:.2}) → ({x_max:.2}, {y_max:.2})");
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

        // ── 5b. Build cell-lookup map for border-point loading (Stage 08) ──
        // Maps (col, row) → index into `retained` so the border-point loader
        // can find neighbour stubs without holding a mutable reference.
        // Only built when overlap is enabled to avoid the allocation overhead
        // on the default (overlap = 0) path.
        let stubs_by_cell: HashMap<(i32, i32), usize> = if config.block_overlap > 0.0 {
            retained
                .iter()
                .enumerate()
                .map(|(i, s)| ((s.col, s.row), i))
                .collect()
        } else {
            HashMap::new()
        };

        // ── 5c. Spill border points to disk sequentially (Stage 08) ───────
        //
        // MEMORY DESIGN: Border points must NOT be held in a Vec<Vec<PointRecord>>
        // spanning all blocks simultaneously.  With small block sizes (e.g. 5 m)
        // a dataset can have thousands of blocks, each with hundreds of border
        // points from up to 8 neighbours.  Accumulating all of them before the
        // parallel phase would consume O(n_blocks × border_density) RAM — the
        // root cause of the OOM crash.
        //
        // Instead, each block's border points are written to a temporary
        // `.border` spill file (same raw binary format as `.spill` files) and
        // only the file path is kept in memory.  The parallel closure loads and
        // immediately drops the border data for one block at a time, bounding
        // peak memory to:
        //
        //   (Rayon thread count) × (canonical_block + border_strip)
        //
        // rather than (all_blocks × border_strip).
        //
        // When overlap is disabled every entry is `None` — zero cost.
        let border_spill_paths: Vec<Option<PathBuf>> = if config.block_overlap > 0.0 {
            let mut paths = Vec::with_capacity(retained.len());
            for stub in &retained {
                let border_pts = load_border_points(
                    &stubs_by_cell,
                    &retained,
                    stub,
                    config.block_size,
                    config.block_overlap,
                )?;

                if border_pts.is_empty() {
                    paths.push(None);
                } else {
                    // Write to a uniquely named `.border` file in the output dir.
                    let path = config
                        .output_dir
                        .join(format!("block_{:05}.border", stub.id));
                    write_border_spill(&path, &border_pts)?;
                    paths.push(Some(path));
                }
            }
            paths
        } else {
            vec![None; retained.len()]
        };

        // ── 6. Optionally load the DTM raster ─────────────────────────────
        let dtm: Option<Arc<DtmView>> = config
            .hag_model
            .as_ref()
            .map(|path| -> Result<Arc<DtmView>> {
                let r = wbraster::Raster::read(path).map_err(|e| {
                    ClassifierError::Raster(format!("failed to load DTM '{}': {e}", path.display()))
                })?;
                Ok(Arc::new(DtmView::from_raster(&r)))
            })
            .transpose()?;

        // ── 7. Per-block parallel processing ─────────────────────────────
        // Zip each stub with its border-spill path.  The closure loads the
        // border file on-demand, processes the block, then deletes the file.
        // Peak memory is bounded to (thread_count × (canonical + border_strip)).
        let block_results: Vec<Result<BlockProcessResult>> = retained
            .into_par_iter()
            .zip(border_spill_paths.into_par_iter())
            .with_min_len(RAYON_MIN_CHUNK)
            .map(|(stub, border_path)| {
                let dtm_ref = dtm.as_deref();

                // Capture metadata before consuming the stub.
                let raw_count = stub.point_count;
                let block_id = stub.id;
                let origin_x = stub.origin_x;
                let origin_y = stub.origin_y;

                // Load canonical point data from spill files for this block only.
                // `block.points` is dropped at the end of this closure so
                // peak memory stays at (thread_count × largest_block_size).
                let block = stub.load()?;

                // (a) Load border points from the pre-written spill file (if any),
                //     then delete the file immediately to free disk space.
                //     When overlap is disabled, `border_path` is None and no I/O
                //     occurs here.
                let border_pts: Vec<PointRecord> = match border_path {
                    Some(ref p) => {
                        let pts = read_border_spill(p)?;
                        let _ = fs::remove_file(p);
                        pts
                    }
                    None => Vec::new(),
                };

                // (b) Build augmented k-d tree.
                //     Merge canonical + border into a single contiguous Vec once
                //     and reuse it for both the index and feature extraction.
                //     Border points are NEVER resampled or written to .feat files.
                let ctx: Vec<PointRecord> = if border_pts.is_empty() {
                    Vec::new() // no overlap: zero-copy path below
                } else {
                    let mut v = Vec::with_capacity(block.points.len() + border_pts.len());
                    v.extend_from_slice(&block.points);
                    v.extend_from_slice(&border_pts);
                    v
                };
                // border_pts no longer needed — drop it to free memory before
                // building the k-d tree (which itself allocates).
                drop(border_pts);

                let index = if ctx.is_empty() {
                    BlockSpatialIndex::build(&block.points)
                } else {
                    BlockSpatialIndex::build(&ctx)
                };

                // (c) Density-gated sampling — canonical points only.
                //     Border points are context-only and must not appear in output.
                let (sampled, sampled_indices, oversampled) =
                    resample_block(&block.points, config.target_points, block_id);

                let sampled_count = sampled.len();

                // (d–f) Extract all features (multi-scale or single-scale).
                //     When overlap is active, `ctx` (canonical + border) is the
                //     neighbourhood context so eigenvalue queries near block edges
                //     draw on real neighbour points.
                //     When overlap is disabled, canonical points only — identical
                //     to the pre-Stage-08 behaviour.
                let search_radii = config.search_radii_effective();
                let features = if ctx.is_empty() {
                    extract_features(
                        &sampled,
                        &block.points,
                        &index,
                        dtm_ref,
                        origin_x,
                        origin_y,
                        config.block_size,
                        &search_radii,
                        config.min_neighbors,
                    )
                } else {
                    extract_features(
                        &sampled,
                        &ctx,
                        &index,
                        dtm_ref,
                        origin_x,
                        origin_y,
                        config.block_size,
                        &search_radii,
                        config.min_neighbors,
                    )
                };
                let n_features = features.first().map_or(0, Vec::len);

                // (g) Serialise to .feat
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

                // (h) Optional debug CSV
                if config.debug_csv {
                    let csv_path = config.output_dir.join(format!("block_{block_id:05}.csv"));
                    write_debug_csv(&csv_path, &features, &search_radii)?;
                }

                let meta = BlockMeta {
                    id: block_id,
                    file: feat_filename,
                    origin_x,
                    origin_y,
                    raw_point_count: raw_count,
                    sampled_point_count: sampled_count,
                    oversampled,
                };

                // Only allocate the indices vec when caller wants them.
                let indices = if capture_indices {
                    sampled_indices
                } else {
                    Vec::new()
                };

                Ok(BlockProcessResult {
                    meta,
                    sampled_indices: indices,
                })
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
            search_radii: config.search_radii_effective(),
            outlier_removal: config.outlier_removal,
            outlier_radius: config.outlier_radius,
            outlier_elev_diff: config.outlier_elev_diff,
            outlier_use_median: config.outlier_use_median,
            block_overlap: config.block_overlap,
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
fn inspect_lidar_header(path: &Path) -> Result<(f64, f64, f64, f64, u64, Option<u32>)> {
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
            Ok((h.min_x, h.min_y, h.max_x, h.max_y, h.point_count(), epsg))
        }
        "copc" => {
            // COPC files carry a standard LAS header; use LasReader for header
            // inspection (CRS/bounds) and CopcReader only for point streaming.
            let f = File::open(path)?;
            let reader = LasReader::new(BufReader::new(f))?;
            let h = reader.header();
            let epsg = reader.crs().and_then(|c| c.epsg);
            Ok((h.min_x, h.min_y, h.max_x, h.max_y, h.point_count(), epsg))
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

// ── Border-point spill I/O (Stage 08) ────────────────────────────────────────
//
// Border points are written to `.border` files using the same compact binary
// layout as the main `.spill` files (31 bytes per point).  This keeps the
// format consistent and avoids introducing a new serialisation dependency.
//
// Layout per point (little-endian):
//   x(f64) y(f64) z(f64) intensity(u16) classification(u8)
//   return_number(u8) number_of_returns(u8) scan_angle(i16) = 31 bytes

/// Bytes per point in a border spill file — identical to the main spill format.
const BORDER_PT_BYTES: usize = 31;

/// Write a slice of `PointRecord`s to a `.border` spill file.
///
/// # Errors
/// Returns `ClassifierError` on any I/O failure.
fn write_border_spill(path: &Path, pts: &[PointRecord]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut buf = [0u8; BORDER_PT_BYTES];
    for pt in pts {
        buf[0..8].copy_from_slice(&pt.x.to_le_bytes());
        buf[8..16].copy_from_slice(&pt.y.to_le_bytes());
        buf[16..24].copy_from_slice(&pt.z.to_le_bytes());
        buf[24..26].copy_from_slice(&pt.intensity.to_le_bytes());
        buf[26] = pt.classification;
        buf[27] = pt.return_number;
        buf[28] = pt.number_of_returns;
        buf[29..31].copy_from_slice(&pt.scan_angle.to_le_bytes());
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a `.border` spill file back into a `Vec<PointRecord>`.
///
/// # Errors
/// Returns [`ClassifierError::SpillCorrupt`] if the file size is not a
/// multiple of `BORDER_PT_BYTES` or if any read fails.
fn read_border_spill(path: &Path) -> Result<Vec<PointRecord>> {
    let metadata = fs::metadata(path).map_err(|_| ClassifierError::SpillCorrupt {
        path: path.display().to_string(),
    })?;
    #[allow(clippy::cast_possible_truncation)]
    let file_bytes = metadata.len() as usize;
    if !file_bytes.is_multiple_of(BORDER_PT_BYTES) {
        return Err(ClassifierError::SpillCorrupt {
            path: path.display().to_string(),
        });
    }
    let n = file_bytes / BORDER_PT_BYTES;
    let mut pts = Vec::with_capacity(n);
    let mut file = File::open(path)?;
    let mut buf = [0u8; BORDER_PT_BYTES];
    for _ in 0..n {
        file.read_exact(&mut buf)?;
        let corrupt = || ClassifierError::SpillCorrupt {
            path: path.display().to_string(),
        };
        let pt = PointRecord {
            x: f64::from_le_bytes(buf[0..8].try_into().map_err(|_| corrupt())?),
            y: f64::from_le_bytes(buf[8..16].try_into().map_err(|_| corrupt())?),
            z: f64::from_le_bytes(buf[16..24].try_into().map_err(|_| corrupt())?),
            intensity: u16::from_le_bytes(buf[24..26].try_into().map_err(|_| corrupt())?),
            classification: buf[26],
            return_number: buf[27],
            number_of_returns: buf[28],
            scan_angle: i16::from_le_bytes(buf[29..31].try_into().map_err(|_| corrupt())?),
            ..PointRecord::default()
        };
        pts.push(pt);
    }
    Ok(pts)
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
        "x_norm",
        "y_norm",
        "z_norm",
        "intensity_norm",
        "return_ratio",
        "scan_angle_norm",
        "hag",
    ];
    let eigen_names = [
        "linearity",
        "planarity",
        "sphericity",
        "omnivariance",
        "curvature",
    ];
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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BlockManifest serde round-trip ────────────────────────────────────────

    /// Verify that `block_overlap` survives a JSON round-trip and that an older
    /// manifest without the field deserialises to `0.0` (backward-compatible).
    #[test]
    fn test_manifest_block_overlap_round_trip() {
        let manifest = BlockManifest {
            source: "test.las".to_string(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            grid_cols: 4,
            grid_rows: 4,
            grid_x_min: 0.0,
            grid_y_min: 0.0,
            search_radii: vec![],
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap: 12.5,
            blocks: vec![],
        };

        let json = serde_json::to_string(&manifest).unwrap();
        assert!(
            json.contains("\"block_overlap\":12.5"),
            "field missing from JSON"
        );

        let de: BlockManifest = serde_json::from_str(&json).unwrap();
        assert!((de.block_overlap - 12.5).abs() < 1e-9);
    }

    /// Older `blocks.json` files that pre-date Stage 08 lack `block_overlap`.
    /// Deserialisation must succeed and default to `0.0`.
    #[test]
    fn test_manifest_block_overlap_default_on_missing() {
        // Minimal JSON without block_overlap field.
        let json = r#"{
            "source": "old.las",
            "block_size": 50.0,
            "target_points": 1024,
            "min_density": 1.0,
            "search_radius": 1.0,
            "min_neighbors": 8,
            "crs_epsg": null,
            "blocks": []
        }"#;

        let de: BlockManifest = serde_json::from_str(json).unwrap();
        assert_eq!(de.block_overlap, 0.0, "missing field should default to 0.0");
    }

    // ── load_border_points geometry ───────────────────────────────────────────

    /// `load_border_points` with an empty `stubs_by_cell` map (overlap disabled
    /// path) must return an empty Vec without error.
    #[test]
    fn test_load_border_points_no_neighbours() {
        use std::collections::HashMap;
        use wblidar::PointRecord;

        let stubs_by_cell: HashMap<(i32, i32), usize> = HashMap::new();
        let all_stubs: Vec<crate::preprocessing::block_partitioner::BlockStub> = vec![];

        // Build a real stub via BlockPartitioner so we have a valid target.
        let dir = tempfile::tempdir().unwrap();
        use crate::preprocessing::block_partitioner::BlockPartitioner;
        let mut partitioner = BlockPartitioner::new(0.0, 0.0, 50.0, 50.0, 50.0, dir.path());
        let mut pt = PointRecord::default();
        pt.x = 25.0;
        pt.y = 25.0;
        pt.z = 10.0;
        partitioner.add_point(pt).unwrap();
        let stubs = partitioner.finalize_stubs().unwrap();
        assert_eq!(stubs.len(), 1);

        // With an empty stubs_by_cell every neighbour lookup misses → empty border.
        let target = &stubs[0];
        let result = load_border_points(&stubs_by_cell, &all_stubs, target, 50.0, 5.0).unwrap();
        assert!(result.is_empty(), "no neighbours → empty border");
    }

    // ── CLI validation ────────────────────────────────────────────────────────

    /// `block_overlap` defaults to `0.0` in `PreprocessConfig`.
    #[test]
    fn test_preprocess_config_default_overlap() {
        let cfg = crate::preprocessing::PreprocessConfig::default();
        assert_eq!(cfg.block_overlap, 0.0);
    }

    // ── Border spill I/O round-trip ───────────────────────────────────────────

    /// Write a set of PointRecords to a `.border` file and read them back.
    /// All serialised fields must survive the round-trip exactly.
    #[test]
    fn test_border_spill_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.border");

        let pts: Vec<PointRecord> = (0..20)
            .map(|i| PointRecord {
                x: i as f64 * 1.5,
                y: i as f64 * 0.7,
                z: i as f64 * 0.3,
                intensity: (i * 1000) as u16,
                classification: (i % 8) as u8,
                return_number: 1,
                number_of_returns: 2,
                scan_angle: (i as i16) - 10,
                ..PointRecord::default()
            })
            .collect();

        write_border_spill(&path, &pts).unwrap();
        let recovered = read_border_spill(&path).unwrap();

        assert_eq!(recovered.len(), pts.len(), "point count must match");
        for (a, b) in pts.iter().zip(recovered.iter()) {
            assert!((a.x - b.x).abs() < 1e-12, "x mismatch");
            assert!((a.y - b.y).abs() < 1e-12, "y mismatch");
            assert!((a.z - b.z).abs() < 1e-12, "z mismatch");
            assert_eq!(a.intensity, b.intensity, "intensity mismatch");
            assert_eq!(a.classification, b.classification, "classification mismatch");
            assert_eq!(a.return_number, b.return_number, "return_number mismatch");
            assert_eq!(
                a.number_of_returns, b.number_of_returns,
                "number_of_returns mismatch"
            );
            assert_eq!(a.scan_angle, b.scan_angle, "scan_angle mismatch");
        }
    }

    /// An empty border spill file round-trips to an empty Vec.
    #[test]
    fn test_border_spill_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.border");
        write_border_spill(&path, &[]).unwrap();
        let recovered = read_border_spill(&path).unwrap();
        assert!(recovered.is_empty(), "empty write → empty read");
    }
}

// ── Stage 08: border-point loader ────────────────────────────────────────────

/// Collect points from the up-to-8 grid neighbours of `target` that fall
/// within the expanded bounding box `[origin - overlap, origin + block_size + overlap]`.
///
/// This function is called **sequentially** before the Rayon parallel phase so
/// that all spill files are guaranteed to exist (no concurrent `load()` calls
/// have deleted them yet).  The returned `Vec<PointRecord>` is written to a
/// `.border` spill file by the caller; it is not held in memory across blocks.
///
/// Points that belong to the target block itself are not included — only
/// genuine cross-boundary neighbours are returned.
fn load_border_points(
    stubs_by_cell: &HashMap<(i32, i32), usize>,
    all_stubs: &[BlockStub],
    target: &BlockStub,
    block_size: f64,
    overlap: f64,
) -> Result<Vec<PointRecord>> {
    // Expanded bounding box of the target block in projection units.
    let x_lo = target.origin_x - overlap;
    let x_hi = target.origin_x + block_size + overlap;
    let y_lo = target.origin_y - overlap;
    let y_hi = target.origin_y + block_size + overlap;

    let mut border: Vec<PointRecord> = Vec::new();

    // Iterate over all 8 cardinal + diagonal neighbours.
    for dc in -1_i32..=1 {
        for dr in -1_i32..=1 {
            if dc == 0 && dr == 0 {
                continue; // skip the target block itself
            }
            let key = (target.col + dc, target.row + dr);
            if let Some(&idx) = stubs_by_cell.get(&key) {
                let neighbour = &all_stubs[idx];
                // Read neighbour spill files without deleting them.
                let pts = neighbour.read_points()?;
                for pt in pts {
                    // Keep only points that fall inside the expanded bbox.
                    if pt.x >= x_lo && pt.x <= x_hi && pt.y >= y_lo && pt.y <= y_hi {
                        border.push(pt);
                    }
                }
            }
        }
    }

    Ok(border)
}
