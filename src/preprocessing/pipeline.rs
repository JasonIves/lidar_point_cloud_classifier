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
    lite_point::LitePoint,
    normalizer::{resample_block, sample_halo, DtmView, ZNormalization},
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
    #[serde(default)]
    pub oversample_jitter: f64,
    /// Whether the legacy per-block `z_norm` normalisation was used instead
    /// of the default whole-file absolute range (z_norm bug fix, Stage 37
    /// follow-up). `false` (default) means the fixed/global mode was used.
    #[serde(default)]
    pub z_norm_use_block_relative: bool,
    /// Halo budget fraction φ used for this run (Stage 45). `0.0` (default,
    /// backward-compatible) means no halo rows were written.
    #[serde(default)]
    pub halo_fraction: f64,
    pub blocks: Vec<BlockMeta>,
}

/// File name of the intermediate ground-only LAS produced by the Stage 38
/// auto-DTM ground filter, written under `PreprocessConfig::output_dir`.
const AUTO_GROUND_LAS: &str = "_auto_ground.las";

/// File name of the auto-generated bare-earth DTM raster (Stage 38), written
/// under `PreprocessConfig::output_dir`.
const AUTO_DTM_TIF: &str = "_auto_dtm.tif";

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
    /// Number of trailing rows in the `.feat` payload that are halo
    /// (overlap-margin) samples rather than canonical core samples
    /// (Stage 45). `0` (default, backward-compatible) = all-core block.
    #[serde(default)]
    pub n_halo: usize,
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
    /// sampled output **core** point row.  Length equals
    /// `meta.sampled_point_count − meta.n_halo`.
    pub sampled_indices: Vec<usize>,
    /// Raw ASPRS classification bytes of the halo rows (Stage 45), in the
    /// same order as the `.feat` halo rows.  Populated only when
    /// `capture_indices` is true (labeled pipeline); empty otherwise.
    pub halo_classifications: Vec<u8>,
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
        // When enabled, `wbtools_oss::LidarRemoveOutliersTool` is invoked directly
        // via the `wbcore::Tool` trait.  The tool writes a temp `_outlier_cleaned.las`
        // file in the output directory; all subsequent steps (header inspection,
        // point streaming) use that cleaned file as the effective input.  The temp
        // file is deleted once the pipeline run completes.
        //
        // See docs/stages/stage-04-outlier-removal.md (original design) and
        // docs/stages/stage-30-whitebox-git-dependency-integration.md (restoration).
        let outlier_cleaned_path = config.output_dir.join("_outlier_cleaned.las");
        let input_path_owned: PathBuf;
        let input_path: &Path = if config.outlier_removal {
            eprintln!(
                "[preprocessing] outlier removal: radius={:.2}, elev_diff={:.2}, use_median={}",
                config.outlier_radius, config.outlier_elev_diff, config.outlier_use_median
            );
            run_outlier_removal(&config.input, &outlier_cleaned_path, config)?;
            eprintln!(
                "[preprocessing] outlier removal: wrote {}",
                outlier_cleaned_path.display()
            );
            input_path_owned = outlier_cleaned_path.clone();
            &input_path_owned
        } else {
            &config.input
        };

        // ── 2. Open the LiDAR file and inspect the header ───────────────────────────
        // NOTE: header inspection is intentionally performed *before* the new
        // Step 1c eigenvalue pre-pass (below) rather than strictly after it as
        // the Stage 30 doc's "Approved Design" narratively numbers the steps.
        // The pre-pass's memory-budget decision (Step 5b) needs the header's
        // point count, so the single `inspect_lidar_header` call is shared by
        // both Step 1c and the grid-geometry computation that follows —
        // avoiding a second, redundant header read.
        let (x_min, y_min, x_max, y_max, z_min, z_max, total_points, crs_epsg) =
            inspect_lidar_header(input_path)?;

        // ── Z-normalisation strategy resolution (z_norm bug fix, Stage 37
        //    follow-up) ────────────────────────────────────────────────────
        // Resolved once here, from the header's whole-file z bounds, so
        // every block uses the same absolute reference regardless of which
        // tile a point lands in — the fix for the "patchwork quilt"
        // tile-boundary discontinuity caused by the legacy per-block
        // z_norm behaviour. `--z-norm-block-relative` opts back into the
        // legacy neighbour-dependent mode for reproducibility/comparison.
        let z_norm_strategy: ZNormalization = if config.z_norm_use_block_relative {
            ZNormalization::BlockMinMax
        } else {
            ZNormalization::Global { z_min, z_max }
        };

        // ── 1c. Eigenvalue-feature pre-pass (Stage 30, Step 5b/5c/5d) ───────────────
        // Runs `wbtools_oss::LidarEigenvalueFeaturesTool` once over the whole
        // (possibly outlier-cleaned) input file when the header-derived point
        // count fits within `config.eigen_memory_budget_bytes`; otherwise the
        // memory-gated spatial-split path (Stage 30 Step 5d) is used instead
        // — see `run_eigenvalue_prepass_split`.
        //
        // As of Step 5e+5f+5g, this table is joined into per-block feature
        // extraction below (Step 7) by each point's original stream index
        // (`block.point_indices[i]`), replacing the prior per-block, per-
        // radius local eigenvalue computation entirely.
        let eigen_table = run_eigenvalue_prepass(
            input_path,
            &config.output_dir,
            config,
            total_points,
            (x_min, y_min, x_max, y_max),
        )?;

        if eigen_table.is_empty() {
            eprintln!("[preprocessing] eigen pre-pass: 0 rows (empty input)");
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean_linearity: f64 =
                eigen_table.iter().map(|r| f64::from(r[3])).sum::<f64>() / eigen_table.len() as f64;
            eprintln!(
                "[preprocessing] eigen pre-pass: {} rows parsed (mean linearity = {mean_linearity:.4})",
                eigen_table.len()
            );
        }
        let eigen_table = Arc::new(eigen_table);

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

        // ── 3. Stream points into the block partitioner ───────────────────────────
        // `input_path` is either the original file or the outlier-cleaned temp
        // file from Step 1b; either way it is streamed without loading the
        // full file into memory.
        let mut partitioner = BlockPartitioner::new(
            x_min,
            y_min,
            x_max,
            y_max,
            config.block_size,
            &config.output_dir,
        );
        stream_points(input_path, &mut partitioner)?;

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

        // ── 6. Resolve the DTM raster (Stage 38 — 3-way priority) ─────────
        // Priority: (1) explicit --hag-model wins; (2) else if auto_dtm is on,
        // auto-generate a bare-earth DTM from the (possibly outlier-cleaned)
        // input; (3) else None → block-min-z proxy.
        //
        // `auto_dtm_path` tracks the auto-generated `_auto_dtm.tif` so it (and
        // its `_auto_ground.las` sibling) can be cleaned up in Step 9 unless
        // `config.keep_auto_dtm` is set.
        let mut auto_dtm_path: Option<PathBuf> = None;
        let dtm: Option<Arc<DtmView>> = if let Some(path) = config.hag_model.as_ref() {
            let r = wbraster::Raster::read(path).map_err(|e| {
                ClassifierError::Raster(format!("failed to load DTM '{}': {e}", path.display()))
            })?;
            Some(Arc::new(DtmView::from_raster(&r)))
        } else if config.auto_dtm {
            eprintln!(
                "[preprocessing] auto-DTM: generating bare-earth DTM (resolution={:.3})",
                config.auto_dtm_resolution
            );
            let dtm_path = run_auto_dtm(input_path, &config.output_dir, config)?;
            eprintln!("[preprocessing] auto-DTM: wrote {}", dtm_path.display());
            let r = wbraster::Raster::read(&dtm_path).map_err(|e| {
                ClassifierError::Raster(format!(
                    "failed to load auto-generated DTM '{}': {e}",
                    dtm_path.display()
                ))
            })?;
            auto_dtm_path = Some(dtm_path);
            Some(Arc::new(DtmView::from_raster(&r)))
        } else {
            None
        };

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

                // (a) Load border points (index-carrying pairs) from the
                //     pre-written spill file (if any), then delete the file
                //     immediately to free disk space.  When overlap is
                //     disabled, `border_path` is None and no I/O occurs here.
                let border_pts: Vec<(u64, LitePoint)> = match border_path {
                    Some(ref p) => {
                        let pts = read_border_spill(p)?;
                        let _ = fs::remove_file(p);
                        pts
                    }
                    None => Vec::new(),
                };

                // (b) Halo sampling (Stage 45): reserve `n_halo_target` rows of
                //     the fixed-N tensor for points sampled from the border
                //     strip (subsample-only, never oversampled, never jittered
                //     — duplicates add no cross-boundary context).  When the
                //     strip is sparse (or absent, e.g. dataset-boundary
                //     blocks), all available halo points are taken and the
                //     core sample backfills the remainder, so the tensor is
                //     always exactly `target_points` rows.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss
                )]
                let n_halo_target =
                    (config.halo_fraction * config.target_points as f64).round() as usize;
                let halo_pts: Vec<(u64, LitePoint)> = if n_halo_target > 0 {
                    sample_halo(&border_pts, n_halo_target, block_id)
                } else {
                    Vec::new()
                };
                let n_halo = halo_pts.len();
                // The border strip is no longer needed once the halo sample
                // is drawn — free it before the larger allocations below.
                drop(border_pts);

                // (c) Density-gated core sampling to `target_points − n_halo`
                //     (one seeded call, computed after halo actuals are
                //     known; with `halo_fraction == 0` this degenerates to
                //     the pre-Stage-45 full-N core sample, bit-identical).
                let core_target = config.target_points.saturating_sub(n_halo);
                let (core_sampled, sampled_indices, oversampled) = resample_block(
                    &block.points,
                    core_target,
                    block_id,
                    config.oversample_jitter,
                );

                // Combined row layout written to the .feat payload:
                // [core rows | halo rows].
                let mut sampled: Vec<LitePoint> = core_sampled;
                sampled.extend(halo_pts.iter().map(|&(_, pt)| pt));
                let sampled_count = sampled.len();

                // (d) Look up each row's precomputed eigenvalue row from the
                //     whole-file pre-pass table via its original stream
                //     index — core rows via `block.point_indices[sampled_idx]`
                //     (existing join), halo rows via the index carried in the
                //     border spill record (identical join, identical
                //     zero-row fallback for a table miss).
                let mut eigen_rows: Vec<[f32; 10]> = sampled_indices
                    .iter()
                    .map(|&sampled_idx| {
                        // On 32-bit targets a `u64` point index could in principle
                        // exceed `usize::MAX`; fall back to an out-of-range sentinel
                        // (`usize::MAX`) rather than truncating or panicking — the
                        // subsequent `.get()` lookup simply misses and yields the
                        // all-zero default row, exactly as for any other cache miss.
                        let point_idx =
                            usize::try_from(block.point_indices[sampled_idx]).unwrap_or(usize::MAX);
                        eigen_table.get(point_idx).copied().unwrap_or([0.0f32; 10])
                    })
                    .collect();
                eigen_rows.extend(halo_pts.iter().map(|&(original_idx, _)| {
                    let point_idx = usize::try_from(original_idx).unwrap_or(usize::MAX);
                    eigen_table.get(point_idx).copied().unwrap_or([0.0f32; 10])
                }));

                // (e–f) Extract full feature vectors: scalar (from `sampled`)
                //     combined with the precomputed eigenvalue rows above.
                let features = extract_features(
                    &sampled,
                    &eigen_rows,
                    dtm_ref,
                    origin_x,
                    origin_y,
                    config.block_size,
                    config.hag_normalization,
                    z_norm_strategy,
                );
                let n_features = features.first().map_or(0, Vec::len);

                // (g) Serialise to .feat (v2 header records the halo row count)
                let feat_filename = format!("block_{block_id:05}.feat");
                let feat_path = config.output_dir.join(&feat_filename);
                write_feat_file(
                    &feat_path,
                    block_id,
                    sampled_count,
                    n_features,
                    n_halo,
                    origin_x,
                    origin_y,
                    &features,
                )?;

                // (h) Optional debug CSV
                if config.debug_csv {
                    let csv_path = config.output_dir.join(format!("block_{block_id:05}.csv"));
                    write_debug_csv(&csv_path, &features)?;
                }

                let meta = BlockMeta {
                    id: block_id,
                    file: feat_filename,
                    origin_x,
                    origin_y,
                    raw_point_count: raw_count,
                    sampled_point_count: sampled_count,
                    oversampled,
                    n_halo,
                };

                // Only allocate the indices/classifications vecs when the
                // caller wants them (labeled pipeline).
                let indices = if capture_indices {
                    sampled_indices
                } else {
                    Vec::new()
                };
                let halo_classes: Vec<u8> = if capture_indices {
                    halo_pts.iter().map(|&(_, pt)| pt.classification).collect()
                } else {
                    Vec::new()
                };

                Ok(BlockProcessResult {
                    meta,
                    sampled_indices: indices,
                    halo_classifications: halo_classes,
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
            outlier_removal: config.outlier_removal,
            outlier_radius: config.outlier_radius,
            outlier_elev_diff: config.outlier_elev_diff,
            outlier_use_median: config.outlier_use_median,
            block_overlap: config.block_overlap,
            oversample_jitter: config.oversample_jitter,
            z_norm_use_block_relative: config.z_norm_use_block_relative,
            halo_fraction: config.halo_fraction,
            blocks: block_metas,
        };

        let manifest_path = config.output_dir.join("blocks.json");
        let manifest_file = File::create(&manifest_path)?;
        serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)?;
        eprintln!("[preprocessing] wrote {}", manifest_path.display());

        // ── 9. Clean up the outlier-removal temp file, if one was created ─────────
        if config.outlier_removal {
            let _ = fs::remove_file(&outlier_cleaned_path);
        }

        // ── 9b. Clean up auto-DTM intermediates (Stage 38) ────────────────
        // The auto-generated `_auto_dtm.tif` (and its `_auto_ground.las`
        // sibling) are transient artifacts; delete them unless the user asked
        // to keep them via `--keep-auto-dtm`.
        if let Some(dtm_path) = auto_dtm_path {
            if config.keep_auto_dtm {
                eprintln!(
                    "[preprocessing] auto-DTM: keeping intermediates ({} + _auto_ground.las)",
                    dtm_path.display()
                );
            } else {
                let _ = fs::remove_file(&dtm_path);
                let _ = fs::remove_file(config.output_dir.join(AUTO_GROUND_LAS));
            }
        }

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
#[allow(clippy::type_complexity)]
fn inspect_lidar_header(path: &Path) -> Result<(f64, f64, f64, f64, f64, f64, u64, Option<u32>)> {
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
                h.min_x,
                h.min_y,
                h.max_x,
                h.max_y,
                h.min_z,
                h.max_z,
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
                h.min_x,
                h.min_y,
                h.max_x,
                h.max_y,
                h.min_z,
                h.max_z,
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
///
/// Maintains a running 0-based point-index counter (`idx`), incremented once
/// per point read across all three format branches (`las`/`laz`/`copc`), and
/// passes it to `BlockPartitioner::add_point`. This counter's ordering must
/// match the ordering used by `wbtools_oss::LidarEigenvalueFeaturesTool`'s own
/// `point_num` field (both simply count points in input-stream order), so
/// that a block's points can later be joined against the eigenvalue pre-pass
/// table by index (Stage 30, point-index-join extension / Step 5e).
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
    let mut idx: u64 = 0;

    match ext.as_str() {
        "las" => {
            let f = File::open(path)?;
            let mut reader = LasReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(idx, LitePoint::from(&pt))?;
                idx += 1;
            }
        }
        "laz" => {
            let f = File::open(path)?;
            let mut reader = LazReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(idx, LitePoint::from(&pt))?;
                idx += 1;
            }
        }
        "copc" => {
            use wblidar::copc::CopcReader;
            let mut reader = CopcReader::open_path(path)?;
            while reader.read_point(&mut pt)? {
                partitioner.add_point(idx, LitePoint::from(&pt))?;
                idx += 1;
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

/// Run the `wbtools_oss::LidarRemoveOutliersTool` directly via the `wbcore::Tool`
/// trait, writing a cleaned copy of `input` to `output_path`.
///
/// This restores the original Stage 04 design (see
/// `docs/stages/stage-04-outlier-removal.md`) after the 2026-06-17 build-speed
/// workaround (`outlier_filter.rs`) was reverted in Stage 30 — see
/// `docs/stages/stage-30-whitebox-git-dependency-integration.md`.
///
/// The tool is invoked with an explicit `output` path so the result is written
/// to disk rather than the in-memory lidar store; the caller then treats
/// `output_path` as the effective input for all subsequent pipeline steps.
///
/// # Errors
/// Returns [`ClassifierError::Tool`] if validation or execution fails.
fn run_outlier_removal(input: &Path, output_path: &Path, config: &PreprocessConfig) -> Result<()> {
    use serde_json::json;
    use wbcore::{AllowAllCapabilities, RecordingProgressSink, Tool, ToolArgs, ToolContext};
    use wbtools_oss::tools::LidarRemoveOutliersTool;

    let mut args: ToolArgs = ToolArgs::new();
    args.insert("input".to_string(), json!(input.display().to_string()));
    args.insert(
        "output".to_string(),
        json!(output_path.display().to_string()),
    );
    args.insert("search_radius".to_string(), json!(config.outlier_radius));
    args.insert("elev_diff".to_string(), json!(config.outlier_elev_diff));
    args.insert("use_median".to_string(), json!(config.outlier_use_median));

    let progress = RecordingProgressSink::new();
    let capabilities = AllowAllCapabilities;
    let ctx = ToolContext {
        progress: &progress,
        capabilities: &capabilities,
    };

    let tool = LidarRemoveOutliersTool;
    tool.validate(&args)?;
    tool.run(&args, &ctx)?;

    Ok(())
}

// ── Automatic ground DTM helper (Stage 38) ────────────────────────────────────

/// Auto-generate a bare-earth DTM raster from `input`, returning the path to
/// the written `_auto_dtm.tif` under `output_dir` (Stage 38 — Option A).
///
/// Two Whitebox tools are invoked directly via the `wbcore::Tool` trait,
/// mirroring the invocation pattern established by [`run_outlier_removal`]:
///
/// 1. `wbtools_oss::ImprovedGroundPointFilterTool` in *filter* mode
///    (`classify = false`) writes a ground-only LAS (`_auto_ground.las`) —
///    only points the progressive-morphology filter judges to be bare earth
///    are retained.  All ground-filter parameters other than `block_size`
///    are intentionally left at the tool's own defaults per the Stage 38
///    "streamlined surface" decision; only `--dtm-resolution` is exposed to
///    users (mapped to both `block_size` here and `resolution` below).
/// 2. `wbtools_oss::LidarTinGriddingTool` interpolates the ground-only LAS's
///    `elevation` onto a regular grid at `config.auto_dtm_resolution`,
///    writing the `_auto_dtm.tif` raster consumed by `DtmView::from_raster`.
///
/// The caller (`run_internal`) is responsible for deleting both intermediate
/// files unless `config.keep_auto_dtm` is set (Step 9b).
///
/// # Errors
/// Returns [`ClassifierError::Tool`] if validation or execution of either
/// underlying tool fails.
fn run_auto_dtm(input: &Path, output_dir: &Path, config: &PreprocessConfig) -> Result<PathBuf> {
    use serde_json::json;
    use wbcore::{AllowAllCapabilities, RecordingProgressSink, Tool, ToolArgs, ToolContext};
    use wbtools_oss::tools::{ImprovedGroundPointFilterTool, LidarTinGriddingTool};

    let ground_path = output_dir.join(AUTO_GROUND_LAS);
    let dtm_path = output_dir.join(AUTO_DTM_TIF);

    let progress = RecordingProgressSink::new();
    let capabilities = AllowAllCapabilities;

    // ── Stage 1: extract bare-earth points into a ground-only LAS ─────────
    {
        let mut args: ToolArgs = ToolArgs::new();
        args.insert("input".to_string(), json!(input.display().to_string()));
        args.insert(
            "output".to_string(),
            json!(ground_path.display().to_string()),
        );
        args.insert("block_size".to_string(), json!(config.auto_dtm_resolution));
        // Filter mode (not classify): write only the retained ground points.
        args.insert("classify".to_string(), json!(false));

        let ctx = ToolContext {
            progress: &progress,
            capabilities: &capabilities,
        };

        let tool = ImprovedGroundPointFilterTool;
        tool.validate(&args)?;
        tool.run(&args, &ctx)?;
    }

    // ── Stage 2: TIN-grid the ground points into a bare-earth DTM raster ──
    {
        let mut args: ToolArgs = ToolArgs::new();
        args.insert(
            "input".to_string(),
            json!(ground_path.display().to_string()),
        );
        args.insert("output".to_string(), json!(dtm_path.display().to_string()));
        args.insert("resolution".to_string(), json!(config.auto_dtm_resolution));
        args.insert("interpolation_parameter".to_string(), json!("elevation"));

        let ctx = ToolContext {
            progress: &progress,
            capabilities: &capabilities,
        };

        let tool = LidarTinGriddingTool;
        tool.validate(&args)?;
        tool.run(&args, &ctx)?;
    }

    Ok(dtm_path)
}

// ── Eigenvalue-feature pre-pass helper (Stage 30, Step 5b/5c/5d) ─────────────

/// Bytes per record in a `.eigen` sidecar file written by
/// `wbtools_oss::LidarEigenvalueFeaturesTool`: `u64` point_num (8 bytes) +
/// 10×`f32` (40 bytes) = 48 bytes/record.
const EIGEN_RECORD_BYTES: usize = 48;

/// Run the `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass (Stage 30,
/// Step 5b/5c/5d).
///
/// Estimates the in-memory size the tool's own whole-cloud point buffer would
/// require (`total_points * size_of::<wblidar::PointRecord>()`) and compares it
/// against `config.eigen_memory_budget_bytes`.
///
/// - When the estimate is **within budget**, invokes the tool once over the
///   entire (possibly outlier-cleaned) input file, parses the resulting
///   `.eigen` binary sidecar into an in-memory `Vec<[f32; 10]>` indexed by
///   original-file point index (0-based, matching the tool's own `point_num`
///   field, which is written in stream order), and deletes both sidecar
///   files (`.eigen` and `.eigen.json`).
/// - When the estimate **exceeds budget**, dispatches to
///   [`run_eigenvalue_prepass_split`] — the memory-gated spatial-split path
///   (Stage 30 Step 5d) — which returns a table of the same shape.
///
/// `bbox` is `(x_min, y_min, x_max, y_max)`, the same header-derived bounding
/// box used for grid-geometry computation elsewhere in `run_internal()`; it
/// is the basis for choosing the wider split axis when splitting is required.
///
/// # Errors
/// Returns [`ClassifierError::Tool`] if validation or execution of the
/// underlying tool fails, [`ClassifierError::Pipeline`] if a `.eigen` file
/// is malformed, and [`ClassifierError::Io`] on file I/O failure.
fn run_eigenvalue_prepass(
    input: &Path,
    output_dir: &Path,
    config: &PreprocessConfig,
    total_points: u64,
    bbox: (f64, f64, f64, f64),
) -> Result<Vec<[f32; 10]>> {
    use serde_json::json;
    use wbcore::{AllowAllCapabilities, RecordingProgressSink, Tool, ToolArgs, ToolContext};
    use wbtools_oss::tools::LidarEigenvalueFeaturesTool;

    // ── 5b: memory estimation + whole-file-vs-split decision ─────────────
    let point_record_size = std::mem::size_of::<PointRecord>();
    #[allow(clippy::cast_possible_truncation)]
    let total_points_usize = total_points as usize;
    let estimated_bytes = total_points_usize.saturating_mul(point_record_size);

    eprintln!(
        "[preprocessing] eigen pre-pass: {total_points} points × {point_record_size} bytes \
         ≈ {estimated_bytes} bytes (budget: {} bytes)",
        config.eigen_memory_budget_bytes
    );

    if estimated_bytes > config.eigen_memory_budget_bytes {
        let n_strips = compute_n_strips(estimated_bytes, config.eigen_memory_budget_bytes);
        eprintln!(
            "[preprocessing] eigen pre-pass: estimate exceeds budget — splitting into \
             {n_strips} spatial strips (Stage 30 Step 5d)"
        );
        return run_eigenvalue_prepass_split(
            input,
            output_dir,
            config,
            total_points,
            bbox,
            n_strips,
        );
    }

    // ── 5c: whole-file invocation ──────────────────────────────────────────
    let eigen_path = output_dir.join("_eigen_prepass.eigen");
    let json_path = output_dir.join("_eigen_prepass.eigen.json");

    let mut args: ToolArgs = ToolArgs::new();
    args.insert("input".to_string(), json!(input.display().to_string()));
    args.insert("num_neighbours".to_string(), json!(7));
    args.insert("search_radius".to_string(), json!(config.search_radius));
    args.insert(
        "output".to_string(),
        json!(eigen_path.display().to_string()),
    );

    let progress = RecordingProgressSink::new();
    let capabilities = AllowAllCapabilities;
    let ctx = ToolContext {
        progress: &progress,
        capabilities: &capabilities,
    };

    let tool = LidarEigenvalueFeaturesTool;
    tool.validate(&args)?;
    tool.run(&args, &ctx)?;

    let table = read_eigen_file(&eigen_path, total_points)?;

    // Clean up sidecars — they are a transient pre-pass artefact, not part of
    // the published pipeline output.
    let _ = fs::remove_file(&eigen_path);
    let _ = fs::remove_file(&json_path);

    Ok(table)
}

/// Compute the number of spatial strips needed so that, roughly, each
/// strip's own point count fits within `budget_bytes` (Stage 30, Step 5d).
///
/// `ceil(estimated_bytes / budget_bytes)`, clamped to a minimum of 2 (this
/// helper is only ever called once the caller has already determined
/// `estimated_bytes > budget_bytes`, so the true minimum is always ≥ 2 in
/// practice; the clamp is a defensive guard against rounding edge cases and
/// a misconfigured `0` budget).
fn compute_n_strips(estimated_bytes: usize, budget_bytes: usize) -> usize {
    if budget_bytes == 0 {
        // Degenerate: the CLI validates `--eigen-memory-budget-mb >= 1`, but
        // guard against a division by zero regardless of how this helper is
        // ever called directly (e.g. from a unit test).
        return estimated_bytes.max(2);
    }
    let n = estimated_bytes.div_ceil(budget_bytes);
    n.max(2)
}

/// One spatial strip's working state during the Stage 30 Step 5d split pass:
/// its core/extended coordinate ranges along the chosen split axis, the
/// temp LAS writer its selected points are streamed into, and the
/// `(original_point_index, is_core)` tag recorded for each point written (in
/// the exact order it was written, which is the same order
/// `wbtools_oss::LidarEigenvalueFeaturesTool` will assign as that strip's own
/// local `point_num` when it later processes this strip's temp file).
struct EigenSplitStrip {
    core_lo: f64,
    core_hi: f64,
    ext_lo: f64,
    ext_hi: f64,
    is_last: bool,
    writer: Box<dyn wblidar::io::PointWriter>,
    tags: Vec<(u64, bool)>,
    las_path: PathBuf,
}

/// Mirror `wblidar::frontend::infer_stream_writer_config_from_source` (private
/// there; also reproduced independently in `src/output/las_writer.rs`) using
/// only the public `LasReader` API, so each per-strip temp LAS file preserves
/// the source's point-data format, scale/offset, and CRS.
fn infer_writer_config_from_source(input: &Path) -> Result<wblidar::las::writer::WriterConfig> {
    use std::io::BufReader;
    use wblidar::las::writer::WriterConfig;
    use wblidar::las::LasReader;

    let reader = LasReader::new(BufReader::new(File::open(input)?))?;
    let hdr = reader.header();
    Ok(WriterConfig {
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
    })
}

/// Route one point to every strip whose *extended* (core + overlap) range
/// contains it, writing it to that strip's temp LAS file and recording
/// whether the point falls in that strip's *core* range (Stage 30, Step 5d).
///
/// A point may be written to more than one strip when it falls in the
/// overlap region shared by two adjacent strips — this is intentional: each
/// strip needs border context from its neighbours for correct neighbourhood
/// queries near its own edges, but only the strip whose *core* range actually
/// contains the point will have that point's row retained during stitching.
fn route_point_to_strips(
    pt: &PointRecord,
    idx: u64,
    axis_is_x: bool,
    strips: &mut [EigenSplitStrip],
) -> Result<()> {
    let coord = if axis_is_x { pt.x } else { pt.y };
    for strip in strips.iter_mut() {
        if coord >= strip.ext_lo && coord <= strip.ext_hi {
            let is_core = if strip.is_last {
                coord >= strip.core_lo && coord <= strip.core_hi
            } else {
                coord >= strip.core_lo && coord < strip.core_hi
            };
            strip.writer.write_point(pt)?;
            strip.tags.push((idx, is_core));
        }
    }
    Ok(())
}

/// The memory-gated spatial-split eigenvalue pre-pass path (Stage 30, Step
/// 5d), used when the whole-file estimate in [`run_eigenvalue_prepass`]
/// exceeds `config.eigen_memory_budget_bytes`.
///
/// Splits the input spatially along the **wider** bounding-box axis into
/// `n_strips` roughly equal-width strips, each extended by an overlap buffer
/// of `config.search_radius * 2.0` (a safety margin within the doc's
/// suggested `1.5`–`2.0` range) shared with the adjacent strip(s). Each
/// strip's selected points (core + overlap) are streamed into its own temp
/// LAS file under `output_dir/_eigen_split_cache/`, `LidarEigenvalueFeaturesTool`
/// is invoked once per strip, and only each strip's *core*-region rows are
/// kept — border-only rows (present purely to give correct neighbourhood
/// context near the strip's edges) are discarded. All strips' core rows are
/// stitched back together into a single `Vec<[f32; 10]>` indexed by
/// **original full-file** point index, identical in shape to the whole-file
/// path's return value.
///
/// Temp-file hygiene mirrors the `.spill` file convention already
/// established by `BlockPartitioner`: every strip's temp LAS file and
/// `.eigen`/`.eigen.json` sidecars are deleted immediately after that
/// strip's rows are consumed. A startup check **warns** (does not silently
/// delete) about any pre-existing files found in the cache directory,
/// signalling a possible prior interrupted run.
///
/// # Errors
/// Returns [`ClassifierError::Tool`] if validation or execution of the
/// underlying tool fails for any strip, [`ClassifierError::Pipeline`] if a
/// strip's `.eigen` file is malformed, and [`ClassifierError::Io`] on file
/// I/O failure (including reading/writing the per-strip temp LAS files).
#[allow(clippy::too_many_lines)]
fn run_eigenvalue_prepass_split(
    input: &Path,
    output_dir: &Path,
    config: &PreprocessConfig,
    total_points: u64,
    bbox: (f64, f64, f64, f64),
    n_strips: usize,
) -> Result<Vec<[f32; 10]>> {
    use serde_json::json;
    use wbcore::{AllowAllCapabilities, RecordingProgressSink, Tool, ToolArgs, ToolContext};
    use wblidar::io::{PointReader, PointWriter};
    use wblidar::las::writer::LasWriter;
    use wbtools_oss::tools::LidarEigenvalueFeaturesTool;

    let (x_min, y_min, x_max, y_max) = bbox;
    let dx = x_max - x_min;
    let dy = y_max - y_min;
    // Split along the wider axis, per the Approved Design's "simple, pragmatic
    // split" directive — no attempt at density-aware or sampling-based splits.
    let axis_is_x = dx >= dy;
    let (lo, hi) = if axis_is_x {
        (x_min, x_max)
    } else {
        (y_min, y_max)
    };
    let span = (hi - lo).max(f64::EPSILON);
    #[allow(clippy::cast_precision_loss)]
    let strip_width = span / n_strips as f64;
    // Overlap buffer: >= search_radius, with a ×2 safety margin (within the
    // doc's suggested ×1.5–2 range) — required for correctness, since a
    // point near a strip's core boundary must still see its true full
    // neighbourhood out to `search_radius` when the tool processes that
    // strip's temp file.
    let overlap = (config.search_radius * 2.0).max(0.0);

    let cache_dir = output_dir.join("_eigen_split_cache");
    fs::create_dir_all(&cache_dir)?;

    // Startup check: warn (do not silently delete) about any pre-existing
    // files in the cache directory — mirrors `BlockPartitioner::new()`'s
    // `.spill` stale-file warning convention.
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            eprintln!(
                "[warn] stale eigen-split cache file found (prior interrupted run?): {}",
                entry.path().display()
            );
        }
    }

    // ── Build one strip descriptor + temp LAS writer per strip ───────────
    let writer_cfg = infer_writer_config_from_source(input)?;
    let mut strips: Vec<EigenSplitStrip> = Vec::with_capacity(n_strips);
    for i in 0..n_strips {
        #[allow(clippy::cast_precision_loss)]
        let i_f = i as f64;
        let is_last = i + 1 == n_strips;
        let core_lo = lo + i_f * strip_width;
        let core_hi = if is_last {
            hi
        } else {
            lo + (i_f + 1.0) * strip_width
        };
        let ext_lo = (core_lo - overlap).max(lo);
        let ext_hi = (core_hi + overlap).min(hi);

        let las_path = cache_dir.join(format!("strip_{i:04}.las"));
        let writer: Box<dyn PointWriter> = Box::new(LasWriter::new(
            BufWriter::new(File::create(&las_path)?),
            writer_cfg.clone(),
        )?);

        strips.push(EigenSplitStrip {
            core_lo,
            core_hi,
            ext_lo,
            ext_hi,
            is_last,
            writer,
            tags: Vec::new(),
            las_path,
        });
    }

    // ── Single pass over the input, routing each point to every strip whose
    //    extended range contains it ────────────────────────────────────────
    {
        use std::io::BufReader;
        use wblidar::las::LasReader;
        use wblidar::laz::LazReader;

        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let mut pt = PointRecord::default();
        let mut idx: u64 = 0;

        match ext.as_str() {
            "las" => {
                let f = File::open(input)?;
                let mut reader = LasReader::new(BufReader::new(f))?;
                while reader.read_point(&mut pt)? {
                    route_point_to_strips(&pt, idx, axis_is_x, &mut strips)?;
                    idx += 1;
                }
            }
            "laz" => {
                let f = File::open(input)?;
                let mut reader = LazReader::new(BufReader::new(f))?;
                while reader.read_point(&mut pt)? {
                    route_point_to_strips(&pt, idx, axis_is_x, &mut strips)?;
                    idx += 1;
                }
            }
            "copc" => {
                use wblidar::copc::CopcReader;
                let mut reader = CopcReader::open_path(input)?;
                while reader.read_point(&mut pt)? {
                    route_point_to_strips(&pt, idx, axis_is_x, &mut strips)?;
                    idx += 1;
                }
            }
            _ => {
                return Err(ClassifierError::UnsupportedFormat {
                    path: input.display().to_string(),
                });
            }
        }
    }

    // Finish (flush + close) every strip's temp LAS writer before invoking
    // the tool on any of them.
    for strip in &mut strips {
        strip.writer.finish()?;
    }

    // ── Prepare the full-length output table ──────────────────────────────
    #[allow(clippy::cast_possible_truncation)]
    let total_points_usize = total_points as usize;
    let mut full_table = vec![[0.0f32; 10]; total_points_usize];

    // ── Run the tool once per strip; retain only core rows; clean up ──────
    for strip in &strips {
        if strip.tags.is_empty() {
            // No points routed to this strip at all (possible for a very
            // narrow / empty strip) — nothing to process, just discard the
            // (empty) temp LAS file.
            let _ = fs::remove_file(&strip.las_path);
            continue;
        }

        let eigen_path = strip.las_path.with_extension("eigen");
        let json_path = PathBuf::from(format!("{}.json", eigen_path.display()));

        let mut args: ToolArgs = ToolArgs::new();
        args.insert(
            "input".to_string(),
            json!(strip.las_path.display().to_string()),
        );
        args.insert("num_neighbours".to_string(), json!(7));
        args.insert("search_radius".to_string(), json!(config.search_radius));
        args.insert(
            "output".to_string(),
            json!(eigen_path.display().to_string()),
        );

        let progress = RecordingProgressSink::new();
        let capabilities = AllowAllCapabilities;
        let ctx = ToolContext {
            progress: &progress,
            capabilities: &capabilities,
        };

        let tool = LidarEigenvalueFeaturesTool;
        tool.validate(&args)?;
        tool.run(&args, &ctx)?;

        #[allow(clippy::cast_possible_truncation)]
        let strip_point_count = strip.tags.len() as u64;
        let strip_table = read_eigen_file(&eigen_path, strip_point_count)?;

        for (local_idx, (orig_idx, is_core)) in strip.tags.iter().enumerate() {
            if *is_core {
                if let Some(row) = strip_table.get(local_idx) {
                    #[allow(clippy::cast_possible_truncation)]
                    let orig_idx_usize = *orig_idx as usize;
                    full_table[orig_idx_usize] = *row;
                }
            }
        }

        // Clean up this strip's temp files immediately after its rows are
        // consumed — mirrors the `.spill` file "write → read once → delete"
        // lifecycle.
        let _ = fs::remove_file(&eigen_path);
        let _ = fs::remove_file(&json_path);
        let _ = fs::remove_file(&strip.las_path);
    }

    // Best-effort: remove the cache directory itself if it's now empty. Not
    // an error if it still contains stale files from elsewhere (e.g. the
    // stale-leftover warning case above) — `remove_dir` only succeeds on an
    // empty directory.
    let _ = fs::remove_dir(&cache_dir);

    Ok(full_table)
}

/// Parse a `wbtools_oss::LidarEigenvalueFeaturesTool` `.eigen` binary sidecar
/// into an in-memory table indexed by original-file point index.
///
/// The tool writes one 48-byte record per point in strict input stream order
/// (`point_num` is the 0-based index, monotonically increasing with no gaps),
/// so the returned `Vec<[f32; 10]>` can be indexed directly by point index.
///
/// `expected_points` is used only for a diagnostic warning if the actual
/// record count differs (e.g. due to withheld points) — it is not a hard
/// requirement, since the actual on-disk record count is authoritative.
///
/// # Errors
/// Returns [`ClassifierError::Pipeline`] if the file size is not a multiple
/// of `EIGEN_RECORD_BYTES`, or if any `point_num` is out of range for the
/// file's own record count.
fn read_eigen_file(path: &Path, expected_points: u64) -> Result<Vec<[f32; 10]>> {
    let metadata = fs::metadata(path)?;
    #[allow(clippy::cast_possible_truncation)]
    let file_bytes = metadata.len() as usize;
    if !file_bytes.is_multiple_of(EIGEN_RECORD_BYTES) {
        return Err(ClassifierError::Pipeline(format!(
            "eigen pre-pass output '{}' has size {file_bytes} bytes, not a multiple of \
             {EIGEN_RECORD_BYTES} bytes/record",
            path.display()
        )));
    }
    let n = file_bytes / EIGEN_RECORD_BYTES;
    #[allow(clippy::cast_possible_truncation)]
    if n as u64 != expected_points {
        eprintln!(
            "[warn] eigen pre-pass output '{}' contains {n} records, header reported \
             {expected_points} points",
            path.display()
        );
    }

    let mut table = vec![[0.0f32; 10]; n];
    let mut file = File::open(path)?;
    let mut buf = [0u8; EIGEN_RECORD_BYTES];
    for i in 0..n {
        file.read_exact(&mut buf)?;
        let corrupt = || {
            ClassifierError::Pipeline(format!(
                "eigen pre-pass output '{}' corrupt at record {i}",
                path.display()
            ))
        };
        let point_num = u64::from_le_bytes(buf[0..8].try_into().map_err(|_| corrupt())?);
        #[allow(clippy::cast_possible_truncation)]
        let idx = point_num as usize;
        if idx >= n {
            return Err(ClassifierError::Pipeline(format!(
                "eigen pre-pass output '{}' has out-of-range point_num {point_num} at record {i}",
                path.display()
            )));
        }
        let mut row = [0.0f32; 10];
        for (j, chunk) in buf[8..48].chunks_exact(4).enumerate() {
            row[j] = f32::from_le_bytes(chunk.try_into().map_err(|_| corrupt())?);
        }
        table[idx] = row;
    }
    Ok(table)
}

// ── Border-point spill I/O (Stage 08, index-carrying as of Stage 45) ─────────

//
// Border points are written to `.border` files using the same compact binary
// layout as the main `.spill` files (39 bytes per point: u64 original-file
// stream index + the 31-byte LitePoint record).  This is an internal
// temporary format (written and deleted within a single run), so upgrading
// it from the pre-Stage-45 31-byte layout requires no migration path.
//
// The original index is required to join halo rows against the whole-file
// eigenvalue pre-pass table (Stage 45 — the same join used for canonical
// points in Step 7d).
//
// Layout per point (little-endian):
//   original_index(u64) x(f64) y(f64) z(f64) intensity(u16) classification(u8)
//   return_number(u8) number_of_returns(u8) scan_angle(i16) = 39 bytes

/// Bytes per point in a border spill file — identical to the main spill format.
const BORDER_PT_BYTES: usize = 39;

/// Write a slice of `(original_index, LitePoint)` pairs to a `.border` spill file.
///
/// # Errors
/// Returns `ClassifierError` on any I/O failure.
fn write_border_spill(path: &Path, pts: &[(u64, LitePoint)]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut buf = [0u8; BORDER_PT_BYTES];
    for (idx, pt) in pts {
        buf[0..8].copy_from_slice(&idx.to_le_bytes());
        buf[8..16].copy_from_slice(&pt.x.to_le_bytes());
        buf[16..24].copy_from_slice(&pt.y.to_le_bytes());
        buf[24..32].copy_from_slice(&pt.z.to_le_bytes());
        buf[32..34].copy_from_slice(&pt.intensity.to_le_bytes());
        buf[34] = pt.classification;
        buf[35] = pt.return_number;
        buf[36] = pt.number_of_returns;
        buf[37..39].copy_from_slice(&pt.scan_angle.to_le_bytes());
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a `.border` spill file back into a `Vec<(original_index, LitePoint)>`.
///
/// # Errors
/// Returns [`ClassifierError::SpillCorrupt`] if the file size is not a
/// multiple of `BORDER_PT_BYTES` or if any read fails.
fn read_border_spill(path: &Path) -> Result<Vec<(u64, LitePoint)>> {
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
        let idx = u64::from_le_bytes(buf[0..8].try_into().map_err(|_| corrupt())?);
        let pt = LitePoint {
            x: f64::from_le_bytes(buf[8..16].try_into().map_err(|_| corrupt())?),
            y: f64::from_le_bytes(buf[16..24].try_into().map_err(|_| corrupt())?),
            z: f64::from_le_bytes(buf[24..32].try_into().map_err(|_| corrupt())?),
            intensity: u16::from_le_bytes(buf[32..34].try_into().map_err(|_| corrupt())?),
            classification: buf[34],
            return_number: buf[35],
            number_of_returns: buf[36],
            scan_angle: i16::from_le_bytes(buf[37..39].try_into().map_err(|_| corrupt())?),
        };
        pts.push((idx, pt));
    }
    Ok(pts)
}

// ── Output serialisation helpers ──────────────────────────────────────────────

/// Write a `.feat` binary block file (format v2, Stage 45).
///
/// ## File layout
/// ```text
/// [header — 41 bytes]                 (v1 was 37 bytes, all-core rows)
///   magic:      4 bytes  = b"WBFT"
///   version:    u8       = 2
///   n_points:   u32 LE  (= target_points N, core + halo)
///   n_features: u32 LE  (= 17)
///   block_id:   u64 LE
///   origin_x:   f64 LE
///   origin_y:   f64 LE
///   n_halo:     u32 LE  (trailing rows that are halo samples; 0 = all-core)
/// [data]
///   f32[n_points × n_features]  row-major, little-endian
///   rows [0 .. n_points − n_halo)        = core sampled points
///   rows [n_points − n_halo .. n_points) = halo (overlap-margin) samples
/// ```
#[allow(clippy::too_many_arguments)] // flat (path, header fields, payload) write signature
fn write_feat_file(
    path: &Path,
    block_id: u64,
    n_points: usize,
    n_features: usize,
    n_halo: usize,
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
    #[allow(clippy::cast_possible_truncation)]
    w.write_all(&(n_halo as u32).to_le_bytes())?;

    // Data — write each row as raw f32 bytes
    for row in features {
        let bytes: &[u8] = bytemuck::cast_slice(row.as_slice());
        w.write_all(bytes)?;
    }
    w.flush()?;

    Ok(())
}

/// Write a debug CSV file with the fixed Stage 30 (Step 5e+5f+5g) column layout:
/// 7 scalar columns followed by the 10 fixed eigenvalue-feature columns
/// produced by the whole-file pre-pass.
fn write_debug_csv(path: &Path, features: &[Vec<f32>]) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let cols = [
        "x_norm",
        "y_norm",
        "z_norm",
        "intensity_norm",
        "return_ratio",
        "scan_angle_norm",
        "hag",
        "lambda1",
        "lambda2",
        "lambda3",
        "linearity",
        "planarity",
        "sphericity",
        "omnivariance",
        "eigentropy",
        "slope",
        "residual",
    ];
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

// ── Stage 08: border-point loader ────────────────────────────────────────────

/// Collect points from the up-to-8 grid neighbours of `target` that fall
/// within the expanded bounding box `[origin - overlap, origin + block_size + overlap]`.
///
/// This function is called **sequentially** before the Rayon parallel phase so
/// that all spill files are guaranteed to exist (no concurrent `load()` calls
/// have deleted them yet).  The returned `Vec<(u64, LitePoint)>` pairs
/// (original-file stream index + point) are written to a `.border` spill file
/// by the caller; they are not held in memory across blocks.  The index is
/// required to join halo rows against the whole-file eigenvalue pre-pass
/// table (Stage 45).
///
/// Points that belong to the target block itself are not included — only
/// genuine cross-boundary neighbours are returned.
fn load_border_points(
    stubs_by_cell: &HashMap<(i32, i32), usize>,
    all_stubs: &[BlockStub],
    target: &BlockStub,
    block_size: f64,
    overlap: f64,
) -> Result<Vec<(u64, LitePoint)>> {
    // Expanded bounding box of the target block in projection units.
    let x_lo = target.origin_x - overlap;
    let x_hi = target.origin_x + block_size + overlap;
    let y_lo = target.origin_y - overlap;
    let y_hi = target.origin_y + block_size + overlap;

    let mut border: Vec<(u64, LitePoint)> = Vec::new();

    // Iterate over all 8 cardinal + diagonal neighbours.
    for dc in -1_i32..=1 {
        for dr in -1_i32..=1 {
            if dc == 0 && dr == 0 {
                continue; // skip the target block itself
            }
            let key = (target.col + dc, target.row + dr);
            if let Some(&idx) = stubs_by_cell.get(&key) {
                let neighbour = &all_stubs[idx];
                // Read neighbour spill files (with original indices) without
                // deleting them.
                let pts = neighbour.read_points_indexed()?;
                for (orig_idx, pt) in pts {
                    // Keep only points that fall inside the expanded bbox.
                    if pt.x >= x_lo && pt.x <= x_hi && pt.y >= y_lo && pt.y <= y_hi {
                        border.push((orig_idx, pt));
                    }
                }
            }
        }
    }

    Ok(border)
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
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap: 12.5,
            oversample_jitter: 0.0,
            z_norm_use_block_relative: false,
            halo_fraction: 0.25,
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
    // The default value is the literal 0.0 constant (serde's `#[serde(default)]`
    // on an f64 field), so exact equality is deterministic and safe here.
    #[allow(clippy::float_cmp)]
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
        use crate::preprocessing::block_partitioner::BlockPartitioner;
        use std::collections::HashMap;

        let stubs_by_cell: HashMap<(i32, i32), usize> = HashMap::new();
        let all_stubs: Vec<crate::preprocessing::block_partitioner::BlockStub> = vec![];

        // Build a real stub via BlockPartitioner so we have a valid target.
        let dir = tempfile::tempdir().unwrap();
        let mut partitioner = BlockPartitioner::new(0.0, 0.0, 50.0, 50.0, 50.0, dir.path());

        let pt = LitePoint {
            x: 25.0,
            y: 25.0,
            z: 10.0,
            ..LitePoint::default()
        };
        partitioner.add_point(0, pt).unwrap();
        let stubs = partitioner.finalize_stubs().unwrap();
        assert_eq!(stubs.len(), 1);

        // With an empty stubs_by_cell every neighbour lookup misses → empty border.
        let target = &stubs[0];
        let result = load_border_points(&stubs_by_cell, &all_stubs, target, 50.0, 5.0).unwrap();
        assert!(result.is_empty(), "no neighbours → empty border");
    }

    // ── CLI validation ────────────────────────────────────────────────────────

    /// `block_overlap` defaults to `0.0` in `PreprocessConfig`.
    // Exact-zero comparison against a `Default`-derived literal `0.0` constant;
    // no floating-point arithmetic occurs, so this cannot be a precision issue.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_preprocess_config_default_overlap() {
        let cfg = crate::preprocessing::PreprocessConfig::default();
        assert_eq!(cfg.block_overlap, 0.0);
    }

    // ── Border spill I/O round-trip ───────────────────────────────────────────

    /// Write a set of PointRecords to a `.border` file and read them back.
    /// All serialised fields must survive the round-trip exactly.
    // Test fixture field generation from a small index range (0..20); all
    // casts here operate on small non-negative values well within the target
    // types' ranges, so truncation/sign-loss is not a real concern.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    #[test]
    fn test_border_spill_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.border");

        let pts: Vec<(u64, LitePoint)> = (0..20u64)
            .map(|i| {
                (
                    i,
                    LitePoint {
                        x: i as f64 * 1.5,
                        y: i as f64 * 0.7,
                        z: i as f64 * 0.3,
                        intensity: (i * 1000) as u16,
                        classification: (i % 8) as u8,
                        return_number: 1,
                        number_of_returns: 2,
                        scan_angle: (i as i16) - 10,
                    },
                )
            })
            .collect();

        write_border_spill(&path, &pts).unwrap();
        let recovered = read_border_spill(&path).unwrap();

        assert_eq!(recovered.len(), pts.len(), "point count must match");
        for ((idx_a, a), (idx_b, b)) in pts.iter().zip(recovered.iter()) {
            assert_eq!(idx_a, idx_b, "original index must survive round trip");
            assert!((a.x - b.x).abs() < 1e-12, "x mismatch");
            assert!((a.y - b.y).abs() < 1e-12, "y mismatch");
            assert!((a.z - b.z).abs() < 1e-12, "z mismatch");
            assert_eq!(a.intensity, b.intensity, "intensity mismatch");
            assert_eq!(
                a.classification, b.classification,
                "classification mismatch"
            );
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

    // ── Eigenvalue pre-pass (Stage 30, Step 5b/5c) ────────────────────────────

    #[test]
    fn test_read_eigen_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.eigen");
        let rows: Vec<[f32; 10]> = (0..5)
            .map(|i: usize| {
                let mut r = [0.0f32; 10];
                for (j, v) in r.iter_mut().enumerate() {
                    #[allow(clippy::cast_precision_loss)]
                    let v_val = (i * 10 + j) as f32 * 0.1;
                    *v = v_val;
                }
                r
            })
            .collect();

        {
            let mut writer = BufWriter::new(File::create(&path).unwrap());
            for (point_num, row) in rows.iter().enumerate() {
                writer.write_all(&(point_num as u64).to_le_bytes()).unwrap();
                for v in row {
                    writer.write_all(&v.to_le_bytes()).unwrap();
                }
            }
            writer.flush().unwrap();
        }

        let table = read_eigen_file(&path, 5).unwrap();
        assert_eq!(table.len(), 5);
        for (a, b) in rows.iter().zip(table.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_run_eigenvalue_prepass_whole_file() {
        use wblidar::io::PointWriter;
        use wblidar::las::writer::{LasWriter, WriterConfig};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("input.las");

        let pts: Vec<PointRecord> = (0..20)
            .map(|i: i32| PointRecord {
                x: f64::from(i),
                y: f64::from(i % 5),
                z: f64::from(i % 3) * 0.5,
                intensity: 100,
                classification: 2,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            })
            .collect();

        {
            let cfg = WriterConfig::default();
            let mut writer =
                LasWriter::new(BufWriter::new(File::create(&input_path).unwrap()), cfg).unwrap();
            for pt in &pts {
                writer.write_point(pt).unwrap();
            }
            writer.finish().unwrap();
        }

        let config = PreprocessConfig {
            search_radius: 5.0,
            ..PreprocessConfig::default()
        };

        let bbox = (0.0, 0.0, 19.0, 4.0);
        let table =
            run_eigenvalue_prepass(&input_path, dir.path(), &config, pts.len() as u64, bbox)
                .unwrap();
        assert_eq!(table.len(), pts.len());

        // Sidecar files must be cleaned up.
        assert!(!dir.path().join("_eigen_prepass.eigen").exists());
        assert!(!dir.path().join("_eigen_prepass.eigen.json").exists());
    }

    // ── Eigenvalue pre-pass memory-gated spatial split (Stage 30, Step 5d) ────

    #[test]
    fn test_compute_n_strips_ceils_and_clamps_to_two() {
        assert_eq!(compute_n_strips(100, 40), 3); // ceil(2.5) = 3
        assert_eq!(compute_n_strips(81, 40), 3); // ceil(2.025) = 3
        assert_eq!(compute_n_strips(41, 40), 2); // ceil(1.025) = 2
        assert_eq!(compute_n_strips(80, 40), 2); // exactly 2
        assert_eq!(compute_n_strips(0, 40), 2); // clamped to minimum of 2
    }

    #[test]
    fn test_run_eigenvalue_prepass_split_produces_full_length_table() {
        use wblidar::io::PointWriter;
        use wblidar::las::writer::{LasWriter, WriterConfig};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("input.las");

        // 60 points spread along X so a 3-way split has a clear "wider axis".
        let pts: Vec<PointRecord> = (0..60)
            .map(|i: i32| PointRecord {
                x: f64::from(i),
                y: f64::from(i % 4),
                z: f64::from(i % 3) * 0.5,
                intensity: 100,
                classification: 2,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            })
            .collect();

        {
            let cfg = WriterConfig::default();
            let mut writer =
                LasWriter::new(BufWriter::new(File::create(&input_path).unwrap()), cfg).unwrap();
            for pt in &pts {
                writer.write_point(pt).unwrap();
            }
            writer.finish().unwrap();
        }

        let config = PreprocessConfig {
            search_radius: 5.0,
            ..PreprocessConfig::default()
        };

        let bbox = (0.0, 0.0, 59.0, 3.0);
        let table = run_eigenvalue_prepass_split(
            &input_path,
            dir.path(),
            &config,
            pts.len() as u64,
            bbox,
            3,
        )
        .unwrap();

        assert_eq!(table.len(), pts.len());

        // The split cache directory contents must be cleaned up (each
        // strip's temp LAS + sidecars are removed as soon as consumed).
        let cache_dir = dir.path().join("_eigen_split_cache");
        let remaining: Vec<_> = fs::read_dir(&cache_dir)
            .map(|it| it.flatten().collect::<Vec<_>>())
            .unwrap_or_default();
        assert!(
            remaining.is_empty(),
            "split cache must be emptied after use: {remaining:?}"
        );
    }

    #[test]
    fn test_run_eigenvalue_prepass_dispatches_to_split_when_over_budget() {
        use wblidar::io::PointWriter;
        use wblidar::las::writer::{LasWriter, WriterConfig};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("input.las");

        let pts: Vec<PointRecord> = (0..40)
            .map(|i: i32| PointRecord {
                x: f64::from(i),
                y: f64::from(i % 5),
                z: f64::from(i % 3) * 0.5,
                intensity: 100,
                classification: 2,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            })
            .collect();

        {
            let cfg = WriterConfig::default();
            let mut writer =
                LasWriter::new(BufWriter::new(File::create(&input_path).unwrap()), cfg).unwrap();
            for pt in &pts {
                writer.write_point(pt).unwrap();
            }
            writer.finish().unwrap();
        }

        // Budget deliberately set to force exactly 2 strips:
        // ceil((n * size_of::<PointRecord>()) / budget) == 2.
        let point_record_size = std::mem::size_of::<PointRecord>();
        let estimated = point_record_size * pts.len();
        let budget = estimated / 2 + 1;

        let config = PreprocessConfig {
            search_radius: 5.0,
            eigen_memory_budget_bytes: budget,
            ..PreprocessConfig::default()
        };

        let bbox = (0.0, 0.0, 39.0, 4.0);
        let table =
            run_eigenvalue_prepass(&input_path, dir.path(), &config, pts.len() as u64, bbox)
                .unwrap();
        assert_eq!(table.len(), pts.len());
    }

    #[test]
    fn test_run_eigenvalue_prepass_split_warns_about_stale_cache_files_but_does_not_delete_them() {
        use wblidar::io::PointWriter;
        use wblidar::las::writer::{LasWriter, WriterConfig};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("input.las");

        let pts: Vec<PointRecord> = (0..20)
            .map(|i: i32| PointRecord {
                x: f64::from(i),
                y: f64::from(i % 4),
                z: 0.0,
                intensity: 50,
                classification: 2,
                return_number: 1,
                number_of_returns: 1,
                ..PointRecord::default()
            })
            .collect();

        {
            let cfg = WriterConfig::default();
            let mut writer =
                LasWriter::new(BufWriter::new(File::create(&input_path).unwrap()), cfg).unwrap();
            for pt in &pts {
                writer.write_point(pt).unwrap();
            }
            writer.finish().unwrap();
        }

        // Simulate a stale leftover from a prior interrupted run, using a
        // filename that will not collide with this run's own strip files
        // (which are always named `strip_%04d.las`) — this run only ever
        // deletes files it itself created.
        let cache_dir = dir.path().join("_eigen_split_cache");
        fs::create_dir_all(&cache_dir).unwrap();
        let stale_path = cache_dir.join("leftover_from_prior_run.tmp");
        fs::write(&stale_path, b"stale").unwrap();

        let config = PreprocessConfig {
            search_radius: 5.0,
            ..PreprocessConfig::default()
        };

        let bbox = (0.0, 0.0, 19.0, 3.0);
        let table = run_eigenvalue_prepass_split(
            &input_path,
            dir.path(),
            &config,
            pts.len() as u64,
            bbox,
            2,
        )
        .unwrap();
        assert_eq!(table.len(), pts.len());

        // The pre-existing stale file must still be present -- we only *warn*,
        // never silently delete leftovers from a prior interrupted run.
        assert!(
            stale_path.exists(),
            "stale leftover file must not be deleted"
        );
    }

    // ── Automatic ground DTM (Stage 38) ───────────────────────────────────────

    /// Auto-DTM is the default no-external-DTM path: `auto_dtm` defaults to
    /// `true`, `auto_dtm_resolution` to `DEFAULT_DTM_RESOLUTION` (1.0), and
    /// `keep_auto_dtm` to `false`.
    // Exact comparison against the `Default`-derived literal `1.0` constant;
    // no floating-point arithmetic occurs, so precision is not a concern.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_preprocess_config_default_auto_dtm() {
        let cfg = PreprocessConfig::default();
        assert!(cfg.auto_dtm, "auto-DTM must default to enabled");
        assert_eq!(
            cfg.auto_dtm_resolution,
            crate::preprocessing::DEFAULT_DTM_RESOLUTION,
            "default resolution must be DEFAULT_DTM_RESOLUTION"
        );
        assert_eq!(
            crate::preprocessing::DEFAULT_DTM_RESOLUTION,
            1.0,
            "DEFAULT_DTM_RESOLUTION must be 1.0"
        );
        assert!(
            !cfg.keep_auto_dtm,
            "intermediates must be deleted by default"
        );
    }

    /// `run_auto_dtm` on a small synthetic bare-earth cloud produces a raster
    /// that is readable by `wbraster::Raster::read` and wrappable in a
    /// `DtmView` (the exact consumption path used by `run_internal`).
    #[test]
    fn test_run_auto_dtm_produces_readable_raster() {
        use wblidar::io::PointWriter;
        use wblidar::las::writer::{LasWriter, WriterConfig};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("ground.las");

        // Dense, gently sloped bare-earth grid (26×26 = 676 points over a
        // 25×25 m footprint at 1 m spacing) so the ground filter retains a
        // solid ground surface and TIN gridding yields a populated raster.
        let mut pts: Vec<PointRecord> = Vec::with_capacity(26 * 26);
        for ix in 0..26_i32 {
            for iy in 0..26_i32 {
                pts.push(PointRecord {
                    x: f64::from(ix),
                    y: f64::from(iy),
                    z: 100.0 + f64::from(ix) * 0.05,
                    intensity: 100,
                    classification: 2,
                    return_number: 1,
                    number_of_returns: 1,
                    ..PointRecord::default()
                });
            }
        }

        {
            let cfg = WriterConfig::default();
            let mut writer =
                LasWriter::new(BufWriter::new(File::create(&input_path).unwrap()), cfg).unwrap();
            for pt in &pts {
                writer.write_point(pt).unwrap();
            }
            writer.finish().unwrap();
        }

        let config = PreprocessConfig {
            auto_dtm: true,
            auto_dtm_resolution: 1.0,
            ..PreprocessConfig::default()
        };

        let dtm_path = run_auto_dtm(&input_path, dir.path(), &config).unwrap();
        assert!(dtm_path.exists(), "auto-DTM raster must be written");
        assert_eq!(dtm_path, dir.path().join(AUTO_DTM_TIF));

        // The raster must be readable and wrappable exactly as run_internal does.
        let raster = wbraster::Raster::read(&dtm_path).unwrap();
        let _view = DtmView::from_raster(&raster);

        // The intermediate ground-only LAS must also have been produced.
        assert!(
            dir.path().join(AUTO_GROUND_LAS).exists(),
            "intermediate ground LAS must exist after run_auto_dtm"
        );

        drop(pts);
    }
}
