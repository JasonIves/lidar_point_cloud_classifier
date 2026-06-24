#![allow(clippy::missing_errors_doc, clippy::doc_markdown,
         clippy::cast_precision_loss, clippy::cast_possible_truncation,
         clippy::cast_sign_loss, clippy::cast_lossless, clippy::cast_possible_wrap)]
//! and emits per-block `.lbl` files alongside the existing `.feat` files.
//!
//! Each `.lbl` file is a raw `u8[n_points]` byte array where each byte is
//! the remapped model class index for the corresponding sampled point.
//!
//! The `preprocess-labeled` CLI sub-command calls [`run_labeled_pipeline`].

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{ClassifierError, Result};
use crate::preprocessing::{
    pipeline::{BlockManifest, BlockMeta},
    PreprocessConfig, PreprocessingPipeline,
};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Extends [`BlockMeta`] with label file path, class distribution, and
/// macro-tile assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledBlockMeta {
    #[serde(flatten)]
    pub meta: BlockMeta,
    /// Path (filename only) of the sibling `.lbl` file.
    pub lbl_file: String,
    /// Macro-tile ID for spatially-disjoint train/val splitting.
    pub macro_tile_id: u32,
    /// Per-class point counts using remapped model indices as keys.
    pub class_distribution: HashMap<String, u64>,
}

/// Top-level `labeled_blocks.json` manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledBlockManifest {
    pub source: String,
    pub block_size: f64,
    pub target_points: usize,
    pub min_density: f64,
    pub search_radius: f64,
    /// Search radii used for multi-scale eigenvalue features.
    /// Empty means single-scale using `search_radius`.
    #[serde(default)]
    pub search_radii: Vec<f64>,
    pub min_neighbors: usize,
    pub crs_epsg: Option<u32>,
    /// ASPRS code (string key) → model class index mapping embedded for traceability.
    pub label_map: HashMap<String, u8>,
    /// Coarse spatial grid metadata for the train/val macro-tile split.
    pub spatial_tile_grid: SpatialTileGrid,
    pub blocks: Vec<LabeledBlockMeta>,
}

/// Describes the NxN macro-tile grid applied to the dataset bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialTileGrid {
    pub cols: usize,
    pub rows: usize,
    pub bbox_min_x: f64,
    pub bbox_min_y: f64,
    pub bbox_max_x: f64,
    pub bbox_max_y: f64,
}

/// Configuration for the labeled preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct LabeledPreprocessConfig {
    pub preprocess: PreprocessConfig,
    /// ASPRS code → model class index.  Codes absent from the map fall back to
    /// the Unassigned index (7 by default).
    pub label_map: HashMap<u8, u8>,
    /// NxN macro-tile grid resolution (default: 4).
    pub tile_grid: usize,
}

impl LabeledPreprocessConfig {
    /// Build the default 8-class ASPRS label remapping from Stage 02.
    #[must_use]
    pub fn default_label_map() -> HashMap<u8, u8> {
        let mut m = HashMap::new();
        m.insert(2, 0); // Ground
        m.insert(3, 1); // Low Vegetation
        m.insert(4, 2); // Medium Vegetation
        m.insert(5, 3); // High Vegetation
        m.insert(6, 4); // Building
        m.insert(9, 5); // Water
        m.insert(7, 6); // Low Point (noise)
        m.insert(1, 7); // Unassigned
        m
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Run the labeled preprocessing pipeline.
///
/// Wraps [`PreprocessingPipeline::run_with_indices`] and emits one `.lbl` file
/// per retained block alongside the existing `.feat` files.  Writes
/// `labeled_blocks.json` to `config.preprocess.output_dir`.
///
/// Blocks where all points carry ASPRS classification code `0` (never
/// classified) are silently dropped from the output manifest.
///
/// # Errors
/// Returns `ClassifierError` on any I/O or format error.
pub fn run_labeled_pipeline(
    config: &LabeledPreprocessConfig,
) -> Result<LabeledBlockManifest> {
    let output_dir = &config.preprocess.output_dir;
    fs::create_dir_all(output_dir)?;

    // ── 1. Run the Stage 01 pipeline with index capture ───────────────────
    let (base_manifest, process_results) =
        PreprocessingPipeline::run_with_indices(&config.preprocess)?;

    // ── 2. Compute dataset bounding box from block origins ────────────────
    let (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) =
        compute_bbox(&base_manifest, config.preprocess.block_size);

    let tile_grid = config.tile_grid.max(1);
    let grid_meta = SpatialTileGrid {
        cols: tile_grid,
        rows: tile_grid,
        bbox_min_x,
        bbox_min_y,
        bbox_max_x,
        bbox_max_y,
    };

    // ── 3. Re-stream input file to collect classification bytes ───────────
    let raw_class_map = stream_classifications(
        &config.preprocess.input,
        &base_manifest,
        config.preprocess.block_size,
    )?;

    // ── 4. Per-block: write .lbl file, compute distribution ───────────────
    let unassigned_idx: u8 = *config.label_map.get(&1).unwrap_or(&7);

    let mut labeled_blocks = Vec::new();

    for proc in &process_results {
        let id = proc.meta.id;

        // Retrieve raw classification bytes for this block's raw points.
        let Some(raw_classes) = raw_class_map.get(&id) else { continue };

        // Map sampled indices → remapped model class labels.
        let labels: Vec<u8> = proc
            .sampled_indices
            .iter()
            .map(|&raw_idx| {
                let asprs = raw_classes.get(raw_idx).copied().unwrap_or(0);
                remap(asprs, &config.label_map, unassigned_idx)
            })
            .collect();

        // Drop blocks where all points carry ASPRS code 0 (never classified).
        let all_unclassified = proc
            .sampled_indices
            .iter()
            .all(|&ri| raw_classes.get(ri).copied().unwrap_or(0) == 0);
        if all_unclassified {
            continue;
        }

        // Write .lbl file.
        let lbl_filename = format!("block_{id:05}.lbl");
        let lbl_path = output_dir.join(&lbl_filename);
        write_lbl_file(&lbl_path, &labels)?;

        // Compute class distribution (remapped indices as string keys).
        let mut dist: HashMap<String, u64> = HashMap::new();
        for &l in &labels {
            *dist.entry(l.to_string()).or_insert(0) += 1;
        }

        // Compute macro_tile_id.
        let macro_tile_id = compute_macro_tile_id(
            proc.meta.origin_x,
            proc.meta.origin_y,
            &grid_meta,
            tile_grid,
        );

        labeled_blocks.push(LabeledBlockMeta {
            meta: proc.meta.clone(),
            lbl_file: lbl_filename,
            macro_tile_id,
            class_distribution: dist,
        });
    }

    // Sort by block id for deterministic output.
    labeled_blocks.sort_by_key(|b| b.meta.id);

    eprintln!(
        "[labeled-preprocess] retained {} labeled blocks ({} dropped as all-unclassified)",
        labeled_blocks.len(),
        process_results.len() - labeled_blocks.len()
    );

    // ── 5. Build label_map for manifest (string keys) ─────────────────────
    let label_map_str: HashMap<String, u8> = config
        .label_map
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    // ── 6. Write labeled_blocks.json ──────────────────────────────────────
    let manifest = LabeledBlockManifest {
        source: base_manifest.source.clone(),
        block_size: base_manifest.block_size,
        target_points: base_manifest.target_points,
        min_density: base_manifest.min_density,
        search_radius: base_manifest.search_radius,
        search_radii: base_manifest.search_radii.clone(),
        min_neighbors: base_manifest.min_neighbors,
        crs_epsg: base_manifest.crs_epsg,
        label_map: label_map_str,
        spatial_tile_grid: grid_meta,
        blocks: labeled_blocks,
    };

    let manifest_path = output_dir.join("labeled_blocks.json");
    let f = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(BufWriter::new(f), &manifest)?;
    eprintln!("[labeled-preprocess] wrote {}", manifest_path.display());

    Ok(manifest)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Remap an ASPRS classification byte to a model class index.
fn remap(asprs: u8, map: &HashMap<u8, u8>, unassigned: u8) -> u8 {
    *map.get(&asprs).unwrap_or(&unassigned)
}

/// Write a raw `u8[n_points]` label file.
fn write_lbl_file(path: &Path, labels: &[u8]) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(labels)?;
    w.flush()?;
    Ok(())
}

/// Compute dataset bounding box padded by block_size.
fn compute_bbox(manifest: &BlockManifest, block_size: f64) -> (f64, f64, f64, f64) {
    if manifest.blocks.is_empty() {
        return (0.0, 0.0, block_size, block_size);
    }
    let min_x = manifest
        .blocks
        .iter()
        .map(|b| b.origin_x)
        .fold(f64::INFINITY, f64::min);
    let min_y = manifest
        .blocks
        .iter()
        .map(|b| b.origin_y)
        .fold(f64::INFINITY, f64::min);
    let max_x = manifest
        .blocks
        .iter()
        .map(|b| b.origin_x)
        .fold(f64::NEG_INFINITY, f64::max)
        + block_size;
    let max_y = manifest
        .blocks
        .iter()
        .map(|b| b.origin_y)
        .fold(f64::NEG_INFINITY, f64::max)
        + block_size;
    (min_x, min_y, max_x, max_y)
}

/// Compute the macro-tile ID for a block at `(origin_x, origin_y)`.
fn compute_macro_tile_id(
    origin_x: f64,
    origin_y: f64,
    grid: &SpatialTileGrid,
    tile_grid: usize,
) -> u32 {
    let w = (grid.bbox_max_x - grid.bbox_min_x).max(1e-9);
    let h = (grid.bbox_max_y - grid.bbox_min_y).max(1e-9);
    let tile_w = w / tile_grid as f64;
    let tile_h = h / tile_grid as f64;

    let col = ((origin_x - grid.bbox_min_x) / tile_w)
        .floor()
        .clamp(0.0, (tile_grid - 1) as f64) as u32;
    let row = ((origin_y - grid.bbox_min_y) / tile_h)
        .floor()
        .clamp(0.0, (tile_grid - 1) as f64) as u32;

    row * tile_grid as u32 + col
}

/// Stream the input LiDAR file and collect classification bytes per block.
///
/// Returns a `HashMap<block_id, Vec<u8>>` where each `Vec<u8>` is the raw
/// `classification` byte for every raw point in that block (in streaming order).
fn stream_classifications(
    input_path: &PathBuf,
    manifest: &BlockManifest,
    block_size: f64,
) -> Result<HashMap<u64, Vec<u8>>> {
    use std::fs::File;
    use std::io::BufReader;
    use wblidar::io::PointReader;
    use wblidar::las::LasReader;
    use wblidar::laz::LazReader;

    // Use the header-derived grid geometry stored in the manifest.  Never
    // re-derive grid_cols from retained block origins: if the density filter
    // dropped trailing columns, re-derivation would produce a smaller value
    // and corrupt block IDs for every row after the first.
    if manifest.grid_cols == 0 {
        return Err(ClassifierError::Pipeline(
            "blocks.json is missing grid_cols — re-run preprocessing to regenerate it".to_string(),
        ));
    }
    let grid_cols = manifest.grid_cols as u64;
    let grid_rows = manifest.grid_rows as u64;
    let x_min = manifest.grid_x_min;
    let y_min = manifest.grid_y_min;

    let mut result: HashMap<u64, Vec<u8>> = HashMap::new();
    // Pre-allocate entries for every retained block in the manifest.
    for b in &manifest.blocks {
        result.insert(b.id, Vec::new());
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let mut pt = wblidar::PointRecord::default();

    match ext.as_str() {
        "las" => {
            let f = File::open(input_path)?;
            let mut reader = LasReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                route_point(&pt, x_min, y_min, block_size, grid_cols, grid_rows, &mut result);
            }
        }
        "laz" => {
            let f = File::open(input_path)?;
            let mut reader = LazReader::new(BufReader::new(f))?;
            while reader.read_point(&mut pt)? {
                route_point(&pt, x_min, y_min, block_size, grid_cols, grid_rows, &mut result);
            }
        }
        "copc" => {
            use wblidar::copc::CopcReader;
            let mut reader = CopcReader::open_path(input_path)?;
            while reader.read_point(&mut pt)? {
                route_point(&pt, x_min, y_min, block_size, grid_cols, grid_rows, &mut result);
            }
        }
        _ => {
            return Err(ClassifierError::UnsupportedFormat {
                path: input_path.display().to_string(),
            })
        }
    }

    Ok(result)
}

/// Route a single point's classification byte into the correct block bucket.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn route_point(
    pt: &wblidar::PointRecord,
    x_min: f64,
    y_min: f64,
    block_size: f64,
    grid_cols: u64,
    grid_rows: u64,
    result: &mut HashMap<u64, Vec<u8>>,
) {
    let col = ((pt.x - x_min) / block_size).floor() as i64;
    let row = ((pt.y - y_min) / block_size).floor() as i64;
    if col < 0 || row < 0 || col as u64 >= grid_cols || row as u64 >= grid_rows {
        return;
    }
    let block_id = crate::preprocessing::block_id(row, col, grid_cols as i64);
    if let Some(vec) = result.get_mut(&block_id) {
        vec.push(pt.classification);
    }
}



// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbl_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.lbl");
        let labels: Vec<u8> = vec![0, 1, 2, 3, 7, 4, 5, 6];
        write_lbl_file(&path, &labels).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes, labels);
    }

    #[test]
    fn test_label_remap_unknown_code() {
        let map = LabeledPreprocessConfig::default_label_map();
        let unassigned = 7u8;
        // ASPRS 42 is not in the default map → falls back to unassigned (7)
        let result = remap(42, &map, unassigned);
        assert_eq!(result, unassigned);
        // ASPRS 2 = Ground → model index 0
        assert_eq!(remap(2, &map, unassigned), 0);
    }

    #[test]
    fn test_macro_tile_assignment() {
        // 4x4 grid over [0,400] x [0,400].
        let grid = SpatialTileGrid {
            cols: 4,
            rows: 4,
            bbox_min_x: 0.0,
            bbox_min_y: 0.0,
            bbox_max_x: 400.0,
            bbox_max_y: 400.0,
        };
        // Bottom-left corner → tile (col=0, row=0) = id 0
        assert_eq!(compute_macro_tile_id(0.0, 0.0, &grid, 4), 0);
        // Second column, first row → id 1
        assert_eq!(compute_macro_tile_id(100.0, 0.0, &grid, 4), 1);
        // First column, second row → id 4
        assert_eq!(compute_macro_tile_id(0.0, 100.0, &grid, 4), 4);
        // Top-right corner → tile (col=3, row=3) = id 15
        assert_eq!(compute_macro_tile_id(399.0, 399.0, &grid, 4), 15);
        // Clamped to max tile: origin beyond bbox
        assert_eq!(compute_macro_tile_id(500.0, 500.0, &grid, 4), 15);
    }

    #[test]
    fn test_labeled_manifest_fields() {
        // Verify LabeledBlockManifest serialises all required fields.
        let manifest = LabeledBlockManifest {
            source: "test.las".to_string(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            search_radii: vec![],
            min_neighbors: 8,
            crs_epsg: Some(32617),
            label_map: {
                let mut m = HashMap::new();
                m.insert("2".to_string(), 0u8);
                m
            },
            spatial_tile_grid: SpatialTileGrid {
                cols: 4, rows: 4,
                bbox_min_x: 0.0, bbox_min_y: 0.0,
                bbox_max_x: 400.0, bbox_max_y: 400.0,
            },
            blocks: vec![LabeledBlockMeta {
                meta: BlockMeta {
                    id: 42,
                    file: "block_00042.feat".to_string(),
                    origin_x: 100.0, origin_y: 200.0,
                    raw_point_count: 500,
                    sampled_point_count: 500,
                    oversampled: false,
                },
                lbl_file: "block_00042.lbl".to_string(),
                macro_tile_id: 6,
                class_distribution: {
                    let mut d = HashMap::new();
                    d.insert("0".to_string(), 100u64);
                    d
                },
            }],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("lbl_file"));
        assert!(json.contains("class_distribution"));
        assert!(json.contains("macro_tile_id"));
        assert!(json.contains("spatial_tile_grid"));
        assert!(json.contains("label_map"));
    }
}
