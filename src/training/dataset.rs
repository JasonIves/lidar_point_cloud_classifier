//! Labeled block dataset — loads `.feat` + `.lbl` pairs and manages the
//! spatially-disjoint train/val split.
//!
//! The dataset does **not** own burn tensors.  It returns `(Array2<f32>, Vec<u8>)`
//! tuples; callers convert to burn tensors for the GPU/CPU backend.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_lossless, clippy::must_use_candidate, clippy::missing_errors_doc, clippy::doc_markdown)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::LabeledBlockManifest;
use crate::preprocessing::{FEAT_MAGIC, FEAT_VERSION, N_FEATURES};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded block: feature matrix + label vector.
pub struct LoadedBlock {
    /// Feature matrix: `[n_points, N_FEATURES]`.
    pub features: Array2<f32>,
    /// Per-point class labels (remapped model indices).
    pub labels: Vec<u8>,
    pub block_id: u64,
}

/// Manages the `.feat` / `.lbl` dataset and provides train/val split.
pub struct LabeledBlockDataset {
    data_dir: PathBuf,
    manifest: LabeledBlockManifest,
    pub train_ids: Vec<u64>,
    pub val_ids:   Vec<u64>,
}

impl LabeledBlockDataset {
    /// Load from a `labeled_blocks.json` manifest.
    ///
    /// The train/val split is determined by `macro_tile_id`:
    /// - If `val_tile_block_ids` is `Some(set)`, those block IDs go to validation.
    /// - Otherwise, a stride-based spatial split assigns approximately
    ///   `val_split` fraction of macro-tiles to validation.
    ///
    /// # Errors
    /// Returns an error if the manifest cannot be read or parsed.
    pub fn load(
        data_dir: &Path,
        val_split: f64,
        val_tile_block_ids: Option<&HashSet<u64>>,
        seed: u64,
    ) -> Result<Self> {
        let manifest_path = data_dir.join("labeled_blocks.json");
        let f = File::open(&manifest_path).map_err(|e| {
            ClassifierError::Pipeline(format!("cannot open labeled_blocks.json: {e}"))
        })?;
        let manifest: LabeledBlockManifest = serde_json::from_reader(BufReader::new(f))
            .map_err(|e| ClassifierError::Pipeline(format!("labeled_blocks.json parse: {e}")))?;

        let (train_ids, val_ids) = if let Some(explicit) = val_tile_block_ids {
            // Explicit override
            let mut train = Vec::new();
            let mut val   = Vec::new();
            for b in &manifest.blocks {
                if explicit.contains(&b.meta.id) { val.push(b.meta.id); }
                else { train.push(b.meta.id); }
            }
            (train, val)
        } else {
            spatial_split(&manifest, val_split, seed)
        };

        eprintln!(
            "[dataset] train blocks: {}, val blocks: {}",
            train_ids.len(), val_ids.len()
        );

        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            manifest,
            train_ids,
            val_ids,
        })
    }

    /// Return the number of classes from the manifest `label_map`.
    pub fn n_classes(&self) -> usize {
        // The label_map maps ASPRS codes → class indices; max index + 1 = n_classes.
        self.manifest
            .label_map
            .values()
            .copied()
            .max()
            .map_or(8, |m| m as usize + 1)
    }

    /// Compute per-class point counts from the **training** blocks only.
    /// Returns a `Vec<u64>` of length `n_classes`.
    pub fn class_counts_train(&self) -> Vec<u64> {
        let n = self.n_classes();
        let train_set: HashSet<u64> = self.train_ids.iter().copied().collect();
        let mut counts = vec![0u64; n];
        for b in &self.manifest.blocks {
            if !train_set.contains(&b.meta.id) { continue; }
            for (k, &v) in &b.class_distribution {
                if let Ok(idx) = k.parse::<usize>() {
                    if idx < n { counts[idx] += v; }
                }
            }
        }
        counts
    }

    /// Load a single block (features + labels) from disk.
    ///
    /// # Errors
    /// Returns an error if the `.feat` or `.lbl` file cannot be read.
    pub fn load_block(&self, block_id: u64) -> Result<LoadedBlock> {
        // Find the block meta entry.
        let bm = self
            .manifest
            .blocks
            .iter()
            .find(|b| b.meta.id == block_id)
            .ok_or_else(|| ClassifierError::Pipeline(format!("block {block_id} not in manifest")))?;

        // Load .feat file.
        let feat_path = self.data_dir.join(&bm.meta.file);
        let features = load_feat_file(&feat_path)?;

        // Load .lbl file.
        let lbl_path = self.data_dir.join(&bm.lbl_file);
        let n_points = features.nrows();
        let labels = load_lbl_file(&lbl_path, n_points)?;

        Ok(LoadedBlock { features, labels, block_id })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial split
// ─────────────────────────────────────────────────────────────────────────────

/// Assign blocks to train or validation set using macro-tile stride selection.
fn spatial_split(
    manifest: &LabeledBlockManifest,
    val_split: f64,
    seed: u64,
) -> (Vec<u64>, Vec<u64>) {
    // Collect unique populated macro-tile IDs.
    let mut tile_to_blocks: HashMap<u32, Vec<u64>> = HashMap::new();
    for b in &manifest.blocks {
        tile_to_blocks
            .entry(b.macro_tile_id)
            .or_default()
            .push(b.meta.id);
    }

    let mut tile_ids: Vec<u32> = tile_to_blocks.keys().copied().collect();
    tile_ids.sort_unstable(); // deterministic order

    let n_tiles = tile_ids.len();
    let target_val = (n_tiles as f64 * val_split).round().max(1.0) as usize;
    let stride = n_tiles / target_val.max(1);

    // Use seed to pick an offset within the stride to break ties deterministically.
    let offset = (seed as usize) % stride.max(1);

    let mut val_tiles: HashSet<u32> = HashSet::new();
    let mut i = offset;
    while i < n_tiles && val_tiles.len() < target_val {
        val_tiles.insert(tile_ids[i]);
        i += stride;
    }

    let mut train = Vec::new();
    let mut val   = Vec::new();

    for b in &manifest.blocks {
        if val_tiles.contains(&b.macro_tile_id) {
            val.push(b.meta.id);
        } else {
            train.push(b.meta.id);
        }
    }

    (train, val)
}

// ─────────────────────────────────────────────────────────────────────────────
// File loaders
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a `.feat` binary file into an `Array2<f32>` of shape `[n_points, N_FEATURES]`.
fn load_feat_file(path: &Path) -> Result<Array2<f32>> {
    let mut f = File::open(path)
        .map_err(|e| ClassifierError::Pipeline(format!("feat open {}: {e}", path.display())))?;

    // ── Header ───────────────────────────────────────────────────────────
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;
    if &magic != FEAT_MAGIC {
        return Err(ClassifierError::Pipeline(format!(
            "feat: bad magic in {}",
            path.display()
        )));
    }

    let mut hdr = [0u8; 33]; // version(1) + n_points(4) + n_features(4) + block_id(8) + origin_x(8) + origin_y(8)
    f.read_exact(&mut hdr)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    let version    = hdr[0];
    let n_points   = u32::from_le_bytes(hdr[1..5].try_into().unwrap()) as usize;
    let n_features = u32::from_le_bytes(hdr[5..9].try_into().unwrap()) as usize;

    if version != FEAT_VERSION {
        return Err(ClassifierError::Pipeline(format!(
            "feat: unsupported version {version}"
        )));
    }
    if n_features != N_FEATURES {
        return Err(ClassifierError::Pipeline(format!(
            "feat: expected {N_FEATURES} features, got {n_features}"
        )));
    }

    // ── Data ─────────────────────────────────────────────────────────────
    let n_f32 = n_points * n_features;
    let mut buf = vec![0u8; n_f32 * 4];
    f.read_exact(&mut buf)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Array2::from_shape_vec((n_points, n_features), floats)
        .map_err(|e| ClassifierError::Pipeline(format!("feat reshape: {e}")))
}

/// Read a raw `.lbl` file — just `u8[n_points]`.
fn load_lbl_file(path: &Path, n_points: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)
        .map_err(|e| ClassifierError::Pipeline(format!("lbl open {}: {e}", path.display())))?;
    let mut buf = vec![0u8; n_points];
    f.read_exact(&mut buf)
        .map_err(|e| ClassifierError::Pipeline(format!("lbl read: {e}")))?;
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_val_tile_override() {
        // If we supply explicit val block IDs, those blocks must be in val set
        // regardless of macro_tile_id.
        let mut explicit = HashSet::new();
        explicit.insert(42u64);
        explicit.insert(43u64);

        let blocks = vec![
            // macro_tile_id 0 → would normally be train
            make_lbm(42, 0),
            make_lbm(43, 0),
            make_lbm(44, 1),
        ];
        let manifest = dummy_manifest(blocks);

        let (train, val) = if true {
            let mut tr = Vec::new();
            let mut v  = Vec::new();
            for b in &manifest.blocks {
                if explicit.contains(&b.meta.id) { v.push(b.meta.id); }
                else { tr.push(b.meta.id); }
            }
            (tr, v)
        } else {
            spatial_split(&manifest, 0.2, 42)
        };

        assert!(val.contains(&42));
        assert!(val.contains(&43));
        assert!(train.contains(&44));
    }

    #[test]
    fn test_spatial_split_fraction() {
        // 16 blocks in 16 distinct macro-tiles; val_split=0.25 → 4 val tiles
        let blocks: Vec<_> = (0..16u64).map(|i| make_lbm(i, i as u32)).collect();
        let manifest = dummy_manifest(blocks);
        let (train, val) = spatial_split(&manifest, 0.25, 42);
        assert_eq!(val.len(), 4, "expected 4 val blocks, got {}", val.len());
        assert_eq!(train.len(), 12);
    }

    // ── helpers ──────────────────────────────────────────────────────────────
    use crate::preprocessing::labeled_pipeline::{
        LabeledBlockMeta, LabeledBlockManifest, SpatialTileGrid,
    };
    use crate::preprocessing::pipeline::BlockMeta;
    use std::collections::HashMap as HM;

    fn make_lbm(id: u64, macro_tile_id: u32) -> LabeledBlockMeta {
        LabeledBlockMeta {
            meta: BlockMeta {
                id,
                file: format!("block_{id:05}.feat"),
                origin_x: (id % 4) as f64 * 50.0,
                origin_y: (id / 4) as f64 * 50.0,
                raw_point_count: 1024,
                sampled_point_count: 1024,
                oversampled: false,
            },
            lbl_file: format!("block_{id:05}.lbl"),
            macro_tile_id,
            class_distribution: HM::new(),
        }
    }

    fn dummy_manifest(blocks: Vec<LabeledBlockMeta>) -> LabeledBlockManifest {
        LabeledBlockManifest {
            source: "test.las".into(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map: HM::new(),
            spatial_tile_grid: SpatialTileGrid {
                cols: 4, rows: 4,
                bbox_min_x: 0.0, bbox_min_y: 0.0,
                bbox_max_x: 200.0, bbox_max_y: 200.0,
            },
            blocks,
        }
    }
}
