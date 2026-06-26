//! Labeled block dataset — loads `.feat` + `.lbl` pairs and manages the
//! spatially-disjoint train/val split.
//!
//! Supports **multiple input directories** (one per preprocessed LiDAR file).
//! Block IDs within a single directory are `row * grid_cols + col` and can
//! collide across directories.  The dataset encodes a per-directory index into
//! the high 32 bits of each `GlobalBlockId` to guarantee global uniqueness
//! without changing any on-disk format.  `trainer.rs` consumes `Vec<u64>` and
//! calls `load_block(id)` unchanged; the composite key is transparent to it.
//!
//! See `docs/stages/stage-05-multi-directory-dataset.md` for full design rationale.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::manual_is_multiple_of
)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::LabeledBlockManifest;
use crate::preprocessing::{
    n_features_for_radii, FEAT_MAGIC, FEAT_VERSION, N_EIGEN_FEATURES_PER_RADIUS, N_FEATURES,
    N_SCALAR_FEATURES,
};

// ─────────────────────────────────────────────────────────────────────────────
// Composite block ID
// ─────────────────────────────────────────────────────────────────────────────

/// Encode `(dir_index, local_block_id)` into a single `u64`.
///
/// Layout: `high 32 bits = dir_index`, `low 32 bits = local_block_id`.
/// The local ID field supports up to ~4 billion local block IDs per directory;
/// for a 50 m block size over a 200 km × 200 km area the grid has ~16 M cells,
/// well within that limit.
#[inline]
fn make_global_id(dir_idx: usize, local_id: u64) -> u64 {
    ((dir_idx as u64) << 32) | (local_id & 0xFFFF_FFFF)
}

/// Decode a `GlobalBlockId` back into `(dir_index, local_block_id)`.
#[inline]
fn decode_global_id(gid: u64) -> (usize, u64) {
    ((gid >> 32) as usize, gid & 0xFFFF_FFFF)
}

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

/// One preprocessed-LiDAR-file directory: path + parsed manifest.
struct DirEntry {
    path: PathBuf,
    manifest: LabeledBlockManifest,
}

/// Manages the `.feat` / `.lbl` dataset across one or more preprocessing
/// directories and provides a spatially-disjoint train/val split.
///
/// Block IDs returned in `train_ids` and `val_ids` are **composite
/// `GlobalBlockId`** values (`dir_index << 32 | local_block_id`).  Pass them
/// directly to [`load_block`]; do not interpret the raw bits externally.
pub struct LabeledBlockDataset {
    dirs: Vec<DirEntry>,
    /// Validated common class count across all directories.
    n_classes_inner: usize,
    /// Feature count derived from manifest `search_radii`.
    /// 12 for single-radius (backward-compatible), 7+5×N for N radii.
    n_features_inner: usize,
    pub train_ids: Vec<u64>,
    pub val_ids: Vec<u64>,
}

impl LabeledBlockDataset {
    /// Load from one or more `preprocess-labeled` output directories.
    ///
    /// Each directory must contain a `labeled_blocks.json` manifest.  All
    /// directories must have been preprocessed with the same label map
    /// (same `n_classes`); a mismatch is a hard error.
    ///
    /// For a single directory this is identical to the previous single-dir API.
    ///
    /// # Errors
    /// Returns an error if any manifest cannot be read, parsed, or if the
    /// `n_classes` values are inconsistent across directories.
    pub fn load(
        data_dirs: &[PathBuf],
        val_split: f64,
        val_tile_block_ids: Option<&HashSet<u64>>,
        seed: u64,
    ) -> Result<Self> {
        if data_dirs.is_empty() {
            return Err(ClassifierError::Pipeline(
                "at least one --data-dir is required".into(),
            ));
        }

        // Load all manifests and validate class consistency.
        let mut dirs = Vec::with_capacity(data_dirs.len());
        let mut validated_n_classes: Option<usize> = None;

        for (idx, dir) in data_dirs.iter().enumerate() {
            let manifest_path = dir.join("labeled_blocks.json");
            let f = File::open(&manifest_path).map_err(|e| {
                ClassifierError::Pipeline(format!("cannot open {}: {e}", manifest_path.display()))
            })?;
            let manifest: LabeledBlockManifest = serde_json::from_reader(BufReader::new(f))
                .map_err(|e| {
                    ClassifierError::Pipeline(format!(
                        "labeled_blocks.json parse error in {}: {e}",
                        dir.display()
                    ))
                })?;

            let nc = manifest
                .label_map
                .values()
                .copied()
                .max()
                .map_or(8, |m| m as usize + 1);

            match validated_n_classes {
                None => validated_n_classes = Some(nc),
                Some(expected) if expected != nc => {
                    return Err(ClassifierError::Pipeline(format!(
                        "n_classes mismatch: directory 0 has {expected} classes \
                         but directory {idx} ('{}') has {nc} classes. \
                         Re-preprocess all directories with the same --label-map.",
                        dir.display()
                    )));
                }
                _ => {}
            }

            dirs.push(DirEntry {
                path: dir.clone(),
                manifest,
            });
        }

        let n_classes_inner = validated_n_classes.unwrap_or(8);

        // Derive n_features from the first manifest's search_radii.
        // Empty search_radii = single-scale = N_FEATURES (12) for backward compat.
        let n_features_inner = {
            let radii = &dirs[0].manifest.search_radii;
            if radii.is_empty() {
                N_FEATURES
            } else {
                n_features_for_radii(radii.len())
            }
        };

        // Validate that all directories agree on n_features_inner.
        for (idx, entry) in dirs.iter().enumerate().skip(1) {
            let nf = if entry.manifest.search_radii.is_empty() {
                N_FEATURES
            } else {
                n_features_for_radii(entry.manifest.search_radii.len())
            };
            if nf != n_features_inner {
                return Err(ClassifierError::Pipeline(format!(
                    "n_features mismatch: directory 0 has {n_features_inner} features \
                     but directory {idx} ('{}') has {nf} features. \
                     Re-preprocess all directories with the same --search-radii.",
                    entry.path.display()
                )));
            }
        }

        // Build train/val split.  If an explicit override is supplied, apply it
        // to all directories; warn if multiple dirs are in use since the raw
        // block IDs are ambiguous without the dir prefix.
        let (train_ids, val_ids) = if let Some(explicit) = val_tile_block_ids {
            if dirs.len() > 1 {
                eprintln!(
                    "[dataset] warning: --val-tile-blocks with multiple --data-dir \
                     entries matches local block IDs against all directories. \
                     Consider per-run val splits instead."
                );
            }
            // Match explicit IDs against local block IDs in each directory.
            let mut train = Vec::new();
            let mut val = Vec::new();
            for (dir_idx, entry) in dirs.iter().enumerate() {
                for b in &entry.manifest.blocks {
                    let gid = make_global_id(dir_idx, b.meta.id);
                    if explicit.contains(&b.meta.id) {
                        val.push(gid);
                    } else {
                        train.push(gid);
                    }
                }
            }
            (train, val)
        } else {
            // Independent spatial split per directory; results are concatenated.
            let mut train = Vec::new();
            let mut val = Vec::new();
            for (dir_idx, entry) in dirs.iter().enumerate() {
                let (local_train, local_val) = spatial_split(&entry.manifest, val_split, seed);
                for id in local_train {
                    train.push(make_global_id(dir_idx, id));
                }
                for id in local_val {
                    val.push(make_global_id(dir_idx, id));
                }
            }
            (train, val)
        };

        if dirs.len() == 1 {
            eprintln!(
                "[dataset] train blocks: {}, val blocks: {}",
                train_ids.len(),
                val_ids.len()
            );
        } else {
            eprintln!(
                "[dataset] {} directories — train blocks: {}, val blocks: {}",
                dirs.len(),
                train_ids.len(),
                val_ids.len()
            );
        }

        Ok(Self {
            dirs,
            n_classes_inner,
            n_features_inner,
            train_ids,
            val_ids,
        })
    }

    /// Return the validated common class count across all loaded directories.
    pub fn n_classes(&self) -> usize {
        self.n_classes_inner
    }

    /// Return the feature count per point (7 + 5 × n_radii).
    pub fn n_features(&self) -> usize {
        self.n_features_inner
    }

    /// Compute per-class point counts from the **training** blocks only.
    /// Returns a `Vec<u64>` of length `n_classes()`.
    pub fn class_counts_train(&self) -> Vec<u64> {
        let n = self.n_classes_inner;
        let train_set: HashSet<u64> = self.train_ids.iter().copied().collect();
        let mut counts = vec![0u64; n];
        for (dir_idx, entry) in self.dirs.iter().enumerate() {
            for b in &entry.manifest.blocks {
                let gid = make_global_id(dir_idx, b.meta.id);
                if !train_set.contains(&gid) {
                    continue;
                }
                for (k, &v) in &b.class_distribution {
                    if let Ok(idx) = k.parse::<usize>() {
                        if idx < n {
                            counts[idx] += v;
                        }
                    }
                }
            }
        }
        counts
    }

    /// Load a single block (features + labels) from disk.
    ///
    /// `block_id` must be a `GlobalBlockId` as returned by `train_ids` or
    /// `val_ids` — do not pass raw local block IDs here.
    ///
    /// # Errors
    /// Returns an error if the `.feat` or `.lbl` file cannot be read, or if
    /// the composite ID refers to an out-of-range directory.
    pub fn load_block(&self, block_id: u64) -> Result<LoadedBlock> {
        let (dir_idx, local_id) = decode_global_id(block_id);

        let entry = self.dirs.get(dir_idx).ok_or_else(|| {
            ClassifierError::Pipeline(format!(
                "load_block: directory index {dir_idx} out of range \
                 (only {} directories loaded)",
                self.dirs.len()
            ))
        })?;

        let bm = entry
            .manifest
            .blocks
            .iter()
            .find(|b| b.meta.id == local_id)
            .ok_or_else(|| {
                ClassifierError::Pipeline(format!(
                    "block {local_id} not found in manifest for '{}'",
                    entry.path.display()
                ))
            })?;

        let feat_path = entry.path.join(&bm.meta.file);
        let features = load_feat_file(&feat_path)?;

        let lbl_path = entry.path.join(&bm.lbl_file);
        let n_points = features.nrows();
        let labels = load_lbl_file(&lbl_path, n_points)?;

        Ok(LoadedBlock {
            features,
            labels,
            block_id,
        })
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
    let mut val = Vec::new();

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

    let version = hdr[0];
    // Fixed-size sub-slices of a [u8; 33] array: try_into cannot fail, but we
    // propagate as Pipeline error rather than unwrap() to satisfy the no-panics rule.
    let corrupt = || ClassifierError::Pipeline("feat: header byte slice conversion failed".into());
    let n_points =
        u32::from_le_bytes(<[u8; 4]>::try_from(&hdr[1..5]).map_err(|_| corrupt())?) as usize;
    let n_features =
        u32::from_le_bytes(<[u8; 4]>::try_from(&hdr[5..9]).map_err(|_| corrupt())?) as usize;

    if version != FEAT_VERSION {
        return Err(ClassifierError::Pipeline(format!(
            "feat: unsupported version {version}"
        )));
    }
    // Accept any positive n_features (multi-scale or legacy 12).
    if n_features == 0
        || !matches!(n_features, f if (f - N_SCALAR_FEATURES) % N_EIGEN_FEATURES_PER_RADIUS == 0)
    {
        return Err(ClassifierError::Pipeline(format!(
            "feat: n_features={n_features} is not a valid value (expected 7 + 5×N)"
        )));
    }

    // ── Data ─────────────────────────────────────────────────────────────
    let n_f32 = n_points * n_features;
    let mut buf = vec![0u8; n_f32 * 4];
    f.read_exact(&mut buf)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    // chunks_exact(4) guarantees each chunk is exactly 4 bytes; the try_into
    // cannot fail, but we use an array copy instead of try_into to avoid any
    // unwrap() in production code.
    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
    fn test_global_id_round_trip() {
        // Encoding and decoding must be lossless for representative inputs.
        for (dir, local) in [(0usize, 0u64), (1, 42), (3, 65535), (255, 4_000_000)] {
            let gid = make_global_id(dir, local);
            let (d2, l2) = decode_global_id(gid);
            assert_eq!(d2, dir, "dir mismatch for ({dir}, {local})");
            assert_eq!(l2, local, "local mismatch for ({dir}, {local})");
        }
    }

    #[test]
    fn test_global_ids_are_distinct_across_dirs() {
        // Same local ID in two directories must produce different global IDs.
        let gid0 = make_global_id(0, 42);
        let gid1 = make_global_id(1, 42);
        assert_ne!(gid0, gid1);
    }

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
            let mut v = Vec::new();
            for b in &manifest.blocks {
                if explicit.contains(&b.meta.id) {
                    v.push(b.meta.id);
                } else {
                    tr.push(b.meta.id);
                }
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
        LabeledBlockManifest, LabeledBlockMeta, SpatialTileGrid,
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
            search_radii: vec![],
            min_neighbors: 8,
            crs_epsg: None,
            label_map: HM::new(),
            spatial_tile_grid: SpatialTileGrid {
                cols: 4,
                rows: 4,
                bbox_min_x: 0.0,
                bbox_min_y: 0.0,
                bbox_max_x: 200.0,
                bbox_max_y: 200.0,
            },
            blocks,
        }
    }
}
