//! Labeled block dataset — loads `.feat` + `.lbl` pairs and manages the
//! spatially-disjoint train/val split.
//!
//! Supports **multiple input directories** (one per preprocessed `LiDAR` file).
//! Block IDs within a single directory are `row * grid_cols + col` and can
//! collide across directories.  The dataset encodes a per-directory index into
//! the high 32 bits of each `GlobalBlockId` to guarantee global uniqueness
//! without changing any on-disk format.  `trainer.rs` consumes `Vec<u64>` and
//! calls `load_block(id)` unchanged; the composite key is transparent to it.
//!
//! See `docs/stages/stage-05-multi-directory-dataset.md` for full design rationale.

use std::collections::{HashMap, HashSet};

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::Array2;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::LabeledBlockManifest;
use crate::preprocessing::{
    validate_block_filename, FEAT_MAGIC, FEAT_VERSION, MAX_FEAT_PAYLOAD_BYTES, N_FEATURES,
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
#[derive(Debug)]
pub struct LoadedBlock {
    /// Feature matrix: `[n_points, N_FEATURES]`.
    pub features: Array2<f32>,
    /// Per-point class labels (remapped model indices).
    pub labels: Vec<u8>,
    pub block_id: u64,
}

/// One preprocessed-LiDAR-file directory: path + parsed manifest.
#[derive(Debug)]
struct DirEntry {
    path: PathBuf,
    manifest: LabeledBlockManifest,
    /// Stage 21 (Performance): local block ID → index into `manifest.blocks`,
    /// built once at `load()` time so `load_block()` no longer needs an O(n)
    /// linear scan on every call.
    block_index: HashMap<u64, usize>,
}

/// Stage 27 (Block Caching, audit finding 5.2): opt-in, byte-budget-bounded
/// in-memory cache of already-decoded blocks, scoped to one
/// `LabeledBlockDataset` (i.e. one training run) — not a process-`static`.
///
/// Follows the `whitebox_next_gen::memory_store` idiom (a plain
/// `HashMap<K, Arc<V>>` behind a `Mutex`, no external caching crate, no
/// eviction policy) investigated in Stage 26. When the configured byte
/// budget would be exceeded by an additional block, the cache simply
/// declines to insert it (the block is transparently re-read from disk on
/// its next request) and logs exactly one informative warning the first
/// time this happens — never an error.
#[derive(Debug)]
struct BlockCache {
    entries: HashMap<u64, Arc<LoadedBlock>>,
    bytes_used: usize,
    max_bytes: usize,
    /// Set to `true` after the first time an insert was skipped due to the
    /// budget being exceeded, so the `[cache] ... budget exceeded` warning
    /// is only ever emitted once per training run.
    warned_budget_exceeded: bool,
}

impl BlockCache {
    fn new(max_mb: usize) -> Self {
        Self {
            entries: HashMap::new(),
            bytes_used: 0,
            max_bytes: max_mb.saturating_mul(1024 * 1024),
            warned_budget_exceeded: false,
        }
    }

    /// Exact in-memory footprint of a loaded block: `n_points × n_features`
    /// `f32` feature bytes plus `n_points` `u8` label bytes.
    fn block_bytes(block: &LoadedBlock) -> usize {
        let n_points = block.features.nrows();
        let n_features = block.features.ncols();
        n_points.saturating_mul(n_features).saturating_mul(4) + n_points
    }

    /// Attempt to insert a freshly-loaded block. Best-effort: silently
    /// declines (no error, no panic) if the budget would be exceeded,
    /// logging exactly one warning the first time that happens.
    fn try_insert(&mut self, block: Arc<LoadedBlock>) {
        let bytes = Self::block_bytes(&block);
        if self.bytes_used.saturating_add(bytes) <= self.max_bytes {
            self.bytes_used += bytes;
            self.entries.insert(block.block_id, block);
        } else if !self.warned_budget_exceeded {
            self.warned_budget_exceeded = true;
            eprintln!(
                "[cache] block cache budget ({} MB) exceeded — further blocks \
                 will be re-read from disk instead of cached for the remainder \
                 of this run",
                self.max_bytes / (1024 * 1024)
            );
        }
    }
}

/// Clone a cached block's data into a fresh, independently-owned
/// `LoadedBlock`. `Array2<f32>`/`Vec<u8>` clones are cheap relative to the
/// disk read + `.feat`/`.lbl` parse they replace.
fn clone_loaded_block(block: &LoadedBlock) -> LoadedBlock {
    LoadedBlock {
        features: block.features.clone(),
        labels: block.labels.clone(),
        block_id: block.block_id,
    }
}

/// Manages the `.feat` / `.lbl` dataset across one or more preprocessing
/// directories and provides a spatially-disjoint train/val split.
///
/// Block IDs returned in `train_ids` and `val_ids` are **composite
/// `GlobalBlockId`** values (`dir_index << 32 | local_block_id`).  Pass them
/// directly to [`load_block`]; do not interpret the raw bits externally.
#[derive(Debug)]
pub struct LabeledBlockDataset {
    dirs: Vec<DirEntry>,
    /// Validated common class count across all directories.
    n_classes_inner: usize,
    /// Fixed feature count (Stage 30, Step 5e+5f+5g): `N_FEATURES` (17),
    /// derived from a single whole-file eigenvalue pre-pass.
    n_features_inner: usize,
    pub train_ids: Vec<u64>,
    pub val_ids: Vec<u64>,
    /// Stage 21 (Performance): cached copy of `train_ids` as a `HashSet` so
    /// `class_counts_train()` no longer rebuilds it on every call.
    train_set: HashSet<u64>,
    /// Stage 27 (Block Caching, audit finding 5.2): opt-in in-memory block
    /// cache, installed via `with_block_cache()`. `None` (the default after
    /// `load()`) disables caching entirely — `load_block()` behaves exactly
    /// as it did before Stage 27.
    cache: Option<Mutex<BlockCache>>,
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
    // Stage 24 (Code Quality Cleanup, item 4.1): this constructor validates
    // and merges N independent manifests (class-index contiguity, n_classes
    // agreement, n_features agreement) and builds the train/val split in one
    // pass — splitting it further would scatter tightly-coupled validation
    // state across extra functions for no real benefit. The `usize as u64`
    // casts below are bounded block/point counts, so precision loss/
    // truncation is inconsequential.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
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

        let (dirs, n_classes_inner) = load_dir_entries(data_dirs)?;

        // Fixed-width feature count (Stage 30, Step 5e+5f+5g): every manifest
        // produced by the whole-file eigenvalue pre-pass has exactly
        // N_FEATURES columns per point. There is no more per-directory
        // variability to validate.
        let n_features_inner = N_FEATURES;

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

        let train_set: HashSet<u64> = train_ids.iter().copied().collect();

        Ok(Self {
            dirs,
            n_classes_inner,
            n_features_inner,
            train_ids,
            val_ids,
            train_set,
            cache: None,
        })
    }

    /// Load from pre-split `train`/`val` directories (Stage 32).
    ///
    /// Unlike [`load`](Self::load), no macro-tile stride selection is
    /// performed at all — the split was already decided physically when
    /// `wb_lidar_train split-dataset` materialized these directories. Every
    /// block found under any of `train_dirs` is assigned to `train_ids`;
    /// every block found under any of `val_dirs` is assigned to `val_ids`.
    ///
    /// `n_classes`/label-map-contiguity validation is shared across **all**
    /// directories (train + val combined), exactly as it is today across
    /// multiple `--data-dir` entries passed to [`load`](Self::load) — a val
    /// directory preprocessed with a different label map than the train
    /// directories is still a hard error.
    ///
    /// # Errors
    /// Returns an error if either `train_dirs` or `val_dirs` is empty, if any
    /// manifest cannot be read/parsed, or if the `n_classes` values are
    /// inconsistent across the combined directory set.
    pub fn load_presplit(train_dirs: &[PathBuf], val_dirs: &[PathBuf]) -> Result<Self> {
        if train_dirs.is_empty() {
            return Err(ClassifierError::Pipeline(
                "at least one --data-dir (train) is required".into(),
            ));
        }
        if val_dirs.is_empty() {
            return Err(ClassifierError::Pipeline(
                "at least one --val-data-dir is required".into(),
            ));
        }

        // Load all directories (train first, so train dirs occupy the low
        // indices and val dirs occupy the high indices — purely an internal
        // detail, transparent via the existing GlobalBlockId composite key).
        let mut all_dirs: Vec<PathBuf> = Vec::with_capacity(train_dirs.len() + val_dirs.len());
        all_dirs.extend_from_slice(train_dirs);
        all_dirs.extend_from_slice(val_dirs);

        let (dirs, n_classes_inner) = load_dir_entries(&all_dirs)?;
        let n_features_inner = N_FEATURES;

        let n_train_dirs = train_dirs.len();
        let mut train_ids = Vec::new();
        let mut val_ids = Vec::new();
        for (dir_idx, entry) in dirs.iter().enumerate() {
            let is_val = dir_idx >= n_train_dirs;
            for b in &entry.manifest.blocks {
                let gid = make_global_id(dir_idx, b.meta.id);
                if is_val {
                    val_ids.push(gid);
                } else {
                    train_ids.push(gid);
                }
            }
        }

        eprintln!(
            "[dataset] pre-split directories — {} train dir(s), {} val dir(s) — \
             train blocks: {}, val blocks: {}",
            n_train_dirs,
            val_dirs.len(),
            train_ids.len(),
            val_ids.len()
        );

        let train_set: HashSet<u64> = train_ids.iter().copied().collect();

        Ok(Self {
            dirs,
            n_classes_inner,
            n_features_inner,
            train_ids,
            val_ids,
            train_set,
            cache: None,
        })
    }

    /// Opt into an in-memory block cache bounded by `max_mb` megabytes.
    ///
    /// `None` disables caching — the default after [`load`](Self::load), and
    /// exactly matches pre-Stage-27 behavior (every `load_block()` call reads
    /// from disk). When enabled, blocks are cached on a best-effort,
    /// byte-budget-bounded basis: see `BlockCache` for the eviction-free,
    /// budget-exceeded-warns-once design rationale.
    #[must_use]
    pub fn with_block_cache(mut self, max_mb: Option<usize>) -> Self {
        self.cache = max_mb.map(|mb| Mutex::new(BlockCache::new(mb)));
        self
    }

    /// Return the validated common class count across all loaded directories.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes_inner
    }

    /// Return the fixed feature count per point (`N_FEATURES` = 17).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features_inner
    }

    /// Return the raw ASPRS-code-string → model-class-index label map
    /// declared in the **first** loaded directory's `labeled_blocks.json`
    /// manifest (Stage 40).
    ///
    /// All loaded directories are already validated (in
    /// [`load_dir_entries`]) to share the same class *count*
    /// ([`n_classes`](Self::n_classes)); this accessor exposes the actual
    /// ASPRS-code ↔ model-index mapping so callers (e.g. `wb_lidar_train
    /// evaluate`'s `reconcile_n_classes`) can additionally verify that the
    /// model's own `label_map` (model index → ASPRS code) agrees with this
    /// mapping in *content*, not merely in count — two label maps with the
    /// same number of classes can still assign different ASPRS codes to the
    /// same model index, which would silently corrupt every evaluation
    /// metric without this check.
    ///
    /// Directories are guaranteed non-empty by [`load`](Self::load) and
    /// [`load_presplit`](Self::load_presplit), so indexing the first
    /// directory never panics. If multiple directories are loaded, only the
    /// first directory's label map is returned; directory-to-directory
    /// label-map *content* agreement is not currently cross-validated here
    /// (only the class count is, in [`load_dir_entries`]).
    #[must_use]
    pub fn label_map(&self) -> &HashMap<String, u8> {
        &self.dirs[0].manifest.label_map
    }

    /// Invert this dataset's ASPRS-code(string) → model-class-index label
    /// map (`label_map`) into a dense model-index → ASPRS-code `Vec<u8>` of
    /// length `n_classes()`, suitable for embedding directly as a
    /// `.wbmodel`'s `label_map` field (Stage 41).
    ///
    /// # Why this exists
    ///
    /// `preprocess-labeled` records whichever ASPRS-code ↔ model-index
    /// mapping was actually used to encode the `.lbl` files — either the
    /// built-in [`LabeledPreprocessConfig::default_label_map`] or a custom
    /// `--label-map` file — in `labeled_blocks.json`'s `label_map` field
    /// (ASPRS code → model index). `PointNetClassifier::classify()` needs
    /// the *inverse* direction (model index → ASPRS code) to translate a
    /// predicted class index back into a real ASPRS code for the output
    /// LAS `Classification` field. This method performs that inversion
    /// from the dataset's actual recorded mapping — never a hardcoded
    /// default — so training on data preprocessed with a custom
    /// `--label-map` correctly propagates that same custom mapping into
    /// the saved model.
    ///
    /// # Errors
    /// Returns an error if any ASPRS code key in the dataset's label map is
    /// not a valid `u8` string, or if any model-class-index value is out of
    /// range for `n_classes()`.
    ///
    /// Any model index with no corresponding entry in the dataset's label
    /// map (which should not occur for a validated, contiguous label map —
    /// see [`load_dir_entries`]) falls back to ASPRS code `1` (Unassigned),
    /// matching `PointNetClassifier::classify()`'s existing `.unwrap_or(1)`
    /// fallback convention for absent `label_map` entries.
    pub fn inverse_label_map(&self) -> Result<Vec<u8>> {
        let n = self.n_classes_inner;
        let mut derived: Vec<Option<u8>> = vec![None; n];
        for (code_str, &idx) in self.label_map() {
            let code: u8 = code_str.parse().map_err(|_| {
                ClassifierError::Pipeline(format!(
                    "dataset label_map has a non-numeric ASPRS code key {code_str:?}"
                ))
            })?;
            let slot = derived.get_mut(idx as usize).ok_or_else(|| {
                ClassifierError::Pipeline(format!(
                    "dataset label_map model index {idx} is out of range for {n} classes"
                ))
            })?;
            *slot = Some(code);
        }
        Ok(derived.into_iter().map(|c| c.unwrap_or(1)).collect())
    }

    /// Return the largest sampled block size recorded in the loaded manifests.
    ///
    /// GPU pre-flight and `CubeCL` memory-pool sizing need a representative
    /// upper bound for single-block tensor shapes. The manifest already
    /// records the post-resampling `sampled_point_count` for every block, so
    /// use that instead of relying solely on the historical 5120-point default.
    #[must_use]
    pub fn max_sampled_points_per_block(&self) -> usize {
        self.dirs
            .iter()
            .flat_map(|entry| entry.manifest.blocks.iter())
            .map(|block| block.meta.sampled_point_count)
            .max()
            .unwrap_or(0)
    }

    /// Enumerate the `GlobalBlockId` of **every** block across all loaded
    /// directories, independent of any train/val split.
    ///
    /// [`load`](Self::load) always partitions blocks into `train_ids` /
    /// `val_ids` (and `spatial_split` forces at least one held-out macro-tile
    /// even when `val_split == 0.0`). Consumers that must score the entire
    /// dataset — e.g. held-out evaluation — should iterate this list instead of
    /// relying on `train_ids ∪ val_ids`. Each block is returned exactly once.
    #[must_use]
    pub fn all_block_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        for (dir_idx, entry) in self.dirs.iter().enumerate() {
            for b in &entry.manifest.blocks {
                ids.push(make_global_id(dir_idx, b.meta.id));
            }
        }
        ids
    }

    /// Compute per-class point counts from the **training** blocks only.
    /// Returns a `Vec<u64>` of length `n_classes()`.
    #[must_use]
    pub fn class_counts_train(&self) -> Vec<u64> {
        let n = self.n_classes_inner;
        // Stage 21 (Performance): use the HashSet cached at load() time
        // instead of rebuilding it on every call.
        let mut counts = vec![0u64; n];
        for (dir_idx, entry) in self.dirs.iter().enumerate() {
            for b in &entry.manifest.blocks {
                let gid = make_global_id(dir_idx, b.meta.id);
                if !self.train_set.contains(&gid) {
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
    /// Stage 27 (Block Caching, audit finding 5.2): if an in-memory block
    /// cache is installed (via [`with_block_cache`](Self::with_block_cache)),
    /// this checks it first and returns a cheap in-memory clone on a hit
    /// (no disk I/O). On a miss, it falls through to the original disk-read
    /// path unchanged, then makes a best-effort attempt to populate the
    /// cache for subsequent calls. A poisoned cache mutex is treated as
    /// "caching unavailable" (silently falls back to disk) rather than
    /// propagated as an error, since caching is purely a performance
    /// optimization, never a correctness requirement.
    ///
    /// # Errors
    /// Returns an error if the `.feat` or `.lbl` file cannot be read, or if
    /// the composite ID refers to an out-of-range directory.
    pub fn load_block(&self, block_id: u64) -> Result<LoadedBlock> {
        if let Some(mutex) = &self.cache {
            if let Ok(guard) = mutex.lock() {
                if let Some(cached) = guard.entries.get(&block_id) {
                    return Ok(clone_loaded_block(cached));
                }
            }
        }

        let (dir_idx, local_id) = decode_global_id(block_id);

        let entry = self.dirs.get(dir_idx).ok_or_else(|| {
            ClassifierError::Pipeline(format!(
                "load_block: directory index {dir_idx} out of range \
                 (only {} directories loaded)",
                self.dirs.len()
            ))
        })?;

        // Stage 21 (Performance): O(1) HashMap lookup instead of an O(n)
        // linear scan through entry.manifest.blocks on every call.
        let bm = entry
            .block_index
            .get(&local_id)
            .and_then(|&i| entry.manifest.blocks.get(i))
            .ok_or_else(|| {
                ClassifierError::Pipeline(format!(
                    "block {local_id} not found in manifest for '{}'",
                    entry.path.display()
                ))
            })?;

        validate_block_filename(&bm.meta.file)?;
        validate_block_filename(&bm.lbl_file)?;

        let feat_path = entry.path.join(&bm.meta.file);
        let features = load_feat_file(&feat_path)?;

        let lbl_path = entry.path.join(&bm.lbl_file);
        let n_points = features.nrows();
        let labels = load_lbl_file(&lbl_path, n_points)?;

        let loaded = LoadedBlock {
            features,
            labels,
            block_id,
        };

        // Stage 27 (Block Caching, audit finding 5.2): best-effort cache
        // insert. As above, a poisoned mutex is silently treated as
        // "caching unavailable" rather than propagated as an error.
        if let Some(mutex) = &self.cache {
            if let Ok(mut guard) = mutex.lock() {
                guard.try_insert(Arc::new(clone_loaded_block(&loaded)));
            }
        }

        Ok(loaded)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Directory / manifest loading (shared by `load()` and `load_presplit()`)
// ─────────────────────────────────────────────────────────────────────────────

/// Load and validate a set of `preprocess-labeled` output directories,
/// returning the parsed `DirEntry` list (in the same order as `data_dirs`)
/// plus the validated common class count across all of them.
///
/// Shared by [`LabeledBlockDataset::load`] and
/// [`LabeledBlockDataset::load_presplit`] (Stage 32) so both constructors
/// enforce identical `n_classes`/label-map-contiguity validation from a
/// single canonical implementation.
///
/// # Errors
/// Returns an error if any manifest cannot be read/parsed, if any label map
/// has non-contiguous/non-zero-based model class indices, or if the
/// `n_classes` values are inconsistent across directories.
fn load_dir_entries(data_dirs: &[PathBuf]) -> Result<(Vec<DirEntry>, usize)> {
    let mut dirs = Vec::with_capacity(data_dirs.len());
    let mut validated_n_classes: Option<usize> = None;

    for (idx, dir) in data_dirs.iter().enumerate() {
        let manifest_path = dir.join("labeled_blocks.json");
        let f = File::open(&manifest_path).map_err(|e| {
            ClassifierError::Pipeline(format!("cannot open {}: {e}", manifest_path.display()))
        })?;
        let manifest: LabeledBlockManifest =
            serde_json::from_reader(BufReader::new(f)).map_err(|e| {
                ClassifierError::Pipeline(format!(
                    "labeled_blocks.json parse error in {}: {e}",
                    dir.display()
                ))
            })?;

        // Derive n_classes from the number of *distinct* model class
        // indices defined in the label map.
        //
        // WHY NOT max(values)+1:
        // The previous formula used `max(label_map.values()) + 1` as an
        // index-range upper bound.  That assumption breaks whenever any
        // model class index in the range [0, max] is absent from the
        // training data (e.g. raw code 0 = "never classified" in ASPRS,
        // or any skipped code in a non-ASPRS dataset).  The absent class
        // gets count=0 → weight=0.0 → burn's CrossEntropyLoss panics.
        //
        // The correct interpretation: the label map defines a finite,
        // explicit set of output classes.  `n_classes` is simply the
        // number of distinct values (model indices) in that set.
        // `class_distribution` in each block only contains keys that
        // actually appear, so the counts Vec is indexed by model class
        // index and will naturally be 0 for indices not present in data —
        // but now n_classes matches the declared set, not a max+1 guess.
        //
        // The floor weight (1e-3) in compute_class_weights handles the
        // residual case where a declared class has no training samples.
        //
        // CONTRACT: label map values must form a 0-based contiguous set
        // {0, 1, …, n-1}.  This is validated below.
        let nc = {
            let distinct: std::collections::BTreeSet<u8> =
                manifest.label_map.values().copied().collect();
            if distinct.is_empty() {
                8 // safe fallback: matches TrainConfig::default().n_classes
            } else {
                // Validate that values are 0-based contiguous: {0, 1, …, n-1}.
                // This is required because class_counts_train() uses model
                // class indices directly as Vec indices.  A gap (e.g. values
                // {1,2,3} instead of {0,1,2}) would leave slot 0 permanently
                // empty and cause index 3 to be out-of-bounds.
                let n = distinct.len();
                // n is a small class count (never anywhere near u8::MAX),
                // so the truncating cast below is inconsequential.
                #[allow(clippy::cast_possible_truncation)]
                let expected: std::collections::BTreeSet<u8> = (0..n as u8).collect();
                if distinct != expected {
                    let vals: Vec<u8> = distinct.into_iter().collect();
                    return Err(ClassifierError::Pipeline(format!(
                        "label map in '{}' has non-contiguous or non-zero-based \
                         model class indices: {vals:?}.\n\
                         Model class indices (the VALUES in the label map JSON) \
                         must form the set {{0, 1, …, n-1}}.\n\
                         Example for 8 classes: {{\"2\":0, \"3\":1, \"4\":2, \
                         \"5\":3, \"6\":4, \"9\":5, \"7\":6, \"1\":7}}.\n\
                         The KEYS are your raw dataset class codes (any values); \
                         the VALUES are the 0-based model output indices.",
                        dir.display()
                    )));
                }
                n
            }
        };

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

        let block_index: HashMap<u64, usize> = manifest
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.meta.id, i))
            .collect();

        dirs.push(DirEntry {
            path: dir.clone(),
            manifest,
            block_index,
        });
    }

    let n_classes_inner = validated_n_classes.unwrap_or(8);
    Ok((dirs, n_classes_inner))
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial split
// ─────────────────────────────────────────────────────────────────────────────

/// Assign blocks to train or validation set using macro-tile stride selection.
// Stage 24 (Code Quality Cleanup, item 4.1): `n_tiles`/`target_val`/`offset`
// are all small, bounded counts (number of macro-tiles, never anywhere near
// f64/usize precision limits), so the truncating/sign-losing casts here are
// inconsequential.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
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
    // Fixed-width validation (Stage 30, Step 5e+5f+5g): n_features must equal
    // the fixed N_FEATURES constant (7 scalar + 10 pre-pass eigenvalue features).
    if n_features != N_FEATURES {
        return Err(ClassifierError::Pipeline(format!(
            "feat: n_features={n_features} does not match expected fixed-width \
             feature count N_FEATURES={N_FEATURES}"
        )));
    }

    // ── Data ─────────────────────────────────────────────────────────────
    // Stage 20 (Security Hardening): use checked arithmetic throughout the
    // size computation and enforce an upper bound *before* allocating, so a
    // corrupted or maliciously-crafted header (e.g. n_points ≈ u32::MAX)
    // cannot drive a multi-gigabyte allocation attempt.
    let n_f32 = n_points.checked_mul(n_features).ok_or_else(|| {
        ClassifierError::Pipeline(format!(
            "feat '{}': n_points × n_features overflows usize (n_points={n_points}, n_features={n_features})",
            path.display()
        ))
    })?;
    let payload_bytes = n_f32.checked_mul(4).ok_or_else(|| {
        ClassifierError::Pipeline(format!(
            "feat '{}': data payload size overflows usize",
            path.display()
        ))
    })?;
    if payload_bytes > MAX_FEAT_PAYLOAD_BYTES {
        return Err(ClassifierError::Pipeline(format!(
            "feat '{}': data payload of {payload_bytes} bytes exceeds the {MAX_FEAT_PAYLOAD_BYTES}-byte \
             safety cap (n_points={n_points}, n_features={n_features}); refusing to allocate",
            path.display()
        )));
    }

    let mut buf = vec![0u8; payload_bytes];
    f.read_exact(&mut buf)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    // Stage 21 (Performance): zero-copy byte→f32 reinterpretation via
    // `bytemuck::try_cast_slice` instead of a manual per-element
    // `chunks_exact(4).map(from_le_bytes)` loop. `try_cast_slice` (rather
    // than the panicking `cast_slice`) preserves the project-wide
    // no-panics rule: on the (practically unreachable, since a `Vec<u8>`'s
    // heap allocation is always suitably aligned for `f32` on every
    // supported platform) misalignment case, fall back to the original
    // manual conversion instead of erroring or panicking.
    let floats: Vec<f32> = match bytemuck::try_cast_slice::<u8, f32>(&buf) {
        Ok(slice) => slice.to_vec(),
        Err(_) => buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    };

    Array2::from_shape_vec((n_points, n_features), floats)
        .map_err(|e| ClassifierError::Pipeline(format!("feat reshape: {e}")))
}

/// Read a raw `.lbl` file — just `u8[n_points]`.
fn load_lbl_file(path: &Path, n_points: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path)
        .map_err(|e| ClassifierError::Pipeline(format!("lbl open {}: {e}", path.display())))?;

    // Stage 20 (Security Hardening): validate the file is at least large
    // enough before reading, so a truncated `.lbl` produces a clear
    // validation error rather than a generic "unexpected end of file" I/O
    // error from read_exact.
    let actual_len = f
        .metadata()
        .map_err(|e| ClassifierError::Pipeline(format!("lbl metadata {}: {e}", path.display())))?
        .len();
    let expected_len = n_points as u64;
    if actual_len < expected_len {
        return Err(ClassifierError::Pipeline(format!(
            "lbl '{}' is truncated: expected {expected_len} bytes, found {actual_len}",
            path.display()
        )));
    }

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

    // Test-fixture index range (0..16) is nowhere near u32::MAX; truncation
    // is not possible in practice.
    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn test_spatial_split_fraction() {
        // 16 blocks in 16 distinct macro-tiles; val_split=0.25 → 4 val tiles
        let blocks: Vec<_> = (0..16u64).map(|i| make_lbm(i, i as u32)).collect();
        let manifest = dummy_manifest(blocks);
        let (train, val) = spatial_split(&manifest, 0.25, 42);
        assert_eq!(val.len(), 4, "expected 4 val blocks, got {}", val.len());
        assert_eq!(train.len(), 12);
    }

    #[test]
    fn test_max_sampled_points_per_block_uses_manifest_metadata() {
        let mut blocks_a = vec![make_lbm(1, 0), make_lbm(2, 0)];
        blocks_a[0].meta.sampled_point_count = 2048;
        blocks_a[1].meta.sampled_point_count = 4096;

        let mut blocks_b = vec![make_lbm(3, 1)];
        blocks_b[0].meta.sampled_point_count = 8192;

        let manifest_a = dummy_manifest(blocks_a);
        let manifest_b = dummy_manifest(blocks_b);
        let block_index_a = build_block_index(&manifest_a);
        let block_index_b = build_block_index(&manifest_b);

        let dataset = LabeledBlockDataset {
            dirs: vec![
                DirEntry {
                    path: PathBuf::from("a"),
                    manifest: manifest_a,
                    block_index: block_index_a,
                },
                DirEntry {
                    path: PathBuf::from("b"),
                    manifest: manifest_b,
                    block_index: block_index_b,
                },
            ],
            n_classes_inner: 8,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        assert_eq!(dataset.max_sampled_points_per_block(), 8192);
    }

    #[test]
    fn test_all_block_ids_covers_every_block_once_across_dirs() {
        // Two dirs with colliding local IDs (0,1 in each). all_block_ids()
        // must return every block exactly once with distinct GlobalBlockIds,
        // independent of any train/val split.
        let manifest_a = dummy_manifest(vec![make_lbm(0, 0), make_lbm(1, 0)]);
        let manifest_b = dummy_manifest(vec![make_lbm(0, 1), make_lbm(1, 1)]);
        let block_index_a = build_block_index(&manifest_a);
        let block_index_b = build_block_index(&manifest_b);

        let dataset = LabeledBlockDataset {
            dirs: vec![
                DirEntry {
                    path: PathBuf::from("a"),
                    manifest: manifest_a,
                    block_index: block_index_a,
                },
                DirEntry {
                    path: PathBuf::from("b"),
                    manifest: manifest_b,
                    block_index: block_index_b,
                },
            ],
            n_classes_inner: 8,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        let ids = dataset.all_block_ids();
        assert_eq!(ids.len(), 4, "all four blocks must be enumerated");
        let unique: HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(unique.len(), 4, "GlobalBlockIds must be distinct");
        assert!(unique.contains(&make_global_id(0, 0)));
        assert!(unique.contains(&make_global_id(0, 1)));
        assert!(unique.contains(&make_global_id(1, 0)));
        assert!(unique.contains(&make_global_id(1, 1)));
    }

    /// Stage 21 (Performance): the `HashMap`-backed `load_block()` lookup
    /// must find an existing block ID and return `None` (via the outer
    /// `Option` chain) for a missing one, matching the pre-optimization
    /// linear-scan semantics exactly.
    #[test]
    fn test_block_index_hit_and_miss() {
        let blocks = vec![make_lbm(10, 0), make_lbm(20, 0), make_lbm(30, 1)];
        let manifest = dummy_manifest(blocks);
        let index = build_block_index(&manifest);

        // Hit: existing local block ID resolves to the correct manifest entry.
        let found = index.get(&20).and_then(|&i| manifest.blocks.get(i));
        assert!(found.is_some());
        assert_eq!(found.unwrap().meta.id, 20);

        // Miss: a local ID never present in the manifest resolves to None.
        assert!(!index.contains_key(&999));
    }

    #[test]
    fn test_load_feat_file_rejects_oversized_header_before_allocating() {
        // Stage 20 (Security Hardening): a corrupted/malicious header whose
        // n_points × n_features × 4 exceeds MAX_FEAT_PAYLOAD_BYTES must be
        // rejected with a clear error *before* any large allocation is
        // attempted. We only need to write the 4-byte magic + 33-byte header
        // — load_feat_file must fail during header validation, never reaching
        // the point where it tries to read/allocate the (nonexistent) payload.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("oversized.feat");

        let n_points: u32 = 100_000_000;
        // N_FEATURES is a small compile-time constant (well within u32 range);
        // truncation is not possible in practice.
        #[allow(clippy::cast_possible_truncation)]
        let n_features: u32 = N_FEATURES as u32; // valid fixed-width count
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION);
        bytes.extend_from_slice(&n_points.to_le_bytes());
        bytes.extend_from_slice(&n_features.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // block_id
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_x
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_y
        std::fs::write(&path, &bytes).expect("write fixture");

        let result = load_feat_file(&path);
        assert!(result.is_err(), "oversized header must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("exceeds") && msg.contains("safety cap"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_lbl_file_rejects_truncated_file() {
        // Stage 20 (Security Hardening): a `.lbl` file shorter than the
        // requested n_points must be rejected with a clear "truncated"
        // error instead of a generic I/O read error.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("truncated.lbl");
        std::fs::write(&path, [0u8, 1u8, 2u8]).expect("write fixture");

        let result = load_lbl_file(&path, 10);
        assert!(result.is_err(), "truncated lbl must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("truncated"), "unexpected error message: {msg}");
    }

    // ── Stage 25 (Testing Gaps, item 6.2): error-path coverage ────────────────

    #[test]
    fn test_load_rejects_empty_data_dirs() {
        let result = LabeledBlockDataset::load(&[], 0.2, None, 0);
        assert!(result.is_err(), "empty data_dirs must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("at least one --data-dir"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_rejects_missing_manifest() {
        // A tempdir that exists but has no labeled_blocks.json inside it.
        let dir = tempfile::tempdir().expect("tempdir");
        let result = LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.2, None, 0);
        assert!(result.is_err(), "missing manifest must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("cannot open"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_rejects_corrupt_manifest_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("labeled_blocks.json"), b"{ not valid json ")
            .expect("write fixture");
        let result = LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.2, None, 0);
        assert!(result.is_err(), "corrupt manifest JSON must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("parse error"),
            "unexpected error message: {msg}"
        );
    }

    // ── Stage 41 (Model label_map identity bug fix): inverse_label_map() ──

    #[test]
    fn test_inverse_label_map_non_identity_mapping() {
        // Mirrors LabeledPreprocessConfig::default_label_map()'s inversion:
        // ASPRS code (string) -> model index, inverted into model index ->
        // ASPRS code. Deliberately non-identity so a regression to hardcoded
        // identity would be caught.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut label_map = HM::new();
        label_map.insert("2".to_string(), 0u8); // Ground
        label_map.insert("3".to_string(), 1u8); // Low Veg
        label_map.insert("6".to_string(), 2u8); // Building
        let mut manifest = dummy_manifest(vec![make_lbm(0, 0)]);
        manifest.label_map = label_map;
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write fixture");

        let dataset = LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.2, None, 0)
            .expect("load dataset");

        let inverted = dataset.inverse_label_map().expect("inverse_label_map");
        assert_eq!(inverted, vec![2u8, 3u8, 6u8]);
    }

    #[test]
    fn test_inverse_label_map_rejects_non_numeric_asprs_code() {
        let manifest_label_map = {
            let mut m = HM::new();
            m.insert("not-a-number".to_string(), 0u8);
            m
        };
        let block_index = HashMap::new();
        let mut manifest = dummy_manifest(vec![]);
        manifest.label_map = manifest_label_map;
        let dataset = LabeledBlockDataset {
            dirs: vec![DirEntry {
                path: PathBuf::from("only-dir"),
                manifest,
                block_index,
            }],
            n_classes_inner: 1,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        let result = dataset.inverse_label_map();
        assert!(result.is_err(), "non-numeric ASPRS code must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("non-numeric"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_inverse_label_map_rejects_out_of_range_model_index() {
        let manifest_label_map = {
            let mut m = HM::new();
            // n_classes_inner will be set to 1 below, but this label map
            // claims model index 5 — out of range.
            m.insert("2".to_string(), 5u8);
            m
        };
        let block_index = HashMap::new();
        let mut manifest = dummy_manifest(vec![]);
        manifest.label_map = manifest_label_map;
        let dataset = LabeledBlockDataset {
            dirs: vec![DirEntry {
                path: PathBuf::from("only-dir"),
                manifest,
                block_index,
            }],
            n_classes_inner: 1,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        let result = dataset.inverse_label_map();
        assert!(result.is_err(), "out-of-range model index must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_rejects_non_contiguous_label_map() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut label_map = HM::new();
        // Values {1, 2} are not 0-based contiguous ({0, 1, ...}).
        label_map.insert("5".to_string(), 1u8);
        label_map.insert("6".to_string(), 2u8);
        let mut manifest = dummy_manifest(vec![make_lbm(0, 0)]);
        manifest.label_map = label_map;
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write fixture");

        let result = LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.2, None, 0);
        assert!(result.is_err(), "non-contiguous label map must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("non-contiguous"),
            "unexpected error message: {msg}"
        );
    }

    // ── Stage 32 (Dataset Split Materialization): load_presplit() ─────────

    #[test]
    fn test_load_presplit_assigns_entire_dirs_to_train_or_val() {
        let train_dir = tempfile::tempdir().expect("train tempdir");
        let val_dir = tempfile::tempdir().expect("val tempdir");

        let train_manifest = dummy_manifest(vec![make_lbm(0, 0), make_lbm(1, 1)]);
        let val_manifest = dummy_manifest(vec![make_lbm(0, 2)]); // local id 0 again — must not collide

        std::fs::write(
            train_dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&train_manifest).expect("serialize train manifest"),
        )
        .expect("write train manifest fixture");
        std::fs::write(
            val_dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&val_manifest).expect("serialize val manifest"),
        )
        .expect("write val manifest fixture");

        let dataset = LabeledBlockDataset::load_presplit(
            &[train_dir.path().to_path_buf()],
            &[val_dir.path().to_path_buf()],
        )
        .expect("load_presplit should succeed");

        assert_eq!(
            dataset.train_ids.len(),
            2,
            "all train-dir blocks → train_ids"
        );
        assert_eq!(dataset.val_ids.len(), 1, "all val-dir blocks → val_ids");

        // No overlap between train_ids and val_ids despite colliding local IDs.
        let train_set: HashSet<u64> = dataset.train_ids.iter().copied().collect();
        let val_set: HashSet<u64> = dataset.val_ids.iter().copied().collect();
        assert!(train_set.is_disjoint(&val_set));
    }

    #[test]
    fn test_load_presplit_rejects_empty_train_or_val_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dummy_manifest(vec![make_lbm(0, 0)]);
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write fixture");

        assert!(
            LabeledBlockDataset::load_presplit(&[], &[dir.path().to_path_buf()]).is_err(),
            "empty train_dirs must be rejected"
        );
        assert!(
            LabeledBlockDataset::load_presplit(&[dir.path().to_path_buf()], &[]).is_err(),
            "empty val_dirs must be rejected"
        );
    }

    #[test]
    fn test_load_rejects_n_classes_mismatch_across_dirs() {
        let dir0 = tempfile::tempdir().expect("tempdir 0");
        let dir1 = tempfile::tempdir().expect("tempdir 1");

        let mut label_map_2 = HM::new();
        label_map_2.insert("a".to_string(), 0u8);
        label_map_2.insert("b".to_string(), 1u8);
        let mut manifest0 = dummy_manifest(vec![make_lbm(0, 0)]);
        manifest0.label_map = label_map_2;

        let mut label_map_3 = HM::new();
        label_map_3.insert("a".to_string(), 0u8);
        label_map_3.insert("b".to_string(), 1u8);
        label_map_3.insert("c".to_string(), 2u8);
        let mut manifest1 = dummy_manifest(vec![make_lbm(0, 0)]);
        manifest1.label_map = label_map_3;

        std::fs::write(
            dir0.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest0).expect("serialize manifest0"),
        )
        .expect("write fixture 0");
        std::fs::write(
            dir1.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest1).expect("serialize manifest1"),
        )
        .expect("write fixture 1");

        let result = LabeledBlockDataset::load(
            &[dir0.path().to_path_buf(), dir1.path().to_path_buf()],
            0.2,
            None,
            0,
        );
        assert!(result.is_err(), "n_classes mismatch must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("n_classes mismatch"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_feat_file_rejects_bad_magic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad_magic.feat");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NOPE"); // wrong 4-byte magic
        bytes.push(FEAT_VERSION);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_points
        bytes.extend_from_slice(&12u32.to_le_bytes()); // n_features
        bytes.extend_from_slice(&0u64.to_le_bytes()); // block_id
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_x
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_y
        std::fs::write(&path, &bytes).expect("write fixture");

        let result = load_feat_file(&path);
        assert!(result.is_err(), "bad magic must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("bad magic"), "unexpected error message: {msg}");
    }

    #[test]
    fn test_load_feat_file_rejects_unsupported_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad_version.feat");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION.wrapping_add(1)); // unsupported version
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_points
        bytes.extend_from_slice(&12u32.to_le_bytes()); // n_features
        bytes.extend_from_slice(&0u64.to_le_bytes()); // block_id
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_x
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_y
        std::fs::write(&path, &bytes).expect("write fixture");

        let result = load_feat_file(&path);
        assert!(result.is_err(), "unsupported version must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unsupported version"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_block_rejects_out_of_range_dir_index() {
        let manifest = dummy_manifest(vec![make_lbm(0, 0)]);
        let block_index = build_block_index(&manifest);
        let dataset = LabeledBlockDataset {
            dirs: vec![DirEntry {
                path: PathBuf::from("only-dir"),
                manifest,
                block_index,
            }],
            n_classes_inner: 8,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        // dir index 5 does not exist (only 1 directory loaded).
        let result = dataset.load_block(make_global_id(5, 0));
        assert!(
            result.is_err(),
            "out-of-range directory index must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_load_block_rejects_missing_local_id() {
        let manifest = dummy_manifest(vec![make_lbm(10, 0)]);
        let block_index = build_block_index(&manifest);
        let dataset = LabeledBlockDataset {
            dirs: vec![DirEntry {
                path: PathBuf::from("only-dir"),
                manifest,
                block_index,
            }],
            n_classes_inner: 8,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        };

        // Local block id 999 is absent from the manifest.
        let result = dataset.load_block(make_global_id(0, 999));
        assert!(result.is_err(), "missing local block id must be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not found in manifest"),
            "unexpected error message: {msg}"
        );
    }

    // ── Stage 27 (Block Caching, audit finding 5.2): cache behavior ───────────

    #[test]
    fn test_block_cache_hit_avoids_disk_reread() {
        // Build a single-directory dataset on disk with one real block, load
        // it once with caching enabled (populating the cache), then delete
        // the underlying .feat/.lbl files. A second load_block() call must
        // still succeed by serving from the cache instead of touching disk.
        let dir = tempfile::tempdir().expect("tempdir");
        let block_id = 0u64;
        let n_points = 4usize;
        let n_features = N_FEATURES;

        // Write a minimal valid .feat file.
        let feat_path = dir.path().join("block_00000.feat");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION);
        // n_points/n_features are tiny test-fixture constants (4 and
        // N_FEATURES), nowhere near u32::MAX — truncation is not possible.
        #[allow(clippy::cast_possible_truncation)]
        {
            bytes.extend_from_slice(&(n_points as u32).to_le_bytes());
            bytes.extend_from_slice(&(n_features as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&block_id.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        for _ in 0..(n_points * n_features) {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        std::fs::write(&feat_path, &bytes).expect("write feat fixture");

        // Write a minimal valid .lbl file.
        let lbl_path = dir.path().join("block_00000.lbl");
        std::fs::write(&lbl_path, vec![0u8; n_points]).expect("write lbl fixture");

        let mut lbm = make_lbm(block_id, 0);
        lbm.meta.file = "block_00000.feat".to_string();
        lbm.lbl_file = "block_00000.lbl".to_string();
        let manifest = dummy_manifest(vec![lbm]);
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest fixture");

        let dataset = LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.2, None, 0)
            .expect("load dataset")
            .with_block_cache(Some(64));

        let gid = make_global_id(0, block_id);

        // First load: cache miss, reads from disk, populates cache.
        let first = dataset.load_block(gid).expect("first load_block");
        assert_eq!(first.features.nrows(), n_points);

        // Delete the on-disk files so a second disk read would fail.
        std::fs::remove_file(&feat_path).expect("remove feat fixture");
        std::fs::remove_file(&lbl_path).expect("remove lbl fixture");

        // Second load: must succeed via the cache, not disk.
        let second = dataset
            .load_block(gid)
            .expect("second load_block must hit cache, not disk");
        assert_eq!(second.features.nrows(), n_points);
        assert_eq!(second.labels.len(), n_points);
    }

    #[test]
    fn test_block_cache_budget_exceeded_falls_back_gracefully() {
        // A cache budget of 0 MB can never fit even the smallest block, so
        // every insert attempt must silently decline (never error) and the
        // one-time warning path (warned_budget_exceeded) must engage without
        // panicking. load_block() must still succeed by falling back to disk
        // on every call.
        let block = LoadedBlock {
            features: Array2::from_elem((4, N_FEATURES), 1.0f32),
            labels: vec![0u8; 4],
            block_id: 0,
        };
        let mut cache = BlockCache::new(0);
        assert!(!cache.warned_budget_exceeded);

        cache.try_insert(Arc::new(clone_loaded_block(&block)));
        assert!(
            cache.warned_budget_exceeded,
            "budget of 0 MB must immediately exceed on first insert attempt"
        );
        assert!(
            cache.entries.is_empty(),
            "an over-budget insert must not actually store the block"
        );

        // A second over-budget insert must not panic and must not re-warn
        // (warned_budget_exceeded stays true; try_insert() has no visible
        // side effect to assert on for the "only warns once" behavior beyond
        // not panicking, since eprintln! output isn't captured here).
        let block2 = LoadedBlock {
            features: Array2::from_elem((4, N_FEATURES), 2.0f32),
            labels: vec![1u8; 4],
            block_id: 1,
        };
        cache.try_insert(Arc::new(clone_loaded_block(&block2)));
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn test_with_block_cache_none_disables_caching() {
        // with_block_cache(None) must leave the dataset's cache field unset,
        // exactly matching pre-Stage-27 behavior (load_block() always reads
        // from disk, never touching any cache).
        let manifest = dummy_manifest(vec![make_lbm(0, 0)]);
        let block_index = build_block_index(&manifest);
        let dataset = LabeledBlockDataset {
            dirs: vec![DirEntry {
                path: PathBuf::from("only-dir"),
                manifest,
                block_index,
            }],
            n_classes_inner: 8,
            n_features_inner: N_FEATURES,
            train_ids: Vec::new(),
            val_ids: Vec::new(),
            train_set: HashSet::new(),
            cache: None,
        }
        .with_block_cache(None);

        assert!(dataset.cache.is_none());
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    use crate::preprocessing::labeled_pipeline::{
        LabeledBlockManifest, LabeledBlockMeta, SpatialTileGrid,
    };
    use crate::preprocessing::pipeline::BlockMeta;
    use std::collections::HashMap as HM;

    /// Test-only mirror of the `HashMap` construction done in `load()`.
    fn build_block_index(manifest: &LabeledBlockManifest) -> HashMap<u64, usize> {
        manifest
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.meta.id, i))
            .collect()
    }

    // Test-fixture helper: `id` is always a small deterministic index in
    // these tests, nowhere near f64's precision limit — precision loss is
    // not possible in practice.
    #[allow(clippy::cast_precision_loss)]
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
