//! Three-way (train/val/test) spatially-disjoint, optionally class-stratified
//! macro-tile split for labeled block datasets (Stage 32), generalized to
//! merge multiple `preprocess-labeled` output directories into a single
//! globally-stratified split (Stage 33).
//!
//! This module computes *which* macro-tiles (and therefore which blocks) go
//! into each of up to three subsets. It does not touch any files on disk —
//! see `crate::cli::split_dataset_cmd` for the tool that physically
//! materializes the result into separate directories.
//!
//! See `docs/stages/stage-32-dataset-split-materialization.md` and
//! `docs/stages/stage-33-multi-input-dataset-split.md` for the full design
//! rationale.

use std::collections::{BTreeSet, HashMap};

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::LabeledBlockManifest;

/// Relative weight given to matching the requested train/val/test size
/// fractions during stratified assignment, versus matching per-class
/// proportions. Chosen so the user's explicit `--val-split`/`--test-split`
/// request is respected as the primary constraint, with class balance acting
/// as a secondary refinement.
const SIZE_WEIGHT: f64 = 4.0;
const CLASS_WEIGHT: f64 = 1.0;

/// Local (non-global) block IDs assigned to each of the three subsets.
///
/// Used for the single-manifest case; see [`MultiThreeWaySplit`] for the
/// multi-input case, which additionally tracks which input manifest each
/// block came from.
#[derive(Debug, Clone, Default)]
pub struct ThreeWaySplit {
    pub train_ids: Vec<u64>,
    pub val_ids: Vec<u64>,
    pub test_ids: Vec<u64>,
}

/// `(source_dir_index, original_block_id)` pairs assigned to each of the
/// three subsets, computed across one or more merged input manifests
/// (Stage 33). `source_dir_index` is the position of the originating
/// manifest within the `manifests` slice passed to
/// [`three_way_spatial_split_multi`].
#[derive(Debug, Clone, Default)]
pub struct MultiThreeWaySplit {
    pub train: Vec<(usize, u64)>,
    pub val: Vec<(usize, u64)>,
    pub test: Vec<(usize, u64)>,
}

/// Compute a spatially-disjoint, macro-tile-based 3-way split for a single
/// manifest.
///
/// `test_split == 0.0` produces an empty `test_ids` (equivalent to a pure
/// 2-way train/val split). When `stratify_classes` is `true`, macro-tiles are
/// assigned via a greedy cost-minimizing heuristic that balances both the
/// requested size fractions and each subset's aggregate per-class
/// proportions (using each block's `class_distribution`, already computed at
/// `preprocess-labeled` time). When `false`, a pure spatial stride-selection
/// (matching the existing 2-way `training::dataset::spatial_split`
/// algorithm) is used instead.
///
/// This is a thin wrapper around [`three_way_spatial_split_multi`] with a
/// single-element manifest slice — see that function for the general,
/// multi-input algorithm. Kept as a separate function (rather than requiring
/// every caller to wrap a single manifest in a slice) for API stability and
/// because it is the common case.
///
/// # Errors
/// Returns an error if `val_split`/`test_split` are outside `[0.0, 1.0)` or
/// if `val_split + test_split >= 1.0` (which would leave no blocks for
/// training).
pub fn three_way_spatial_split(
    manifest: &LabeledBlockManifest,
    val_split: f64,
    test_split: f64,
    seed: u64,
    stratify_classes: bool,
) -> Result<ThreeWaySplit> {
    let multi =
        three_way_spatial_split_multi(&[manifest], val_split, test_split, seed, stratify_classes)?;
    // A single-manifest call always tags every block with source_dir_index
    // == 0, so stripping that index is lossless here.
    Ok(ThreeWaySplit {
        train_ids: multi.train.into_iter().map(|(_, id)| id).collect(),
        val_ids: multi.val.into_iter().map(|(_, id)| id).collect(),
        test_ids: multi.test.into_iter().map(|(_, id)| id).collect(),
    })
}

/// Compute a spatially-disjoint, macro-tile-based 3-way split across one or
/// more merged `preprocess-labeled` manifests (Stage 33).
///
/// Each input manifest's blocks are grouped by a composite
/// `(source_dir_index, macro_tile_id)` key rather than bare `macro_tile_id`,
/// since `macro_tile_id` is only meaningful relative to its own source
/// file's local bounding box/grid — two different source files' tile `0`
/// are unrelated tiles that must never be merged together. All manifests
/// must be mutually compatible (see [`validate_manifest_compatibility`]);
/// class-stratification then balances against the true *combined* global
/// per-class proportions across all inputs, not each input's local
/// distribution alone.
///
/// # Errors
/// Returns an error if `manifests` is empty, if `val_split`/`test_split` are
/// outside `[0.0, 1.0)`, if `val_split + test_split >= 1.0`, or if the
/// manifests are not mutually compatible (see
/// [`validate_manifest_compatibility`]).
pub fn three_way_spatial_split_multi(
    manifests: &[&LabeledBlockManifest],
    val_split: f64,
    test_split: f64,
    seed: u64,
    stratify_classes: bool,
) -> Result<MultiThreeWaySplit> {
    if !(0.0..1.0).contains(&val_split) || !val_split.is_finite() {
        return Err(ClassifierError::Pipeline(
            "val_split must be in the range [0.0, 1.0) and finite".into(),
        ));
    }
    if !(0.0..1.0).contains(&test_split) || !test_split.is_finite() {
        return Err(ClassifierError::Pipeline(
            "test_split must be in the range [0.0, 1.0) and finite".into(),
        ));
    }
    if val_split + test_split >= 1.0 {
        return Err(ClassifierError::Pipeline(
            "val_split + test_split must be < 1.0 (the training set would otherwise be empty)"
                .into(),
        ));
    }

    let n_classes = validate_manifest_compatibility(manifests)?;

    let mut tile_to_blocks: HashMap<(usize, u32), Vec<(usize, u64)>> = HashMap::new();
    for (dir_idx, manifest) in manifests.iter().enumerate() {
        for b in &manifest.blocks {
            tile_to_blocks
                .entry((dir_idx, b.macro_tile_id))
                .or_default()
                .push((dir_idx, b.meta.id));
        }
    }

    if stratify_classes {
        Ok(stratified_assign_multi(
            manifests,
            &tile_to_blocks,
            n_classes,
            val_split,
            test_split,
            seed,
        ))
    } else {
        Ok(non_stratified_assign_multi(
            &tile_to_blocks,
            val_split,
            test_split,
            seed,
        ))
    }
}

/// Validate that all supplied manifests were preprocessed with mutually
/// compatible settings, and return the shared, validated class count.
///
/// Fields describing *how* the data was preprocessed (`label_map`,
/// `block_size`, `target_points`, `min_density`, `search_radius`,
/// `min_neighbors`, `crs_epsg`) must be identical across every manifest —
/// merging manifests preprocessed with different settings would silently
/// produce a meaningless or misleading split. Fields describing *where* the
/// data came from (`source`, `spatial_tile_grid`) are intentionally not
/// validated — they legitimately differ per source file.
///
/// # Errors
/// Returns an error if `manifests` is empty, or if any manifest disagrees
/// with the first manifest on one of the fields listed above (the error
/// message names the offending input's index and field).
pub fn validate_manifest_compatibility(manifests: &[&LabeledBlockManifest]) -> Result<usize> {
    let Some(first) = manifests.first() else {
        return Err(ClassifierError::Pipeline(
            "at least one input manifest is required".into(),
        ));
    };
    let n_classes = derive_n_classes(first);

    for (idx, m) in manifests.iter().enumerate().skip(1) {
        if m.label_map != first.label_map {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s label_map does not match input manifest 0's label_map \
                 — all --input directories must have been preprocessed with the same label \
                 mapping"
            )));
        }
        if f64_mismatch(m.block_size, first.block_size) {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s block_size ({}) does not match input manifest 0's \
                 block_size ({})",
                m.block_size, first.block_size
            )));
        }
        if m.target_points != first.target_points {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s target_points ({}) does not match input manifest 0's \
                 target_points ({})",
                m.target_points, first.target_points
            )));
        }
        if f64_mismatch(m.min_density, first.min_density) {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s min_density ({}) does not match input manifest 0's \
                 min_density ({})",
                m.min_density, first.min_density
            )));
        }
        if f64_mismatch(m.search_radius, first.search_radius) {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s search_radius ({}) does not match input manifest 0's \
                 search_radius ({})",
                m.search_radius, first.search_radius
            )));
        }
        if m.min_neighbors != first.min_neighbors {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s min_neighbors ({}) does not match input manifest 0's \
                 min_neighbors ({})",
                m.min_neighbors, first.min_neighbors
            )));
        }
        if m.crs_epsg != first.crs_epsg {
            return Err(ClassifierError::Pipeline(format!(
                "input manifest {idx}'s crs_epsg ({:?}) does not match input manifest 0's \
                 crs_epsg ({:?})",
                m.crs_epsg, first.crs_epsg
            )));
        }
    }

    Ok(n_classes)
}

/// Config-value equality check for `f64` fields copied verbatim through the
/// pipeline (not derived via runtime float arithmetic), so exact-ish
/// comparison is meaningful here. A tiny epsilon avoids any theoretical
/// float round-trip artifact from JSON (de)serialization.
fn f64_mismatch(a: f64, b: f64) -> bool {
    (a - b).abs() > 1e-9
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-stratified path — pure spatial stride selection
// ─────────────────────────────────────────────────────────────────────────────

fn non_stratified_assign_multi(
    tile_to_blocks: &HashMap<(usize, u32), Vec<(usize, u64)>>,
    val_split: f64,
    test_split: f64,
    seed: u64,
) -> MultiThreeWaySplit {
    let mut tile_keys: Vec<(usize, u32)> = tile_to_blocks.keys().copied().collect();
    tile_keys.sort_unstable();

    let val_tiles = select_stride_subset(&tile_keys, val_split, seed);

    let remaining: Vec<(usize, u32)> = tile_keys
        .iter()
        .copied()
        .filter(|t| !val_tiles.contains(t))
        .collect();
    let test_tiles = select_stride_subset(&remaining, test_split, seed);

    let mut result = MultiThreeWaySplit::default();
    for key in &tile_keys {
        let ids = tile_to_blocks.get(key).cloned().unwrap_or_default();
        if val_tiles.contains(key) {
            result.val.extend(ids);
        } else if test_tiles.contains(key) {
            result.test.extend(ids);
        } else {
            result.train.extend(ids);
        }
    }
    result
}

/// Select a deterministic, evenly-strided subset of `tile_keys` covering
/// (approximately) `frac` of them. Mirrors
/// `training::dataset::spatial_split`'s selection rule exactly (for the
/// single-manifest, `u32`-keyed case), so that
/// `three_way_spatial_split(..., test_split = 0.0, stratify_classes = false)`
/// produces identical val/train tile assignment to the existing 2-way split.
/// Generalized over any `Ord + Copy` tile-key type so the same logic serves
/// both the single-manifest (`u32`) and multi-input (`(usize, u32)`) cases.
// n_tiles/target/stride/offset are small, bounded macro-tile counts, never
// anywhere near f64/usize precision limits — the casts below are
// inconsequential (same rationale as `training::dataset::spatial_split`).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn select_stride_subset<T: Ord + Copy>(tile_keys: &[T], frac: f64, seed: u64) -> BTreeSet<T> {
    let mut selected = BTreeSet::new();
    if frac <= 0.0 || tile_keys.is_empty() {
        return selected;
    }
    let n_tiles = tile_keys.len();
    let target = (n_tiles as f64 * frac).round().max(1.0) as usize;
    let stride = n_tiles / target.max(1);
    let offset = (seed as usize) % stride.max(1);

    let mut i = offset;
    while i < n_tiles && selected.len() < target {
        selected.insert(tile_keys[i]);
        i += stride;
    }
    selected
}

// ─────────────────────────────────────────────────────────────────────────────
// Stratified path — greedy, class-balance-aware bin assignment
// ─────────────────────────────────────────────────────────────────────────────

struct TileInfo {
    key: (usize, u32),
    counts: Vec<u64>,
    total: u64,
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn stratified_assign_multi(
    manifests: &[&LabeledBlockManifest],
    tile_to_blocks: &HashMap<(usize, u32), Vec<(usize, u64)>>,
    n_classes: usize,
    val_split: f64,
    test_split: f64,
    seed: u64,
) -> MultiThreeWaySplit {
    let mut block_dist: HashMap<(usize, u64), &HashMap<String, u64>> = HashMap::new();
    for (dir_idx, manifest) in manifests.iter().enumerate() {
        for b in &manifest.blocks {
            block_dist.insert((dir_idx, b.meta.id), &b.class_distribution);
        }
    }

    let mut tile_keys: Vec<(usize, u32)> = tile_to_blocks.keys().copied().collect();
    tile_keys.sort_unstable();

    let mut tiles: Vec<TileInfo> = Vec::with_capacity(tile_keys.len());
    let mut global_counts = vec![0u64; n_classes];
    for &key in &tile_keys {
        let mut counts = vec![0u64; n_classes];
        for bref in &tile_to_blocks[&key] {
            if let Some(dist) = block_dist.get(bref) {
                for (k, &v) in *dist {
                    if let Ok(idx) = k.parse::<usize>() {
                        if idx < n_classes {
                            counts[idx] += v;
                            global_counts[idx] += v;
                        }
                    }
                }
            }
        }
        let total: u64 = counts.iter().sum();
        tiles.push(TileInfo { key, counts, total });
    }

    let grand_total: u64 = global_counts.iter().sum();
    let global_props: Vec<f64> = if grand_total == 0 {
        vec![0.0; n_classes]
    } else {
        global_counts
            .iter()
            .map(|&c| c as f64 / grand_total as f64)
            .collect()
    };

    // Deterministic seeded shuffle to break ties, followed by a stable sort
    // descending by total tile point count (largest-first greedy bin-packing
    // — assigning the biggest, highest-impact tiles first minimizes final
    // imbalance versus assigning small tiles first).
    seeded_shuffle(&mut tiles, seed);
    tiles.sort_by_key(|t| std::cmp::Reverse(t.total));

    let train_frac = 1.0 - val_split - test_split;
    let fracs = [train_frac, val_split, test_split]; // 0=train, 1=val, 2=test

    let mut split_counts: [Vec<u64>; 3] = [
        vec![0u64; n_classes],
        vec![0u64; n_classes],
        vec![0u64; n_classes],
    ];
    let mut split_totals: [u64; 3] = [0, 0, 0];
    let mut assigned: [Vec<(usize, u32)>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    let grand_f = (grand_total.max(1)) as f64;

    for tile in &tiles {
        let mut best_split = 0usize;
        let mut best_cost = f64::INFINITY;

        for (s, &frac) in fracs.iter().enumerate() {
            if frac <= 0.0 {
                continue; // split disabled (e.g. test_split == 0.0)
            }
            let new_total = split_totals[s] + tile.total;
            let size_cost = (new_total as f64 / grand_f - frac).powi(2);

            let class_cost = if new_total == 0 {
                0.0
            } else {
                (0..n_classes)
                    .map(|c| {
                        let new_c = split_counts[s][c] + tile.counts[c];
                        (new_c as f64 / new_total as f64 - global_props[c]).powi(2)
                    })
                    .sum::<f64>()
            };

            let cost = SIZE_WEIGHT * size_cost + CLASS_WEIGHT * class_cost;
            if cost < best_cost {
                best_cost = cost;
                best_split = s;
            }
        }

        split_totals[best_split] += tile.total;
        for (dst, src) in split_counts[best_split].iter_mut().zip(tile.counts.iter()) {
            *dst += src;
        }
        assigned[best_split].push(tile.key);
    }

    let mut result = MultiThreeWaySplit::default();
    for key in &assigned[0] {
        result.train.extend(tile_to_blocks[key].clone());
    }
    for key in &assigned[1] {
        result.val.extend(tile_to_blocks[key].clone());
    }
    for key in &assigned[2] {
        result.test.extend(tile_to_blocks[key].clone());
    }
    result
}

/// Derive the model class count from a manifest's label map. Falls back to
/// `8` (matching `TrainConfig::default().n_classes`) if the label map is
/// empty. Unlike `training::dataset::LabeledBlockDataset::load`, this does
/// not hard-error on a non-contiguous label map — it is only used here to
/// size cost-calculation vectors for a splitting heuristic, not to index
/// per-class loss weights, so a best-effort count is acceptable.
fn derive_n_classes(manifest: &LabeledBlockManifest) -> usize {
    let distinct: BTreeSet<u8> = manifest.label_map.values().copied().collect();
    if distinct.is_empty() {
        8
    } else {
        distinct.len()
    }
}

/// Small, deterministic, non-cryptographic seeded shuffle (Fisher-Yates using
/// an inline `xorshift64*`-style generator). Not for cryptographic use —
/// purely to break ties deterministically before the largest-first stable
/// sort that actually drives greedy assignment order. Avoids adding a `rand`
/// dependency edge for this one-off, non-security-sensitive ordering step
/// (per AGENTS.md "Minimal & Thoughtful Dependencies").
#[allow(clippy::cast_possible_truncation)]
fn seeded_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = items.len();
    for i in (1..n).rev() {
        // n (a tile count) never approaches u64::MAX, so the truncating cast
        // back to usize below is inconsequential in practice.
        let j = (next_rand() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;
    use crate::preprocessing::labeled_pipeline::{LabeledBlockMeta, SpatialTileGrid};
    use crate::preprocessing::pipeline::BlockMeta;
    use std::collections::HashMap as HM;
    use std::collections::HashSet;

    fn make_block(id: u64, macro_tile_id: u32, class_counts: &[(u8, u64)]) -> LabeledBlockMeta {
        let mut dist = HM::new();
        let mut total = 0u64;
        for &(c, n) in class_counts {
            dist.insert(c.to_string(), n);
            total += n;
        }
        LabeledBlockMeta {
            meta: BlockMeta {
                id,
                file: format!("block_{id:05}.feat"),
                origin_x: 0.0,
                origin_y: 0.0,
                raw_point_count: total as usize,
                sampled_point_count: total as usize,
                oversampled: false,
            },
            lbl_file: format!("block_{id:05}.lbl"),
            macro_tile_id,
            class_distribution: dist,
        }
    }

    fn dummy_manifest(blocks: Vec<LabeledBlockMeta>, n_classes: usize) -> LabeledBlockManifest {
        let mut label_map = HM::new();
        for c in 0..n_classes {
            // Cast is safe: n_classes is a small test-fixture constant.
            #[allow(clippy::cast_possible_truncation)]
            label_map.insert(c.to_string(), c as u8);
        }
        LabeledBlockManifest {
            source: "test.las".into(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map,
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

    #[test]
    fn test_non_stratified_fraction_semantics_match_2way() {
        // 16 blocks in 16 distinct macro-tiles; val_split=0.25, test_split=0.0
        // must produce 4 val blocks / 12 train blocks / 0 test blocks —
        // identical fraction semantics to the existing 2-way
        // `training::dataset::spatial_split` (see
        // `test_spatial_split_fraction` in that module).
        let blocks: Vec<_> = (0..16u64)
            .map(|i| make_block(i, i as u32, &[(0, 10)]))
            .collect();
        let manifest = dummy_manifest(blocks, 1);
        let split = three_way_spatial_split(&manifest, 0.25, 0.0, 42, false).unwrap();
        assert_eq!(split.val_ids.len(), 4, "expected 4 val blocks");
        assert_eq!(split.train_ids.len(), 12, "expected 12 train blocks");
        assert_eq!(split.test_ids.len(), 0, "expected 0 test blocks");
    }

    #[test]
    fn test_three_way_split_disjoint_and_complete() {
        let blocks: Vec<_> = (0..30u64)
            .map(|i| make_block(i, (i % 10) as u32, &[(0, 5), (1, 5)]))
            .collect();
        let manifest = dummy_manifest(blocks, 2);

        for stratify in [false, true] {
            let split = three_way_spatial_split(&manifest, 0.2, 0.2, 7, stratify).unwrap();

            let mut all: Vec<u64> = Vec::new();
            all.extend(&split.train_ids);
            all.extend(&split.val_ids);
            all.extend(&split.test_ids);
            all.sort_unstable();
            let expected: Vec<u64> = (0..30u64).collect();
            assert_eq!(
                all, expected,
                "stratify={stratify}: union of all three subsets must equal the full \
                 block set with no duplicates"
            );

            let train_set: HashSet<u64> = split.train_ids.iter().copied().collect();
            let val_set: HashSet<u64> = split.val_ids.iter().copied().collect();
            let test_set: HashSet<u64> = split.test_ids.iter().copied().collect();
            assert!(
                train_set.is_disjoint(&val_set),
                "stratify={stratify}: train/val must be disjoint"
            );
            assert!(
                train_set.is_disjoint(&test_set),
                "stratify={stratify}: train/test must be disjoint"
            );
            assert!(
                val_set.is_disjoint(&test_set),
                "stratify={stratify}: val/test must be disjoint"
            );
        }
    }

    #[test]
    fn test_rejects_out_of_range_fractions() {
        let manifest = dummy_manifest(vec![make_block(0, 0, &[(0, 1)])], 1);
        assert!(
            three_way_spatial_split(&manifest, 1.0, 0.0, 0, false).is_err(),
            "val_split == 1.0 must be rejected"
        );
        assert!(
            three_way_spatial_split(&manifest, 0.5, 0.5, 0, false).is_err(),
            "val_split + test_split == 1.0 must be rejected"
        );
        assert!(
            three_way_spatial_split(&manifest, -0.1, 0.0, 0, false).is_err(),
            "negative val_split must be rejected"
        );
    }

    #[test]
    fn test_stratification_reduces_class_imbalance() {
        // 10 macro-tiles: tiles 0..7 are ~99% class 0, tiles 7..10 are ~99%
        // class 1. With val_split=0.3, seed=3: n_tiles=10, target=3,
        // stride=10/3=3, offset=3%3=0 -> the non-stratified path selects
        // tiles {0, 3, 6}, all from the class-0-heavy group, producing a
        // val subset with almost no class-1 representation despite the
        // dataset overall containing a substantial class-1 fraction.
        let mut blocks = Vec::new();
        for i in 0..7u64 {
            blocks.push(make_block(i, i as u32, &[(0, 100), (1, 1)]));
        }
        for i in 7..10u64 {
            blocks.push(make_block(i, i as u32, &[(0, 1), (1, 100)]));
        }
        let manifest = dummy_manifest(blocks, 2);

        let non_strat = three_way_spatial_split(&manifest, 0.3, 0.0, 3, false).unwrap();
        let strat = three_way_spatial_split(&manifest, 0.3, 0.0, 3, true).unwrap();

        let global_props = global_class_props(&manifest, 2);
        let dev_non_strat = split_deviation(&manifest, &non_strat.val_ids, &global_props, 2);
        let dev_strat = split_deviation(&manifest, &strat.val_ids, &global_props, 2);

        assert!(
            dev_strat < dev_non_strat,
            "stratified val-set class-proportion deviation ({dev_strat}) should be \
             lower than the non-stratified deviation ({dev_non_strat})"
        );
    }

    // ── multi-input (Stage 33) tests ────────────────────────────────────

    #[test]
    fn test_multi_input_parity_with_single_manifest() {
        let blocks: Vec<_> = (0..20u64)
            .map(|i| make_block(i, (i % 7) as u32, &[(0, 5), (1, 3)]))
            .collect();
        let manifest = dummy_manifest(blocks, 2);

        for stratify in [false, true] {
            let single = three_way_spatial_split(&manifest, 0.25, 0.1, 11, stratify).unwrap();
            let multi =
                three_way_spatial_split_multi(&[&manifest], 0.25, 0.1, 11, stratify).unwrap();

            let mut multi_train: Vec<u64> = multi.train.iter().map(|&(_, id)| id).collect();
            let mut multi_val: Vec<u64> = multi.val.iter().map(|&(_, id)| id).collect();
            let mut multi_test: Vec<u64> = multi.test.iter().map(|&(_, id)| id).collect();
            multi_train.sort_unstable();
            multi_val.sort_unstable();
            multi_test.sort_unstable();

            let mut single_train = single.train_ids.clone();
            let mut single_val = single.val_ids.clone();
            let mut single_test = single.test_ids.clone();
            single_train.sort_unstable();
            single_val.sort_unstable();
            single_test.sort_unstable();

            assert_eq!(
                multi_train, single_train,
                "stratify={stratify}: multi-input single-manifest train set must match"
            );
            assert_eq!(
                multi_val, single_val,
                "stratify={stratify}: multi-input single-manifest val set must match"
            );
            assert_eq!(
                multi_test, single_test,
                "stratify={stratify}: multi-input single-manifest test set must match"
            );
            assert!(
                multi.train.iter().all(|&(d, _)| d == 0),
                "single-manifest call must tag every block with dir_idx == 0"
            );
        }
    }

    #[test]
    fn test_multi_input_disjoint_and_complete_with_colliding_ids() {
        // Two manifests, each with locally-colliding block ids 0..10.
        let blocks_a: Vec<_> = (0..10u64)
            .map(|i| make_block(i, (i % 5) as u32, &[(0, 5), (1, 5)]))
            .collect();
        let blocks_b: Vec<_> = (0..10u64)
            .map(|i| make_block(i, (i % 5) as u32, &[(0, 5), (1, 5)]))
            .collect();
        let manifest_a = dummy_manifest(blocks_a, 2);
        let manifest_b = dummy_manifest(blocks_b, 2);

        for stratify in [false, true] {
            let split =
                three_way_spatial_split_multi(&[&manifest_a, &manifest_b], 0.2, 0.2, 5, stratify)
                    .unwrap();

            let mut all: Vec<(usize, u64)> = Vec::new();
            all.extend(&split.train);
            all.extend(&split.val);
            all.extend(&split.test);
            all.sort_unstable();

            let mut expected: Vec<(usize, u64)> = (0..10u64).map(|i| (0, i)).collect();
            expected.extend((0..10u64).map(|i| (1, i)));
            expected.sort_unstable();

            assert_eq!(
                all, expected,
                "stratify={stratify}: union of (dir_idx, block_id) pairs must equal the full \
                 merged block set with no duplicates, despite colliding local ids"
            );

            let train_set: HashSet<(usize, u64)> = split.train.iter().copied().collect();
            let val_set: HashSet<(usize, u64)> = split.val.iter().copied().collect();
            let test_set: HashSet<(usize, u64)> = split.test.iter().copied().collect();
            assert!(train_set.is_disjoint(&val_set));
            assert!(train_set.is_disjoint(&test_set));
            assert!(val_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_multi_input_stratification_uses_combined_global_balance() {
        // File A: 5 tiles, all ~99% class 0. File B: 5 tiles, all ~99%
        // class 1. Neither file alone has a balanced class mix, but the
        // *combined* dataset is ~50/50. A per-file-only stratifier would see
        // each file as (almost) single-class and could not do better than
        // the non-stratified split for that file; the merged multi-input
        // stratifier must use the true combined global proportions.
        let blocks_a: Vec<_> = (0..5u64)
            .map(|i| make_block(i, i as u32, &[(0, 100), (1, 1)]))
            .collect();
        let blocks_b: Vec<_> = (0..5u64)
            .map(|i| make_block(i, i as u32, &[(0, 1), (1, 100)]))
            .collect();
        let manifest_a = dummy_manifest(blocks_a, 2);
        let manifest_b = dummy_manifest(blocks_b, 2);

        let non_strat =
            three_way_spatial_split_multi(&[&manifest_a, &manifest_b], 0.3, 0.0, 3, false).unwrap();
        let strat =
            three_way_spatial_split_multi(&[&manifest_a, &manifest_b], 0.3, 0.0, 3, true).unwrap();

        // Combined global class proportions (~50/50 by construction).
        let global_props = [0.5, 0.5];

        let dev_non_strat = multi_split_deviation(
            &[&manifest_a, &manifest_b],
            &non_strat.val,
            &global_props,
            2,
        );
        let dev_strat =
            multi_split_deviation(&[&manifest_a, &manifest_b], &strat.val, &global_props, 2);

        assert!(
            dev_strat < dev_non_strat,
            "combined-global-aware stratified val-set deviation ({dev_strat}) should be \
             lower than the non-stratified deviation ({dev_non_strat})"
        );
    }

    #[test]
    fn test_validate_manifest_compatibility_rejects_mismatches() {
        let base = dummy_manifest(vec![make_block(0, 0, &[(0, 1)])], 2);

        let mut wrong_label_map = base.clone();
        wrong_label_map.label_map.insert("99".to_string(), 5);
        assert!(validate_manifest_compatibility(&[&base, &wrong_label_map]).is_err());

        let mut wrong_block_size = base.clone();
        wrong_block_size.block_size = 999.0;
        assert!(validate_manifest_compatibility(&[&base, &wrong_block_size]).is_err());

        let mut wrong_target_points = base.clone();
        wrong_target_points.target_points = 1;
        assert!(validate_manifest_compatibility(&[&base, &wrong_target_points]).is_err());

        let mut wrong_min_density = base.clone();
        wrong_min_density.min_density = 999.0;
        assert!(validate_manifest_compatibility(&[&base, &wrong_min_density]).is_err());

        let mut wrong_search_radius = base.clone();
        wrong_search_radius.search_radius = 999.0;
        assert!(validate_manifest_compatibility(&[&base, &wrong_search_radius]).is_err());

        let mut wrong_min_neighbors = base.clone();
        wrong_min_neighbors.min_neighbors = 999;
        assert!(validate_manifest_compatibility(&[&base, &wrong_min_neighbors]).is_err());

        let mut wrong_crs = base.clone();
        wrong_crs.crs_epsg = Some(1234);
        assert!(validate_manifest_compatibility(&[&base, &wrong_crs]).is_err());

        // A manifest identical to `base` (differing only in `source`, which
        // is intentionally not validated) must be accepted.
        let mut compatible = base.clone();
        compatible.source = "different_file.las".into();
        assert!(validate_manifest_compatibility(&[&base, &compatible]).is_ok());

        assert!(
            validate_manifest_compatibility(&[]).is_err(),
            "an empty manifest slice must be rejected"
        );
    }

    // ── test helpers ─────────────────────────────────────────────────────

    fn global_class_props(manifest: &LabeledBlockManifest, n_classes: usize) -> Vec<f64> {
        let mut counts = vec![0u64; n_classes];
        for b in &manifest.blocks {
            for (k, &v) in &b.class_distribution {
                if let Ok(idx) = k.parse::<usize>() {
                    if idx < n_classes {
                        counts[idx] += v;
                    }
                }
            }
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            vec![0.0; n_classes]
        } else {
            counts.iter().map(|&c| c as f64 / total as f64).collect()
        }
    }

    fn split_deviation(
        manifest: &LabeledBlockManifest,
        ids: &[u64],
        global_props: &[f64],
        n_classes: usize,
    ) -> f64 {
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        let mut counts = vec![0u64; n_classes];
        for b in &manifest.blocks {
            if !id_set.contains(&b.meta.id) {
                continue;
            }
            for (k, &v) in &b.class_distribution {
                if let Ok(idx) = k.parse::<usize>() {
                    if idx < n_classes {
                        counts[idx] += v;
                    }
                }
            }
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        (0..n_classes)
            .map(|c| {
                let p = counts[c] as f64 / total as f64;
                (p - global_props[c]).powi(2)
            })
            .sum()
    }

    fn multi_split_deviation(
        manifests: &[&LabeledBlockManifest],
        refs: &[(usize, u64)],
        global_props: &[f64],
        n_classes: usize,
    ) -> f64 {
        let ref_set: HashSet<(usize, u64)> = refs.iter().copied().collect();
        let mut counts = vec![0u64; n_classes];
        for (dir_idx, manifest) in manifests.iter().enumerate() {
            for b in &manifest.blocks {
                if !ref_set.contains(&(dir_idx, b.meta.id)) {
                    continue;
                }
                for (k, &v) in &b.class_distribution {
                    if let Ok(idx) = k.parse::<usize>() {
                        if idx < n_classes {
                            counts[idx] += v;
                        }
                    }
                }
            }
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        (0..n_classes)
            .map(|c| {
                let p = counts[c] as f64 / total as f64;
                (p - global_props[c]).powi(2)
            })
            .sum()
    }
}
