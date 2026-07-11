# Stage 33 — Multi-Input Merged-Manifest Dataset Split

## Goal

Stage 32's `wb_lidar_train split-dataset` accepts exactly **one** `--input`
directory. When a user runs `preprocess-labeled` once per source `.laz`/`.las`
file (the documented workaround for the per-file block-ID collision issue —
see below), the resulting `labeled_blocks.json` files live in separate
directories, each describing only *that file's* blocks, macro-tile grid, and
per-tile class distributions.

Today, the only way to combine such per-file outputs into a single train/val/
test split is to run `split-dataset` **once per directory, independently**.
This has a real correctness gap when `--stratify-classes` (the default) is
requested: each independent run only sees that one file's local class
distribution, not the dataset's true global distribution. A rare class that
happens to be concentrated in a single file gets stratified as if it were the
*entire* dataset's distribution for that file's share of the split, rather
than being balanced against the real global proportions across all files.
Small per-file block counts also give the greedy stratifier very little room
to work with, compounding the problem.

This stage generalizes `split-dataset` to accept **multiple** `--input`
directories in one invocation, merging their manifests before computing a
single, globally-stratified 3-way split — while still producing one physical
`train/`, `val/`, and (optional) `test/` output directory tree, each loadable
exactly as before via `LabeledBlockDataset::load_presplit()`.

### Why not fix this at the `preprocess-labeled` layer instead?

An alternative would be teaching `preprocess-labeled` to append to (rather
than overwrite) an existing `labeled_blocks.json` and renumber blocks to
avoid ID collisions across repeated invocations. That would be a more
invasive change to an existing, working command and its on-disk format, and
would not obviously generalize better than solving the problem once, at the
one place that actually needs a *global* view: the splitter. Per AGENTS.md's
"Greenfield" spirit for this sub-project and to keep the change surface
minimal, this stage only touches the new `split-dataset` command and its
supporting `dataset_split` module — `preprocess-labeled`/`labeled_pipeline.rs`
are unchanged.

### Explicitly NOT a gap (clarified during planning, no code changes needed)

`LabeledBlockDataset::load_presplit()` (Stage 32) already correctly merges
**all** blocks from **all** supplied `--val-data-dir` directories (and
separately, all `--data-dir` directories) into single combined `train_ids`/
`val_ids` sets before any training epoch runs — validation aggregation across
multiple pre-split directories already works. The gap this stage closes is
strictly upstream of that: `split-dataset` itself only ever saw one input
directory when deciding *how* to split, so if you fed it >1 directory's worth
of data by running it >1 times, each run's stratification decision was made
in ignorance of the others.

---

## Inputs & Outputs

### CLI change: `wb_lidar_train split-dataset --input` becomes repeatable

```
wb_lidar_train split-dataset
    --input   <dir>     Directory produced by `preprocess-labeled`
                         (must contain labeled_blocks.json). REPEATABLE —
                         pass --input once per source directory to merge
                         them into a single global split.
    --output  <dir>     Output directory; train/, val/, [test/] subdirectories
                         are created inside it
    [--val-split  <f64>]   Fraction of macro-tiles -> validation (default: 0.20)
    [--test-split <f64>]   Fraction of macro-tiles -> test (default: 0.0, disabled)
    [--seed <u64>]         Seed for deterministic tie-breaking (default: 42)
    [--no-stratify-classes] Disable class-stratified assignment; use pure
                            spatial macro-tile stride selection instead
                            (default: stratification is ON)
    [--move]               Move files instead of copying (default: copy)
```

This is **Option A** from the planning discussion: explicit, repeated
`--input <dir>` flags (mirroring the existing `--data-dir`/`--val-data-dir`
repeatable pattern in `train_cmd.rs`), rather than a single `--input-root`
auto-discovery flag. The caller is expected to already know its list of
per-file output directories (e.g. from the same loop that invoked
`preprocess-labeled` once per source file) and pass them explicitly.

A single `--input` continues to work exactly as before — the single-manifest
case is not a special CLI mode, it is simply the `n == 1` case of the general
multi-input path.

### Compatibility validation across inputs

Before computing any split, all supplied manifests must agree on the fields
that describe *how* the data was preprocessed (as opposed to *where* it came
from): `label_map` (identical key→value mapping), `block_size`,
`target_points`, `min_density`, `search_radius`, `min_neighbors`, `crs_epsg`.
A mismatch on any of these produces an immediate, specific error naming the
offending input's index and field (e.g. `"input manifest 2's block_size
(30.0) does not match input manifest 0's block_size (50.0)"`) — merging
manifests preprocessed with different settings would silently produce a
meaningless or misleading split.

`spatial_tile_grid` and `source` are **not** validated for equality — they
legitimately differ per source file (different files have different spatial
extents and origin paths) and are retained in the output purely for
informational/debugging purposes (see below).

### Output layout — unchanged shape, new provenance semantics

```
<output>/
  train/
    block_00000.feat, block_00000.lbl, ...   (freshly renumbered, see below)
    labeled_blocks.json
  val/
    ...
  test/                       (only if --test-split > 0.0)
    ...
```

Each subset's `labeled_blocks.json` is a `LabeledBlockManifest` with:
- `source`: a comma-joined list of every input manifest's original `source`
  value (e.g. `"tile_001.laz, tile_002.laz"`), so provenance of a merged
  split is still recoverable.
- `block_size`/`target_points`/`min_density`/`search_radius`/`min_neighbors`/
  `crs_epsg`/`label_map`: taken from the first input manifest (already
  validated identical across all inputs above).
- `spatial_tile_grid`: taken from the first input manifest as-is. This field
  is **purely informational** in the merged case — no code in this repository
  (searched: only `dataset_split.rs` and `labeled_pipeline.rs` construct or
  read `spatial_tile_grid`, and neither `training/dataset.rs` nor any other
  consumer reads it back out of a loaded manifest) actually uses this field
  after the split is materialized, so carrying over one representative grid
  is a documented, harmless simplification rather than a correctness issue.
- `blocks`: each block's `macro_tile_id` and `class_distribution` are
  retained unchanged from its original per-file manifest for debugging
  visibility, but **`macro_tile_id` is no longer globally unique or
  comparable across blocks from different original inputs** post-merge (it
  was only ever meaningful relative to its own source file's local bounding
  box/grid) — this is documented here and via a doc-comment at the relevant
  struct/field, not enforced in code (renaming the field or adding a second
  field was judged unnecessary complexity for a debug-only value).

### Fresh sequential IDs at materialization time

Because two different `--input` directories' local block IDs are computed
independently (each from that file's own header bounding box — see
`preprocessing::block_id()`/`labeled_pipeline::stream_classifications()`),
they can and do collide (e.g. both directories may contain a
`block_00000.feat`). To make the merged output directories unambiguous and
collision-free regardless of how many inputs are merged or how their local
IDs happen to line up, every output subset (`train/`, `val/`, `test/`) is
renumbered from scratch at write time:

1. Collect the subset's assigned `(source_dir_index, original_block_id)`
   pairs.
2. Sort them deterministically by `(source_dir_index, original_block_id)`.
3. Assign fresh sequential IDs `0, 1, 2, ...` in that sorted order.
4. Write each block's `.feat`/`.lbl` files under the new
   `block_{new_id:05}.feat`/`.lbl` names in the destination subset directory
   (copied/moved from its original source directory + original filename).
5. The written-out `labeled_blocks.json`'s `LabeledBlockMeta.meta.id` and
   `.meta.file`/`.lbl_file` reflect the **new** id/filenames; all other
   per-block fields (`origin_x`/`origin_y`/`raw_point_count`/
   `sampled_point_count`/`oversampled`/`macro_tile_id`/`class_distribution`)
   are carried over unchanged from the original block.

This sorted, deterministic renumbering means re-running `split-dataset` with
the same inputs/flags/seed always produces byte-identical output filenames
for a given logical block, even though absolute numeric IDs are freshly
assigned each run (they are not guaranteed to match the *original* per-file
IDs, by design).

---

## Algorithm

### `src/preprocessing/dataset_split.rs` changes

- New: `pub fn validate_manifest_compatibility(manifests: &[&LabeledBlockManifest]) -> Result<usize>`
  — validates the fields listed above are identical across all manifests;
  returns the shared, validated `n_classes` (derived from the first
  manifest's `label_map`, same rule as existing `derive_n_classes()`).
- New: `pub struct MultiThreeWaySplit { pub train: Vec<(usize, u64)>, pub val: Vec<(usize, u64)>, pub test: Vec<(usize, u64)> }`
  — each tuple is `(source_dir_index, original_block_id)`.
- New: `pub fn three_way_spatial_split_multi(manifests: &[&LabeledBlockManifest], val_split: f64, test_split: f64, seed: u64, stratify_classes: bool) -> Result<MultiThreeWaySplit>`
  — same fraction validation as the existing single-manifest function, then
  calls `validate_manifest_compatibility()`, then groups blocks by a
  composite `(source_dir_index, macro_tile_id)` key (instead of bare
  `macro_tile_id`) so tiles from different source files are never confused
  with one another even if their raw `macro_tile_id` values coincide. The
  non-stratified stride-selection and stratified greedy cost-minimization
  algorithms are otherwise **identical in spirit** to Stage 32's — generalized
  to operate on the composite tile key type and composite block reference
  type throughout (`select_stride_subset<T: Ord + Copy>`, `TileInfo { key:
  (usize, u32), .. }`, `block_dist: HashMap<(usize, u64), ..>`).
- Changed (behavior-preserving refactor): `pub fn three_way_spatial_split(manifest: &LabeledBlockManifest, ...) -> Result<ThreeWaySplit>`
  keeps its **exact existing public signature** and becomes a thin wrapper:
  calls `three_way_spatial_split_multi(&[manifest], ...)` and strips the
  (always-`0`) `source_dir_index` from each resulting tuple to reconstruct a
  plain `ThreeWaySplit`. All 4 existing unit tests for this function continue
  to exercise it unchanged and must continue to pass without modification —
  this is a **regression/parity guarantee**, not just a design goal.

### `src/cli/split_dataset_cmd.rs` changes

- `--input` parsing changes from `Option<PathBuf>` (single) to `Vec<PathBuf>`
  (repeatable, at least one required).
- Reads and parses every input directory's `labeled_blocks.json` into a
  `Vec<LabeledBlockManifest>`, then builds `Vec<&LabeledBlockManifest>` to
  call `three_way_spatial_split_multi()`.
- `materialize_split()`/`write_subset()` are generalized to accept the full
  `inputs: &[PathBuf]` list and `manifests: &[LabeledBlockManifest]`, resolve
  each `(dir_idx, block_id)` back to its correct source directory + original
  block metadata via a `HashMap<(usize, u64), &LabeledBlockMeta>` lookup, and
  perform the fresh-sequential-ID renumbering described above at copy/move
  time.
- `print_usage()` updated to document `--input` as repeatable.

---

## Definition of Done (DoD)

1. `three_way_spatial_split_multi()` called with a single manifest produces
   identical block-id results (after stripping the constant `dir_idx = 0`) to
   `three_way_spatial_split()` called directly on that same manifest — parity
   test.
2. All 4 pre-existing `three_way_spatial_split()` unit tests from Stage 32
   continue to pass **unmodified**.
3. Every `(dir_idx, block_id)` pair across 2+ synthetic input manifests
   (including manifests with deliberately colliding local block IDs) appears
   in **exactly one** of train/val/test — disjointness + completeness test.
4. Stratified multi-input mode measurably reduces aggregate class-proportion
   deviation (computed against the *true combined* global class proportions
   across all input manifests) versus running the equivalent non-stratified
   multi-input split, on a synthetic fixture where two "files" have
   complementary skewed per-tile class mixes.
5. `validate_manifest_compatibility()` rejects mismatched `label_map`,
   `block_size`, `target_points`, `min_density`, `search_radius`,
   `min_neighbors`, and `crs_epsg` with a clear, field-naming error — one
   test per field.
6. End-to-end `split-dataset`-equivalent test (calling `materialize_split()`
   directly, matching the existing Stage 32 test style): 2 synthetic input
   directories with overlapping local block IDs materialize into a single
   merged `train/`/`val/` output tree with **no filename collisions**, and
   the result is loadable via `LabeledBlockDataset::load_presplit()` with the
   expected total block counts.
7. `--move` semantics preserved: only deletes source files after a given
   block's both files copy successfully, regardless of which input directory
   the block came from.
8. `cargo build --all-targets --all-features` — zero errors.
9. `cargo clippy --all-targets --all-features -- -D warnings` — zero
   warnings.
10. `cargo clippy --all-targets --features training -- -D warnings` — zero
    warnings.
11. `cargo test --all-features` — all tests (existing + new) pass.
12. `cargo fmt -- --check` — clean.
13. This document accurately reflects the landed implementation (see
    "Implementation Status" below).

---

## Implementation Status

**Status: Complete.** All items in the Definition of Done above are
satisfied by the landed implementation.

### Files touched

- `src/preprocessing/dataset_split.rs` — added `MultiThreeWaySplit`,
  `three_way_spatial_split_multi()`, `validate_manifest_compatibility()`,
  `f64_mismatch()`, `non_stratified_assign_multi()`,
  `stratified_assign_multi()` (replacing the prior single-manifest
  `non_stratified_assign()`/`stratified_assign()`), generalized
  `select_stride_subset<T: Ord + Copy>()` and `TileInfo { key: (usize, u32),
  .. }`. `three_way_spatial_split()` was refactored into a thin wrapper
  around `three_way_spatial_split_multi(&[manifest], ...)` with its public
  signature unchanged. 5 new unit tests added; all 4 pre-existing Stage 32
  unit tests kept unmodified.
- `src/cli/split_dataset_cmd.rs` — `--input` changed from a single
  `Option<PathBuf>` to a repeatable `Vec<PathBuf>` (at least one required);
  `run()` now parses every input's `labeled_blocks.json`, builds manifest
  references, and calls `three_way_spatial_split_multi()`;
  `materialize_split()`/`write_subset()` signatures generalized to accept
  `inputs: &[PathBuf]` + `manifests: &[LabeledBlockManifest]` and perform
  fresh sequential ID renumbering (sorted by `(dir_idx, original_block_id)`)
  at copy/move time; `print_usage()` updated. 1 new negative-path test
  (`test_run_rejects_missing_input_even_with_output`) and 1 new end-to-end
  merge test (`test_multi_input_merge_materializes_no_filename_collisions_and_loadable`)
  added; the 3 pre-existing Stage 32 tests were updated to the new
  multi-input call signatures (same assertions/behavior, no test deleted).
- `docs/stages/stage-33-multi-input-dataset-split.md` — this document
  (new).

### Verification results

1. **Parity test** (`test_multi_input_parity_with_single_manifest`) — ✅
   passes for both `stratify_classes = false` and `true`; confirms
   `three_way_spatial_split_multi(&[manifest], ...)` (after stripping the
   constant `dir_idx = 0`) produces identical train/val/test id sets to
   `three_way_spatial_split(manifest, ...)`.
2. **Pre-existing Stage 32 regression tests** — ✅ all 4 continue to pass
   unmodified: `test_non_stratified_fraction_semantics_match_2way`,
   `test_three_way_split_disjoint_and_complete`,
   `test_rejects_out_of_range_fractions`,
   `test_stratification_reduces_class_imbalance`.
3. **Disjointness/completeness with colliding local IDs**
   (`test_multi_input_disjoint_and_complete_with_colliding_ids`) — ✅ passes
   for both stratify modes; every `(dir_idx, block_id)` pair across two
   10-block manifests with fully-colliding local IDs (`0..10` in each)
   appears in exactly one of train/val/test.
4. **Combined global stratification improvement**
   (`test_multi_input_stratification_uses_combined_global_balance`) — ✅
   passes; with file A ~99% class 0 and file B ~99% class 1 (combined
   ~50/50), the stratified val-set deviation from the true combined global
   proportions is measurably lower than the non-stratified deviation.
5. **`validate_manifest_compatibility()` field-mismatch rejection**
   (`test_validate_manifest_compatibility_rejects_mismatches`) — ✅ passes;
   one assertion per field (`label_map`, `block_size`, `target_points`,
   `min_density`, `search_radius`, `min_neighbors`, `crs_epsg`), plus
   confirms a `source`-only difference is accepted and an empty manifest
   slice is rejected.
6. **End-to-end merge materialization**
   (`test_multi_input_merge_materializes_no_filename_collisions_and_loadable`)
   — ✅ passes; two synthetic input directories with colliding local block
   IDs (`0..6` each) materialize into a single merged `train/`/`val/` tree
   with zero filename collisions, exactly 12 blocks written, and the result
   loads correctly via `LabeledBlockDataset::load_presplit()` with matching
   combined counts.
7. **`--move` semantics** (`test_move_deletes_source_files_after_success`) —
   ✅ passes on the updated multi-input call signature; source files are
   only removed after both files of a block copy successfully.
8. `cargo build --all-targets --all-features` — ✅ zero errors.
9. `cargo clippy --all-targets --all-features -- -D warnings` — ✅ zero
   warnings (one `clippy::case_sensitive_file_extension_comparisons` finding
   in a new test helper was fixed during development by switching to
   `Path::extension().is_some_and(...eq_ignore_ascii_case("feat"))`).
10. `cargo clippy --all-targets --features training -- -D warnings` — ✅
    zero warnings.
11. `cargo test --all-features` — ✅ **119** lib tests + **1** integration
    test pass (up from Stage 32's 113 lib tests: +5 in `dataset_split.rs`,
    +2 in `split_dataset_cmd.rs`, `test_run_rejects_missing_input_even_with_output`
    accounting for the remaining new addition), 0 failed.
12. `cargo test --features training` — ✅ identical result: 119 lib tests +
    1 integration test pass, 0 failed.
13. `cargo fmt -- --check` — ✅ clean (one auto-fix pass via `cargo fmt` was
    needed for a handful of line-wrapping diffs in newly-added test code,
    re-verified clean afterward).

### Notes / deliberate simplifications carried forward

- `spatial_tile_grid` in a merged subset's output manifest is taken as-is
  from the first input manifest — this is documented above as an
  intentional, harmless simplification since no code in this repository
  reads that field back out of a loaded manifest.
- A block's `macro_tile_id` in the merged output manifest is retained
  unchanged from its original per-file value for debugging visibility only;
  it is not globally unique or comparable across blocks originating from
  different inputs post-merge (documented above; not enforced in code).


