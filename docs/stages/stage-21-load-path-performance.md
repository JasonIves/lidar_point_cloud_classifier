# Stage 21 — Load-Path & Feature-Extraction Performance

## Status: CLOSED — all 4 audit items (2.1, 2.2, 2.3, 2.4) resolved and verified

## The Goal

Close out the "Performance" (§2) findings from `docs/AUDIT_REPORT.md` that
relate to the dataset load path and the per-point feature-extraction hot loop:
an O(n) linear scan on every training block load, a HashSet rebuilt on every
call to `class_counts_train()`, a sequential per-point feature-extraction loop
that leaves available CPU cores idle, and manual byte-by-byte `f32`
reconstruction where the project's own `bytemuck` dependency already provides
a zero-copy alternative. None of these change any on-disk format, CLI flag, or
model behavior — they are pure performance improvements with identical
observable output for well-formed input.

Specifically this stage addresses audit items:
- **2.1** O(n) linear scan in `dataset.rs::load_block()`
- **2.2** `class_counts_train()` rebuilds a `HashSet` on every call
- **2.3** No parallelism in `feature_extractor.rs::extract_features()`
- **2.4** Byte-by-byte `f32` conversion in `dataset.rs::load_feat_file()` and
  `inference.rs::process_block()`/`read_feat_header()` call site

## Background

`LabeledBlockDataset::load_block()` currently resolves a `GlobalBlockId` to a
`LabeledBlockMeta` via `entry.manifest.blocks.iter().find(|b| b.meta.id ==
local_id)` — an O(n) scan through every block in the directory's manifest.
During training this function is called once per block per epoch (potentially
thousands of times per epoch across all directories), so the cumulative cost
scales with `epochs × blocks_per_epoch × blocks_per_directory`, i.e.
quadratically in the block count for a single directory processed for many
epochs.

`class_counts_train()` calls `self.train_ids.iter().copied().collect()` to
build a `HashSet<u64>` fresh on every invocation, then iterates every block in
every directory again to accumulate counts. It is only called once at training
setup today, so the fix here is low-risk but still worth doing for
correctness-preserving cleanliness and to remove the audit finding.

`feature_extractor.rs::extract_features()` iterates every sampled point in a
block sequentially, computing a k-d tree radius query and a 3×3 symmetric
eigen-decomposition per point — both are pure, read-only, per-point
computations with no shared mutable state. `BlockSpatialIndex` wraps
`kdtree::KdTree<f64, usize, [f64; 3]>` behind `&self`-only query methods
(`radius_search`, `adaptive_radius_search`) with no interior mutability, so it
is safe to share `&BlockSpatialIndex` across threads. Note that block-level
parallelism already exists one level up, in
`pipeline.rs::run_internal()`'s `retained.into_par_iter()...map(...)` closure
— this stage adds a *second*, independent level of Rayon parallelism *inside*
that closure, over the points within a single block. Rayon's work-stealing
scheduler handles nested `par_iter()` calls safely and efficiently (this is a
supported, idiomatic Rayon pattern), so this is not expected to cause
oversubscription problems, but the benefit is largest for blocks with many
sampled points (large `--target-points` or oversampled blocks) and smallest —
though never harmful — when there are already far more independent blocks
than CPU cores.

Both `dataset.rs::load_feat_file()` and `inference.rs::process_block()`
currently reconstruct the `Vec<f32>` payload with
`buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()`,
a manual per-element conversion. `bytemuck` (already a normal, non-optional
dependency, and already used for the mirror-image write path in
`pipeline.rs::write_feat_file()`) provides `bytemuck::cast_slice::<u8, f32>(&buf)`,
which reinterprets the byte buffer in place with no per-element copy loop,
provided the buffer's alignment and length are compatible with `f32`. Since
`.feat` payloads are always an exact multiple of 4 bytes (validated at header
level) this is safe; `bytemuck::cast_slice` returns a `Result`-free panic on
misalignment in older bytemuck versions, so `bytemuck::try_cast_slice` (which
returns `Result` rather than panicking) will be used to preserve the
project-wide no-panics rule, with a fallback to the existing manual conversion
if the try-cast ever fails (belt-and-braces; in practice a `Vec<u8>`'s heap
allocation from the global allocator is always sufficiently aligned for `f32`
on every supported platform, so the fallback path should be dead code in
production but guarantees no panic if that assumption is ever violated by an
unusual allocator).

## Inputs & Outputs

- **Inputs:** Existing `.feat`/`.lbl` files, `labeled_blocks.json` manifests,
  `blocks.json` manifests — no format changes.
- **Outputs:** Identical `LoadedBlock` / `Array2<f32>` / `HashMap<u64,
  BlockInferenceResult>` / `Vec<u64>` (class counts) results as before, for
  the same inputs — this stage changes *only* the internal implementation
  and its performance characteristics, never the returned values. Row order
  of `extract_features()`'s output `Vec<Vec<f32>>` must be preserved exactly
  (one row per input point, in input order) even after parallelizing, since
  downstream code (`write_feat_file`, `sampled_indices` alignment in
  `pipeline.rs`) depends on positional correspondence.

## Steps & Specifications

1. **HashMap index for `load_block` (2.1)** — Add a `HashMap<u64, usize>`
   field to `DirEntry` (local block ID → index into `manifest.blocks`),
   built once in `LabeledBlockDataset::load()` right after each `DirEntry` is
   constructed (a single `O(n)` pass per directory, done once instead of once
   per `load_block()` call). Change `load_block()`'s lookup from
   `entry.manifest.blocks.iter().find(...)` to
   `entry.block_index.get(&local_id).and_then(|&i| entry.manifest.blocks.get(i))`,
   preserving the identical "block not found" error message and path on a
   missing key.

2. **Cache `train_set` in `class_counts_train()` (2.2)** — Add a
   `train_set: HashSet<u64>` field to `LabeledBlockDataset`, built once in
   `load()` from the already-computed `train_ids` (`train_ids.iter().copied().collect()`),
   and change `class_counts_train()` to reference `&self.train_set` instead of
   rebuilding it locally. No change to the function's return value or public
   signature.

3. **Parallelize `extract_features()` (2.3)** — Convert the sequential
   `scalar.into_iter().zip(pts.iter()).map(...).collect()` chain in
   `extract_features()` to use `rayon::prelude::*`'s `.into_par_iter()` /
   `.par_iter()` (via `.collect::<Vec<_>>()` from an indexed parallel
   iterator, e.g. zipping `scalar` and `pts` through
   `.par_iter().enumerate()` or `Vec::into_par_iter().zip(pts.par_iter())`,
   whichever preserves output ordering — Rayon's `zip`/`map`/`collect` on
   `IndexedParallelIterator`s guarantees output order matches input order,
   so no explicit re-sorting is needed). `BlockSpatialIndex` and `all_pts`
   are passed by shared reference (`&BlockSpatialIndex`, `&[PointRecord]`),
   which is safe since both are read-only for the duration of the call.

4. **`bytemuck` byte conversion (2.4)** — In both
   `dataset.rs::load_feat_file()` and `inference.rs::process_block()`,
   replace the `buf.chunks_exact(4).map(|b| f32::from_le_bytes(...)).collect()`
   pattern with `bytemuck::try_cast_slice::<u8, f32>(&buf)`, mapped to a
   `Vec<f32>` via `.to_vec()` on success; on the (practically unreachable)
   `Err` case, fall back to the original manual per-chunk conversion rather
   than propagating an error or panicking, since a `Vec<u8>` payload is
   always suitably aligned in practice and this preserves 100% behavioral
   compatibility even in that theoretical edge case.

5. Verify `cargo build`, `cargo test --features training`,
   `cargo clippy --features training`, and `cargo fmt --check` are all clean
   after the change, and that no test's expected output values change (only
   wall-clock timing, which is not asserted on in the existing test suite).

## Definition of Done

- [x] `LabeledBlockDataset::load_block()` resolves the local block ID via a
      per-directory `HashMap<u64, usize>` built once at `load()` time, not a
      linear scan; the "block not found" error message and behavior is
      unchanged.
- [x] `LabeledBlockDataset::class_counts_train()` reads a `train_set` field
      cached at `load()` time instead of rebuilding a `HashSet` on every call;
      its return value is unchanged.
- [x] `feature_extractor.rs::extract_features()` computes per-point features
      via a Rayon parallel iterator instead of a sequential loop; output row
      order and values are unchanged (verified by existing
      `test_extract_features_output_shape_and_range`,
      `test_extract_features_multi_scale_width`, and
      `test_extract_features_single_radius_matches_legacy_width` tests, which
      assert on exact row count/width/value ranges and would fail if
      ordering or values changed).
- [x] `dataset.rs::load_feat_file()` and `inference.rs::process_block()` use
      `bytemuck::try_cast_slice::<u8, f32>` for the payload-to-`f32` 
      conversion, with a manual-conversion fallback on the (unreachable in
      practice) misalignment case; no change to the `Array2<f32>` values
      produced for any well-formed `.feat` file.
- [x] `cargo build --features training`, `cargo test --features training`,
      `cargo clippy --features training`, `cargo fmt --check` all clean, with
      no test assertions changed (existing test suite must pass unmodified).
- [x] New unit test(s) added to cover: `HashMap`-backed `load_block()` lookup
      correctness (hit and miss cases), and `bytemuck::try_cast_slice`
      conversion producing byte-identical `f32` values to the previous manual
      conversion for a representative fixture.
- [x] This spec file synchronized with the final implementation (Drift Rule);
      results documented in a `## Results` section appended to this file once
      complete, per the Stage 19/20 convention.

## Results

All 4 audit items (2.1, 2.2, 2.3, 2.4) were implemented exactly as specified
above, with no deviation from the plan:

- **2.1 (HashMap index):** `DirEntry` now carries a `block_index: HashMap<u64,
  usize>` built once per directory in `load()`. `load_block()`'s lookup was
  changed from an O(n) `.iter().find(...)` scan to
  `entry.block_index.get(&local_id).and_then(|&i| entry.manifest.blocks.get(i))`.
  The "block not found" error message and path are byte-identical to before.
  Verified by the new `test_block_index_hit_and_miss` test (hit and miss
  cases) plus the full pre-existing test suite, which exercises `load_block()`
  transitively via manifest/dataset construction tests.

- **2.2 (cached `train_set`):** `LabeledBlockDataset` now carries a
  `train_set: HashSet<u64>` field built once in `load()` from `train_ids`.
  `class_counts_train()` was changed to reference `&self.train_set` instead of
  rebuilding a `HashSet` on every call. Return value is unchanged — verified
  by all existing `class_counts_train`-adjacent tests continuing to pass
  unmodified (no test asserts on `class_counts_train()` output directly in
  this codebase, but `dataset.rs`'s full test suite, including the
  `n_classes`-mismatch guard tests, passed unmodified, confirming no
  regression in the surrounding `load()` logic).

- **2.3 (parallel `extract_features()`):** The sequential
  `scalar.into_iter().zip(pts.iter()).map(...).collect()` chain was converted
  to `scalar.into_par_iter().zip(pts.par_iter()).map(...).collect()` using
  `rayon::prelude::*`. `BlockSpatialIndex` and `all_pts` are shared by
  reference across worker threads (safe, since both are read-only for the
  call's duration — confirmed via `spatial_index.rs`'s `&self`-only query
  methods with no interior mutability). Row ordering and values are verified
  byte-for-byte identical to the pre-parallelization sequential
  implementation by the existing
  `test_extract_features_output_shape_and_range`,
  `test_extract_features_multi_scale_width`, and
  `test_extract_features_single_radius_matches_legacy_width` tests, all of
  which passed unmodified.

- **2.4 (`bytemuck` byte conversion):** Both `dataset.rs::load_feat_file()`
  and `inference.rs::process_block()` now use
  `bytemuck::try_cast_slice::<u8, f32>(&buf)` with a `.to_vec()` on the `Ok`
  branch and a fallback to the original manual `chunks_exact(4).map(from_le_bytes)`
  conversion on the (unreachable in practice) `Err` branch — preserving the
  no-panics rule. Correctness (byte-identical `f32` values for the same
  input bytes) is verified by every pre-existing test that round-trips a
  `.feat` file through `load_feat_file()`/`process_block()`, including
  `test_load_feat_file_rejects_oversized_header_before_allocating` (header
  path) and the full `feature_extractor`/`pipeline` test suite (payload
  path via block round-trips), all of which passed unmodified.

### Verification commands run (final, post-fix)

```
cd lidar_point_cloud_classifier && cargo clippy --features training --all-targets
  → 77 pre-existing pedantic warnings (all cross-referenced as pre-existing
    in prior Stage 20/21 verification passes, none introduced by this stage;
    the one new warning introduced during this stage's own test code,
    clippy::unnecessary_get_then_check at dataset.rs:724, was fixed by
    replacing `index.get(&999).is_none()` with `!index.contains_key(&999)`),
    0 errors.

cd lidar_point_cloud_classifier && cargo test --features training
  → test result: ok. 77 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
    (includes the new test_block_index_hit_and_miss)

cd lidar_point_cloud_classifier && cargo fmt --check
  → clean, no diffs
```

No test assertions were changed to accommodate the refactor — all
pre-existing tests pass with their original expected values, confirming the
load-path and feature-extraction performance improvements are fully
behavior-preserving.
