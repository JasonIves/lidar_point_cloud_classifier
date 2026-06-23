# Stage 05 — Multi-Directory Training Dataset

**Status:** COMPLETE — 2026-06-23
**Approved:** 2026-06-23
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier

---

## Goal

Allow the `train` sub-command to consume blocks preprocessed from **multiple LiDAR
source files** by accepting repeated `--data-dir` flags, each pointing to a distinct
`preprocess-labeled` output directory.  Each directory is self-contained (its own
`labeled_blocks.json`, `.feat`, and `.lbl` files) and can be added or removed without
touching any existing directory.

---

## Problem Statement

The original pipeline was strictly one-file-in → one manifest-out → one training run.
Block IDs are assigned per-file as `row * grid_cols + col` relative to each file's own
bounding box, so two files processed separately produce colliding IDs and file names
(`block_00000.feat` from file A and from file B are different blocks).

A naive "just point at multiple directories" approach therefore breaks `load_block`
because the same numeric ID can exist in multiple directories.

---

## Design Decisions

### D1 — Per-directory output, in-memory merge (chosen)

Each source file is preprocessed independently to its own output directory.
The **dataset loader** merges all directories into a single virtual dataset at training
time.  Nothing is moved, renamed, or copied on disk.

**Rejected alternatives:**
- *Upstream merge via `lidar_join`*: Creates large intermediate files; no longer needed.
- *Batch preprocessing with globally-unique IDs*: Requires changing the on-disk format
  and all consumers of block IDs.  Heavier scope, harder to roll back.

### D2 — Composite GlobalBlockId (u64)

Block IDs remain `u64` at all API boundaries, so `trainer.rs` requires no changes.
A composite key encodes the directory index in the high 32 bits and the file-local
block ID in the low 32 bits:

```
GlobalBlockId = (dir_index as u64) << 32  |  local_block_id
```

**Capacity**: low 32 bits support up to 4 billion local IDs per directory — sufficient
for any realistic LiDAR acquisition area at any block size.  High 32 bits support up
to 4 billion directories.  `decode_global_id` is the single point of truth for
unpacking.

**Why not a `(usize, u64)` tuple?**: Would require changing `Vec<u64>` in the trainer
loop and all downstream code.  The composite u64 is a zero-cost encoding that keeps
the API surface unchanged.

### D3 — Per-directory independent spatial split

Each directory receives an independent `spatial_split` call, and the results are
concatenated.  This preserves the spatial-disjointness guarantee: validation blocks
within each file come from geographically separated macro-tiles.  A cross-directory
split (treating all blocks as one pool) was rejected because it would require a
globally consistent tile grid across acquisitions, which is impossible in general.

### D4 — Class consistency validation at load time

All directories must produce the same `n_classes` (derived from their respective
`label_map` fields).  A mismatch is a hard error at dataset load time with a clear
message identifying which directories disagree.  This prevents silent label index
mismatches from corrupting training.

---

## Changed Files

| File | Nature of change |
|---|---|
| `src/training/dataset.rs` | Core refactor: `DirEntry` struct, composite key helpers, multi-manifest load, updated `load_block` / `class_counts_train` / `n_classes` |
| `src/cli/train_cmd.rs` | `--data-dir` becomes repeatable; `data_dirs: Vec<PathBuf>` replaces `data_dir: Option<PathBuf>`; metrics anchor uses `data_dirs[0]` |

`trainer.rs` and all preprocessing files are **unchanged**.

---

## CLI Interface

Single directory (backward-compatible, unchanged behavior):
```
wb_lidar_train train --data-dir ./data/labeled/mar19_manifest ...
```

Multiple directories:
```
wb_lidar_train train \
  --data-dir ./data/labeled/mar19_manifest \
  --data-dir ./data/labeled/apr05_manifest \
  ...
```

---

## Constraints and Known Limitations

1. **All directories must share the same `n_classes`**.  If a new acquisition is
   preprocessed with a different `--label-map`, re-preprocess all datasets with a
   common map before combining.

2. **The `--val-tile-blocks` explicit override** supplies raw block IDs without a
   directory prefix.  When using multiple directories, this flag is ambiguous and
   should not be combined with multi-directory mode.  A warning is emitted.

3. **Metrics output** defaults to `data_dirs[0].parent()/metrics/metrics.csv`.

---

## Definition of Done

1. `cargo build --features training` ✓
2. `cargo clippy -- -D warnings` ✓
3. `cargo test --features training` — all existing tests + 2 new key-encoding tests ✓
4. Single `--data-dir` produces identical results to prior behavior ✓
5. Two `--data-dir` flags train successfully and report combined block count ✓
