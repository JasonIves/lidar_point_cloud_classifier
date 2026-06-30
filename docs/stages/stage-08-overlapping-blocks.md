# Stage 08 — Overlapping Block Partitioning

## Goal

Eliminate the spatial edge effect that degrades eigenvalue feature quality for
points near block boundaries.  The current contiguous grid assigns each point to
exactly one cell; points at the edge of a block have fewer spatial neighbours
within that block, producing unreliable local geometry descriptors
(linearity, planarity, sphericity, omnivariance, curvature).

This stage introduces a configurable **overlap radius** (`block_overlap`) that
augments each block's k-d tree with border points read from adjacent blocks'
already-written spill files.  The augmented context is used **only for feature
extraction** — block ownership, resampling, and all output files remain
unchanged.

---

## Inputs & Outputs

### New configuration parameter

| Parameter | Type | Default | CLI flag |
|---|---|---|---|
| `block_overlap` | `f64` (projection units) | `0.0` (disabled) | `--block-overlap <f64>` |

**Recommended value**: `block_size / 2`.  At this setting, every point within
the block has at least `block_size / 2` metres of neighbour context in all
directions, which fully covers any neighbourhood radius ≤ `block_size / 2`.

**Constraint**: `0.0 ≤ block_overlap < block_size`.  An overlap ≥ `block_size`
would load the entire dataset into every block's k-d tree.

### Manifest changes

`blocks.json` and `labeled_blocks.json` gain one new field:

```json
{
  "block_overlap": 0.0,
  ...
}
```

Both fields carry `#[serde(default)]` so existing manifests without the field
deserialise correctly (treated as `0.0`).

### `.feat` / `.lbl` file format

**Unchanged.**  The overlap border points are never written to output files.
Point counts, feature layout, and binary format are identical to Stage 01–07.

---

## Algorithm Steps

### Phase 1 — Contiguous block partitioning (unchanged)

1. Stream all points into `BlockPartitioner` via `floor()` assignment.
2. `finalize_stubs()` — flush all in-memory cells to spill files on disk.
3. Density filter on stub point counts (unchanged).

### Phase 2 — Overlap derivation (new, inserted between Phase 1 and tree build)

For each retained block stub (in the existing Rayon parallel closure):

4. Load canonical block points from its own spill files (`stub.load()`).
5. **[NEW]** If `block_overlap > 0.0`, call `load_border_points(stubs_map, stub, block_size, block_overlap)`:
   - Identify the up-to-8 neighbouring cells: `(col±1, row±1)` and the 4 cardinal neighbours.
   - For each neighbour cell that exists in `stubs_map`:
     - Read its spill file(s) directly (read-only; the neighbour owns and will delete its own files).
     - Filter to points within the expanded bounding box:
       ```
       x ∈ [origin_x - overlap, origin_x + block_size + overlap]
       y ∈ [origin_y - overlap, origin_y + block_size + overlap]
       ```
   - Return the collected border `Vec<PointRecord>`.
6. Build `BlockSpatialIndex` from **canonical + border points** (augmented k-d tree).
7. Resample — **canonical points only** (border points excluded from sampling).
8. Extract features using the augmented k-d tree (border points provide neighbourhood context).
9. Write `.feat` file — **canonical sampled points only** (unchanged).

### Phase 3 — Manifest and output (unchanged)

10. Write `blocks.json` with `block_overlap` field added.

---

## Implementation Details

### `BlockPartitioner` changes

A new method `stubs_by_cell(stubs: &[BlockStub]) -> HashMap<(i32,i32), &BlockStub>` is
added as a free function in `block_partitioner.rs` to build a lookup map from
`(col, row)` to stub reference.  This is used by the border-point loader.

A new free function `load_border_points` in `pipeline.rs` (not in
`block_partitioner.rs`, to keep the partitioner focused on partitioning):

```rust
fn load_border_points(
    stubs_by_cell: &HashMap<(i32, i32), usize>,  // cell → index into stubs slice
    all_stubs: &[BlockStub],
    target: &BlockStub,
    block_size: f64,
    overlap: f64,
) -> Result<Vec<PointRecord>>
```

This function:
- Iterates over the 8 neighbours of `(target.col, target.row)`.
- For each neighbour present in `stubs_by_cell`, reads its spill files.
- Filters points to the expanded bounding box.
- Returns the collected border points.

**Important**: The neighbour's spill files are read but **not deleted** here.
Each block stub's `load()` method deletes its own spill files.  The border
loader uses a separate read path that does not consume the stub.

### `PreprocessConfig` changes

```rust
pub struct PreprocessConfig {
    // ... existing fields ...
    /// Overlap radius in projection units added to each block's k-d tree context.
    /// Border points from adjacent blocks within this radius are included for
    /// feature extraction but are never resampled or written to .feat files.
    /// Default: 0.0 (disabled). Recommended: block_size / 2.
    pub block_overlap: f64,
}
```

Default: `0.0`.

### `BlockManifest` changes

```rust
pub struct BlockManifest {
    // ... existing fields ...
    #[serde(default)]
    pub block_overlap: f64,
}
```

### `LabeledPreprocessConfig` and `LabeledBlockManifest`

Mirror the same `block_overlap` field.  The labeled pipeline passes
`block_overlap` through to the underlying `PreprocessConfig` and records it in
`labeled_blocks.json`.

### CLI flags

Both `preprocess_cmd.rs` and `preprocess_labeled_cmd.rs` gain:

```
--block-overlap <f64>   Overlap radius in projection units for border-point
                        augmentation (default: 0.0, recommended: block_size/2).
                        Must be in [0.0, block_size).
```

---

## Definition of Done (DoD)

1. **Regression**: `--block-overlap 0.0` (default) produces **bit-identical**
   `.feat` output to the pre-Stage-08 pipeline for the same input.
2. **Augmentation**: `--block-overlap 25.0` with `--block-size 50.0` produces
   `.feat` files with the same point count but a larger k-d tree during feature
   extraction (verified by unit test counting points in the augmented index).
3. `block_overlap` is recorded in `blocks.json` (via `BlockManifest`).
   `labeled_blocks.json` inherits the value through the base pipeline; the
   labeled manifest does not duplicate the field (it is already in `blocks.json`).
4. Existing `blocks.json` files without `block_overlap` deserialise correctly
   (treated as `0.0`) — verified by `test_manifest_block_overlap_default_on_missing`.
5. Range validation rejects `overlap < 0.0`, `overlap >= block_size`, and
   non-finite values — enforced in both `preprocess_cmd.rs` and
   `preprocess_labeled_cmd.rs`.
6. All 60 tests pass (`cargo test --features training`).
7. `cargo clippy --features training -- -D warnings` → zero errors. ✅
8. `cargo fmt --check` → clean. ✅
9. This stage spec file is synchronised with the implementation. ✅

### Implementation status: **COMPLETE** (2026-06-26) — Hotfix applied (2026-06-26)

Files modified:
- `src/preprocessing/mod.rs` — `block_overlap: f64` field in `PreprocessConfig`
- `src/preprocessing/block_partitioner.rs` — `BlockStub::read_points()` (non-destructive read)
- `src/preprocessing/pipeline.rs` — `BlockManifest::block_overlap`, `stubs_by_cell` map,
  sequential border-point pre-loading, augmented k-d tree in parallel closure,
  `load_border_points()` free function, 4 new unit tests;
  **hotfix**: eliminated redundant `block.points.clone()` in the overlap branch
  (see "Memory Hotfix" section below)
- `src/output/las_writer.rs` — `block_overlap: 0.0` in test helper `BlockManifest`
- `src/cli/preprocess_cmd.rs` — `--block-overlap` flag, validation, help text
- `src/cli/preprocess_labeled_cmd.rs` — `--block-overlap` flag, validation, help text

---

## Memory Hotfix — Border-Point Accumulation OOM (2026-06-26, revised)

### Symptom

Running `preprocess-labeled` with `--block-overlap` enabled and a small
`--block-size` (e.g. `5.0`) caused an immediate OOM crash:

```
memory allocation of 44040192 bytes failed
error: process didn't exit successfully: STATUS_STACK_BUFFER_OVERRUN
```

The `STATUS_STACK_BUFFER_OVERRUN` (Windows exit code `0xc0000409`) is a
secondary symptom: the allocator panic unwinds through deeply nested Rayon
closures, overflowing the thread stack.  The root cause is the heap allocation
failure.  The crash persisted even after the initial double-clone fix (see
below), because the dominant memory consumer was elsewhere.

### Root Cause — Primary: `border_points_per_block` global accumulation

The original Step 5c built a `Vec<Vec<PointRecord>>` holding **all border
points for all blocks simultaneously** before the parallel phase began:

```rust
// WRONG — O(n_blocks × border_density) RAM before any block is processed
let border_points_per_block: Vec<Vec<PointRecord>> = retained.iter()
    .map(|stub| load_border_points(...).unwrap_or_default())
    .collect();
```

With `--block-size 5.0` on a real LiDAR file, there can be thousands of
blocks.  Each block loads border points from up to 8 neighbours.  At typical
LiDAR density (10 pts/m²), a 5 m block has ~250 canonical points; each of 8
neighbours contributes ~250 border points = 2 000 border points per block.
With 1 000 blocks × 2 000 border points × ~80 bytes/`PointRecord` = **160 MB**
accumulated before a single block is processed — and this scales with dataset
size, not thread count.

### Root Cause — Secondary: double-clone in parallel closure

The parallel closure also cloned `block.points` twice when `border_pts` was
non-empty (once for the k-d tree, once for the feature-extraction context),
adding another `2 × (N + B)` allocation per active thread.

### Fix — Border-point spill to disk

Border points are now written to per-block `.border` spill files (same 31-byte
binary layout as `.spill` files) during the sequential phase.  Only the file
path is kept in memory.  The parallel closure loads and immediately deletes the
`.border` file for one block at a time:

```
Sequential phase (Step 5c):
  for each block:
    border_pts = load_border_points(...)   // transient Vec — one block at a time
    write_border_spill("block_XXXXX.border", &border_pts)
    paths.push(Some(path))                 // only the path stays in RAM
    // border_pts dropped here

Parallel phase (Step 7):
  for each (stub, border_path) in parallel:
    block = stub.load()                    // canonical points
    border_pts = read_border_spill(path)   // load border from disk
    fs::remove_file(path)                  // delete immediately
    ctx = canonical + border               // single merged Vec
    drop(border_pts)                       // free before k-d tree build
    index = BlockSpatialIndex::build(&ctx)
    features = extract_features(&sampled, &ctx, &index, ...)
    // ctx, block.points dropped at end of closure
```

**Memory bound (overlap enabled):**

| Phase | Peak RAM |
|---|---|
| Sequential border spill | `O(1 block × border_strip)` — one block at a time |
| Parallel processing | `threads × (canonical + border_strip)` |

This is hardware-independent and scales correctly regardless of dataset size,
block size, or thread count.

### New files / functions

| Symbol | Location | Purpose |
|---|---|---|
| `BORDER_PT_BYTES` | `pipeline.rs` | 31 bytes/point constant (matches spill format) |
| `write_border_spill()` | `pipeline.rs` | Write `&[PointRecord]` to `.border` file |
| `read_border_spill()` | `pipeline.rs` | Read `.border` file → `Vec<PointRecord>` |
| `border_spill_paths` | `pipeline.rs` Step 5c | `Vec<Option<PathBuf>>` replacing `Vec<Vec<PointRecord>>` |

Two new unit tests cover the border spill I/O:
- `test_border_spill_round_trip` — 20-point write/read with field verification
- `test_border_spill_empty` — empty slice round-trips to empty Vec

### Confounding Factor

The test command also used `--block-size 5.0` (vs. the typical `50.0`), which
produces ~100× more blocks and causes each block to be oversampled to
`--target-points 4096` (far above the raw point count in a 5 m cell).  This
amplifies per-block allocation and makes the accumulation bug visible at thread
counts that would be safe with normal block sizes.  The fix is correct
regardless of block size.

---

## Relationship to Prior Stages

- **Stage 01** (spatial preprocessing): Block partitioning logic in
  `block_partitioner.rs` is **not modified**.  The overlap derivation is a
  separate read-only phase in `pipeline.rs`.
- **Stage 03** (labeled pipeline): `labeled_pipeline.rs` mirrors the overlap
  logic; label assignment is unaffected because labels come from canonical
  points only.
- **Stage 06** (multi-scale features): The augmented k-d tree is passed to
  `extract_features` unchanged; multi-scale radii benefit from the larger
  neighbourhood context automatically.
