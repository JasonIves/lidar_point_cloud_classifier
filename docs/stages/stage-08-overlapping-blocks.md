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
6. All 58 tests pass (`cargo test --features training`).
7. `cargo clippy --features training -- -D warnings` → zero errors. ✅
8. `cargo fmt --check` → clean. ✅
9. This stage spec file is synchronised with the implementation. ✅

### Implementation status: **COMPLETE** (2026-06-26)

Files modified:
- `src/preprocessing/mod.rs` — `block_overlap: f64` field in `PreprocessConfig`
- `src/preprocessing/block_partitioner.rs` — `BlockStub::read_points()` (non-destructive read)
- `src/preprocessing/pipeline.rs` — `BlockManifest::block_overlap`, `stubs_by_cell` map,
  sequential border-point pre-loading, augmented k-d tree in parallel closure,
  `load_border_points()` free function, 4 new unit tests
- `src/output/las_writer.rs` — `block_overlap: 0.0` in test helper `BlockManifest`
- `src/cli/preprocess_cmd.rs` — `--block-overlap` flag, validation, help text
- `src/cli/preprocess_labeled_cmd.rs` — `--block-overlap` flag, validation, help text

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
