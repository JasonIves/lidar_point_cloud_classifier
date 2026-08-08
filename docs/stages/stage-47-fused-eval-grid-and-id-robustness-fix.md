# Stage 47 — Fused-Eval Grid-Geometry & Block-Id Robustness Fix

**Status:** Implemented
**Date:** 2026-08-07
**Depends on:** Stage 44 (classify-time prediction fusion / `evaluate --fused-eval`),
Stage 32 (dataset-split materialization), Stage 33 (multi-input dataset split)
**Supersedes (partially):** Stage 44 §"Algorithm Steps" step 11 (grid derivation) and
`derive_grid_and_radius`'s original bounding-box-from-retained-blocks approach; see the
cross-reference note added to `stage-44-classify-time-prediction-fusion.md`'s
"Implementation Status & Deviations" section.

---

## The Goal

Fix a production bug reported after a period of live testing use: **`evaluate
--fused-eval` metrics come through as all-zero** (per-class TP/FP/FN/mIoU/accuracy all
`0`, i.e. `MetricsAccumulator`'s `total_points == 0` fallback), while the non-fused
`evaluate` path on the same held-out data works correctly. Diagnosis surfaced **two
independent, compounding root causes**, both fixed by this stage.

### Root Cause #1 — grid geometry re-derived from *retained* block origins

`LabeledBlockManifest` (unlike `BlockManifest`) carried no persisted `grid_cols` /
`grid_rows` / `grid_x_min` / `grid_y_min`. Stage 44's `derive_grid_and_radius` therefore
re-derived the grid's bounds and width purely from the min/max origin of whichever
blocks happened to survive `preprocess-labeled`'s min-density filter. Whenever an edge
column/row is dropped (a common, expected outcome of density filtering near a tile's
physical boundary), the re-derived `grid_cols`/`x_min` disagree with the TRUE grid that
was used to assign each retained block's `meta.id = row * true_grid_cols + col` at
preprocessing time. `build_vote_structures`'s pre-fix map (keyed directly by `meta.id`)
and the query-time lookup in `fused_label` (keyed by `block_id(row, col,
re-derived_grid_cols)` recomputed from the query point's own coordinates) then disagree
— including for a block's **own self-vote** — so *no* vote is ever found for *any*
point, and every metric collapses to the `total_points == 0` zero-fallback.

### Root Cause #2 — `split-dataset` unconditionally renumbers every block id

Independently of density filtering, `split_dataset_cmd.rs::materialize_split()`
unconditionally assigns every output block a **fresh sequential `id`** when writing
each `train/val/test` subset — on **every** `split-dataset` invocation, not only when
merging multiple `--input` sources. This breaks the `meta.id == row * grid_cols + col`
invariant that the pre-fix map-keying (`map.insert(gid, result)` keyed by raw
`meta.id`) depended on, **unconditionally** — so any `evaluate --fused-eval` run
against a `split-dataset` output (the normal, expected workflow for held-out test
evaluation) was broken independent of Root Cause #1.

Both defects manifest identically to the user: fused-eval metrics are always zero.

---

## Inputs & Outputs

No CLI surface changes. This stage is a pure correctness fix; flags, defaults, and
output formats introduced by Stage 44 are unchanged.

### Data format change: `LabeledBlockManifest` gains 4 fields

```rust
pub struct LabeledBlockManifest {
    // ...existing fields unchanged...
    #[serde(default)]
    pub grid_cols: u32,
    #[serde(default)]
    pub grid_rows: u32,
    #[serde(default)]
    pub grid_x_min: f64,
    #[serde(default)]
    pub grid_y_min: f64,
    pub blocks: Vec<LabeledBlockMeta>,
}
```

`#[serde(default)]` makes this backward-compatible: a pre-Stage-47
`labeled_blocks.json` on disk deserializes with all four fields zeroed, which is the
same sentinel `split-dataset` now writes for a spatially-incoherent multi-input merge
(see below) — both cases are rejected identically and explicitly at `evaluate
--fused-eval` time rather than silently producing wrong grid geometry.

---

## Steps & Specifications

### Part 1 — Persist true grid geometry in `LabeledBlockManifest`

`src/preprocessing/labeled_pipeline.rs`: `run_labeled_pipeline` already computes an
authoritative `grid_cols`/`grid_rows`/`grid_x_min`/`grid_y_min` on the *unfiltered*
`base_manifest` (the same geometry `BlockManifest` — used by plain
`preprocess`/`classify` — has always carried) before the min-density filter drops any
blocks. This stage copies those four values straight into the emitted
`LabeledBlockManifest`, so they describe the TRUE grid regardless of which blocks were
subsequently retained.

### Part 2 — Key the fusion vote map by spatially-derived `block_id`, not `meta.id`

`src/cli/evaluate_cmd.rs`:

- `derive_grid_and_radius(dataset, fusion_radius)` (signature simplified — no longer
  takes `all_ids`) now calls a new `LabeledBlockDataset::manifest_grid()` accessor and
  builds `GridGeometry` directly from the manifest's persisted fields. It **never**
  inspects the set of retained blocks to determine grid extent.
- `build_vote_structures` computes, for each block, its own `(row, col)` from its
  **own persisted spatial origin** (`sm.origin_x`, `sm.origin_y`) against the
  authoritative `GridGeometry`, and inserts into the vote map under
  `block_id(row, col, grid.grid_cols)` — **not** under the block's raw `meta.id`. This
  key is, by construction, identical to whatever key `fused_label`'s own lookup
  computes for a query point that falls inside that same block (both sides now derive
  the key from the same authoritative grid + the point's/block's own coordinates), so
  the map is self-consistent regardless of what `meta.id` happens to contain upstream.

### Part 3 — Propagate grid geometry through `split-dataset` (single-input case)

`src/cli/split_dataset_cmd.rs::write_subset()`: when exactly one `--input` manifest
feeds the split (`manifests.len() == 1`), the subset's `grid_cols`/`grid_rows`/
`grid_x_min`/`grid_y_min` are copied straight from that single source manifest. A
single input has exactly one coherent grid; `materialize_split`'s unconditional
`id`-renumbering (Stage 32) no longer matters for fused-eval correctness because Part 2
never trusts `meta.id` for map-keying — only the block's own origin and the persisted
grid geometry matter, and both survive the split untouched.

### Part 4 — Reject (via natural error) multi-input-merged splits

When a split merges blocks from **more than one** `--input` source (Stage 33), each
source file has its own independent, unrelated grid — there is no single coherent grid
to propagate. `write_subset()` zeroes all four fields (`0, 0, 0.0, 0.0`) in this case —
the same sentinel used for a pre-Stage-47 manifest. `LabeledBlockDataset::manifest_grid()`
explicitly checks for `grid_cols == 0 || grid_rows == 0` and returns a descriptive
`ClassifierError::Pipeline`:

```
labeled_blocks.json is missing grid_cols/grid_rows — required for --fused-eval. This
means either the manifest predates Stage 47 (re-run preprocess-labeled), or it is a
split-dataset output that merged blocks from multiple distinct source files, which have
no single coherent grid (re-run split-dataset with a single --input, or evaluate without
--fused-eval).
```

`evaluate --fused-eval` against such a manifest now fails fast with this message
instead of silently deriving a meaningless grid (Root Cause #1's original failure mode)
or silently succeeding with wrong geometry.

### New struct: `ManifestGridMeta` (`src/training/dataset.rs`)

```rust
#[derive(Debug, Clone, Copy)]
pub struct ManifestGridMeta {
    pub x_min: f64,
    pub y_min: f64,
    pub block_size: f64,
    pub grid_cols: u32,
    pub grid_rows: u32,
}
```

Deliberately a plain, independent struct rather than reusing `model::fusion::GridGeometry`
directly, to preserve the existing module dependency direction (`training` does not
depend on `model`). `evaluate_cmd.rs` converts it into a `GridGeometry` at the one call
site that needs it.

---

## Module Touch List

| File | Change |
|---|---|
| `src/preprocessing/labeled_pipeline.rs` | `LabeledBlockManifest` gains 4 grid fields; `run_labeled_pipeline` populates them from `base_manifest` |
| `src/training/dataset.rs` | New `ManifestGridMeta` + `LabeledBlockDataset::manifest_grid()` accessor (errors on zeroed/missing grid) |
| `src/cli/evaluate_cmd.rs` | `derive_grid_and_radius` rewritten to use `manifest_grid()` (no more bounding-box re-derivation); `build_vote_structures` re-keys the vote map by spatially-derived `block_id` instead of raw `meta.id` |
| `src/cli/split_dataset_cmd.rs` | `write_subset()` propagates grid fields for single-input splits; zeroes them for multi-input merges, via two extracted helpers (`subset_grid_geometry`, `join_manifest_sources`) kept `write_subset` under clippy's `too_many_lines`/`similar_names` thresholds |
| `src/preprocessing/dataset_split.rs` | Test-fixture `dummy_manifest` updated with the 4 new required fields |
| `src/cli/fix_label_map_cmd.rs` | Test-fixture manifest literal updated with the 4 new required fields |
| `tests/training_integration.rs` | Test-fixture manifest literal updated with the 4 new required fields |

No changes to: `.feat`/`.lbl` binary formats, `model::fusion` (the shared
candidacy/weight/accumulate routine itself is correct and untouched — only how its
input map is populated/queried at the two call sites changes), `classify`'s fusion path
(uses `BlockManifest`/`proc.meta.id` directly, which was never renumbered — unaffected
by either root cause), or `whitebox_next_gen`.

---

## Definition of Done

- [x] `LabeledBlockManifest` gains `grid_cols`/`grid_rows`/`grid_x_min`/`grid_y_min`
      with `#[serde(default)]` (backward-compatible deserialization of pre-Stage-47
      manifests).
- [x] `run_labeled_pipeline` populates the 4 fields from the pre-density-filter
      `base_manifest` (`test_labeled_manifest_fields`).
- [x] `LabeledBlockDataset::manifest_grid()` returns the persisted grid, or a
      descriptive `Err` when `grid_cols == 0 || grid_rows == 0`.
- [x] `derive_grid_and_radius` no longer inspects `all_ids`/retained block bounds at
      all — grid geometry comes solely from `manifest_grid()`.
- [x] `build_vote_structures` keys the vote map by `block_id` derived from each
      block's own persisted spatial origin against the authoritative grid — never by
      raw `meta.id`.
- [x] **Regression test — Root Cause #1** (`test_fused_eval_survives_density_dropped_edge_column`):
      a manifest with a true 3-column grid where column 0 is absent (density-dropped)
      and the two retained blocks' `meta.id` values follow the TRUE 3-wide grid
      (`1`, `2`, not a re-derived 2-wide `0`, `1`) — fused evaluation succeeds with
      `n_points == 6` and correct per-class accuracy (previously: `n_points == 0`,
      all-zero metrics).
- [x] **Regression test — Root Cause #2** (`test_fused_eval_survives_arbitrary_non_canonical_block_ids`):
      two blocks with correct spatial origins but arbitrary, non-canonical `meta.id`
      values (`77`, `13` — simulating post-`split-dataset` renumbering) — fused
      evaluation produces results identical to the canonical-id fixture
      (`test_fused_eval_two_blocks_band_split`), proving `meta.id` content has zero
      effect on correctness.
- [x] `split-dataset` single-`--input` splits propagate `grid_cols`/`grid_rows`/
      `grid_x_min`/`grid_y_min` unchanged (`test_single_input_split_propagates_grid_geometry_unchanged`).
- [x] `split-dataset` multi-`--input` merges zero all 4 grid fields on every output
      subset (`test_multi_input_merge_zeroes_grid_geometry`), and the resulting
      manifest is confirmed to make `LabeledBlockDataset::manifest_grid()` return
      `Err` when loaded.
- [x] All pre-existing tests pass unchanged after the fixture updates required by the
      new mandatory manifest fields (`cargo test --features training`).
- [x] `cargo clippy --all-targets --features training -- -D warnings` → zero warnings.
- [x] `cargo fmt --check` → clean.
- [x] This spec file is synchronized with the implementation (AGENTS.md
      living-synchronization contract); cross-reference added to Stage 44's
      "Implementation Status & Deviations" section.

---

## Verification Log

- `cargo test --features training` — 238 tests pass (237 unit + 1 integration),
  including the 2 new `evaluate_cmd.rs` root-cause regression tests and the 2 new
  `split_dataset_cmd.rs` grid-propagation/rejection regression tests.
- A pre-existing, unrelated merge-conflict artifact (`=======placeholder=======`)
  found in `labeled_pipeline.rs` during this work, which broke `cargo build`
  independent of this stage's changes, was also fixed as part of restoring a clean
  build baseline.
- `cargo clippy --all-targets --features training -- -D warnings` initially flagged
  two lints against `write_subset()`'s grid-geometry block introduced by Part 3/4:
  `clippy::similar_names` (the `grid_cols`/`grid_rows`/`grid_x_min`/`grid_y_min`
  tuple-destructuring binding names) and `clippy::too_many_lines` (112/100 lines).
  Fixed by extracting a `SubsetGridGeometry { cols, rows, x_min, y_min }` struct plus
  two helper functions, `subset_grid_geometry(manifests)` and
  `join_manifest_sources(manifests)`, called from `write_subset` — field access
  (`grid.cols`, etc.) instead of a same-scope tuple destructuring resolves the
  similar-names lint, and the extraction brings `write_subset` comfortably under the
  100-line threshold. Re-verified clean after this refactor: zero clippy warnings,
  `cargo fmt` idempotent (no diff), 238/238 tests passing.

## Alternatives Considered (and rejected)

| Alternative | Rejection rationale |
|---|---|
| Re-derive grid geometry more robustly (e.g. GCD of retained origins) instead of persisting it | Still fundamentally guesses at information that was already known and computed once at preprocessing time; any guess can be defeated by a sufficiently sparse retained set (e.g. only 2 non-adjacent blocks survive density filtering). Persisting the true value is strictly simpler and exactly correct. |
| Stop `split-dataset` from renumbering block ids at all | Rejected: the renumbering serves a real purpose (Stage 32's global uniqueness contract across merged multi-input subsets) and is depended upon elsewhere (`dataset.rs`'s per-directory local-id contiguity assumptions). Making the *fusion* mechanism not depend on `meta.id` for correctness is strictly less invasive than re-architecting id assignment. |
| Silently fall back to unfused per-block argmax when grid geometry is missing/incoherent | Rejected: silently downgrading `--fused-eval` to non-fused behavior without any indication would reintroduce exactly the kind of silent-wrongness this stage exists to eliminate. An explicit, actionable error is preferred. |
