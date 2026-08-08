# Stage 44 — Classify-Time Prediction Fusion (Block-Overlap Soft Voting)

**Status:** COMPLETE — implementation landed 2026-07-27; spec synchronized
(see "Implementation Status & Deviations" at the end)
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `docs/stages/stage-42-absolute-z-normalization.md` (§"Deferred /
Related Findings" — items 1 and 2, which this stage begins to address),
`docs/stages/stage-08-overlapping-blocks.md` (precedent for `block_overlap`
semantics and regression contracts),
`docs/stages/stage-45-fixed-n-halo-split.md` (follow-up that supplies *direct*
halo predictions to the fusion mechanism defined here)

---

## Goal

Eliminate (or sharply reduce) the "patchwork quilt" / "grid" classification
artifact — block-shaped label discontinuities at block boundaries, where one
physical structure (e.g. a building) is classified as building on one side of
a seam and vegetation on the other — by **reconciling the classifications of
multiple blocks over the same location**.

The mechanism: **distance-weighted soft voting (probability fusion)**. Every
original LiDAR point is labeled by fusing the per-block softmax probability
vectors of *all* blocks whose inference footprint covers it, weighted by how
central the point is to each voting block and how close the block's nearest
sampled point is. This replaces the current single-block, hard-argmax,
nearest-sample label inheritance.

This stage requires **no preprocessing changes, no `.feat` format changes,
and no retraining**. It operates on existing v1 `.feat` files and existing
trained models. With fusion disabled (default-preserving `--fusion-radius 0`
explicitly passed), output is **bit-identical** to the pre-Stage-44 pipeline.

### Root causes addressed (from the Stage 42 architectural review)

1. **Per-tile-only global max-pool** — each block's predictions are made under
   a truncated global context. Fusion does not fix the context itself (that
   is Stage 45's halo), but it converts the resulting hard seam into a smooth,
   centrality-weighted transition in which the block for which the point is
   most interior dominates.
2. **Hard-argmax single-vote labeling** — `model.classify()` collapses logits
   to one `u8` label per sampled point; `write_classified` assigns each
   original point the label of exactly one nearest sampled point in its
   canonical block. No confidence information survives to reconcile
   disagreements. This stage retains per-point softmax probabilities and
   makes reconciliation possible.

---

## Inputs & Outputs

### New CLI flags — `classify` sub-command (`src/cli/classify_cmd.rs`)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fusion-radius <f64>` | projection units | *see below* | Radius beyond each block's canonical rect within which its predictions may vote on points owned by other blocks. `0.0` disables fusion (legacy single-block behavior). Constraint: `0.0 ≤ fusion_radius ≤ block_size / 2`. **Default: `manifest.block_overlap` when that value is `> 0.0` (forward-compatible with Stage 45 halo-augmented manifests), else `0.0`.** Explicit flag always overrides the default. |
| `--fusion-temp <f64>` | f64 | `1.0` | Softmax temperature τ applied per block before voting. τ > 1 softens (equalizes) each block's class distribution; τ < 1 sharpens it. Must be finite and `> 0.0`. |

Validation errors (exit non-zero with message) on: negative or non-finite
radius; radius > `block_size / 2` (read from the manifest); non-finite or
non-positive temperature.

### New CLI flags — `evaluate` sub-command (`src/cli/evaluate_cmd.rs`)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--fused-eval` | bool flag | off | Replicate the write-time fusion over labeled validation points instead of per-block argmax. Reports standard metrics **plus a boundary-band vs. interior metric split**. |
| `--fusion-radius <f64>` | projection units | manifest-derived (as above) | Same semantics as `classify`. Only meaningful with `--fused-eval`. |
| `--fusion-temp <f64>` | f64 | `1.0` | Same semantics as `classify`. Only meaningful with `--fused-eval`. |

Passing `--fusion-radius` / `--fusion-temp` without `--fused-eval` is a
usage error (prevents silently ignored flags).

### `src/model/inference.rs` — `BlockInferenceResult` redesign

The result type changes from "k-d tree of labels" to "k-d tree of probability
rows":

```rust
pub struct BlockInferenceResult {
    /// 2-D spatial index: key = `[x, y]` (projection units),
    /// value = row index into `probs`.
    tree: KdTree<f64, u32, [f64; 2]>,
    /// Row-major `[n_points × n_classes]` temperature-softmaxed probability
    /// matrix, in *model class-index* space (NOT yet mapped through
    /// `label_map`).
    probs: Vec<f32>,
    /// Number of rows in `probs` / points in `tree`.
    n_points: usize,
    /// Number of model classes (`probs.len() / n_points`).
    n_classes: usize,
}
```

Public API:

```rust
/// Build from coordinates + raw logits; applies temperature softmax per row.
pub fn from_logits(xs: &[f64], ys: &[f64], logits: &Array2<f32>, temperature: f64) -> Result<Self>;

/// Nearest sampled point to `(qx, qy)`: returns `(squared_distance, prob_row)`
/// in model class-index space, or `None` when the tree is empty.
pub fn nearest_vote(&self, qx: f64, qy: f64) -> Option<(f64, &[f32])>;

pub fn n_classes(&self) -> usize;
```

- Softmax uses the standard max-subtracted, temperature-scaled form:
  `p_i = exp((z_i − z_max) / τ) / Σ_j exp((z_j − z_max) / τ)`.
  At τ = 1.0 this is ordinary softmax. Because softmax is monotone, a block
  voting alone produces exactly today's argmax — preserving the regression
  contract.
- `nearest_label()` and `from_points()` are removed (internal API; tests
  updated). The label-mapping step (model index → ASPRS code via
  `label_map`) moves out of per-block inference to the final consumer
  (writer / evaluator), since argmax now happens *after* fusion.
- `run_inference(manifest, model, feat_dir, temperature)` gains a trailing
  `temperature: f64` parameter. `process_block` calls `model.forward()`
  (logits) instead of `model.classify()` (argmax) and builds the result via
  `from_logits`. **Per-block forward pass, k-d tree size, and block-level
  Rayon parallelism are unchanged**; the only new per-block work is one
  softmax over `N × n_classes` floats (negligible).

### `src/output/las_writer.rs` — signature change

```rust
pub struct FusionConfig {
    /// Voting reach beyond each block's canonical rect (0.0 = disabled).
    pub radius: f64,
}

pub fn write_classified<S: BuildHasher>(
    input_path: &Path,
    output_path: &Path,
    inference_map: &HashMap<u64, BlockInferenceResult, S>,
    manifest: &BlockManifest,
    label_map: &[u8],        // model class index → ASPRS code (from model)
    fusion: &FusionConfig,
) -> Result<()>
```

`label_map` is passed from `classify_cmd` (it owns the loaded model). The
final argmax over the fused probability accumulator is mapped through
`label_map` exactly as `classify()` did before (fallback: ASPRS 1 Unassigned
on empty map — same as today).

### No format / manifest changes

`.feat` files, `blocks.json`, `.wbmodel` files: **all unchanged**. Older
manifests and v1 `.feat` files work without regeneration. The fusion default
reads the *existing* `block_overlap` field (Stage 08, `#[serde(default)]`)
— no new fields are added to the manifest in this stage.

---

## Algorithm Steps

### Phase 1 — Inference (per block, Rayon-parallel, unchanged structure)

1. Load `.feat`, reconstruct `(x, y)` per row — unchanged.
2. `model.forward(features)` → logits `[N, n_classes]` (replaces
   `model.classify()`).
3. Temperature-softmax each row → `probs`.
4. Build `BlockInferenceResult` (k-d tree over all `N` rows → row index).
5. Collect into `inference_map` — unchanged.

### Phase 2 — Streaming write with fusion (per original point `P = (x, y)`)

6. Compute canonical cell `(col, row)` and `block_id` — unchanged.
7. **Interior fast path:** let `dx_in = min(x − ox_A, ox_A + s − x)`,
   `dy_in = min(y − oy_A, oy_A + s − y)` be P's distances to its canonical
   block A's rect edges (`s = block_size`, rect from manifest grid geometry).
   If `fusion.radius == 0.0` **or** `min(dx_in, dy_in) ≥ fusion.radius`,
   no other block can have a nonzero vote (see weight function below) →
   perform the legacy single lookup: `A.nearest_vote(x, y)` → argmax over
   the returned row → `label_map` → assign. If A is absent from the map
   (density-dropped), preserve original classification — unchanged behavior.
8. **Fusion path** (P within `radius` of a canonical edge, or A absent):
   - **Candidacy.** Enumerate candidate cells:
     `cols = floor((x − r − x_min)/s) ..= floor((x + r − x_min)/s)`,
     likewise rows, each clamped to the grid. With `r ≤ s/2` this yields
     **at most 2 columns × 2 rows = 4 candidates** (proof: span =
     `(s + 2r)/s ≤ 2`). Look each up via `block_id()`; skip cells absent
     from `inference_map`.
   - **Centrality weight (distance-to-rect trapezoid).** For candidate *b*
     with canonical rect `[ox, ox+s] × [oy, oy+s]`, per axis:
     `gap_x = max(ox − x, x − (ox + s), 0)` (0 inside the rect), then
     `wx = clamp((r − gap_x) / r, 0, 1)`; `wy` likewise. Centrality
     `t_b = wx · wy`. Properties: **1 everywhere inside b's canonical
     interior** (plateau), linear ramp to **0 at distance r beyond the
     rect**, C⁰-continuous — the blend introduces no seam of its own.
     Skip the candidate if `t_b == 0.0`.
   - **Proximity weight.** `q(d²) = 1 / (d² + σ²)` where `d²` is the
     squared distance from P to b's nearest sampled point (returned by
     `nearest_vote`) and `σ` is the **proximity bandwidth** — by default
     the characteristic inter-sample spacing,
     `σ = block_size / √target_points` (`default_proximity_sigma` in
     `model::fusion`). Inverse-square (Shepard-style) falloff prevents a
     sparse block's distant sample from outvoting the local one, while the
     bandwidth bounds the term so a query point that coincides with a
     block's own sample (d² = 0) cannot dominate the blend by orders of
     magnitude (see deviation note 8 — the fused-eval blind spot).
     `σ` and the power are fixed (no CLI surface); ablation may revisit.
   - **Accumulate.** `vote = t_b · q(d²)`;
     `acc[c] += vote · p_b[c]` for each class `c`; no need to track
     `Σ vote` separately unless normalizing for diagnostics — argmax is
     scale-invariant, so the accumulator is left unnormalized for speed.
   - **Decide.** If any votes accumulated: `idx* = argmax(acc)`;
     `classification = label_map.get(idx*).copied().unwrap_or(1)`.
     If *no* candidate produced a vote (e.g. A absent and no neighbor
     within `r`): preserve original classification — matches legacy
     missing-block behavior.
9. Write the point record — unchanged.

**Emergent benefit:** with fusion enabled, points inside a density-dropped
block that lie within `radius` of a retained neighbor's edge receive
neighbor-voted labels instead of blindly preserving the source
classification. (With v1 `.feat` files these votes come from the neighbor's
nearest *core* sample — small weight, honest magnitude. Stage 45 makes this
channel direct.)

### Phase 3 — Fused evaluation (`evaluate --fused-eval`)

10. After loading the labeled manifest and model (existing path), build the
    same `BlockInferenceResult` map over the labeled `.feat` blocks
    (temperature as configured).
11. Grid geometry: `LabeledBlockManifest` lacks `grid_cols`/`grid_x_min`, so
    derive the grid from block origins: `x_min = min(origin_x)`,
    `y_min = min(origin_y)`, `col = round((origin_x − x_min) / s)`.
    (Partitioner guarantees grid-aligned origins; documented assumption.)
12. For each labeled sampled point (`.feat` row ↔ `.lbl` byte), compute its
    fused predicted class using the *identical* candidacy/weight/accumulate
    routine as Phase 2 (shared free function in `inference.rs` or a new
    `fusion.rs` module — single implementation, two callers).
13. Metrics (`EpochMetrics` machinery, unchanged) are computed twice over the
    same predictions:
    - **Full set** — drop-in replacement metrics (mIoU, per-class IoU,
      precision/recall/F1, confusion matrix), printed in the existing
      format with a `[fused-eval]` banner.
    - **Split by band** — each point's distance-to-canonical-edge is
      recoverable from its `x_norm`/`y_norm` features:
      `d_edge = min(xn, 1−xn, yn, 1−yn) · s`. Points with `d_edge < radius`
      form the **boundary band**; the rest are **interior**. Per-class IoU
      and mIoU are reported for each subset. This is the diagnostic that
      (a) demonstrates seam improvement and (b) gates the Stage 45
      go/no-go decision (boundary-band IoU for object classes vs.
      interior IoU).
14. Default (no `--fused-eval`): behavior, output, and comparability with
    historical results are **unchanged** — per-block argmax evaluation.

### Module touch list

| File | Change |
|---|---|
| `src/model/inference.rs` | `BlockInferenceResult` redesign; `from_logits`, `nearest_vote`; softmax helper; `run_inference` temperature param; shared `fused_label()` routine (candidacy + weights + accumulate) |
| `src/output/las_writer.rs` | `FusionConfig`; `write_classified` new params (`label_map`, `fusion`); fast path + fusion path in stream loop |
| `src/cli/classify_cmd.rs` | `--fusion-radius` / `--fusion-temp` parsing, validation, manifest-derived default; pass-through |
| `src/cli/evaluate_cmd.rs` | `--fused-eval` + shared fusion flags; grid derivation; band-split reporting; flag-combination validation |
| `src/error.rs` | no new variants expected (reuse `Pipeline`) |

No changes to: preprocessing (any module), model forward pass, training,
`.feat`/`.lbl` formats, manifests, or `whitebox_next_gen`.

---

## Performance & Memory Analysis

| Cost center | Delta vs. pre-Stage-44 |
|---|---|
| Inference forward pass | **None** (same N×17 per block; softmax replaces argmax loop — cheaper or equal) |
| `BlockInferenceResult` memory | `u8`/point → `n_classes × f32`/point (e.g. 8 classes → 32 B/pt vs 1 B/pt). At `target_points = 4096`: ~128 KB/block. Same dataset-scaled envelope as the existing `inference_map` (status-quo scaling, modest constant-factor growth). A quantized-`u8` probability representation (`p×255`) is a documented fallback if profiling shows pressure — **not** implemented in this stage. |
| Write pass — interior points | Identical (fast path: one k-d NN query + argmax over ≤8 floats) |
| Write pass — boundary points | ≤4 k-d NN queries + ≤4·n_classes fused multiply-adds + closed-form weights. Boundary fraction at `r = s/4` ≈ 75% of area; at `r = s/8` ≈ 44%. Each extra query is over a ~4k-point 2-D tree (~12 comparisons). Expected wall-clock delta: **single-digit percent**, dominated as today by LAZ I/O. |
| Parallelism / locks | Unchanged — inference Rayon block-parallel, write pass sequential stream, zero synchronization primitives added. |

---

## Alternatives Considered (and rejected)

| Alternative | Rejection rationale |
|---|---|
| **Hard majority voting** on per-block argmax labels | Discards confidence; tie-breaking is arbitrary; cannot express "weak building vs. strong vegetation". |
| **Logit averaging** (mean logits → softmax → argmax) | Raw logit *scale* varies per block with each block's global vector; the largest-magnitude block would dominate multiplicatively. Probability vectors are scale-bounded and comparable; the weighted mixture is the standard ensemble posterior. |
| **K shifted grids** (e.g. 0.5-offset second grid, vote across passes) | Equivalent coverage to halo fusion but costs K preprocess passes, K× `.feat` storage, K× inference (4× at 50% stride). The Stage 45 halo achieves the same at ~1.0× forward cost. |
| **Global probability accumulation map** (reconcile sampled points into a dataset-wide HashMap, then write) | Extra pass + memory scaling with dataset size; fragile float-key identity matching across independently sampled blocks. Per-original-point streaming reconciliation at write time is constant-memory and exact. |
| **Tent weight ramping *inside* the canonical rect** (w→0 at the edge from within) | Neighbor blocks can never vote (their weight is 0 outside their rect) — the mechanism collapses to single-block labeling. The distance-to-rect trapezoid (plateau inside, ramp outside) is the correct generalization. |

---

## Definition of Done (DoD)

1. **Regression contract:** `classify --fusion-radius 0` produces
   **bit-identical** output LAS/LAZ to the pre-Stage-44 pipeline for the
   same input/model/blocks (integration test on a synthetic fixture).
2. **Interior invariance:** with fusion enabled at `r > 0`, points beyond
   `r` from every canonical edge receive labels identical to the
   `--fusion-radius 0` run (unit + integration level).
3. **Softmax correctness:** `from_logits` with τ = 1.0 reproduces
   reference softmax values for a known logit matrix (max-subtraction
   verified on an extreme-magnitude row — no overflow); τ ≠ 1.0
   sharpens/flattens as documented; argmax of each row equals
   `model.classify()`'s former output (monotonicity test).
4. **Weight function tests:** plateau = 1 for interior queries; linear ramp
   across the outer band; 0 at/ beyond `r`; corner product `wx·wy`;
   C⁰ continuity at the rect edge and at the ramp toe.
5. **Candidacy tests:** ≤4 candidates for `r ≤ s/2` at arbitrary query
   positions; correct cells at seams, corners, and grid borders;
   clamped at grid edges; density-dropped (absent) cells skipped.
6. **Blend decision test:** synthetic two-block fixture with deliberately
   contradictory per-block probabilities → a point at the seam receives the
   50/50-blended argmax; a point deep in block A receives A's argmax;
   proximity weight `q` demonstrably demotes a far-sample vote.
7. **Missing-block behavior:** canonical block absent + no neighbors within
   `r` → original classification preserved (matches legacy).
8. **CLI validation:** radius/temperature range errors; manifest-derived
   radius default (with/without `block_overlap`); `evaluate` rejects
   fusion flags without `--fused-eval`.
9. **Fused-eval:** `--fused-eval` runs end-to-end on a labeled fixture;
   band-split output present and arithmetically consistent with the full
   set; default (unfused) output **byte-identical** to pre-Stage-44 for the
   same fixture.
10. All existing tests pass (`cargo test --features training`); new tests
    added per items 2–9.
11. `cargo clippy --features training -- -D warnings` → zero warnings.
12. `cargo fmt --check` → clean.
13. `docs/user/user_guide.md` gains a "Prediction Fusion" section
    (flag reference + when to use it).
14. This spec file is synchronized with the implementation before the stage
    is marked complete (AGENTS.md living-synchronization contract).

---

## Definition-of-Done Gate for Stage 45

This stage ships the measurement instrument (`--fused-eval` band split) that
decides Stage 45's full go-ahead: if boundary-band IoU for object classes
(Building especially) remains materially below interior IoU after fusion,
Stage 45's halo-augmented context is justified; if fusion alone closes the
gap, Stage 45 may be descoped. (Stage 45's spec already exists as a draft;
the gate is advisory, not blocking, per user direction 2026-07-27.)

---

## Implementation Status & Deviations (2026-07-27)

Implemented as specified. Verification: `cargo test --features training` —
209 unit tests + 1 integration test passing (38 new tests added by this
stage); `cargo clippy --all-targets --features training -- -D warnings` —
zero warnings; `cargo fmt --check` — clean.

### Deviations from the spec text above (code is authoritative)

1. **Fusion module location.** The shared candidacy/weight/accumulate
   routine lives in a new `src/model/fusion.rs` module (the spec's allowed
   "new `fusion.rs` module" option), exposing `FusionConfig`,
   `GridGeometry`, `centrality_weight`, and `fused_label`. `model/mod.rs`
   declares it. `las_writer` and `evaluate_cmd` share this single
   implementation.
2. **`evaluate --fused-eval` default radius is `block_size / 4`** (not
   "manifest-derived as above"). `LabeledBlockManifest` does not carry
   `block_overlap`, and the dataset loader does not own the sibling
   `blocks.json`. The spec's recommended pairing (s/4) is therefore applied
   directly; `--fusion-radius` overrides it. `classify` retains the
   spec'd `block_overlap`-derived default.
3. **Fused-eval evaluates each `--data-dir` independently** with its own
   derived grid (single-dir `LabeledBlockDataset::load` per directory,
   metrics accumulated across directories). Blocks from different
   directories describe unrelated spatial regions and local block IDs
   collide across directories by design, so cross-directory voting would be
   meaningless. Spec steps 10–12 are unchanged in every other respect.
4. **`dataset.rs` gained a read-only `block_spatial_meta(gid)` accessor**
   (origin + block size by composite `GlobalBlockId`) to support grid
   derivation and coordinate reconstruction in fused-eval. No behavioural
   change to existing methods.
5. **`run_evaluation_fused` is decomposed** into `derive_grid_and_radius`,
   `build_vote_structures`, and `fused_predictions` helpers (clippy
   function-length/complexity lints); algorithm unchanged.
6. **DoD #1 interpretation.** Bit-identical regression is verified at the
   library level: the `radius = 0` write path is decision-identical to the
   legacy nearest-sample-argmax semantics on synthetic fixtures
   (`test_write_classified_substitutes_classification`,
   `test_fusion_off_matches_legacy_single_block`), and softmax-argmax
   monotonicity parity is proven in
   `test_softmax_argmax_matches_logit_argmax`. A byte-level LAS diff
   against a pre-Stage-44 binary on a real dataset is deferred to manual
   verification, consistent with dataset-dependent DoD items in earlier
   stages.
7. **DoD #9 verification scope.** Fused-eval correctness is verified
   end-to-end on a two-block labeled fixture
   (`test_fused_eval_two_blocks_band_split`): default radius derivation,
   band/interior point partition, and per-subset accuracy all asserted.
   The unfused path is exercised by the pre-existing Stage 39 tests, which
   pass unchanged.
8. **Superseded by Stage 47 (2026-08-07) — grid derivation & vote-map keying.**
   Live testing use surfaced `evaluate --fused-eval` metrics coming through as
   all-zero. Root cause: this stage's `derive_grid_and_radius` (step 11 above)
   re-derived grid geometry from the *retained* blocks' own origins (since
   `LabeledBlockManifest` carried no persisted grid fields), which silently
   diverges from the TRUE grid whenever density filtering drops an edge
   block or column — and, independently, `split-dataset` unconditionally
   renumbers every block's `id` on every output, which breaks the
   `id == row * grid_cols + col` invariant the pre-fix vote-map keying
   (`map.insert(gid, result)`, keyed by raw `meta.id`) depended on. Either
   defect alone causes the vote-map lookup to miss for every block —
   including a block's own self-vote — collapsing every metric to zero.
   **Fixed by Stage 47**: `LabeledBlockManifest` now persists true grid
   geometry from preprocessing time; `derive_grid_and_radius` reads it
   directly instead of re-deriving anything; `build_vote_structures` keys
   the vote map by a `block_id` freshly derived from each block's own
   spatial origin (never trusting `meta.id`); `split-dataset` propagates
   grid geometry for single-input splits and explicitly rejects
   (informative error) multi-input-merged splits for fused-eval, since
   those have no single coherent grid to propagate. See
   `docs/stages/stage-47-fused-eval-grid-and-id-robustness-fix.md` for the
   full diagnosis, fix, and regression tests.
9. **Hotfix — proximity bandwidth σ (real-data validation, 2026-07-27).**

   First-field-validation of `--fused-eval` (10 m blocks, r = 2.5 m)
   produced fused metrics *identical* to unfused. Root cause: in
   fused-eval every query point **is** one of its canonical block's own
   sampled points, so its canonical `nearest_vote` returns `d² = 0`
   exactly, giving the canonical block weight `1/ε = 10⁹` — a home-block
   dictatorship that no neighbour vote can contest. The same 1/d²
   singularity over-favoured the home block (milder) in deployed classify.
   Fix: the proximity term is now `1 / (d² + σ²)` with
   `σ = block_size / √target_points` (`default_proximity_sigma`), bounding
   self-hits to ~2× a one-spacing-away vote and making blends genuine in
   both call sites. `fused_label` gained a `proximity_sigma` parameter;
   `las_writer` and `evaluate_cmd` derive it from the manifest;
   `LabeledBlockDataset::target_points()` was added to support the eval
   derivation. Existing tests were updated to the new signature; all
   behavioural test outcomes are unchanged (the σ term preserves the
   intended relative-weight structure).

### Files modified / added

- `src/model/inference.rs` — `BlockInferenceResult` redesigned (probability
  rows + row-index k-d tree), `from_logits`/`nearest_vote`/`n_classes`,
  temperature softmax, `run_inference`/`process_block` temperature param,
  shared `reconstruct_xy`.
- `src/model/fusion.rs` — **new**; `FusionConfig`, `GridGeometry`,
  `centrality_weight`, `fused_label`.
- `src/model/mod.rs` — `pub mod fusion;`.
- `src/output/las_writer.rs` — fused write path; `write_classified` gains
  `label_map` + `fusion` params; 3 tests updated/added.
- `src/cli/classify_cmd.rs` — `--fusion-radius` / `--fusion-temp`,
  resolution + validation + help; 7 new tests.
- `src/cli/evaluate_cmd.rs` — `--fused-eval` + fusion flags, `run_fused` +
  `run_evaluation_fused` (+3 helpers), band-split reporting; 3 new tests.
- `src/training/dataset.rs` — `BlockSpatialMeta` + `block_spatial_meta()`;
  1 new test.
- `docs/user/user_guide.md` — "Prediction Fusion (Stage 44)" section;
  flag tables for `classify` and `evaluate` updated.
