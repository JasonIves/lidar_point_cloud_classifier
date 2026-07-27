# Stage 45 — Fixed-N Halo Split: Overlapping Blocks as Model Input (`.feat` v2)

**Status:** COMPLETE — implementation landed 2026-07-27 with Amendments 1–3
(see "Approved Amendments" at the end); spec synchronized
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `docs/stages/stage-44-classify-time-prediction-fusion.md`
(prerequisite — supplies the fusion mechanism this stage feeds),
`docs/stages/stage-08-overlapping-blocks.md` (whose border-point channel,
currently **vestigial** — border points are loaded and immediately
`drop()`ed in `pipeline.rs` Step 7 — this stage revives as genuine model
input), `docs/stages/stage-42-absolute-z-normalization.md` (§"Deferred /
Related Findings" item 1 — per-tile global max-pool — which this stage
addresses directly), Stage 30 (whole-file eigen pre-pass; same breaking-change
/retraining precedent), Stage 29 (jitter oversampling — core path unchanged)

---

## Goal

Give every block's PointNet forward pass **cross-boundary context** by
replacing a fraction of each block's core sample rows with **halo rows** —
points sampled from the block's overlap margin (the Stage 08 border strip) —
so that:

1. The block's **global max-pool vector aggregates structure from across its
   boundaries** (a building cut by a seam is seen whole, not halved), fixing
   the per-tile-context root cause of the quilt artifact at the *input* level;
2. Halo points receive **genuine per-point predictions at their own
   locations**, which Stage 44's fusion layer consumes as direct votes in
   neighboring territory (replacing Stage 44's nearest-core-sample
   approximation across the seam).

**The defining constraint — fixed N:** the per-block tensor stays exactly
`target_points × 17`. Halo rows are a **budget reallocation inside N**, not an
addition: `N = N_core + N_halo`. Forward-pass FLOPs, GPU VRAM, training
micro-batch shapes, and `.feat` payload sizes are **unchanged** — the user's
block-size / density / batch-size calibration carries over intact. The honest
cost is (a) core sampling density diluted by the halo fraction and (b) a
one-time re-preprocess + retrain (accepted; same precedent as Stages 30/37/42).

---

## Inputs & Outputs

### New CLI flags — `preprocess` and `preprocess-labeled`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--halo-fraction <f64>` | f64 | `0.0` | Fraction φ of each block's `target_points` rows reserved for halo (overlap-margin) samples: `0.0 ≤ φ ≤ 0.5`. `0.0` disables halo sampling (pre-Stage-45 behavior). Recommended: `0.25`. **Requires `--block-overlap > 0.0`** — the overlap radius *is* the halo reach `r`. Recommended pairing: `--block-overlap = block_size / 4`. |

Validation errors on: φ outside `[0.0, 0.5]` or non-finite; `φ > 0.0` with
`block_overlap == 0.0` (message must state that halo rows are drawn from the
Stage 08 border strip, which only exists when overlap is enabled).

### New CLI flag — `train` sub-command

| Flag | Type | Default | Description |
|---|---|---|---|
| `--halo-loss-weight <f32>` | f32 | `1.0` | Loss weight applied to halo rows (per-point) during training. `1.0` treats halo rows as full training samples (they carry real ground-truth labels and act as free context-shift augmentation). Must be finite and `≥ 0.0`; `0.0` masks halo rows from the loss entirely (context-only mode). |

### `.feat` format v2 (header only — payload layout unchanged in size)

```text
[header — 41 bytes]                     (v1 was 37 bytes)
  magic:      4 bytes  = b"WBFT"
  version:    u8       = 2
  n_points:   u32      (= target_points N, core + halo)
  n_features: u32      (= 17, unchanged)
  block_id:   u64
  origin_x:   f64
  origin_y:   f64
  n_halo:     u32      (rows [n_points − n_halo .. n_points) are halo rows)

[data]
  f32[n_points × n_features]  row-major, point-major order
  rows [0 .. n_points − n_halo)        = core sampled points   (x_norm,y_norm ∈ [0,1])
  rows [n_points − n_halo .. n_points) = halo sampled points   (x_norm,y_norm ∈ [−r/s, 1+r/s])
```

- `FEAT_VERSION` becomes `2`; a `FEAT_VERSION_V1 = 1` constant is retained.
  **All `.feat` readers accept both versions** (v1 ⇒ `n_halo = 0`, 37-byte
  header). Readers: `model/inference.rs::read_feat_header`,
  `training/dataset.rs` block loader.
- Payload size is unchanged (`N × 17 × 4` bytes); `MAX_FEAT_PAYLOAD_BYTES`
  validation unchanged. `n_halo < n_points` is validated on read (a header
  with `n_halo ≥ n_points` is rejected as corrupt).
- Debug CSV (`--debug-csv`): rows follow the same `[core | halo]` ordering;
  no schema change.

### `.lbl` format — **unchanged**

`.lbl` files are headerless `u8[n_points]` arrays. Halo labels are simply the
trailing `n_halo` bytes, positionally aligned with the `.feat` halo rows. The
training dataset loader learns `n_halo` from the sibling `.feat` v2 header and
slices accordingly. v1 `.feat` + `.lbl` pairs remain valid.

### `.border` internal spill format — 31 → 39 bytes/record

The Stage 08 border spill currently stores `LitePoint` only. Halo rows need
each border point's **original stream index** to join against the whole-file
eigen pre-pass table (`eigen_table[original_idx]`, the same join used for core
points in `pipeline.rs` Step 7d). Records become `(u64 index, LitePoint)` =
8 + 31 = 39 bytes, mirroring the existing `.spill` write path
(`write_spill_file` already stores exactly this pair). This is an **internal
temporary format** (written and deleted within a single run); no backward
compatibility is required.

### Manifest changes

| File | Change |
|---|---|
| `blocks.json` (`BlockManifest`) | New `halo_fraction: f64` field, `#[serde(default)]` (= 0.0). `block_overlap` (Stage 08) already records the halo reach `r` — reused, not duplicated. |
| `blocks.json` (`BlockMeta`) | New `n_halo: usize` field, `#[serde(default)]` (= 0). `sampled_point_count` remains the total row count (N). `raw_point_count` remains canonical-only (density semantics unchanged). `oversampled` reflects the **core** sampling path only. |
| `labeled_blocks.json` | `LabeledBlockManifest` gains `halo_fraction` (`#[serde(default)]`); `LabeledBlockMeta` inherits `n_halo` via the flattened `BlockMeta`. |

### `PreprocessConfig` / `LabeledPreprocessConfig`

```rust
/// Halo budget fraction φ ∈ [0.0, 0.5] (Stage 45). Rows per block:
/// n_halo_target = round(φ · target_points); core target = N − n_halo_target.
/// Requires block_overlap > 0.0. Default 0.0 (disabled).
pub halo_fraction: f64,
```

(`LabeledPreprocessConfig` passes it through `PreprocessConfig`, as with
`block_overlap`.)

### `TrainConfig`

```rust
/// Per-point loss weight for halo rows (Stage 45). Default 1.0.
pub halo_loss_weight: f32,
```

### `classify` integration (no new flags)

`read_feat_header` accepts v2; `process_block` is **otherwise unchanged** —
it already builds the inference tree over *all* rows, so halo rows enter the
tree automatically at their true coordinates (which extend past the canonical
rect). Stage 44's fusion then consumes halo votes transparently. The
`--fusion-radius` default of `manifest.block_overlap` (Stage 44 spec) is
exactly the halo reach — the two stages compose with zero additional CLI
surface.

---

## Algorithm Steps

### Phase 1 — Preprocess (`preprocess`, unlabeled)

1. Steps 1–5c (stream, partition, density filter, cell map, **border spill**)
   run as today, with one change: `write_border_spill` / `read_border_spill`
   carry `(u64, LitePoint)` pairs (39-byte records). Activation is unchanged:
   border spills exist iff `block_overlap > 0.0`.
2. **Step 7 (parallel closure) — replaces `drop(border_pts)`:**
   a. `n_halo_target = round(φ · target_points)` (0 when φ = 0).
   b. **Core sampling (existing machinery):**
      `resample_block(&block.points, target_points − n_halo_target, block_id, jitter)`
      — unchanged semantics, including Stage 29 jitter on core padding. Note:
      because the core target is now `< N`, marginal blocks that previously
      oversampled may no longer need to (documented, benign).
   c. **Halo sampling (new, subsample-only):**
      `sample_halo(&border_pts_with_idx, n_halo_target, block_id)`:
      seeded Fisher–Yates partial shuffle (same RNG crate/pattern as
      `resample_block`, seed = `block_id` with a distinct constant mixed in
      to decorrelate from the core stream, e.g. `block_id ^ 0x9E37_79B9_7F4A_7C15`)
      sampling **without replacement**. **Never oversamples, never jitters**:
      duplicate halo rows add no context and no vote diversity; if the border
      strip supplies fewer than `n_halo_target` points (sparse edges, dataset
      boundary blocks), all available halo points are taken and
      **core backfills the remainder** — i.e. core is resampled/topped-up to
      `N − n_halo_actual` via the same seeded `resample_block` call (second
      draw continuing the same RNG stream semantics as a single call to keep
      reproducibility simple: implement as one `resample_block` call whose
      target is computed *after* halo actuals are known).
   d. **Feature extraction:** build the combined vector
      `points = [core_sampled | halo_sampled]` and parallel
      `eigen_rows = [core_eigen | halo_eigen]` (halo eigen via original
      index from the 39-byte spill record — identical join to Step 7d, with
      the same `usize::try_from` + `.get().unwrap_or([0.0; 10])` fallback),
      then a **single** `extract_features(&points, &eigen_rows, dtm, origin_x,
      origin_y, block_size, hag_normalization, z_norm_strategy)` call.
      Per-row math is unchanged; halo `x_norm`/`y_norm` naturally land in
      `[−r/s, 1+r/s]` because normalization uses the *canonical* origin and
      `block_size` (core rows remain bit-identical to a halo-free run of the
      same core sample — important for ablations). z_norm/HAG/eigen are
      whole-file/global by Stages 30/37/42 — no halo special-casing.
   e. **Serialize v2:** `write_feat_file(..., n_halo: n_halo_actual)` writes
      the 41-byte v2 header (`FEAT_VERSION = 2`) and the unchanged
      `N × 17` payload.
3. Manifest written with `halo_fraction` and per-block `n_halo`.

### Phase 2 — Labeled preprocess (`preprocess-labeled`)

4. Halo points are original LiDAR points and carry `classification` in
   `LitePoint`; their labels are remapped through the **same** `remap()`
   path as core points (unknown/unclassified codes → Unassigned index, as
   today). `BlockProcessResult` gains
   `halo_classifications: Vec<u8>` (raw ASPRS bytes, populated only when
   `capture_indices` is true) so the labeled pipeline needs no second LiDAR
   pass — mirroring the Stage 03 Option-A design.
5. `.lbl` write: `labels = [core_remapped | halo_remapped]` (`n_points`
   bytes total, positional with `.feat` rows). Blocks dropped for being
   all-class-0: the check now considers core + halo labels (halo-only
   signal in an all-zero-core block still means no *core* supervision —
   the drop predicate remains **core-only** to preserve Stage 03 semantics;
   documented decision).
6. Class distribution in `LabeledBlockMeta.class_distribution`: **core rows
   only** (keeps split-stratification semantics of Stages 32–36 unchanged);
   a sibling `halo_class_distribution` field is *not* added (no consumer;
   avoid manifest bloat).

### Phase 3 — Training (`train`)

7. Dataset loader reads the v2 header → `n_halo` (v1 ⇒ 0). `LoadedBlock`
   gains `loss_weights: Vec<f32>` of length `n_points`, computed per row
   (see **Amendment 1** for the masking rule): `1.0` for core rows; for
   halo rows, `halo_loss_weight` when the row's reconstructed macro-tile
   equals its own block's, `0.0` when it differs. v1 blocks ⇒ all-`1.0`
   (zero-cost default; existing caches/fixtures unaffected). The Stage 27
   cache clone path (`clone_loaded_block`, `BlockCache::block_bytes`)
   carries the new field (Amendment 2).
8. Trainer loss: per-point cross-entropy is multiplied elementwise by
   `loss_weights` before reduction (mean over **weighted** points; the
   denominator is `Σ loss_weights`, not `n_points`, so a `0.0` weight
   exactly masks that row). Implemented as a manual weighted-mean CE
   (`log_softmax` → gather target log-probs → weighted sum / weight sum),
   numerically identical to the prior burn CE whenever all weights are 1.0
   (i.e. all v1 datasets); it composes with per-class class-weighting by
   multiplying the two weight factors per point. Batch shapes `[B, N, C]` /
   `[B, N]` unchanged; Stage 18 batched-BN path unaffected. Validation
   metrics (`EpochMetrics`) remain computed over **all rows including
   halo** (they are real labeled points; halo rows near boundaries are
   exactly where the model most needs to be right) — documented decision;
   a core-only metric split is available via Stage 44's `--fused-eval`
   instead.
9. `.wbmodel` format: **unchanged** (architecture is unchanged — halo is a
   data-pipeline concern).

### Phase 4 — Inference / classify

10. `read_feat_header` accepts v1/v2; everything downstream is unchanged
    (tree over all rows; Stage 44 fusion consumes halo votes when
    `--fusion-radius` defaults to `manifest.block_overlap > 0`).

### Module touch list

| File | Change |
|---|---|
| `src/preprocessing/mod.rs` | `halo_fraction` config field + default; `FEAT_VERSION = 2`, `FEAT_VERSION_V1 = 1` |
| `src/preprocessing/pipeline.rs` | 39-byte border spill (index-carrying); `sample_halo()`; Step 7 core/halo split + combined `extract_features`; `write_feat_file` v2 header; `BlockMeta.n_halo`, `BlockManifest.halo_fraction`; `BlockProcessResult.halo_classifications` |
| `src/preprocessing/normalizer.rs` | `sample_halo` lives here beside `resample_block` (shared seeded-shuffle pattern) |
| `src/preprocessing/labeled_pipeline.rs` | halo label remap + `.lbl` concatenation; manifest fields |
| `src/model/inference.rs` | `read_feat_header` v1+v2 accept, `n_halo` validated |
| `src/training/dataset.rs` | v2 header read; `LoadedBlock.loss_weights` |
| `src/training/trainer.rs` (and `bridge.rs` as needed) | weighted per-point loss; `TrainConfig.halo_loss_weight` |
| `src/cli/preprocess_cmd.rs`, `preprocess_labeled_cmd.rs` | `--halo-fraction` flag + validation + help |
| `src/cli/train_cmd.rs` | `--halo-loss-weight` flag + validation + help |
| `src/cli/classify_cmd.rs` | none beyond Stage 44 (v2 header support lands in `inference.rs`) |

No changes in `whitebox_next_gen` (Greenfield constraint intact).

---

## Memory & Performance Analysis

| Cost center | Delta vs. pre-Stage-45 |
|---|---|
| Per-block tensor | **None** — N fixed; `N = N_core + N_halo` |
| Forward-pass FLOPs / VRAM / batch | **None** — `[B, N, 17]` unchanged |
| `.feat` size | +4 bytes/block header; payload unchanged |
| Feature extraction | **None** — `extract_features` still processes exactly N rows/block |
| Preprocess CPU / I/O | Border-strip load + index-carrying spill — this I/O **already occurs today** when `--block-overlap > 0` (and its product is currently discarded); halo sampling is a seeded shuffle over the strip. Net new work vs. overlap-enabled baseline ≈ zero; vs. overlap-disabled baseline, one read pass over neighbor spill files (the Stage 08 cost, already bounded and spill-gated: peak RAM stays `threads × (canonical + border_strip)`) |
| Training wall-time | Loss weighting is an elementwise multiply — negligible. One-time re-preprocess + retrain accepted. |
| Core sample density | ×(1 − φ) in block interiors (write-pass nearest-sample spacing × ~1/√(1−φ) ≈ 1.15 at φ = 0.25); seam bands *gain* coverage (own core + neighbor halo). Monitored via DoD interior-IoU non-regression check. |
| `inference_map` memory | Same as Stage 44 (halo rows are just more rows in the same N-sized trees — **total tree size per block unchanged**) |

---

## Breaking Changes & Retraining Requirement

1. `.feat` **format v2**: old binaries that only accept v1 must be updated
   together (this crate's readers accept both). External consumers of
   `.feat` (none are officially supported) must handle the v2 header.
2. **Any model trained on pre-Stage-45 data must be retrained** to consume
   halo-augmented blocks correctly: halo rows extend `x_norm`/`y_norm`
   outside `[0, 1]`, a genuine input-distribution shift. Classifying v2
   blocks with a pre-45 model is *permitted* (it runs — PointNet is
   shape-agnostic) but is **not a supported configuration**: it doubles as
   the optional "45a" ablation (does the input T-Net absorb the range
   shift?), never a deliverable.
3. Preprocess reproducibility: enabling halo changes core RNG consumption
   (core target < N), so `.feat` payloads differ from pre-45 runs even at
   the same seed. With `halo_fraction = 0.0`, the pipeline is
   **bit-identical** to pre-Stage-45 except the v2 header (payload bytes
   unchanged — verified in DoD #2).

---

## Definition of Done (DoD)

1. **Format:** v2 write/read round-trip including `n_halo`; v1 files read
   correctly (`n_halo = 0`) by both readers; `n_halo ≥ n_points` rejected;
   `MAX_FEAT_PAYLOAD_BYTES` enforcement unchanged.
2. **Regression:** `--halo-fraction 0.0` produces `.feat` payloads
   **bit-identical** to the pre-Stage-45 pipeline for the same input/config
   (header differs only in version/`n_halo` bytes — asserted by test).
3. **Halo sampling:** seeded reproducibility across two identical runs;
   subsample-only (no duplicates when strip ≥ target); shortfall ⇒ all
   strip points taken + core backfills to exactly N (shape guarantee under
   sparse/edge conditions); dataset-boundary blocks (no neighbors) yield
   `n_halo = 0` and a full-N core block.
4. **Feature correctness:** a halo row's eigen block equals the whole-file
   pre-pass row for its original index (synthetic fixture with known
   table); halo `x_norm`/`y_norm` fall outside `[0,1]` by the expected
   amount for a known geometry; core rows bit-match a halo-free run of the
   same core sample.
5. **Labeled pipeline:** `.lbl` length == `.feat` `n_points`; halo label
   bytes match the source points' remapped classifications; core-only
   all-class-0 drop predicate preserved; `class_distribution` core-only.
6. **Training:** `LoadedBlock.loss_weights` correct for v1 (all-ones), v2
   (split at `n_core`), and `--halo-loss-weight 0.0` (masked); weighted-mean
   denominator verified on a known fixture; `cargo test --features training`
   suite green including Stage 18 batched-BN and Stage 27 cache tests.
7. **End-to-end fusion:** on a synthetic two-block v2 fixture, classify with
   default flags (fusion radius = `block_overlap`) → a seam point receives a
   blended label influenced by a *halo* row of the neighboring block
   (verified by constructing the fixture so only the halo vote can produce
   the outcome).
8. **CLI validation:** `--halo-fraction` range; φ > 0 without
   `--block-overlap` rejected with an actionable message;
   `--halo-loss-weight` range.
9. **Memory:** preprocess peak RSS on a reference-scale synthetic LAZ stays
   within the Stage 08 bound (`threads × (canonical + border_strip)`,
   verified by design review + the existing spill tests; full-scale memory
   benchmark deferred to first real-data run, as is project custom).
10. `cargo build --features training`, `cargo test --features training`
    (all existing + new tests), `cargo clippy --features training --
    -D warnings`, `cargo fmt --check` — all clean.
11. Docs: `docs/user/user_guide.md` gains halo documentation;
    `PROJECT_SPEC.md` preprocessing section gains a short addendum noting
    the halo-augmented block design (per the Stage 30 addendum precedent);
    this spec is synchronized with the implementation before close
    (AGENTS.md living-synchronization contract).
12. **Success gate (advisory):** using Stage 44's `--fused-eval` band split,
    boundary-band per-class IoU for object classes (Building) improves vs.
    the Stage-44-only baseline, and interior IoU does not regress beyond
    the agreed tolerance (expected ≤ noise at φ = 0.25). Results are
    recorded in this spec's closing notes.

---

## Approved Amendments (2026-07-27)

Folded into this spec after the Stage 44 implementation review; approved by
the user 2026-07-27 with an explicit documentation requirement for
Amendment 1.

### Amendment 1 — Split-aware halo loss masking (mandatory)

**Problem (found in review):** the original draft gave every halo row a real
ground-truth label at `--halo-loss-weight 1.0`. Halo rows are sampled from
*neighboring* blocks' territory, so a training block at a macro-tile split
boundary would ingest validation/test-area labels into the training loss —
a genuine cross-split label-leakage channel, beyond the feature-level
boundary smoothing Stage 32 deliberately accepted.

**Rule (mandatory, not a flag):** in `LabeledBlockDataset::load_block`, each
halo row's macro-tile is reconstructed from its absolute coordinates
(x_norm/y_norm + block origin, against the manifest's `SpatialTileGrid`,
via `labeled_pipeline::compute_macro_tile_id` — hoisted to `pub(crate)`).
A halo row whose macro-tile **differs** from its own block's receives loss
weight **0.0** — it serves as global-pool context only. Same-tile halo rows
receive `--halo-loss-weight` (default 1.0). Core rows are always 1.0.

**Documentation requirement (user direction):** this masking must be
thoroughly documented in code (`dataset.rs`) so a downstream reader
immediately understands *why* some halo rows carry weight 0 — the comment
must name the leakage channel (neighbor-territory labels crossing a
train/val/test macro-tile boundary) and state that the context channel
(global max-pool) is intentionally retained per the Stage 32 precedent.

### Amendment 2 — Cache clone path

`LoadedBlock.loss_weights` must be cloned in `clone_loaded_block` and
counted in `BlockCache::block_bytes` (Stage 27 block cache).

### Amendment 3 — Measurement plan

- The DoD #12 gate baseline is **Stage 44 with the σ-bandwidth fix** (the
  working instrument). Procedure: `--fused-eval` band split on the same
  held-out split for (a) old model + v1 blocks (44-only baseline),
  (b) *optional 45a ablation*: old model + v2 halo blocks (does the input
  T-Net absorb the x_norm range shift without retraining?), (c) retrained
  model + v2 blocks (full 45). Gate: band IoU (Building especially)
  improves vs (a); interior IoU non-regression.
- Halo rows are **always boundary-band members** by construction (their
  normalized coordinates lie outside `[0,1]`, i.e. negative edge distance)
  — expected semantics, not a bug, when reading band splits on v2 data.

---

## Implementation Status (2026-07-27)

Implemented as specified (with Amendments 1–3 folded in). Verification:
`cargo test --features training` — 216 unit tests + 1 integration test
passing; `cargo clippy --all-targets --features training -- -D warnings` —
zero warnings; `cargo fmt --check` — clean.

Files modified: `preprocessing/mod.rs` (`FEAT_VERSION = 2`,
`FEAT_VERSION_V1 = 1`, `halo_fraction`), `normalizer.rs` (`sample_halo`),
`block_partitioner.rs` (`read_points_indexed`), `pipeline.rs` (39-byte
border spill, Step 7 halo split, v2 writer, `BlockManifest.halo_fraction`,
`BlockMeta.n_halo`, `BlockProcessResult.halo_classifications`),
`labeled_pipeline.rs` (halo label remap + `.lbl` concat,
`pub(crate) compute_macro_tile_id`), `model/inference.rs` (v1/v2 header
accept + `n_halo` validation), `training/dataset.rs` (v2 read,
`LoadedBlock.loss_weights` + split-aware mask + `with_halo_loss_weight`,
Stage 27 cache clone/bytes), `training/trainer.rs` (manual weighted-mean
CE replacing the burn CE config, `TrainConfig.halo_loss_weight`),
`cli/train_cmd.rs` (`--halo-loss-weight` + validation + builder wiring),
`cli/preprocess_cmd.rs` / `cli/preprocess_labeled_cmd.rs`
(`--halo-fraction` + validation + help), `cli/split_dataset_cmd.rs`
(manifest field pass-through), and all affected test fixtures/literals
(updated to the v2 header contract and new struct fields).

Next step (per the Amendment 3 measurement plan): re-preprocess + retrain
with `--block-overlap <block_size/4> --halo-fraction 0.25`, then compare
`evaluate --fused-eval` boundary-band vs. interior IoU (Building
especially) against the Stage-44-only baseline (DoD #12 gate), optionally
via the 45a no-retrain ablation first. Results to be recorded here on
validation.
