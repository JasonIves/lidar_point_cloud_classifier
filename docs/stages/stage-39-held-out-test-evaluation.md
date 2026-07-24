# Stage 39 — Held-Out Test-Set Evaluation (`wb_lidar_train evaluate`)

**Status:** COMPLETE — implemented in `src/cli/evaluate_cmd.rs`, wired into
`wb_lidar_train`; `build` / `clippy -D warnings` / `fmt --check` /
`test --features training` all pass. See
`docs/stages/stage-40-tnet-transpose-fix.md` for a follow-up correctness fix
and hardening applied to this stage's `reconcile_n_classes` check.

**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `PROJECT_SPEC.md` (Training / Evaluation),
`docs/stages/stage-32-dataset-split-materialization.md`,
`src/training/metrics.rs`, `src/model/inference.rs`, `src/model/weights.rs`,
`src/training/dataset.rs`, `src/bin/wb_lidar_train.rs`

---

## Goal

Provide a **general, post-training evaluation** entry point that measures a
trained `.wbmodel`'s classification performance on a **labeled, held-out data
directory** (e.g. the `test/` split materialized by
`wb_lidar_train split-dataset`, or any `preprocess-labeled` output directory
the model never saw during training).

The command emits standard segmentation metrics — per-class IoU / Precision /
Recall / F1, mean IoU, overall accuracy, macro-F1 — plus a full confusion
matrix, as **two CSV files**.

This is a **new sub-command on the `wb_lidar_train` binary**:

```text
wb_lidar_train evaluate
    --model         <path.wbmodel>   trained model to evaluate (required)
    --data-dir      <dir>            labeled test dir (repeatable, required)
    --metrics-out   <path.csv>       per-class metrics CSV (required)
    --confusion-out <path.csv>       confusion-matrix CSV (required)
    [--n-classes    <n>]             optional cross-check against model/data
    [--threads      <n>]             Rayon thread-pool size (default: cores)
```

### Problem it solves

There is currently **no way to score a trained model on unseen labeled data**:

- `wb_lidar_classify classify` requires a spatial `BlockManifest`
  (`blocks.json`) tied to one base LAS file and writes a classified LAS — it
  cannot consume `split-dataset` output (which renumbers/merges blocks and emits
  a `LabeledBlockManifest` in `labeled_blocks.json`), and it produces no
  metrics.
- The training-time `validate_epoch()` routine *does* compute the right metrics
  but is private, requires an in-memory burn `BurnPointNet<B>`, and there is no
  `.wbmodel` → burn reverse bridge. It is also coupled to the training loop.

Stage 39 closes that gap for the deployed model.

---

## Decision

Evaluate through the **pure-Rust inference engine** (`model::weights::load_model`
+ `model::pointnet::PointNetClassifier`), **not** through burn/`validate_epoch`.

Rationale:

1. **Measures the actually-deployed model.** The `.wbmodel` inference path is
   what production `classify` runs. The burn `.valid()` path and the
   pure-Rust inference engine are required to agree numerically; this is
   verified by `test_burn_and_ndarray_forward_outputs_agree_after_bridge` in
   `src/training/bridge.rs` (added in Stage 40), which constructs a
   `BurnPointNet`, bridges it to a `.wbmodel`, and asserts the two forward
   passes produce matching logits on the same input. **Note:** the
   `.wbmodel` round-trip test in `weights.rs` does *not* establish this —
   it only round-trips `PointNetClassifier` through serialization and never
   constructs a `BurnPointNet`, so it cannot catch training/inference
   divergence. An earlier version of this doc incorrectly cited that test
   as evidence of Burn↔ndarray agreement; see
   `docs/stages/stage-40-tnet-transpose-fix.md` for the divergence bug this
   conflation allowed to ship undetected, and for the corrected equivalence
   test.
2. **No new reverse bridge.** The only weight bridge that exists is
   burn → `.wbmodel`; reconstructing a `BurnPointNet` from a `.wbmodel` purely
   to evaluate would be dead weight and a new maintenance surface.
3. **Leaner per AGENTS.md.** Reuses existing, already-tested building blocks:
   `LabeledBlockDataset` (loads `.feat`/`.lbl` and validates label maps),
   `PointNetClassifier::forward` (logits), and `MetricsAccumulator` +
   `write_confusion_matrix_csv` (metrics).

### Index-space correctness (critical)

`.lbl` files store **remapped model class indices** (`0..n-1`), the same space
as `MetricsAccumulator`. But `PointNetClassifier::classify()` returns **ASPRS
codes** via `label_map`. Therefore evaluation must compare the **argmax of
`forward()` logits (model index)** against the `.lbl` model indices directly —
it must **not** use `classify()` / `run_inference()`, which would remap
predictions into ASPRS space and mismatch the ground truth.

### Loading the test directory as a flat block list

`LabeledBlockDataset::load(data_dirs, val_split, None, seed)` already parses
`labeled_blocks.json`, validates label-map contiguity and cross-directory
`n_classes` agreement, and partitions blocks into disjoint `train_ids` +
`val_ids`. For evaluation we want **every** block exactly once, so we call
`load(dirs, 0.0, None, 0)` purely for that validation and then iterate the
dedicated `dataset.all_block_ids()` accessor, which enumerates every block
across all directories independent of any split.

**Why not `train_ids ∪ val_ids`:** `load` *always* builds a train/val split,
and `spatial_split` forces at least one held-out val macro-tile even at
`val_split == 0.0` (`target_val = (n_tiles * val_split).round().max(1.0)`).
So `load(dirs, 0.0, …)` still prints a misleading `[dataset] train blocks: N,
val blocks: M` line. Reconstructing the full set from `train_ids ∪ val_ids`
happens to work (the partition is disjoint + complete) but is fragile and
confusing. `all_block_ids()` makes the "score the entire dataset" intent
explicit; evaluation additionally logs a note clarifying that the loader's
split line is an internal artifact. This reuses all existing validation
without a new loader. `load_presplit` is unsuitable here because it
hard-requires both a train and a val directory.

### Consistency checks

- `model.config.n_classes` must equal `dataset.n_classes()` (derived from the
  manifest label map). Mismatch ⇒ hard `Pipeline` error (evaluating a model
  against data with a different class count is meaningless).
- If `--n-classes` is supplied, it is cross-checked against both and any
  disagreement is a hard error. It is otherwise optional (derived).
- Per-point feature width is already enforced to `N_FEATURES` by both the
  dataset loader and `forward()`.
- As of Stage 40, the model/data class-count check is supplemented by a
  **label-map content check**: the dataset's ASPRS-code→model-index map is
  inverted and compared entry-by-entry against the model's
  model-index→ASPRS-code `label_map`. Two models/datasets can agree on class
  *count* while disagreeing on which ASPRS code maps to which model index
  (e.g. because they were preprocessed with different `--label-map` values);
  this previously passed `reconcile_n_classes` silently and would have
  produced meaningless metrics. See
  `docs/stages/stage-40-tnet-transpose-fix.md`.

---

## Outputs

Two CSV files (reviewer-approved "two CSVs, cleaner" shape):

1. **`--metrics-out` — per-class metrics** (one row per class, plus a header):

   ```text
   class_idx,asprs_code,tp,fp,tn,fn,precision,recall,f1,iou
   ```

   `asprs_code` is taken from the model's `label_map[class_idx]` for
   human readability. A trailing comment-free summary line is *not* embedded;
   aggregate scores (mIoU, overall accuracy, macro-F1) are printed to stderr.

2. **`--confusion-out` — confusion matrix** written by the existing
   `metrics::write_confusion_matrix_csv` (rows = true class, cols = predicted
   class), reused verbatim.

A concise summary (blocks/points evaluated, mIoU, overall accuracy, macro-F1,
and a per-class P/R/F1/IoU table) is printed to **stderr**.

---

## Processing model

```text
load_model(.wbmodel)                         ── Arc<PointNetClassifier>
LabeledBlockDataset::load(dirs, 0.0, None, 0)
for each block id in dataset.all_block_ids() ── Rayon-parallel per block:
    load_block → (features [N×17], gt_labels [N] model-idx)
    logits = model.forward(features)         ── [N × n_classes]
    preds[i] = argmax_j logits[i, j]         ── model index (u8)
    yield (preds, gt_labels)
drain sequentially → MetricsAccumulator::accumulate(preds, gt)
compute() → EpochMetrics
write per-class metrics CSV + confusion CSV
print summary to stderr
```

Parallelism mirrors `run_inference`: a lock-free `par_iter().map(...).collect()`
into `Vec<Result<(Vec<u8>, Vec<u8>)>>`, then a sequential drain that propagates
the first error and folds results into the accumulator.

---

## AGENTS.md compliance

- **Spec-first / greenfield:** this doc precedes code; all new code lives in new
  files within `lidar_point_cloud_classifier` (`src/cli/evaluate_cmd.rs`) plus
  minimal additive wiring in `src/cli/mod.rs` and `src/bin/wb_lidar_train.rs`.
  `whitebox_next_gen` core is untouched.
- **No panics in core paths:** all fallible operations bubble `Result`
  (`ClassifierError::Pipeline`); no `unwrap`/`expect`/`panic!` in non-test code.
- **Minimal deps:** no new crates — reuses `rayon`, `ndarray`, and existing
  training/model modules.
- **Feature-gated:** the sub-command compiles only under `--features training`
  (it depends on `LabeledBlockDataset` and `MetricsAccumulator`).
- **Platform-agnostic / CPU:** the inference engine is pure-Rust CPU; no GPU
  dependency. `--threads` sizes the Rayon pool exactly as `classify` does.

---

## Definition of Done

1. `src/cli/evaluate_cmd.rs` implements arg parsing + orchestration as specified.
2. `evaluate` is dispatched from `src/bin/wb_lidar_train.rs`; module registered
   in `src/cli/mod.rs` under `#[cfg(feature = "training")]`.
3. Predictions use argmax of `forward()` (model indices), compared against
   `.lbl` model indices — verified by test.
4. Two CSVs are written; confusion CSV reuses `write_confusion_matrix_csv`.
5. Tests cover: synthetic labeled dir → expected confusion/metrics; n_classes /
   model-vs-data mismatch error; missing model / missing dir error; no panics.
6. `cargo build --features training`, `cargo clippy --features training
   -- -D warnings`, `cargo fmt --check`, and `cargo test --features training`
   all pass.
</content>
<parameter name="task_progress">- [x] Diagnose root cause: T-Net transform transpose mismatch (training vs deployed forward)
- [x] Present finding to user and obtain explicit approval
- [x] Fix TNet::apply in src/model/layers.rs (remove erroneous transpose) + isolated ndarray regression test
- [x] Run cargo build/test/clippy/fmt to verify the fix compiles and does not regress existing tests
- [x] Add a true Burn↔ndarray forward-equivalence regression test (asymmetric T) comparing BurnPointNet vs bridged PointNetClassifier
- [x] Harden reconcile_n_classes (src/cli/evaluate_cmd.rs) to check label_map content, not just count
- [x] Re-verify clippy/fmt after item 2 edits (dataset.rs, evaluate_cmd.rs)
- [ ] Confirm/clarify checkpoint provenance (item 4)
- [x] Correct stage-39 doc claim about round-trip test proving Burn↔ndarray equivalence
- [ ] Write new docs/stages/stage-40-tnet-transpose-fix.md documenting the bug + fix
- [ ] Final full verification pass (build/test/clippy/fmt) after all changes