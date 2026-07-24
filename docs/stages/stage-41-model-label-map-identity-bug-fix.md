# Stage 41 — Model `label_map` Identity Bug Fix

**Status:** COMPLETE — `src/training/trainer.rs` fixed to derive a trained
model's saved `label_map` from the dataset's actual ASPRS-code ↔
model-index mapping instead of hardcoding identity; new
`LabeledBlockDataset::inverse_label_map()` helper added and shared by both
`trainer::train()` and `evaluate_cmd::reconcile_n_classes()`. `build` /
`clippy -D warnings` / `fmt --check` / `test --features training` (159
unit tests + 1 integration test) all pass.

**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `docs/stages/stage-39-held-out-test-evaluation.md`,
`docs/stages/stage-40-tnet-transpose-fix.md`,
`src/training/dataset.rs`, `src/training/trainer.rs`,
`src/cli/evaluate_cmd.rs`, `src/model/pointnet.rs`,
`src/preprocessing/labeled_pipeline.rs`, `tests/training_integration.rs`

---

## Bug report that triggered this stage

After Stage 40 was deployed, the user ran `wb_lidar_train evaluate` against
real held-out data and hit the new (correctly-functioning) Stage 40
`reconcile_n_classes` label-map-**content** check:

> "The model maps this index to ASPRS code X, but the evaluation data's
> label map maps it to ASPRS code Y."

The user had **not** supplied any custom `--label-map` to either
`preprocess-labeled` or `evaluate`, and asked: *"I didn't even supply a
label map. Do we need to add a label map path to the evaluate
functionality?"*

---

## Root cause: `trainer.rs` hardcoded an identity `label_map`

`src/training/trainer.rs::train()` builds the `Vec<u8>` that gets embedded
verbatim as a `.wbmodel`'s `label_map` field. Before this fix, at all three
`save_model_from_burn()` call sites (checkpoint-save, no-checkpoints-configured,
final save with no checkpoint dir), it did:

```rust
let label_map: Vec<u8> = (0u8..config.n_classes as u8).collect();
```

i.e. `[0, 1, 2, ..., n_classes-1]` — an **identity** mapping, completely
independent of the real ASPRS-code ↔ model-index mapping that
`preprocess-labeled` actually used to encode the `.lbl` files (recorded in
each directory's `labeled_blocks.json` under `label_map: HashMap<String,
u8>`, ASPRS-code-string → model-index).

For the codebase's own **default** label map (used whenever
`preprocess-labeled` is run without `--label-map`):

```text
{"2":0, "3":1, "4":2, "5":3, "6":4, "9":5, "7":6, "1":7}
  Ground Low  Med  High Build Water Noise Unassigned
```

the *true* inverse (model-index → ASPRS-code) that should have been saved
is `[2, 3, 4, 5, 6, 9, 7, 1]` — **not** `[0, 1, 2, 3, 4, 5, 6, 7]`.

### Why this is more severe than it first appears

`PointNetClassifier::classify()` (`src/model/pointnet.rs`) is the
production inference path that converts a predicted model-class-index back
into a real ASPRS code for the output LAS `Classification` field:

```rust
let asprs_code = self.label_map.get(best_idx).copied().unwrap_or(1);
```

Because `trainer.rs` always saved an identity `label_map`, **every
`.wbmodel` ever trained by this codebase has been shipping raw internal
model-class-indices (`0..n_classes-1`) as if they were real ASPRS
classification codes** in the final output LAS files — e.g. a point the
model correctly identifies as "Building" (ASPRS code 6, model index 4)
would be written to the output LAS with `Classification = 4`
("Unclassified" territory in ASPRS terms), not `6`.

This is **not** a metrics-only bug like Stage 40's T-Net transpose issue —
it silently corrupts the actual classified point-cloud deliverable for
every model trained to date. The Stage 40 `reconcile_n_classes` check
(comparing the model's `label_map` against the dataset's real mapping) is
what surfaced it, because the identity `label_map` a trained model
contained could never agree with a real (non-identity) dataset label map
— triggering the mismatch error even though the user had done nothing
wrong.

---

## The fix

### 1. New shared helper: `LabeledBlockDataset::inverse_label_map()` (`src/training/dataset.rs`)

```rust
pub fn inverse_label_map(&self) -> Result<Vec<u8>> {
    let n = self.n_classes_inner;
    let mut derived: Vec<Option<u8>> = vec![None; n];
    for (code_str, &idx) in self.label_map() {
        let code: u8 = code_str.parse().map_err(|_| {
            ClassifierError::Pipeline(format!(
                "dataset label_map has a non-numeric ASPRS code key {code_str:?}"
            ))
        })?;
        let slot = derived.get_mut(idx as usize).ok_or_else(|| {
            ClassifierError::Pipeline(format!(
                "dataset label_map model index {idx} is out of range for {n} classes"
            ))
        })?;
        *slot = Some(code);
    }
    Ok(derived.into_iter().map(|c| c.unwrap_or(1)).collect())
}
```

This inverts `self.label_map()` — the dataset's real, as-preprocessed
ASPRS-code(string) → model-index map, populated verbatim from whichever
mapping `preprocess-labeled` actually used (its built-in default *or* a
custom `--label-map` file supplied by the user) — into a dense
model-index → ASPRS-code `Vec<u8>` of length `n_classes()`. Any model index
with no entry in the dataset's label map falls back to ASPRS code `1`
(Unassigned), matching `classify()`'s own `.unwrap_or(1)` fallback
convention.

**Design guarantee (explicitly required by the user before implementation
began):** this method derives the mapping *exclusively* from
`dataset.label_map()` — it never references
`LabeledPreprocessConfig::default_label_map()` or any other hardcoded
mapping anywhere in its implementation. Because `labeled_blocks.json`'s
`label_map` field is populated by `preprocess-labeled` from whichever
mapping was actually used at preprocessing time (default or custom), a
model trained on data preprocessed with a custom `--label-map` will
correctly have that same custom mapping (inverted) propagated into its
saved `.wbmodel`, never silently falling back to the default ASPRS map.

### 2. `src/training/trainer.rs::train()` — the actual bug fix

```rust
// Before (buggy):
let label_map: Vec<u8> = (0u8..config.n_classes as u8).collect();

// After (fixed):
let label_map: Vec<u8> = dataset.inverse_label_map()?;
```

This single line is reused at all three `save_model_from_burn()` call
sites, so checkpoint saves, the final save when no checkpoint directory is
configured, and the final save with a checkpoint directory configured (via
SWA or best-checkpoint copy) all now embed the correct, dataset-derived
`label_map`.

### 3. `src/cli/evaluate_cmd.rs::reconcile_n_classes()` — refactored to reuse the shared helper

The Stage 40 inline inversion loop was replaced with a call to the same
`inverse_label_map()` method, so there is now exactly one implementation of
"invert the dataset's label map" shared by both the training path (what
gets saved into a new model) and the evaluation path (what gets checked
against an existing model):

```rust
let derived = dataset.inverse_label_map()?;

for (idx, expected_code) in derived.iter().enumerate() {
    let model_code = model.label_map.get(idx).copied();
    if model_code != Some(*expected_code) {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: label map mismatch at model class index {idx} — the \
             model maps this index to ASPRS code {model_code:?}, but the \
             evaluation data's label map maps it to ASPRS code {expected_code}. \
             The model and the evaluation data must have been preprocessed \
             with the exact same --label-map, not merely the same class count."
        )));
    }
}
```

This preserves `reconcile_n_classes`'s existing external behavior and error
message exactly; only the internal inversion logic was deduplicated.

---

## New / updated tests

### `src/training/dataset.rs`

- `test_inverse_label_map_non_identity_mapping` — builds a manifest with a
  deliberately non-identity map (`{"2":0, "3":1, "6":2}`), loads it through
  `LabeledBlockDataset::load()`, and asserts `inverse_label_map()` returns
  `[2, 3, 6]`.
- `test_inverse_label_map_rejects_non_numeric_asprs_code` — asserts a
  non-numeric ASPRS code key produces a clear "non-numeric" error.
- `test_inverse_label_map_rejects_out_of_range_model_index` — asserts a
  model index outside `0..n_classes` produces a clear "out of range" error.

### `tests/training_integration.rs`

The synthetic manifest's label map was changed from identity to
deliberately non-identity (`"0"→1`, `"1"→0`) specifically so a regression
back to hardcoded identity would be caught end-to-end:

```rust
let mut label_map = HashMap::new();
label_map.insert("0".to_string(), 1u8);
label_map.insert("1".to_string(), 0u8);
```

and a new assertion checks the actual `.wbmodel` written by a real
`trainer::train()` run:

```rust
let saved_model = lidar_point_cloud_classifier::model::weights::load_model(&output_path)
    .expect("saved model must load");
let expected_label_map = dataset
    .inverse_label_map()
    .expect("inverse_label_map must succeed");
assert_eq!(
    saved_model.label_map, expected_label_map,
    "trained model's label_map must match the dataset's inverted label map, \
     not a hardcoded identity mapping"
);
```

This test would have failed against the pre-fix `trainer.rs` (which would
have saved `[0, 1]`, not the expected `[1, 0]`) and now passes.

---

## Recommendation for previously-trained models

Every `.wbmodel` trained before this fix has an incorrect (identity)
`label_map`, meaning its `classify()`-produced output LAS files contain raw
internal model-class-indices in the `Classification` field rather than
real ASPRS codes. Retraining is the primary recommended remedy.

**Footnote:** for cases where retraining is impractical, a minimal
standalone utility now exists to patch an existing `.wbmodel`'s
`label_map` field in place — without touching any weight tensors — given
the original training data directory(ies): `wb_lidar_train fix-label-map
--model <path.wbmodel> --data-dir <original-train-dir> [--output
<new-path>]` (see `src/cli/fix_label_map_cmd.rs`, and run with `--help` for
full usage). It is intentionally excluded from the primary `train` /
`evaluate` documentation and usage banner, as it is a one-off repair tool,
not a normal part of the training workflow.

---

## Verification

- `cargo build --features training` — passes.
- `cargo clippy --features training -- -D warnings` — zero warnings.
- `cargo test --features training` — all 159 unit tests + 1 integration
  test pass, including the 3 new `inverse_label_map()` unit tests and the
  updated `tests/training_integration.rs` end-to-end assertion.
- `cargo fmt --check` — clean.

## Definition of Done

1. `trainer.rs` derives a trained model's saved `label_map` from the
   dataset's actual ASPRS-code ↔ model-index mapping (via the new
   `LabeledBlockDataset::inverse_label_map()`), never a hardcoded identity
   mapping.
2. The derivation uses whichever mapping was actually recorded by
   `preprocess-labeled` — the built-in default or a custom `--label-map` —
   by construction, since `inverse_label_map()` never references
   `LabeledPreprocessConfig::default_label_map()` directly.
3. `evaluate_cmd.rs::reconcile_n_classes()` reuses the same shared
   `inverse_label_map()` helper instead of duplicating inversion logic.
4. New unit tests cover the inversion logic directly (including two error
   paths), and the existing end-to-end integration test now asserts the
   real, on-disk `.wbmodel` produced by `train()` has the correct
   (non-identity, dataset-derived) `label_map`.
5. `cargo build --features training`, `cargo clippy --features training
   -- -D warnings`, `cargo fmt --check`, and `cargo test --features
   training` all pass.
