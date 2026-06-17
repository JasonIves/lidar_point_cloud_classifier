# Stage 03 — Training Module: Supervised PointNet Training Pipeline

**Status:** COMPLETE — See [stage-03-results.md](stage-03-results.md) for full development record and deviations  
**Approved:** 2026-06-16  
**Implemented:** 2026-06-16  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier  
**Lead Architect:** GitHub Copilot / AI Collaborator

---

## Architectural Decisions (Resolved)

The following decisions were resolved during spec review and are incorporated below
without further qualification.

| # | Decision | Resolution |
|---|---|---|
| 1 | burn version | Pin `burn = "0.16"` |
| 2 | Pipeline extension for label alignment | Option A — extend `run_pipeline()` to return `sampled_indices: Vec<usize>` per block |
| 3 | Batch dimension strategy | Single-block forward pass with gradient accumulation.  Preserves hardware agnosticism: burn/ndarray does not require contiguous batch tensors, and per-block parallelism at the Rayon level scales correctly across CPU cores without a batched forward requiring synchronization. |
| 4 | Train/val split strategy | Spatially-disjoint macro-tile split (default).  Random block splits leak spatial autocorrelation into validation and systematically over-report accuracy; spatial disjointness is the correct default for LiDAR. |
| 5 | Class weighting default | Class-weighted cross-entropy **on by default**.  Opt-out via `--no-class-weights`.  ASPRS LiDAR datasets are always severely class-imbalanced; unweighted loss produces degenerate models that classify everything as Ground. |
| 6 | Checkpoint cadence | `--checkpoint-every <n>` (default: 1 epoch).  `--keep-best-n <n>` (default: **5**).  Retention of 5 checkpoints enables Stochastic Weight Averaging (SWA) over the convergence region as an optional post-training step (`--swa` flag, off by default — see § SWA). |

---

## Goal

Implement the supervised training pipeline that produces `.wbmodel` weight files
consumable by the Stage 02 inference engine.  Training uses labeled LiDAR datasets
(`.las` / `.laz` / `.copc` files where the `classification` field carries ASPRS
ground-truth codes), processes them through an extended preprocessing step to produce
per-block feature + label pairs, and then trains a PointNet classifier using the
`burn` crate (ndarray backend, pure Rust autograd) before exporting weights via the
Stage 02 `.wbmodel` binary format.

The training workflow is isolated behind a `training` Cargo feature flag so that the
default `wb_lidar_classify` inference binary remains lightweight and compiles without
any ML framework overhead.  A separate `wb_lidar_train` binary (`[[bin]]` entry,
`required-features = ["training"]`) exposes the training CLI.

Training is **out of scope for inference**.  The Stage 02 `classify` command and its
`PointNetClassifier` remain completely unchanged.

---

## New Files & Module Layout

```
lidar_point_cloud_classifier/
  src/
    training/                         ← new module, gated by `#[cfg(feature = "training")]`
      mod.rs                          ← pub re-exports; cfg guard
      dataset.rs                      ← LabeledBlockDataset: load .feat + .lbl pairs, train/val split
      burn_model.rs                   ← BurnPointNet<B>: burn model mirroring Stage 02 architecture
      trainer.rs                      ← epoch/batch loop, loss, optimizer step, checkpoint writes
      metrics.rs                      ← mIoU, per-class IoU, F1, confusion matrix
      bridge.rs                       ← extract burn tensor weights → PointNetClassifier → save_model()
      scheduler.rs                    ← cosine annealing LR scheduler
    preprocessing/
      labeled_pipeline.rs             ← .feat + .lbl writer (wraps existing pipeline.rs via Option A)
    cli/
      train_cmd.rs                    ← `train` sub-command arg parser + orchestration
      preprocess_labeled_cmd.rs       ← `preprocess-labeled` sub-command arg parser
    bin/
      wb_lidar_train.rs               ← separate training binary entry point (requires-features training)
  docs/stages/
    stage-03-training-layer.md        ← this file
    stage-03-results.md               ← (created after implementation is complete)
```

**Modified from prior stages (minimal):**

```
  Cargo.toml                          ← add burn optional dep, `training` feature, new [[bin]] entry
  src/lib.rs                          ← add `pub mod training;` behind cfg guard
  src/preprocessing/mod.rs           ← expose `labeled_pipeline` sub-module
  src/preprocessing/pipeline.rs      ← Option A: add `sampled_indices` to block result (retroactive minor extension)
  src/cli/mod.rs                      ← no change; `wb_lidar_train` has its own binary and dispatch
```

---

## New Dependencies

All training dependencies are optional and gated behind the `training` feature.  The
inference binary (`wb_lidar_classify`) is completely unaffected.

```toml
[features]
training = ["dep:burn"]

[dependencies]
# --- (existing Stage 01/02 deps unchanged) ---

# Training framework — optional; only compiled when `--features training`
burn = { version = "0.16", features = ["ndarray", "autodiff"], optional = true }
```

| Crate | Version | Feature gate | Justification |
|---|---|---|---|
| `burn` | `"0.16"` (pinned) | `training` | Autograd, AdamW optimizer, BatchNorm training mode, cross-entropy loss — pure Rust with the ndarray CPU backend.  The `wgpu` feature can be enabled in a future stage for GPU acceleration without changing any model code. |

No additional crates are introduced.  Metrics, the LR scheduler, SWA weight averaging,
and the `.lbl` file format are all implemented from scratch using `std` + `ndarray`.

> **Why burn over a hand-rolled backward pass?**  Differentiating through the full
> PointNet forward pass — including T-Net backward passes through arbitrary 3×3 matrix
> transforms and through BatchNorm in training mode — would require approximately
> 1 000 lines of gradient code with a comparable gradient-check test suite.  `burn`
> provides verified, tested autograd at the cost of one optional, purpose-bounded
> dependency that compiles only for the training feature.  Under the Whitebox
> "Minimal & Thoughtful Dependencies" principle this is the correct trade-off.

---

## CLI

### New Binary: `wb_lidar_train`

```toml
[[bin]]
name = "wb_lidar_train"
path = "src/bin/wb_lidar_train.rs"
required-features = ["training"]
```

The binary dispatches two sub-commands:

```
wb_lidar_train <sub-command> [options]

  preprocess-labeled   Preprocess labeled LiDAR, emitting .feat + .lbl block files
  train                Train a PointNet model and write a .wbmodel file
  help                 Show this message
```

---

### Sub-command: `preprocess-labeled`

```
wb_lidar_train preprocess-labeled
    --input              <path>    LAS, LAZ, or COPC with ground-truth classification field
    --output             <dir>     Output directory for .feat, .lbl, and labeled_blocks.json
    [--block-size        <f64>]    2D cell edge length in projection units (default: 50.0)
    [--target-points     <usize>]  Fixed N points per block after sampling (default: 1024)
    [--min-density       <f64>]    Minimum pts/m²; blocks below threshold discarded (default: 1.0)
    [--search-radius     <f64>]    Base radius for eigenvalue queries (default: 1.0)
    [--min-neighbors     <usize>]  Adaptive radius minimum neighbor count (default: 8)
    [--hag-model         <path>]   Optional DTM raster for Height Above Ground
    [--label-map         <path>]   Optional JSON file remapping ASPRS codes to model class indices
    [--tile-grid         <usize>]  NxN macro-tile grid resolution for spatial split (default: 4)
    [--threads           <usize>]  Rayon thread pool size (default: system cores)
    [--debug-csv]                  Also emit per-block .csv files (development only)
```

Behaviour is identical to the Stage 01 `preprocess` command except that it also:
- Reads the `classification` byte from every `PointRecord` alongside spatial features.
- Applies the `--label-map` remapping (see § Label Remapping below).
- Assigns each block a `macro_tile_id` based on the `--tile-grid` resolution.
- Writes a sibling `.lbl` file for each `.feat` block.
- Writes `labeled_blocks.json` (superset of `blocks.json`; see § Output Formats).

Blocks where **all points carry classification code `0` (never classified)** are
silently discarded so that unlabeled regions are not presented to the trainer.
The `labeled_blocks.json` manifest reflects only the retained blocks.

---

### Sub-command: `train`

```
wb_lidar_train train
    --data-dir            <dir>    Directory produced by `preprocess-labeled`
    --output-model        <path>   Output .wbmodel file (best validation mIoU checkpoint)
    [--n-classes          <u8>]    Number of output classes (default: 8)
    [--label-map          <path>]  JSON remapping file (must match preprocess-labeled run)
    [--epochs             <usize>] Training epochs (default: 50)
    [--batch-size         <usize>] Blocks per gradient-accumulation batch (default: 16)
    [--learning-rate      <f32>]   Initial AdamW learning rate (default: 1e-3)
    [--weight-decay       <f32>]   AdamW weight decay λ (default: 1e-4)
    [--val-split          <f32>]   Approx fraction of macro-tiles held out for validation (default: 0.20)
    [--val-tile-blocks    <path>]  JSON file listing explicit block IDs to use as validation set
                                   (overrides --val-split spatial tiling when provided)
    [--seed               <u64>]   Deterministic seed for macro-tile val assignment (default: 42)
    [--use-feature-tnet]           Enable STN64d feature T-Net (default: disabled)
    [--no-batch-norm]              Disable BatchNorm layers (default: enabled)
    [--no-class-weights]           Disable inverse-frequency class weighting of cross-entropy loss
                                   (default: class-weighted ON — recommended for LiDAR datasets)
    [--checkpoint-dir     <path>]  Directory to save checkpoint .wbmodel files
    [--checkpoint-every   <usize>] Save a checkpoint every N epochs (default: 1)
    [--keep-best-n        <usize>] Retain only the N highest-val-mIoU checkpoints (default: 5)
    [--swa]                        Apply Stochastic Weight Averaging over retained checkpoints
                                   after training completes (default: disabled; see § SWA)
    [--metrics-out        <path>]  Per-epoch metrics CSV (default: <data-dir>/metrics.csv)
    [--threads            <usize>] Rayon thread pool for parallel data loading
```

---

## Output Formats

### `.lbl` Binary File

Each `.feat` block has a sibling `.lbl` file (`block_00042.feat` ↔ `block_00042.lbl`).

```
[no header — raw byte array]
  u8[n_points]  — remapped model class indices, in the same point order as .feat
```

The point count is inferred from the corresponding `.feat` header (`n_points` field).
Storing no header keeps the format minimal and unambiguous — the `.feat` is the
authoritative source of `n_points` and `block_id`.

> **Note on remapping:** The `.lbl` file stores **model class indices** (0-based,
> contiguous), not raw ASPRS codes.  Remapping is applied during `preprocess-labeled`
> so that the trainer never needs to know about ASPRS codes at all.  The mapping is
> persisted in `labeled_blocks.json` for traceability.

---

### `labeled_blocks.json` Manifest

A superset of Stage 01's `blocks.json`.  Adds `lbl_file`, `class_distribution`, and
`macro_tile_id` per block, plus top-level `label_map` and `spatial_tile_grid` fields:

```json
{
  "source":           "training_data.las",
  "block_size":       50.0,
  "target_points":    1024,
  "min_density":      1.0,
  "search_radius":    1.0,
  "min_neighbors":    8,
  "crs_epsg":         32617,
  "label_map":        { "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "9": 5, "7": 6, "1": 7 },
  "spatial_tile_grid": { "cols": 4, "rows": 4, "bbox_min_x": 449000.0, "bbox_min_y": 4849000.0,
                         "bbox_max_x": 450600.0, "bbox_max_y": 4851200.0 },
  "blocks": [
    {
      "id":                  42,
      "file":                "block_00042.feat",
      "lbl_file":            "block_00042.lbl",
      "origin_x":            450000.0,
      "origin_y":            4850000.0,
      "raw_point_count":     1587,
      "sampled_point_count": 1024,
      "oversampled":         false,
      "macro_tile_id":       6,
      "class_distribution":  { "0": 0, "1": 102, "2": 612, "5": 275, "6": 137 }
    }
  ]
}
```

`class_distribution` uses **remapped model indices** as keys and is used by the
trainer to compute per-class inverse-frequency weights.

---

### Label Remapping (`--label-map` JSON)

An optional JSON object mapping ASPRS byte codes (as string keys) to contiguous
model class indices.  If absent, the default 8-class mapping from Stage 02 is applied:

```json
{ "1": 7, "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 6, "9": 5 }
```

Any ASPRS code not present in the map is assigned to index 7 (Unassigned).
The remapping is applied once during `preprocess-labeled` and embedded in
`labeled_blocks.json` — it is not re-applied at training time.

---

## Spatially-Disjoint Train/Val Split

### Motivation

Adjacent LiDAR blocks in the same city block or forest stand share nearly identical
geometric, intensity, and HAG characteristics.  A random block-level split will place
correlated blocks on both sides of the train/val boundary and produce an inflated
validation mIoU that does not generalise.  Withholding entire spatially-contiguous
macro-tiles produces an honest estimate of model performance on unseen terrain.

### Default Spatial Tiling

The dataset bounding box (from `labeled_blocks.json`) is divided into an `N × N`
macro-tile grid (default `N = 4`, giving up to 16 macro-tiles).  Each block is
assigned a `macro_tile_id` at `preprocess-labeled` time:

```
col = clamp(floor((origin_x - bbox_min_x) / macro_tile_width),  0, N-1)
row = clamp(floor((origin_y - bbox_min_y) / macro_tile_height), 0, N-1)
macro_tile_id = row * N + col
```

Macro-tiles are sorted by ID and assigned to validation using a deterministic stride
so that the withheld fraction approximates `--val-split`:

```
target_val_tiles = max(1, round(n_populated_macro_tiles * val_split))
stride           = floor(n_populated_macro_tiles / target_val_tiles)
val_tile_ids     = { tile_ids[i * stride] for i in 0..target_val_tiles }
// ties broken by --seed
```

This assignment is fixed at `preprocess-labeled` time and embedded in
`labeled_blocks.json`, so the `train` command consumes a reproducible split.
Changing the split requires re-running `preprocess-labeled` with different
`--val-split` / `--tile-grid` parameters, or supplying `--val-tile-blocks`.

### Explicit Override (`--val-tile-blocks`)

If `--val-tile-blocks <path>` is provided to `train`, the file must be a JSON array
of block IDs (integers matching the `id` field in `labeled_blocks.json`):

```json
[42, 43, 44, 107, 108]
```

These blocks form the validation set regardless of `macro_tile_id`.  This is the
recommended path when integrating externally-designated benchmark hold-out regions
(e.g., ISPRS Vaihingen test tiles).

---

## BurnPointNet Architecture

### Design Principle: 1:1 Mirror of Stage 02

`BurnPointNet<B>` must replicate the Stage 02 `PointNetClassifier` forward pass with
exact numerical fidelity so that the weight bridge (`bridge.rs`) is deterministic and
unambiguous.  No architectural additions or variations are permitted — this is a
training twin of the inference model, not a research variant.

### Layer Naming Convention

Field names in `BurnPointNet<B>` follow the same naming scheme as the `.wbmodel`
serialisation order, enabling a mechanical, position-stable weight extraction:

| Section | Field names | `.wbmodel` block order |
|---|---|---|
| Input T-Net encoder | `stn3d_enc[0..2]`, `stn3d_bn_enc[0..2]` | T-Net encoder blocks 0–2 |
| Input T-Net FC | `stn3d_fc[0..1]`, `stn3d_bn_fc[0..1]`, `stn3d_fc2` | T-Net FC blocks 0–2 (no BN on fc2) |
| Feature T-Net encoder | `stn64d_enc[0..2]`, `stn64d_bn_enc[0..2]` | Only present when `use_feature_tnet` |
| Feature T-Net FC | `stn64d_fc[0..1]`, `stn64d_bn_fc[0..1]`, `stn64d_fc2` | Only present when `use_feature_tnet` |
| Main encoder | `enc[0..2]`, `bn_enc[0..2]` | Encoder blocks 0–2 |
| Main decoder | `dec[0..1]`, `bn_dec[0..1]` | Decoder blocks 0–1 |
| Final projection | `proj` (no BN) | Final projection block |

### BatchNorm Training / Inference Mode

`burn`'s `BatchNorm` tracks `running_mean` and `running_var` automatically during
the forward pass.  To switch to inference mode (frozen stats), call
`model.valid()` before the validation loop and restore with `model.clone()` for the
next training epoch.  The running stats accumulated during training are extracted by
the weight bridge and stored as `bn_mean` / `bn_var` in `.wbmodel`.

### Forward Pass Shape Contract

```
Input:  Array2<f32>  shape [N, 12]   (same as Stage 02 inference input)
Output: Tensor<B, 2> shape [N, n_classes]  (raw logits — no softmax)
```

The loss function applies log-softmax internally.  The forward pass must not apply
any activation to the final projection layer output.

---

## Training Loop

### Mini-Batch Gradient Accumulation

Because each block is processed independently (no shared batch dimension in the
model), gradient accumulation is used.  This preserves hardware agnosticism: no
batched tensor dimensions are introduced into the model, so the Stage 02 inference
path is completely unchanged.

```
for epoch in 0..n_epochs:
    shuffle train_block_ids (seeded RNG per epoch: seed XOR epoch)

    total_train_loss = 0.0
    n_train_steps    = 0

    for chunk in train_block_ids.chunks(batch_size):
        acc_loss = Tensor::zeros(...)
        for block_id in chunk:
            (features, labels) = dataset.load(block_id)       // [N, 12] + [N]
            logits   = model.forward(features)                 // [N, n_classes]
            loss     = cross_entropy(logits, labels, class_weights)
            acc_loss = acc_loss + loss
        avg_loss = acc_loss / chunk.len()
        grads    = avg_loss.backward()
        lr_now   = cosine_lr(global_step, total_steps)
        model    = optimizer.step(lr_now, model, grads)
        total_train_loss += avg_loss.into_scalar()
        n_train_steps    += 1
        global_step      += 1

    val_metrics = evaluate(model.valid(), val_block_ids)
    epoch_train_loss = total_train_loss / n_train_steps
    log_epoch_metrics(epoch, epoch_train_loss, val_metrics)

    if epoch % checkpoint_every == 0:
        save_checkpoint(model, epoch, val_metrics.miou, checkpoint_dir, keep_best_n)

best_model = top-val-mIoU checkpoint from checkpoints.json

if swa:
    best_model = swa_average(retained_checkpoints)   // see § SWA

save_model_from_burn(best_model, output_model_path)
```

> **Rayon note:** Data loading for a mini-batch chunk (reading `.feat` + `.lbl` pairs)
> is parallelised via `rayon::iter`.  The model forward pass and gradient accumulation
> are sequential — burn/ndarray autograd state is not thread-safe across concurrent
> forward calls on a shared model.

### Class-Weighted Cross-Entropy Loss (Default)

Unless `--no-class-weights` is passed, class frequencies are computed once from
`class_distribution` aggregated across **training blocks only** (validation blocks
are excluded from frequency counts to prevent leakage):

```
count[c] = sum over training blocks of class_distribution[c]
total    = sum over all c of count[c]
weight[c] = total / (n_classes * count[c])    for count[c] > 0
weight[c] = 0.0                               for count[c] == 0  (class absent)
```

Use `burn::nn::loss::CrossEntropyLossConfig::new().with_weights(weights)`.

### Optimizer

`burn::optim::AdamConfig` with:

| Parameter | CLI flag | Default |
|---|---|---|
| `beta_1` | — | `0.9` |
| `beta_2` | — | `0.999` |
| `epsilon` | — | `1e-8` |
| `weight_decay` | `--weight-decay` | `1e-4` |

### Learning Rate Scheduler

Cosine annealing implemented in `scheduler.rs` — no external dependency:

$$lr(t) = lr_{\min} + \frac{1}{2}(lr_{\max} - lr_{\min})\!\left(1 + \cos\!\left(\frac{\pi \cdot t}{T}\right)\right)$$

| Symbol | Meaning |
|---|---|
| $t$ | Current global step (increments every gradient-accumulation batch) |
| $T$ | Total steps = `n_epochs × ceil(n_train_blocks / batch_size)` |
| $lr_{\max}$ | `--learning-rate` |
| $lr_{\min}$ | `1e-6` (hard-coded floor) |

---

## Checkpoint Management

### Retention Policy

At each checkpoint interval, after saving the new `.wbmodel` file, the manager:

1. Loads `<checkpoint_dir>/checkpoints.json` (creates it if absent).
2. Appends the new entry `{ epoch, val_miou, file }`.
3. Sorts the list by `val_miou` descending.
4. Deletes `.wbmodel` files for any entry beyond position `keep_best_n`.
5. Trims the list and re-writes `checkpoints.json`.

Disk usage is bounded to `keep_best_n` checkpoints regardless of epoch count.

### `checkpoints.json` Format

```json
{
  "keep_best_n": 5,
  "checkpoints": [
    { "epoch": 48, "val_miou": 0.872, "file": "checkpoint_epoch_048.wbmodel" },
    { "epoch": 46, "val_miou": 0.868, "file": "checkpoint_epoch_046.wbmodel" },
    { "epoch": 50, "val_miou": 0.865, "file": "checkpoint_epoch_050.wbmodel" },
    { "epoch": 47, "val_miou": 0.861, "file": "checkpoint_epoch_047.wbmodel" },
    { "epoch": 45, "val_miou": 0.859, "file": "checkpoint_epoch_045.wbmodel" }
  ]
}
```

---

## Stochastic Weight Averaging (SWA)

### Motivation

SWA averages the weights of multiple checkpoints drawn from the convergence region of
the loss landscape, finding a flatter, wider minimum than any single checkpoint and
improving generalisation without additional training.  Retaining 5 checkpoints
(default) provides the natural minimum ensemble for SWA.

### Activation

SWA is off by default.  It is activated by passing `--swa` to `train` and runs after
the main training loop completes, before the final `--output-model` is written.

### Algorithm

```
retained = all checkpoints listed in checkpoints.json (up to keep_best_n)
           loaded via the existing load_model() API from Stage 02

for each parameter tensor (weight, bias, bn_gamma, bn_beta, bn_mean, bn_var):
    avg_param[i] = (1 / |retained|) * sum(param[i] from each loaded model)

swa_model = assemble PointNetClassifier from avg_params
save_model(swa_model, output_model_path)
```

SWA operates entirely on serialised `.wbmodel` files via `load_model()` /
`save_model()` from Stage 02.  **No changes to `weights.rs` or `layers.rs` are
required.**  The bridge is not involved in SWA.

> **BatchNorm running stats:** `bn_mean` and `bn_var` stored in `.wbmodel` are
> exponential-moving-average estimates of per-channel statistics accumulated during
> training.  Averaging them across checkpoints is equivalent to averaging the EMA
> estimates from each checkpoint epoch — this is the correct SWA treatment for
> inference-mode BatchNorm and does not require a separate BN re-calibration pass.

---

## Weight Bridge (`bridge.rs`)

The bridge function signature:

```rust
/// Extract weights from a trained `BurnPointNet` and write a `.wbmodel` file.
///
/// # Errors
/// Returns `ClassifierError::Pipeline` if any extracted tensor shape does not match
/// the dimensions expected from `cfg`.
#[cfg(feature = "training")]
pub fn save_model_from_burn<B: AutodiffBackend>(
    model:  &BurnPointNet<B>,
    cfg:    &PointNetConfig,
    path:   &std::path::Path,
) -> crate::Result<()>
```

### Tensor Extraction Pattern (burn 0.16, pinned)

For each `burn::nn::Linear` layer:

```rust
let w: Vec<f32> = layer.weight.val().into_data().value;
// shape: [out_features, in_features] — row-major, matches .wbmodel convention
let b: Vec<f32> = layer.bias.val().into_data().value;
```

For each `burn::nn::BatchNorm` layer:

```rust
let gamma: Vec<f32> = bn.gamma.val().into_data().value;
let beta:  Vec<f32> = bn.beta.val().into_data().value;
let mean:  Vec<f32> = bn.running_mean.val().into_data().value;
let var:   Vec<f32> = bn.running_var.val().into_data().value;
```

> **Implementation note:** `running_mean` and `running_var` may be wrapped in
> `RunningState<Tensor<B, 1>>` in burn 0.16.  If so, access via `.value` on the
> `RunningState`.  Verify during implementation and record any deviation in
> `stage-03-results.md`.

The extracted `Vec<f32>` slices are assembled into a `PointNetClassifier` by directly
constructing the `Linear` and `BatchNorm1d` structs from `src/model/layers.rs`.  This
is then passed to the existing `save_model()` function from `src/model/weights.rs`.
**No changes to `weights.rs` or `layers.rs` are required.**

### Shape Validation

Before constructing the `PointNetClassifier`, validate:

- Each weight matrix shape `[out, in]` matches the expected dims from `PointNetConfig`.
- Each BN parameter length matches the corresponding layer output dim.
- `label_map.len() == cfg.n_classes`.

Return `ClassifierError::Pipeline(msg)` on any mismatch rather than panicking.

---

## Metrics (`metrics.rs`)

### Accumulated per Validation Pass

For each point across all validation blocks, accumulate per-class TP, FP, FN into
`u64` counters (to avoid overflow on large datasets):

```
for each block in val_set:
    predictions: Vec<u8>  = argmax over logit cols, per point
    ground_truth: Vec<u8> = labels from .lbl

    for (pred, true_label) in zip(predictions, ground_truth):
        if pred == true_label: TP[true_label] += 1
        else:
            FP[pred]        += 1
            FN[true_label]  += 1
        confusion_matrix[true_label][pred] += 1
```

### Formulas

Given per-class counts after accumulation:

$$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

$$\text{mIoU} = \frac{1}{|C'|} \sum_{c \in C'} \text{IoU}_c \quad \text{where } C' = \{c \mid \text{TP}_c + \text{FP}_c + \text{FN}_c > 0\}$$

$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$

$$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

$$\text{F1}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

Classes with `TP + FP + FN == 0` (absent from the validation set) are excluded from
the mIoU and F1_macro averages.

### Output Files

| File | Format | Contents |
|---|---|---|
| `metrics.csv` | CSV, one row per epoch | `epoch, train_loss, val_loss, val_miou, IoU_cls_0, ..., IoU_cls_N, F1_macro` |
| `training_summary.json` | JSON | Best epoch, best val_mIoU, per-class IoU/F1, SWA applied (bool), training duration (seconds) |
| `confusion_matrix.csv` | CSV, N×N | Rows = true class, columns = predicted class; written at end of training using final retained model on full validation set |

---

## Labeled Preprocessing Module (`labeled_pipeline.rs`)

The `preprocess-labeled` command reuses the Stage 01 pipeline with a thin wrapper
(Option A — confirmed):

1. Run `pipeline::run_pipeline(config)` exactly as the `preprocess` command does.
   The pipeline now returns `sampled_indices: Vec<usize>` per block (Stage 01 retroactive
   minor extension to `SampledBlock`; see § Modified from prior stages).
2. For each sampled point, use `sampled_indices[i]` to retrieve its original
   `PointRecord` from the raw per-block point list, read the `classification` byte,
   apply the label remapping, and append to the `.lbl` byte buffer.
3. Write the `.lbl` file (raw `u8[n_points]`).
4. Aggregate `class_distribution` counts from the remapped labels.
5. Assign `macro_tile_id` from the block origin and `--tile-grid` parameters.
6. Write `labeled_blocks.json` (a superset of `blocks.json`).

> **Key constraint:** The feature extraction pipeline must not be duplicated.
> `labeled_pipeline.rs` is an orchestration wrapper — it calls `run_pipeline()` and
> consumes its result.  The single source of truth for feature values is `pipeline.rs`.

The `sampled_indices` extension adds one field to an internal struct and does not
change the public API of `run_pipeline()`.  All 27 existing Stage 01/02 tests must
continue to pass without modification.

---

## Module Configuration Guard

All training module code is wrapped in a feature cfg guard:

```rust
// src/lib.rs addition:
#[cfg(feature = "training")]
pub mod training;
```

```rust
// src/training/mod.rs:
#![cfg(feature = "training")]
```

This ensures that `cargo check` (default, no features) and `cargo test` (default)
never compile burn-dependent code, preserving fast iteration on the inference pipeline.

---

## Definition of Done

| # | Criterion | Verification |
|---|---|---|
| 1 | `cargo build --release --features training` — zero errors | Build gate |
| 2 | `cargo clippy --features training -- -D warnings` — zero crate warnings | Clippy gate |
| 3 | `cargo fmt --check` passes | fmt gate |
| 4 | `cargo build --release` (no features) — zero errors; inference binary unaffected | Inference regression gate |
| 5 | `cargo test` (no features) — all 27 Stage 01/02 tests still pass | Regression gate |
| 6 | Unit: `.lbl` file round-trip — write `u8[N]` labels, read back, bit-identical | `test_lbl_round_trip` |
| 7 | Unit: `labeled_blocks.json` — `lbl_file`, `class_distribution`, `macro_tile_id`, `label_map`, and `spatial_tile_grid` fields all present and correct | `test_labeled_manifest_fields` |
| 8 | Unit: label remapping — ASPRS code absent from map → falls back to Unassigned index | `test_label_remap_unknown_code` |
| 9 | Unit: spatial macro-tile assignment — 16 blocks over a known bounding box → correct `macro_tile_id` per block origin | `test_macro_tile_assignment` |
| 10 | Unit: train/val split — `--val-split 0.25` on a 16-macro-tile dataset withholds exactly 4 spatially-contiguous macro-tiles | `test_spatial_split_fraction` |
| 11 | Unit: explicit `--val-tile-blocks` override — provided block IDs are in val set regardless of `macro_tile_id` | `test_explicit_val_tile_override` |
| 12 | Unit: cosine LR scheduler — `lr(0) == lr_max`, `lr(T) ≈ lr_min`, `lr(T/2) ≈ (lr_max+lr_min)/2` | `test_cosine_schedule_values` |
| 13 | Unit: class weight computation — 3-class training set with known counts → hand-calculated `weight[c]` values match | `test_class_weight_computation` |
| 14 | Unit: mIoU — hand-calculated 3-class reference (TP/FP/FN known) vs `metrics::compute_miou` | `test_miou_three_class` |
| 15 | Unit: absent class exclusion — class with zero TP/FP/FN excluded from mIoU average | `test_miou_absent_class_excluded` |
| 16 | Unit: confusion matrix entries — synthetic 4-point prediction set vs expected 3×3 matrix | `test_confusion_matrix_shape` |
| 17 | Unit: weight bridge round-trip — `BurnPointNet` with constant known weights → bridge → `PointNetClassifier` → `save_model` → `load_model` → weight matrices bit-identical | `test_weight_bridge_round_trip` |
| 18 | Unit: SWA averaging — two `PointNetClassifier` instances with known weights → `swa_average` → all parameters are elementwise mean | `test_swa_averaging` |
| 19 | Unit: checkpoint retention — insert 8 checkpoints with varying mIoU into manager with `keep_best_n = 5` → only top-5 by mIoU retained, rest deleted | `test_checkpoint_keeps_best_n` |
| 20 | Integration: 10-epoch training on ≥ 20 synthetic labeled blocks — validation loss is strictly lower in epoch 10 than epoch 1 | `test_training_converges_synthetic` |
| 21 | Integration: `preprocess-labeled` → `train` (10 epochs) → `classify` — output LAS is valid and classification field has been updated | Manual / CI (requires sample LAS dataset) |
| 22 | CLI: `wb_lidar_train --help` and `wb_lidar_train train --help` print correct usage text | Manual |
| 23 | Performance: 10-epoch training on 500 synthetic labeled blocks (N=1024 each) completes in under 10 minutes single-threaded | Benchmark / manual timing |

---

*This document is the authoritative specification for Stage 03.  No code may be written
for this stage until it has been reviewed and approved.  All implementation deviations
must be recorded in `stage-03-results.md` and this document updated to reflect the
as-built state.*
