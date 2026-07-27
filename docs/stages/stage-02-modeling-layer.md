# Stage 02 — Modeling Layer: PointNet Inference Engine

**Status:** COMPLETE — See [stage-02-results.md](stage-02-results.md) for full development record and deviations
**Approved:** 2026-06-15
**Implemented:** 2026-06-15
**Retroactive extension:** 2026-06-16 — Stage 03 added `#[derive(Clone)]` to `PointNetClassifier` to support SWA weight averaging; see Stage 03 results for rationale
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** GitHub Copilot / AI Collaborator

---

## Goal

Implement the PointNet-style point classification inference engine, the `.wbmodel`
weight file format, a `classify` CLI sub-command, and a streaming LAS/LAZ output
writer.  At the end of this stage the tool can accept a pre-trained model file and a
LiDAR input file and produce a classified LAS/LAZ output file with the
`classification` field updated per the model's predictions.

Training is explicitly **out of scope** for Stage 02.  A `.wbmodel` file is treated
as an external artefact (produced by the Stage 03 training module or an offline
export tool).  Stage 02 proves the inference pipeline is correct using unit-tested
components with synthetically initialised weights.

---

## Inputs & Outputs

### CLI Command

```
wb_lidar_classify classify
    --input   <path>         LAS, LAZ, or COPC source file (same formats as preprocess)
    --model   <path>         Pre-trained model weights file (.wbmodel)
    --blocks  <path>         blocks.json manifest produced by preprocess (same run)
    --output  <path>         Classified output file (.las or .laz)
    [--threads <n>]          Rayon thread pool size (default: system cores)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--input` | `PathBuf` | *required* | Original LiDAR source file |
| `--model` | `PathBuf` | *required* | `.wbmodel` weights file |
| `--blocks` | `PathBuf` | *required* | `blocks.json` manifest from the Stage 01 `preprocess` run on the same file |
| `--output` | `PathBuf` | *required* | Output path; extension determines format (`.las` or `.laz`) |
| `--threads` | `Option<usize>` | system cores | Rayon thread pool override |

**Design note:** `classify` requires a prior `preprocess` run on the same input file.
This two-step design keeps classification fast and stateless: the expensive k-d tree
construction and eigenvalue feature extraction are done once during preprocessing and
cached as `.feat` files.  The inference step consumes those cached features directly.

---

## New File & Module Layout

```
lidar_point_cloud_classifier/src/
  model/
    mod.rs              ← public re-exports; replaces Stage 01 stub
    layers.rs           ← Linear, ReLU, BatchNorm1d (inference-mode), TNet (STN3d/STN64d)
    pointnet.rs         ← PointNetClassifier struct + forward() + classify()
    weights.rs          ← .wbmodel binary format: save / load
    inference.rs        ← block-level inference driver: .feat → per-point labels
  output/
    mod.rs              ← module declaration
    las_writer.rs       ← streaming LAS/LAZ writer with substituted classification
  cli/
    classify_cmd.rs     ← `classify` sub-command argument parser + orchestration
  cli/mod.rs            ← add `classify` branch to dispatch table  [MODIFIED]
docs/stages/
  stage-02-modeling-layer.md   ← this file
  stage-02-results.md          ← implementation record and deviations
```

---

## New Dependency

| Crate | Version | Justification |
|---|---|---|
| `ndarray` | `"0.16"` | N-D array library for matrix operations in the forward pass (pure Rust, no system deps, no unsafe in our code paths) |

No other ML framework crates are introduced.  Inference is hand-rolled over `ndarray`
operations.  The `burn` / `dfdx` decision is deferred to Stage 03 (training module),
where heavier compute infrastructure is justified.

**GPU / hardware acceleration:** The forward pass is parallelised at the **block
level** via Rayon (each block's inference is an independent task).  Within-block
inference is pure `ndarray` matrix multiply — portable and hardware-independent.  A
BLAS-accelerated path (e.g. via `ndarray-linalg + openblas`) can be enabled in a
future stage via a Cargo feature flag without changing any inference code.

---

## Model Architecture

### PointNet Scene Segmentation — Stage 02 Baseline

This is a full PointNet segmentation backbone following Qi et al. (2017), with both
the **Input T-Net (STN3d)** and an optional **Feature T-Net (STN64d)**.  Both T-Nets
are mini-PointNet sub-networks that learn transformation matrices applied to the point
coordinates and intermediate features respectively.  Without the Input T-Net the network
has no mechanism to correct for arbitrary scan orientation — omitting it would degrade
accuracy on any dataset where scan geometry varies between acquisitions.

```
Input:  N × 17   (N sampled points, 17 features each — from .feat file)

── Input T-Net (STN3d) ────────────────────────────────────────────────────────
  Extract cols [0,1,2]: N × 3 (x_norm, y_norm, z_norm)
  Mini-encoder (shared MLP, fixed dims):
    Linear(3 → 64)    → BN → ReLU
    Linear(64 → 128)  → BN → ReLU
    Linear(128 → 1024)→ BN → ReLU
  Global max pool: (N, 1024) → (1024,)
  FC decoder:
    Linear(1024 → 512) → BN → ReLU
    Linear(512 → 256)  → BN → ReLU
    Linear(256 → 9)    → reshape to 3×3, then += I₃  (identity initialisation)
  Apply: xyz_cols = xyz_cols @ T1ᵀ
  Replace cols [0,1,2] in feature matrix  →  N × 17 (unchanged shape)

── Encoder Layer 0 ────────────────────────────────────────────────────────────
  Linear(17 → 64) → BN → ReLU
  → local_feat: N × 64        ← saved; will be concatenated with global descriptor

── Feature T-Net (STN64d, enabled when `use_feature_tnet = true`) ─────────────
  Mini-encoder on local_feat:
    Linear(64 → 64)   → BN → ReLU
    Linear(64 → 128)  → BN → ReLU
    Linear(128 → 1024)→ BN → ReLU
  Global max pool: → (1024,)
  FC decoder:
    Linear(1024 → 512) → BN → ReLU
    Linear(512 → 256)  → BN → ReLU
    Linear(256 → 4096) → reshape to 64×64, then += I₆₄
  Apply: local_feat = local_feat @ T2ᵀ  →  N × 64

── Encoder Layers 1+ ──────────────────────────────────────────────────────────
  Linear(64 → 64)   → BN → ReLU
  Linear(64 → 64)   → BN → ReLU
  Linear(64 → 128)  → BN → ReLU
  Linear(128 → 1024)→ BN → ReLU
  → deep_feat: N × 1024

── Global Max Pooling ─────────────────────────────────────────────────────────
  MaxPool over N dimension  →  (1024,) global descriptor
  Broadcast: repeat to N × 1024

── Segmentation Concat ────────────────────────────────────────────────────────
  Concat(local_feat: N × 64, global: N × 1024)  →  N × 1088
  Note: local_feat here is the (optionally T2-transformed) 64-dim features from
  Encoder Layer 0, not the deep encoder output.  This is the architecturally
  correct formulation from the PointNet segmentation paper.

── Shared MLP Decoder (applied independently to each of the N points) ─────────
  Linear(1088 → 512) → BN → ReLU
  Linear(512 → 256)  → BN → ReLU
  Linear(256 → n_classes)   ← no activation; raw logits

── Output ─────────────────────────────────────────────────────────────────────
  Argmax over class dimension  →  N class indices  (u8 ASPRS codes via label map)
```

**Why 1088, not 2048?**  Concatenating `local_feat (64) + global (1024) = 1088`.  A
naive (incorrect) implementation might concatenate the deep encoder output with its
own global max-pooled version (1024+1024=2048), which adds no new information beyond
what the global vector already contains. Concatenating the shallow local features
(post T-Net, pre deep encoder) with the global context is what gives PointNet its
local-global reasoning ability — this is the canonical formulation from the PointNet
segmentation paper (Qi et al., 2017). See Stage 43 for the restoration of these
canonical dimensions; the encoder/decoder had previously been reduced to a
non-canonical `[64, 128, 256]` / `[256, 128]` shape (256-dim global descriptor,
320-dim concat) as an undocumented deviation from Stage 02 through Stage 42.

**T-Net architecture is fixed** (not configurable) — the mini-encoder/decoder dims
`[64, 128, 1024, 512, 256]` are canonical PointNet values and are not stored in the
model header.  Only the `use_feature_tnet` flag controls whether the STN64d block is
present in the weight file.

**As-built note on `relu` / `relu_1d` signatures:**
Both functions take `&Array` references rather than owned values (see deviation #1 in
[stage-02-results.md](stage-02-results.md)).  Call sites use `relu(&h)` not `relu(h)`.

**As-built note on `apply_bn` helper signatures:**
`apply_bn2d` and `apply_bn1d` take `Option<&BatchNorm1d>` rather than `&Option<BatchNorm1d>`
(see deviation #2).  Call sites use `.as_ref()` on the stored `Option` field.



| Parameter | Default | Description |
|---|---|---|
| `n_features_in` | `17` | Input feature count (must match Stage 01 `N_FEATURES`; expanded from 12 to 17 in Stage 30) |
| `encoder_dims` | `[64, 64, 64, 128, 1024]` | Hidden units for each encoder MLP layer (canonical PointNet dims, restored in Stage 43); `encoder_dims[0]` (64) is the local feature dim used in the segmentation concat |
| `decoder_dims` | `[512, 256]` | Hidden units for each decoder MLP layer (before the final class projection; canonical PointNet dims, restored in Stage 43); decoder input dim = `encoder_dims[0] + encoder_dims.last()` = 64+1024 = 1088 |
| `n_classes` | configurable | Number of output classes |
| `use_batch_norm` | `true` | When `false`, BatchNorm layers are identity (useful for unit-test scenarios) |
| `use_input_tnet` | `true` | Must be `true` for any correctly trained PointNet model; `false` disables STN3d and is only valid for debugging |
| `use_feature_tnet` | `false` | When `true`, STN64d is applied to the 64-dim encoder Layer 0 output |

### ASPRS Class Label Mapping

The `.wbmodel` file encodes a `label_map: Vec<u8>` of length `n_classes` that maps
model output index → ASPRS classification code written to the output LAS file.

**Default 8-class mapping:**

| Model index | ASPRS code | Meaning |
|---|---|---|
| 0 | 1 | Unassigned |
| 1 | 2 | Ground |
| 2 | 3 | Low Vegetation |
| 3 | 4 | Medium Vegetation |
| 4 | 5 | High Vegetation |
| 5 | 6 | Building |
| 6 | 9 | Water |
| 7 | 7 | Low Point (noise) |

---

## `.wbmodel` Binary Format

### Layer Block Convention

A **layer block** is the repeated unit used for every Linear layer:
```
  weight:    f32[dim_out × dim_in]   row-major LE
  bias:      f32[dim_out]            LE
  [if layer has BN:]
    bn_gamma:  f32[dim_out]
    bn_beta:   f32[dim_out]
    bn_mean:   f32[dim_out]
    bn_var:    f32[dim_out]
```

### File Layout

```
[Header — variable length]
  magic:             4 bytes  = b"WBML"
  version:           u8       = 1
  n_features_in:     u8       (must equal N_FEATURES = 17 from Stage 01/Stage 30)
  n_encoder_layers:  u8       (length of encoder_dims array, e.g. 5 for [64,64,64,128,1024])
  n_decoder_layers:  u8       (length of decoder_dims array, e.g. 2 for [512,256])
  n_classes:         u8
  use_batch_norm:    u8       (0 = false, 1 = true)
  use_input_tnet:    u8       (0 = disabled, 1 = STN3d present; must be 1 for valid models)
  use_feature_tnet:  u8       (0 = disabled, 1 = STN64d present)
  reserved:          1 byte   = 0x00
  encoder_dims:      u16 × n_encoder_layers  (LE)
  decoder_dims:      u16 × n_decoder_layers  (LE)
  label_map:         u8  × n_classes

[Input T-Net (STN3d) weight blocks — only present when use_input_tnet = 1]
  The STN3d architecture is FIXED.  Dims are not stored; the parser derives them.
  Mini-encoder (input_dim=3, fixed dims [64, 128, 1024]) — 3 layer blocks:
    Layer block: Linear(3→64),    BN if use_batch_norm
    Layer block: Linear(64→128),  BN if use_batch_norm
    Layer block: Linear(128→1024),BN if use_batch_norm
  FC decoder (fixed dims [512, 256, 9]) — 3 layer blocks:
    Layer block: Linear(1024→512),BN if use_batch_norm
    Layer block: Linear(512→256), BN if use_batch_norm
    Layer block: Linear(256→9),   NO BN (final projection)

[Feature T-Net (STN64d) weight blocks — only present when use_feature_tnet = 1]
  Same structure as STN3d but input_dim=64 and final projection outputs 4096 (64×64).
  Mini-encoder (input_dim=64, fixed dims [64, 128, 1024]) — 3 layer blocks
  FC decoder (fixed dims [512, 256, 4096]) — 3 layer blocks (no BN on last)

[Main encoder weight blocks — one per encoder layer in order]
  For each encoder layer i (0..n_encoder_layers):
    Layer block with BN if use_batch_norm
    (layer 0: Linear(n_features_in → encoder_dims[0]))
    (layer k: Linear(encoder_dims[k-1] → encoder_dims[k]))

[Main decoder weight blocks — one per decoder layer + final projection]
  For each decoder layer j (0..n_decoder_layers):
    Layer block with BN if use_batch_norm
    (layer 0 input_dim = encoder_dims[0] + encoder_dims.last()  ← the concat dim)
    (layer k: Linear(decoder_dims[k-1] → decoder_dims[k]))
  Final projection layer (no BN):
    Layer block: Linear(decoder_dims.last() → n_classes), NO BN
```

All multi-byte values are **little-endian**.  The final decoder projection layer never
has BatchNorm regardless of `use_batch_norm`.  T-Net final projection layers also never
have BatchNorm.

---

## Algorithmic Steps

### Step 1 — Load Model

Parse `.wbmodel` header, validate magic and version, read architecture config, deserialise
all weight tensors into `PointNetClassifier`.  Return `ClassifierError::Pipeline` on any
format mismatch (wrong magic, wrong `n_features_in`, truncated tensor, etc.).

### Step 2 — Load Block Manifest

Deserialise `blocks.json` into `BlockManifest`.  Extract block grid geometry
(`block_size`, `origin_x`/`origin_y` per block).  Build a lookup table:
`HashMap<u64, BlockMeta>` keyed by block ID.

### Step 3 — Per-Block Parallel Inference (Rayon)

For each block in `manifest.blocks`, in parallel:

**(a) Load `.feat` file**

Parse the WBFT header (magic `b"WBFT"`, version, `n_points`, `n_features`, `block_id`,
`origin_x`, `origin_y`).  Validate `n_features == N_FEATURES`.  Read
`n_points × n_features` f32 values into an `ndarray::Array2<f32>` with shape
`[n_points, N_FEATURES]`.

**(b) Run PointNet Forward Pass**

```
features: Array2<f32> shape [N, 17]

// ── Input T-Net ──────────────────────────────────────────────────────────────
if use_input_tnet:
  xyz = features.slice(s![.., 0..3]).to_owned()          // [N, 3]
  // mini-encoder
  h = xyz
  for each (W, b, opt_bn) in stn3d_encoder_layers:        // 3 layers [3→64→128→1024]
    h = h.dot(&W.T) + &b
    if opt_bn: h = bn(h)
    h = h.mapv(|x| x.max(0.0))
  // global max pool
  g = h.fold_axis(Axis(0), f32::NEG_INFINITY, f32::max)  // [1024]
  // FC decoder
  for (i, (W, b, opt_bn)) in stn3d_fc_layers.enumerate(): // [1024→512→256→9]
    g = W.dot(&g) + &b
    if opt_bn && i < 2: g = bn_1d(g)                      // no BN on final FC
    if i < 2: g = g.mapv(|x| x.max(0.0))
  // reshape to 3×3, add identity
  T1 = g.into_shape([3, 3]) + Array2::eye(3)              // [3, 3]
  // apply transform: each point's xyz = T1 @ xyz_i  ≡  xyz_mat @ T1.T
  features.slice_mut(s![.., 0..3]).assign(&xyz.dot(&T1.t()))

// ── Encoder Layer 0 ──────────────────────────────────────────────────────────
local_feat = features.dot(&W0.T) + &b0                    // [N, 64]  (input dim 17)
if use_batch_norm: local_feat = bn(local_feat)
local_feat = local_feat.mapv(|x| x.max(0.0))

// ── Feature T-Net (optional) ─────────────────────────────────────────────────
if use_feature_tnet:
  h = local_feat.clone()                                  // [N, 64]
  for each (W, b, opt_bn) in stn64d_encoder_layers:       // [64→64→128→1024]
    h = h.dot(&W.T) + &b
    if opt_bn: h = bn(h)
    h = h.mapv(|x| x.max(0.0))
  g = h.fold_axis(Axis(0), f32::NEG_INFINITY, f32::max)   // [1024]
  for (i, (W, b, opt_bn)) in stn64d_fc_layers.enumerate():
    g = W.dot(&g) + &b
    if opt_bn && i < 2: g = bn_1d(g)
    if i < 2: g = g.mapv(|x| x.max(0.0))
  T2 = g.into_shape([64, 64]) + Array2::eye(64)           // [64, 64]
  local_feat = local_feat.dot(&T2.t())

// ── Encoder Layers 1+ ────────────────────────────────────────────────────────
deep = local_feat.clone()
for each (W, b, opt_bn) in encoder_layers[1..]:           // [64→64→64→128→1024]
  deep = deep.dot(&W.T) + &b
  if opt_bn: deep = bn(deep)
  deep = deep.mapv(|x| x.max(0.0))

// ── Global Max Pooling ───────────────────────────────────────────────────────
global_vec = deep.fold_axis(Axis(0), f32::NEG_INFINITY, f32::max)  // [1024]
global_mat = global_vec broadcast to [N, 1024]

// ── Segmentation Concat ──────────────────────────────────────────────────────
seg_in = concatenate(Axis(1), &[local_feat.view(), global_mat.view()])  // [N, 1088]

// ── Decoder ──────────────────────────────────────────────────────────────────
for (j, (W, b, opt_bn)) in decoder_layers.enumerate():    // [1088→512→256→n_classes]
  seg_in = seg_in.dot(&W.T) + &b
  if opt_bn && j < n_decoder_layers: seg_in = bn(seg_in)  // no BN on final
  if j < n_decoder_layers: seg_in = seg_in.mapv(|x| x.max(0.0))

Logits: seg_in shape [N, n_classes]
Labels: argmax over axis 1 → [N] usize → map via label_map → Vec<u8> ASPRS codes
```

**(c) Reconstruct Sampled-Point Coordinates**

From the feature matrix, reconstruct approximate (x, y) for each sampled point:

```
x_approx = feat[i, 0] as f64 * block_size + origin_x
y_approx = feat[i, 1] as f64 * block_size + origin_y
```

These coordinates have at most `block_size / 65535` error (f32 quantization within
block).  For label-assignment purposes (nearest-neighbor at ≤ 50 m block scale) this
is negligible.

**(d) Build Per-Block 2-D Spatial Index**

Construct a 2-D k-d tree over the N reconstructed (x_approx, y_approx) pairs.  Store
alongside the N ASPRS labels as `BlockInferenceResult { kdtree, labels }`.

**(e) Store Result**

Insert into a `HashMap<u64, BlockInferenceResult>` (protected by `Mutex` during
parallel insert, read-only thereafter).

### Step 4 — Stream Original LAS/LAZ → Write Classified Output

**As-built note on `write_classified` signature:**
`write_classified` is generic over the hasher `S: BuildHasher` so the public API is not
locked to `RandomState` (see deviation #3 in [stage-02-results.md](stage-02-results.md)).

**As-built note on nearest-neighbor strategy:**
`BlockInferenceResult::nearest_label` uses an O(N) linear scan instead of a 2-D k-d tree.
For `N ≤ 4096` sampled points per block, the linear scan is faster in practice than
allocating and querying a `kdtree` instance (see deviation #8).

**As-built note on `infer_stream_writer_config_from_source`:**
Confirmed private in `wblidar::frontend` — not re-exported. Reproduced inline in
`output/las_writer.rs::infer_writer_config()` using the public `LasReader` API
(see deviation #5).



For each original `PointRecord` in stream order:

1. Compute block `(col, row)` from `(x, y)` using the same formula as Stage 01:

   ```
   col = floor((x - x_min) / block_size)
   row = floor((y - y_min) / block_size)
   block_id = row * grid_cols + col
   ```

   where `x_min`, `y_min`, `block_size`, `grid_cols` are read from `blocks.json`.

2. Look up `BlockInferenceResult` for `block_id`.

3. If found: query the 2-D k-d tree with `(x, y)` → get index of nearest sampled
   point → look up `labels[index]` → assign to `point.classification`.

4. If **not found** (point fell in a block that was filtered out during preprocessing):
   leave `point.classification` unchanged (preserve original value).

5. Write the (modified or unmodified) `PointRecord` to the output writer.

After EOF: call `writer.finish()` to back-patch the LAS header with final point count
and bounding box.

### Step 5 — Log Summary

Emit a progress summary to stderr:

```
[classify] blocks processed: 1234  |  points written: 12_450_123
[classify] output: /path/to/classified.laz
```

---

## Module Responsibilities

| Module | Responsibility | Key Public API |
|---|---|---|
| `model/layers` | `Linear`, `BatchNorm1d` (inference), `relu(&Array)`, `relu_1d(&Array)`, `global_max_pool`; `Stn3d`, `Stn64d` T-Net sub-networks | Pure functions / structs; no mutable state after construction |
| `model/pointnet` | Assemble T-Nets + encoder + decoder layers; run full forward pass + `classify()` | `PointNetClassifier::new(config)`, `::forward(features) -> Array2<f32>`, `::classify(features) -> Vec<u8>` |
| `model/weights` | `.wbmodel` binary serialisation / deserialisation including T-Net weight blocks | `save_model(path, model) -> Result<()>`, `load_model(path) -> Result<PointNetClassifier>` |
| `model/inference` | Block-level driver: load `.feat` → run model → build `BlockInferenceResult`; O(N) nearest-label | `run_inference(manifest, &model, feat_dir) -> Result<HashMap<u64, BlockInferenceResult>>` |
| `output/las_writer` | Stream original LAS/LAZ, substitute classification, write output | `write_classified<S>(input, output, inference_map, manifest) -> Result<()>` |
| `cli/classify_cmd` | Argument parsing + pipeline orchestration | `run(&args) -> Result<()>` |

---

## Performance Guardrails

- **Parallel inference:** Each block's forward pass is an independent Rayon task.
  All weight tensors are read-only after loading and shared via `Arc<PointNetClassifier>`.
- **No panics in production:** All `ndarray` index operations are bounds-checked via
  `Result`-returning APIs or pre-validated shapes at load time.  `unwrap()` / `expect()`
  are forbidden in `model/` and `output/` code paths.
- **Memory ceiling:** `BlockInferenceResult` stores `N × (16 + 1)` bytes per block
  (2D coordinates × f32 + 1 label byte).  For 10,000 blocks × 1,024 points this is
  ≈ 170 MB — within the 4 GB target from PROJECT_SPEC.  If a dataset exceeds safe
  memory limits, the HashMap can be spilled to disk using the same mechanism as Stage
  01's block partitioner (future optimization, not required for Stage 02 DoD).
- **Output fidelity:** All non-classification fields in `PointRecord` are written
  without modification.  VLRs, CRS, scale/offset, GPS time, colour, extra bytes are
  all preserved via `infer_stream_writer_config_from_source`.
- **wblidar write API notes** (discovered Stage 01):
  - `infer_stream_writer_config_from_source(input, output)` is available in
    `wblidar::frontend` (not re-exported from `wblidar::` root — call via the
    internal API or mirror its logic).  **Verify public exposure before implementation.**
  - `LasWriter<BufWriter<File>>` / `LazWriter<BufWriter<File>>` both implement
    `PointWriter` with `write_point(&mut self, p: &PointRecord) -> Result<()>` and
    `finish(&mut self) -> Result<()>`.
  - Output format is determined by the `--output` path extension (`.las` or `.laz`).

---

## Definition of Done

| # | Criterion | Verification Method | Status |
|---|---|---|---|
| 1 | `cargo build --release` succeeds | Build | ⏳ |
| 2 | `cargo clippy -- -D warnings` zero warnings | CI | ⏳ |
| 3 | `cargo fmt --check` passes | CI | ⏳ |
| 4 | Unit: `Linear::forward` output shape and value correctness (hand-calculated 3×2 weight matrix) | `cargo test` | ✅ Pass |
| 5 | Unit: `BatchNorm1d` inference-mode produces correct normalised output vs hand-calculated reference | `cargo test` | ✅ Pass |
| 6 | Unit: `relu` zeros all negative values, passes positive values unchanged | `cargo test` | ✅ Pass |
| 7 | Unit: `global_max_pool` returns correct column-wise maximum for a 4×3 synthetic matrix | `cargo test` | ✅ Pass |
| 8 | Unit: `Stn3d::forward` on a `[N, 3]` input returns a `3×3` matrix that, when applied, leaves an already-canonical point cloud numerically close to identity (identity-initialised weights + zero net → pure identity transform) | `cargo test` | ✅ Pass |
| 9 | Unit: `Stn3d::forward` on a known rotated `[N, 3]` synthetic set with hand-crafted weights produces the expected corrective rotation matrix | `cargo test` | ✅ Pass (see deviation #4 in results) |
| 10 | Unit: `Stn64d::forward` output shape is `[64, 64]` for any `[N, 64]` input | `cargo test` | ✅ Pass |
| 11 | Unit: full `PointNetClassifier::forward` with `use_input_tnet=true`, `use_feature_tnet=false` on a `[1024, 17]` input produces output shape `[1024, n_classes]` | `cargo test` | ✅ Pass |
| 12 | Unit: full `PointNetClassifier::forward` with both T-Nets enabled on a `[1024, 17]` input produces output shape `[1024, n_classes]` | `cargo test` | ✅ Pass |
| 13 | Unit: `.wbmodel` round-trip (with T-Nets enabled) — save a randomly initialised model, reload it, run identical input, verify bit-identical output | `cargo test` | ✅ Pass |
| 14 | Unit: label mapping — argmax of known logit array maps to correct ASPRS codes via `label_map` | `cargo test` | ✅ Pass |
| 15 | Unit: `write_classified` — given a 100-point synthetic LAS stream and a pre-populated `BlockInferenceResult`, output LAS contains correct classification values and all other fields are unchanged | `cargo test` | ✅ Pass |
| 16 | Unit: nearest-neighbor label assignment — point placed at sampled-point coordinates returns exact label; point offset by ε returns nearest label | `cargo test` | ✅ Pass |
| 17 | CLI: `wb_lidar_classify classify --help` prints correct usage with all parameters | Manual | ⏳ Deferred |
| 18 | Integration: `preprocess` → `classify` → valid LAS output | Manual (requires sample LAS dataset, deferred) | ⏳ Deferred |

---

## Open Items / Deferred Decisions

1. **wblidar `infer_stream_writer_config_from_source` visibility:** ~~Unconfirmed—verify
   before implementation.~~ **RESOLVED:** Function is private (`fn`, not `pub fn`).
   Reproduced inline in `output/las_writer.rs::infer_writer_config()` using the public
   `LasReader` API. See deviation #5 in [stage-02-results.md](stage-02-results.md).

2. **COPC output:** The `classify` output is constrained to `.las` / `.laz` (same
   as Stage 01's write-path).  COPC write support in wblidar is not confirmed — avoid
   claiming COPC output support until the wblidar API is verified.

3. **Points outside all blocks:** Points that fell below `--min-density` during
   preprocessing have no `BlockInferenceResult`.  Stage 02 preserves their original
   classification value.  A future stage may offer a `--classify-sparse-as <class>`
   flag.

4. **Stage 03 — Training Module:** Defines how a labeled LiDAR dataset is converted
   to a `.wbmodel` file.  Candidate approaches: `burn` with `ndarray` backend (fully
   pure Rust); external PyTorch training with a Rust weight exporter to `.wbmodel`
   format.  The weight format defined in Stage 02 must remain the authoritative format.

5. **BatchNorm at inference without training statistics:** The `use_batch_norm = false`
   path is provided for unit-testing with random weights where running statistics are
   meaningless.  A real trained model will always have `use_batch_norm = true` with
   valid running mean/variance from training.
