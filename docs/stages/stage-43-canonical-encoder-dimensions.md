\
# Stage 43 — Restore Canonical PointNet Encoder Dimensions

**Status:** COMPLETE — implemented and verified; all 15 Definition of Done
items confirmed (see table below). `PointNetClassifier` and `BurnPointNet`
both use the canonical `[64, 64, 64, 128, 1024]` / `[512, 256]` dims via the
shared `CANONICAL_ENCODER_DIMS` / `CANONICAL_DECODER_DIMS` constants; all
build/test/clippy/fmt gates pass (174/174 tests, zero clippy warnings with
`-D warnings`, clean `cargo fmt --check`); `docs/stages/stage-02-modeling-layer.md`
and `docs/user/user_guide.md` §7 are fully reconciled.
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Revision:** v2 — corrects a factual error in the original draft's Step 2
(`BurnPointNet` was assumed to already use `Vec` fields; it does not — see
"Revision history" below) and closes several related documentation gaps.

**Relates to:** `docs/stages/stage-02-modeling-layer.md`,
`src/model/pointnet.rs`, `src/model/layers.rs`, `src/model/weights.rs`,
`src/training/burn_model.rs`, `src/training/bridge.rs`,
`src/training/trainer.rs`, `src/cli/evaluate_cmd.rs`,
`src/cli/fix_label_map_cmd.rs`

---

## Goal

Restore the main encoder and decoder dimensions to match the canonical PointNet
segmentation architecture (Qi et al., 2017), correcting an undocumented design
deviation present since Stage 02.

### Background

The original Stage 02 spec defined the main encoder as `[64, 128, 256]` and the
main decoder as `[256, 128]`, producing a 256-dimensional global feature vector
and a 320-dimensional segmentation context vector. This was a **substantial
reduction** from the canonical PointNet architecture, which uses a 1024-dimensional
global feature vector and a 1088-dimensional segmentation context. The deviation
was never documented, discussed, or approved — it appears to have been an
undocumented design choice made during the initial spec writing.

The T-Net sub-networks (STN3d and STN64d) were **not** affected by this deviation
— they have always used the canonical internal dimensions
(`[3→64→128→1024]` encoder, `[1024→512→256→9]` decoder for STN3d).

### What changes

| Component | Current (deviated) | Canonical (target) |
|-----------|-------------------|-------------------|
| Main encoder dims | `[64, 128, 256]` | `[64, 64, 64, 128, 1024]` |
| Main encoder layer count | 3 | 5 |
| Global descriptor dim | 256 | 1024 |
| Segmentation concat dim | 64 + 256 = 320 | 64 + 1024 = 1088 |
| Main decoder dims | `[256, 128]` | `[512, 256]` |
| Main decoder layer count | 2 | 2 (unchanged) |
| Final projection input dim | 128 | 256 |

### Breaking change

This is a **breaking architecture change**. Every `.wbmodel` file ever produced
by this codebase uses the old 256-dim global descriptor and is incompatible with
the new architecture. All models must be retrained. This is accepted — the user
is currently performing hyperparameter testing and has confirmed retraining is
not a blocker.

### Revision history

- **v1** (superseded): Step 2 incorrectly claimed `BurnPointNet`'s encoder and
  decoder fields were "already" dynamically-sized `Vec` fields, requiring only a
  `new()` update. This is **false** — `BurnPointNet<B>` currently declares 3
  hardcoded named encoder fields (`enc0`/`enc1`/`enc2` + matching `BatchNorm`
  pairs) and 2 hardcoded named decoder fields (`dec0`/`dec1`), with `forward()`
  and `forward_batched()` both hand-unrolling the exact 3-step / 2-step chain.
  Growing the encoder to 5 layers is a genuine **struct refactor**, not a
  parameter change. v1 also referenced a nonexistent `PointNetConfig::default()`
  and understated the required changes to `training/bridge.rs`. v1's
  Performance Considerations table also contained inaccurate parameter-count
  arithmetic. All of these are corrected in this revision (v2).

---

## Inputs & Outputs

### No CLI changes

No new CLI flags are introduced. No old flags are removed. The `PointNetConfig`
struct's `encoder_dims` and `decoder_dims` fields remain `Vec<usize>`; only the
values used to construct them at the production call site (and in test
fixtures) change. The `.wbmodel` binary format's header fields
(`n_encoder_layers`, `n_decoder_layers`, `encoder_dims[]`, `decoder_dims[]`)
are unchanged in structure — only the stored values change, and both are
already read dynamically by length, so no format code changes are needed.

### Affected files

| File | Change |
|------|--------|
| `src/model/pointnet.rs` | Add `CANONICAL_ENCODER_DIMS` / `CANONICAL_DECODER_DIMS` constants; add `impl Default for PointNetConfig` using them; `concat_dim()` needs no code change (already generic) |
| `src/model/layers.rs` | No structural changes (T-Nets already canonical) |
| `src/model/weights.rs` | No structural changes (format already stores dims dynamically) |
| `src/training/burn_model.rs` | **Structural refactor**: convert `BurnPointNet`'s `enc0..enc2`/`bn_enc0..2` fields to a single `Vec<(nn::Linear<B>, nn::BatchNorm<B,1>)>` field (and, for consistency, `dec0`/`dec1`/`bn_dec0..1` to a matching decoder `Vec`); rewrite `new()`, `forward()`, `forward_batched()` to loop over the `Vec` instead of hand-unrolling a fixed number of calls; update `default_cfg()` test fixture to reference the shared constants |
| `src/training/bridge.rs` | **Structural change**: `save_model_from_burn`'s hardcoded `vec![extract_pair(&model.enc0, ...), extract_pair(&model.enc1, ...), extract_pair(&model.enc2, ...)]` must become a loop over `model.encoder_layers` (and similarly for the decoder `Vec`, if refactored); update `default_cfg()` test fixture to reference the shared constants |
| `src/training/trainer.rs` | Update the production `TrainConfig`→`PointNetConfig` construction to reference the shared constants instead of a literal `vec![64, 128, 256]` / `vec![256, 128]`; update the `test_swa_averages_tnet_weights` test fixture (currently a separate, spec-unlisted duplicate of the same literal) to reference the constants too |
| `src/cli/evaluate_cmd.rs` | No change — test fixture uses small unrelated dims (`[4, 8]`/`[4]`) on a raw `PointNetClassifier`, not tied to the canonical/deviated production values |
| `src/cli/fix_label_map_cmd.rs` | No change — same rationale as `evaluate_cmd.rs` |
| `docs/stages/stage-02-modeling-layer.md` | Full reconciliation pass: update architecture diagram, config table, forward-pass pseudocode, `.wbmodel` format examples, and **all** remaining stale `N × 12` / `N_FEATURES = 12` references (pre-existing drift from Stage 30's 17-feature expansion, not just the encoder dims) |
| `docs/user/user_guide.md` | Update architecture diagram in §7 |

---

## Steps & Specifications

### Step 1 — Add canonical dimension constants and a `Default` impl

`PointNetConfig` currently has **no** `Default` implementation; the "default"
encoder/decoder dims are four independently duplicated literal
`vec![64, 128, 256]` / `vec![256, 128]` declarations scattered across
`trainer.rs` (production path), `trainer.rs` (`test_swa_averages_tnet_weights`),
`bridge.rs` (`default_cfg()` test fixture), and `burn_model.rs`
(`default_cfg()` test fixture). This duplication is exactly the kind of
undocumented-deviation risk Stage 43 exists to close, so this stage also fixes
the root cause by introducing a single source of truth.

In `src/model/pointnet.rs`, add:

```rust
/// Canonical PointNet (Qi et al. 2017) main-encoder hidden dims.
/// `encoder_dims[0]` (64) is the "local feature" width used in the
/// segmentation concat; `encoder_dims.last()` (1024) is the global
/// descriptor width.
pub const CANONICAL_ENCODER_DIMS: [usize; 5] = [64, 64, 64, 128, 1024];

/// Canonical PointNet (Qi et al. 2017) main-decoder hidden dims (before the
/// final class-projection layer).
pub const CANONICAL_DECODER_DIMS: [usize; 2] = [512, 256];
```

`PointNetConfig` itself has no other defaultable fields with an unambiguous
"correct" value (`n_features_in`, `n_classes` are dataset-dependent), so a full
`impl Default for PointNetConfig` is **not** added — instead, all four call
sites in Step 3 construct their `PointNetConfig` using
`CANONICAL_ENCODER_DIMS.to_vec()` / `CANONICAL_DECODER_DIMS.to_vec()` directly.
This avoids implying a misleading "default" `PointNetConfig::default()` for
fields that have no sensible default.

`concat_dim()` is already generic — it computes `encoder_dims[0] +
encoder_dims.last()` — so it automatically produces the correct value (1088)
with no code change. Confirmed: `self.encoder_dims.first()` / `.last()` are
used, not a hardcoded index.

The `n_encoder_layers` / `n_decoder_layers` fields in `weights.rs` are already
read dynamically as `Vec` lengths (not hardcoded constants), so no format
changes are needed there.

### Step 2 — Refactor `BurnPointNet` to a dynamically-sized encoder (and decoder)

**Current state (verified against `src/training/burn_model.rs`):**
`BurnPointNet<B>` declares exactly 3 hardcoded named encoder fields and their
matching `BatchNorm` pairs:

```rust
pub enc0: nn::Linear<B>,
pub bn_enc0: nn::BatchNorm<B, 1>,
pub enc1: nn::Linear<B>,
pub bn_enc1: nn::BatchNorm<B, 1>,
pub enc2: nn::Linear<B>,
pub bn_enc2: nn::BatchNorm<B, 1>,
```

and 2 hardcoded named decoder fields:

```rust
pub dec0: nn::Linear<B>,
pub bn_dec0: nn::BatchNorm<B, 1>,
pub dec1: nn::Linear<B>,
pub bn_dec1: nn::BatchNorm<B, 1>,
```

`forward()` and `forward_batched()` both hand-unroll exactly this 3-step
encoder chain and 2-step decoder chain (`self.enc0.forward(...)` →
`self.enc1.forward(...)` → `self.enc2.forward(...)`, etc.). Growing the
encoder to 5 canonical layers (`[64, 64, 64, 128, 1024]`) with this structure
would require adding 2 more pairs of hardcoded named fields and 2 more
hand-unrolled calls — this works but re-introduces the same
"architecture-hardcoded-in-multiple-places" pattern Stage 43 is trying to
eliminate, and permanently couples the number of layers to the struct
definition. Since the decoder happens to stay at exactly 2 layers under the
canonical target, refactoring it is not strictly required by this stage, but
is done anyway **for consistency and future-proofing** (so the encoder and
decoder use the same pattern, and a future dimension change doesn't require
another structural refactor).

**Target structure:**

```rust
#[derive(Module, Debug)]
pub struct BurnPointNet<B: Backend> {
    pub stn3d: Stn3d<B>,
    pub stn64d: Option<Stn64d<B>>,

    /// Main encoder layers, in order. `encoder_layers[0]` produces
    /// `local_feat` (the segmentation-concat "local" branch); the remaining
    /// layers produce the deep/global branch. Length == `cfg.encoder_dims.len()`.
    pub encoder_layers: Vec<(nn::Linear<B>, nn::BatchNorm<B, 1>)>,

    /// Main decoder layers, in order. Length == `cfg.decoder_dims.len()`.
    pub decoder_layers: Vec<(nn::Linear<B>, nn::BatchNorm<B, 1>)>,

    /// Final projection: `decoder_dims.last() → n_classes` (no BN).
    pub proj: nn::Linear<B>,
}
```

burn's `#[derive(Module)]` macro supports `Vec<M: Module<B>>` fields (a
standard, documented burn usage pattern for variable-depth networks) — this
should be confirmed with a quick `cargo build --features training` check
immediately after the struct change, before writing the rest of the refactor,
to fail fast if this assumption is wrong.

**`BurnPointNet::new()`** must build the `Vec` dynamically from
`cfg.encoder_dims` / `cfg.decoder_dims`, mirroring the existing
`PointNetClassifier` (ndarray) construction pattern already used in
`model/pointnet.rs`'s test helper `make_classifier()`:

```rust
let mut encoder_layers = Vec::with_capacity(cfg.encoder_dims.len());
let mut prev = cfg.n_features_in;
for &dim in &cfg.encoder_dims {
    encoder_layers.push((
        LinearConfig::new(prev, dim).init(device),
        BatchNormConfig::new(dim).init(device),
    ));
    prev = dim;
}
// (analogous loop for decoder_layers, starting from `concat_dim`)
```

The existing minimum-length validation is preserved (still guards against
degenerate configs, just no longer tied to an exact count of 3/2):

```rust
if cfg.encoder_dims.is_empty() {
    return Err(ClassifierError::Pipeline(
        "BurnPointNet requires at least 1 encoder_dims entry".into(),
    ));
}
if cfg.decoder_dims.is_empty() {
    return Err(ClassifierError::Pipeline(
        "BurnPointNet requires at least 1 decoder_dims entry".into(),
    ));
}
```

**`forward()` / `forward_batched()`** replace the hand-unrolled chain with a
loop, matching the ndarray `PointNetClassifier::forward()` pattern exactly
(saving the first layer's output as `local_feat`, then continuing the loop for
the remaining layers):

```rust
// Encoder layer 0 → local_feat (fed to the optional Feature T-Net and to the
// segmentation concat).
let (lin0, bn0) = &self.encoder_layers[0];
let local_feat = {
    let h = lin0.forward(input);
    let h = apply_bn2d(h, bn0);
    h.clamp_min(0.0)
};

// Feature T-Net (unchanged)...

// Encoder layers 1+.
let mut deep = local_feat.clone();
for (lin, bn) in self.encoder_layers.iter().skip(1) {
    let h = lin.forward(deep);
    let h = apply_bn2d(h, bn);
    deep = h.clamp_min(0.0);
}

// ... global max pool, concat ...

// Decoder layers.
let mut h = combined;
for (lin, bn) in &self.decoder_layers {
    let hh = lin.forward(h);
    let hh = apply_bn2d(hh, bn);
    h = hh.clamp_min(0.0);
}
self.proj.forward(h)
```

(`forward_batched()` mirrors this with the 3D `apply_bn3d` helper, exactly as
today.)

**`src/training/bridge.rs::save_model_from_burn`** currently hardcodes:

```rust
let encoder_layers = vec![
    extract_pair::<B>(&model.enc0, &model.bn_enc0, cfg.use_batch_norm)?,
    extract_pair::<B>(&model.enc1, &model.bn_enc1, cfg.use_batch_norm)?,
    extract_pair::<B>(&model.enc2, &model.bn_enc2, cfg.use_batch_norm)?,
];
```

This must become a loop over the new `Vec` field:

```rust
let encoder_layers = model
    .encoder_layers
    .iter()
    .map(|(lin, bn)| extract_pair::<B>(lin, bn, cfg.use_batch_norm))
    .collect::<Result<Vec<_>>>()?;
```

with the analogous change for `decoder_layers`. This is a genuine logic change
to `bridge.rs`, not merely a "test fixture dims" update as the original (v1)
spec draft implied — it is called out explicitly here and in the Affected
Files table above.

### Step 3 — Update the production call site and all test fixtures

Update the following to construct `encoder_dims` / `decoder_dims` from the new
canonical constants (`CANONICAL_ENCODER_DIMS.to_vec()` /
`CANONICAL_DECODER_DIMS.to_vec()`) instead of duplicating literal `Vec`s. All
four known call sites are listed (the original v1 draft listed only 2 of
these):

| File | Site | Current literal | New construction |
|------|------|-----------------|-------------------|
| `src/training/trainer.rs` | production `TrainConfig`→`PointNetConfig` (~line 235) | `vec![64, 128, 256]` / `vec![256, 128]` | `CANONICAL_ENCODER_DIMS.to_vec()` / `CANONICAL_DECODER_DIMS.to_vec()` |
| `src/training/trainer.rs` | test `test_swa_averages_tnet_weights` (~line 1349) | `vec![64, 128, 256]` / `vec![256, 128]` | same constants |
| `src/training/bridge.rs` | test `default_cfg()` | `vec![64, 128, 256]` / `vec![256, 128]` | same constants |
| `src/training/burn_model.rs` | test `default_cfg()` | `vec![64, 128, 256]` / `vec![256, 128]` | same constants |

`src/cli/evaluate_cmd.rs` and `src/cli/fix_label_map_cmd.rs` test fixtures use
small unrelated dims (`[4, 8]`/`[4]`) on a raw `PointNetClassifier` with no
`BurnPointNet` length validation involved — **no change** needed at these two
sites.

### Step 4 — Update documentation

**`docs/stages/stage-02-modeling-layer.md`:**

This file has two categories of drift to fix in the same pass (since it is
already being edited):

1. **Stage-43-specific updates** — encoder/decoder dims and derived values:
   - Architecture diagram: update the encoder section to show 5 layers
     (`Linear(17→64)→Linear(64→64)→Linear(64→64)→Linear(64→128)→Linear(128→1024)`),
     the global max-pool section (1024-dim), the segmentation concat
     (`N × 1088`), and the decoder section
     (`Linear(1088→512)→Linear(512→256)→Linear(256→n_classes)`).
   - Config parameter table: `encoder_dims` default `[64, 64, 64, 128, 1024]`,
     `decoder_dims` default `[512, 256]`, decoder input dim note
     `64 + 1024 = 1088`.
   - "Why 320, not 512?" note → "Why 1088, not 2048?" with updated rationale
     (concatenating local `64` + global `1024` = `1088`; the original
     deviation-era spec's hypothetical `512+512` framing no longer applies at
     these dims and should be reworded to explain the *current* 1088 value
     rather than the historical 320 one).
   - Forward-pass pseudocode: update the encoder loop comment
     (`// 5 layers [17→64→64→64→128→1024]`) and decoder loop comment
     (`// [1088→512→256→n_classes]`), and the intermediate shape comments
     (`N × 1024`, `N × 1088`, etc.).
   - `.wbmodel` format documentation: update the worked example
     (`n_encoder_layers`, e.g. `5` for `[64,64,64,128,1024]`) and any
     `encoder_dims[k-1] → encoder_dims[k]` example values.

2. **Pre-existing drift, unrelated to Stage 43 but reconciled in the same
   pass per the AGENTS.md Drift Rule** — every remaining stale
   `N × 12` / `N_FEATURES = 12` reference, dating from before Stage 30
   expanded the feature layout to 17 features. This includes (at minimum):
   the architecture diagram's `Input: N × 12` header, the "Encoder Layer 0"
   diagram line (`Linear(12 → 64)`), the config table's
   `n_features_in` row (`12` → `17`), the forward-pass pseudocode's
   `features: Array2<f32> shape [N, 12]` comment and `Linear(12 → 64)`
   comment, and the `.wbmodel` format header's
   `n_features_in: u8 (must equal N_FEATURES = 12 from Stage 01)` line. All
   such references are updated to `17` so the document is fully consistent
   with the current Stage-30-era feature layout before Stage 43 is marked
   closed.

**`docs/user/user_guide.md` §7:**
- Update the architecture diagram to show the correct encoder/decoder
  dimensions (5 encoder layers ending at 1024, decoder `1088→512→256`).

### Step 5 — Verify

- `cargo build --all-targets --all-features` — zero errors.
- `cargo build --features training` immediately after the `BurnPointNet`
  struct change (Step 2) — fail-fast check that `Vec<M: Module<B>>` fields
  are supported by the installed `burn` version, before the rest of Step 2's
  `forward()`/`forward_batched()`/`bridge.rs` rewrite is done.
- `cargo clippy --all-targets --all-features -- -D warnings` — zero new warnings.
- `cargo test --all-features` — all tests pass (test fixtures updated).
- `cargo fmt --check` — clean.

---

## Performance Considerations

### Model size increase

Parameter counts below are computed directly from
`params = in_dim · out_dim + out_dim` per `Linear` layer (BatchNorm adds a
further `4 · out_dim` params per layer when `use_batch_norm = true`, omitted
here as it is small and identical in relative proportion for both
architectures). Input dimension is the current `N_FEATURES = 17`.

| Metric | Current `[64,128,256]` / `[256,128]` | Canonical `[64,64,64,128,1024]` / `[512,256]` | Factor |
|--------|---------|-----------|--------|
| Encoder parameters | 17·64+64 + 64·128+128 + 128·256+256 = **~42.5K** | 17·64+64 + 64·64+64 + 64·64+64 + 64·128+128 + 128·1024+1024 = **~149.9K** | ~3.5× |
| Decoder parameters (incl. class-proj, `n_classes=8`) | 320·256+256 + 256·128+128 + 128·8+8 = **~116.1K** | 1088·512+512 + 512·256+256 + 256·8+8 = **~691.0K** | ~6.0× |
| Total (encoder + decoder) | **~158.6K** | **~840.9K** | ~5.3× |
| Global descriptor | 256 f32 | 1024 f32 | 4× |

(The v1 draft of this table significantly overstated the growth — claiming
~67K→~1.1M/~16× for the encoder and ~300K→~2.2M/~7× overall — due to an
arithmetic error. The real growth, recomputed above, is smaller and more
benign than originally stated; this does not change the go/no-go
recommendation for the change.)

The T-Net parameters are unchanged (~1.68M for STN3d: mini-encoder
3·64+64 + 64·128+128 + 128·1024+1024 ≈ 141K, FC decoder
1024·512+512 + 512·256+256 + 256·9+9 ≈ 657K, total STN3d ≈ 798K; STN64d
similarly ≈ 802K when enabled — these totals are unaffected by Stage 43 and
included here only as context, not as a claim being corrected).

### Memory impact

The largest intermediate activation is the main encoder's final 1024-wide
activation (previously 256-wide): `N × 1024 × 4 bytes` ≈ 4 MB per block at
`N = 1024`. With `--forward-batch-size 8`, this is ≈ 32 MB — well within the
8 GB VRAM / 4 GB CPU-memory budgets referenced elsewhere in this codebase's
docs.

### Training speed

The larger encoder/decoder will increase per-block forward/backward time.
The existing gradient accumulation and micro-batching strategy is unaffected.
Given the ~5.3× total parameter growth (not ~7× as originally estimated), the
user should expect a training-epoch slowdown on a similar order — plan for
roughly 2-4× longer training epochs as a conservative estimate — which is
acceptable given the current hyperparameter testing workflow.

---

## Definition of Done

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `cargo build --release --features training` — zero errors | Build gate |
| 2 | `cargo clippy --all-targets --all-features -- -D warnings` — zero new warnings | Clippy gate |
| 3 | `cargo test --all-features` — all tests pass | Test gate |
| 4 | `cargo fmt --check` — clean | Format gate |
| 5 | `model::pointnet::CANONICAL_ENCODER_DIMS == [64, 64, 64, 128, 1024]` and `CANONICAL_DECODER_DIMS == [512, 256]` | Unit test |
| 6 | No literal `vec![64, 128, 256]` / `vec![256, 128]` (or the new canonical literals) remains duplicated across `trainer.rs`, `bridge.rs`, `burn_model.rs` — all four call sites reference the shared constants | Manual review / `grep` check |
| 7 | `PointNetConfig { encoder_dims: CANONICAL_ENCODER_DIMS.to_vec(), decoder_dims: CANONICAL_DECODER_DIMS.to_vec(), .. }.concat_dim() == 1088` | Unit test |
| 8 | Forward pass (`PointNetClassifier::forward`, ndarray) on a `[1024, 17]` input with canonical dims produces output shape `[1024, n_classes]` | Existing/updated shape test |
| 9 | `BurnPointNet::forward` (non-batched) on a `[1024, 17]` input with canonical dims produces output shape `[1024, n_classes]` | Existing/updated shape test |
| 10 | `BurnPointNet::forward_batched` on a `[B, N, 17]` input with canonical dims produces output shape `[B, N, n_classes]`, and identical input blocks across the batch dimension produce identical per-block outputs (no cross-block pooling leakage) | Existing/updated shape test (`test_forward_batched_identical_blocks_are_consistent`, updated to canonical dims) |
| 11 | `.wbmodel` round-trip — save/load produces bit-identical forward pass with canonical dims | Existing round-trip test, updated fixture |
| 12 | Burn↔ndarray forward equivalence — `BurnPointNet` bridged to `PointNetClassifier` produces matching logits with canonical dims (`test_burn_and_ndarray_forward_outputs_agree_after_bridge`, updated fixture) | Existing bridge test |
| 13 | `BurnPointNet`'s `encoder_layers` / `decoder_layers` are `Vec` fields (confirmed via struct definition review, not just `new()`); `training/bridge.rs::save_model_from_burn` extracts them via a loop, not hardcoded `.enc0`/`.enc1`/`.enc2` field access | Manual code review |
| 14 | `stage-02-modeling-layer.md` fully reconciled: correct architecture diagram/dims for Stage 43, **and** all stale `N × 12`/`N_FEATURES = 12` references updated to 17 | Manual review |
| 15 | `user_guide.md` §7 updated with correct architecture diagram | Manual review |

---

*This document is the authoritative specification for Stage 43. Per the
AGENTS.md synchronization rule, the code and this spec must remain in sync.*
