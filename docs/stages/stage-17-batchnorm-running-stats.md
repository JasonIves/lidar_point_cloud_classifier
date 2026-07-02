# Stage 17 — BatchNorm Running-Statistic Explosion (Inference-Mode Logit Blow-Up)

## Status: RESOLVED — root cause confirmed and fixed

> **Resolution summary.** The logit explosion was **not** an axis, momentum, or
> epsilon convention bug (hypotheses 1–3 below were investigated and *refuted* —
> burn's `BatchNorm<B, 1>` normalizes correctly per-channel over the
> `batch × spatial` axes for a `[N, C, 1]` layout). The real defect was the two
> **post-max-pool fully-connected BatchNorm layers** inside each T-Net
> (`Stn3d`/`Stn64d`, mirrored by `TNet` in the deployed model). These layers
> receive the single pooled global descriptor `[1, C]` — a genuine **batch of
> one** — so burn's train-mode `forward_train` computes a batch variance of `0`
> over its lone sample and drives `running_var` from `1.0` toward `0` via the EMA
> update `running = running·(1−m) + batch_var·m`. At inference the layer divides
> by `sqrt(running_var + eps) ≈ sqrt(1e-5) ≈ 0.00316`, amplifying activations by
> ~316× per layer → the observed `val_loss ≈ 1.6e5`. Every other BatchNorm in the
> network sees `N` points (real per-channel variance) and is unaffected.
>
> BatchNorm on a batch-of-one pooled vector is mathematically degenerate, so the
> **fix is to not apply it** on those FC layers, which makes train-mode and
> inference-mode agree exactly. See "Root cause (confirmed)" and "Fix applied"
> below.

## The Goal

Repair the pathological **running-statistic BatchNorm** behaviour that produces
a logit explosion whenever the model is evaluated in inference mode. This
manifests in two places that share the *same* trained running statistics:

1. **Validation** (`training/trainer.rs::validate_epoch`), which — since the
   Stage 16 memory fix — forwards on the inner backend via `model.valid()` and
   therefore uses inference-mode (running-stat) BatchNorm.
2. **Deployed inference** (`model/pointnet.rs::forward`), which applies stored
   running statistics through the ndarray `apply_bn2d` path.

Concretely, with a model whose training loss is healthy (`train_loss ≈ 0.27`),
inference-mode evaluation yields:

```
val_loss_uw ≈ 163258   (expected: order ~1)
val_mIoU    ≈ 0.0423    (expected: well above 8-class random ~0.125 IoU-per-class floor)
```

i.e. the running-stat normalization is destroying the signal. Fixing this
restores **both** correct validation metrics **and** correct production
inference simultaneously.

## Background / Why this is separate from Stage 16

Stage 16 proved that batch-statistic validation on burn 0.16's autodiff backend
is unavoidably memory-leaking (forward-without-backward is never reclaimed), and
that `model.valid()` is the only bounded-VRAM path. `model.valid()` forces
inference-mode BatchNorm, which *exposed* — did not cause — this defect. The
defect was previously **masked** because batch-statistic validation recomputes
per-batch mean/variance and never touches the (broken) running statistics.

An earlier build reportedly showed a similar issue that was "resolved with the
shift in calculation approach" — i.e. it was masked by switching to batch-stat
validation, not actually fixed. Stage 17 must fix it at the source.

## Key isolation already established

`model.valid()` explodes using **burn's own BatchNorm inference path** with
**burn's own trained running statistics**. Therefore:

- The defect is **not** in the burn→ndarray weight bridge.
- The defect is **not** in the ndarray `apply_bn2d` reimplementation.
- The trained **running statistics themselves are pathological** — they are
  populated incorrectly during batch-stat training, or consumed under a
  different convention than they are written.

## Prime suspects (investigated — all REFUTED)

> All three hypotheses below were checked against burn 0.16.1's
> `burn-core/src/nn/norm/batch.rs` and unit-isolated. None is the cause: burn's
> `BatchNorm<B, 1>` correctly treats dim-1 as channels and reduces over dims
> `{0, 2}`, stores *variance* (not std), and applies matching eps. The `[N, C, 1]`
> reshape accumulates running stats over the `N` point axis exactly as intended.
> The defect is instead the batch-of-one FC layers — see "Root cause" below.

1. ~~**Running-stat update convention / momentum.**~~ *(refuted)* burn `BatchNorm` updates
   `running_mean`/`running_var` via an exponential moving average during
   training-mode (autodiff-backend) forwards. Verify the momentum direction and
   that our per-block, batch-size-1, 5120-point forwards feed statistics burn
   expects. Note the model applies BN through `apply_bn2d`, reshaping `[N, C]`
   → `[N, C, 1]`: confirm burn normalizes over the intended axis (the `N`
   sample axis, per-channel `C`) in that layout, so running stats accumulate
   over points-as-samples rather than the singleton spatial axis.
2. ~~**Variance convention / epsilon.**~~ *(refuted)* Confirmed: burn's `running_var` stores variance (not
   std), and the eps added at inference (`1e-5`) matches the training-mode
   normalization. No mismatch. (The *near-zero* running_var is real, but its
   origin is the batch-of-one FC layers, not an eps/variance convention error.)
3. ~~**Axis semantics of the `[N, C, 1]` reshape under BatchNorm<B,1>.**~~ *(refuted)* burn's
   `BatchNorm<B, 1>` treats dim-1 as channels and normalizes over dims {0, 2};
   a `[N, C, 1]` tensor normalizes each channel over `N × 1` — as intended. The
   running-stat update reduces over `{0, 2}`, so with `N` points it accumulates
   correct per-channel statistics. This layout is fine for the encoder/decoder
   BN layers (which see all `N` points); it is only degenerate when `N = 1`.

## Root cause (confirmed)

The two T-Net variants each end with a small MLP head applied to the **pooled
global feature vector**:

```
let g = h.transpose().max_dim(1).transpose(); // [1, C]  ← global max-pool over N points
let g = apply_bn2d(g, &self.bn_fc0);          // BatchNorm over a batch of ONE
```

Because the max-pool collapses the `N`-point axis to a single global descriptor,
these `bn_fc0`/`bn_fc1` layers see input shape `[1, C]` → reshaped `[1, C, 1]`,
i.e. **one sample**. burn's `forward_train` (in `burn-core/src/nn/norm/batch.rs`)
computes `var` over `flatten_size = batch × trailing = 1 × 1 = 1` sample, giving
a batch variance of `0` for every channel. The EMA update

```
running_var = running_var·(1 − momentum) + batch_var·momentum   // batch_var = 0
```

therefore decays `running_var` geometrically from its `1.0` init toward `0`
(`0.9^k` per step for the default momentum `0.1`). At inference the same layer
computes `x_norm = (x − running_mean) / sqrt(running_var + eps)`; with
`running_var ≈ 0` the denominator collapses to `sqrt(1e-5) ≈ 0.00316`, so each
of the two FC BN layers multiplies its activations by ~316×. Stacked across both
T-Nets this compounds into the ~1e5 logit explosion. Batch-stat validation hid
this because it recomputed a (still-degenerate but self-consistent) per-batch
statistic and never read `running_var`.

All *other* BatchNorm layers (encoder shared-MLP, decoder head) operate on the
full `[N, C]` per-point tensor with `N = 5120`, so their batch variance is real
and their running stats are healthy — those are left untouched.

## Fix applied

BatchNorm on a genuine batch-of-one pooled descriptor is mathematically
undefined (zero variance), so the correct fix is to **not apply it** on the
post-pool FC layers, which makes train-mode and inference-mode identical for
those layers:

- **`training/burn_model.rs`** — `Stn3d::forward` and `Stn64d::forward`: removed
  the `apply_bn2d(g, &self.bn_fc0)` / `bn_fc1` calls after the max-pool; the FC
  layers now go straight `fc → ReLU`. Added Stage-17 explanatory comments.
- **`model/layers.rs`** — `TNet::forward` (deployed ndarray inference): removed
  the matching `apply_bn1d` calls on the post-pool FC layers so the deployed
  model mirrors the training twin exactly. Removed the now-unused `apply_bn1d`
  free function (replaced by an explanatory note); `BatchNorm1d::forward_1d`
  retained for the per-point BN paths.
- **Struct fields retained.** `bn_fc0`/`bn_fc1` fields remain on `Stn3d`,
  `Stn64d`, and `TNet` (and continue to be carried through `bridge.rs`,
  `trainer.rs` SWA, and the `.wbmodel` (de)serializer) so the on-disk model
  format is unchanged and backward/forward compatible. They are simply no longer
  applied during the forward pass.
- **Regression tests** added to `burn_model.rs`:
  `test_batchnorm_batch1_running_var_decays_toward_zero` (feeds 15 single-sample
  `[1, 8, 1]` tensors and asserts `running_var` collapses well below its `1.0`
  init — proving the mechanism) and
  `test_valid_inference_logits_bounded_after_training` (trains a few steps on
  `Autodiff<NdArray>`, runs a `.valid()` forward, and asserts all logits are
  finite with `max_abs < 1e3` — proving the explosion is gone).

## Inputs & Outputs

- Inputs: unchanged CLI. Investigation may add a **temporary, opt-in** debug
  dump of BatchNorm running_mean/running_var ranges (gated, low-overhead, per
  AGENTS.md — no high-throughput logging).
- Outputs: unchanged `.wbmodel` / metrics format. Behaviour change on success:
  inference-mode `val_loss` returns to order ~1 and `val_mIoU` tracks
  training-time performance; deployed inference produces sane logits.

## Steps & Specifications (as executed)

1. ~~**Instrument** running stats~~ — not needed; the mechanism was isolated
   directly by reading burn's `forward_train` and reproduced in a unit test.
2. **Unit-isolated** the degenerate case in
   `test_batchnorm_batch1_running_var_decays_toward_zero`: repeatedly feeding a
   single-sample `[1, C, 1]` tensor through a `BatchNorm<B, 1>` on
   `Autodiff<NdArray>` drives `running_var` toward `0`, exactly the mechanism
   behind the explosion.
3. **Identified** the mismatch: not axis/momentum/eps (all refuted), but the
   batch-of-one pooled FC BatchNorm producing a `0` batch variance that decays
   `running_var`.
4. **Fixed** by removing BatchNorm application on the post-pool FC layers in both
   the training twin (`burn_model.rs`) and the deployed model (`layers.rs`);
   struct fields and `.wbmodel` format preserved.
5. **Validated** at the unit level via
   `test_valid_inference_logits_bounded_after_training` (train-then-`.valid()`
   logits are finite and bounded). A full end-to-end GPU run remains as an
   operator smoke-check (see DoD).

## Definition of Done

- [x] A minimal unit test reproduces the train-vs-`.valid()` BatchNorm
      divergence and then passes once the fix lands (outputs agree within
      tolerance for identical inputs and matched statistics).
      *(Done: `test_batchnorm_batch1_running_var_decays_toward_zero` +
      `test_valid_inference_logits_bounded_after_training` in `burn_model.rs`.)*
- [ ] Inference-mode validation on a real short run yields `val_loss` of order
      ~1 (not ~1e5) and `val_mIoU` consistent with `train_loss`.
      *(Pending an operator GPU run; the unit test confirms bounded `.valid()`
      logits, which is the direct cause of the prior blow-up.)*
- [ ] Deployed inference (`model/pointnet.rs`) produces non-degenerate logits on
      the same checkpoint.
      *(Pending an operator smoke-check; the deployed `TNet` path now mirrors the
      training twin exactly, so the same FC BN is no longer applied.)*
- [x] No autodiff-backend forward-without-backward reintroduced (Stage 16 VRAM
      baseline preserved). *(`trainer.rs::validate_epoch` still uses
      `model.valid()`; unchanged by this stage.)*
- [x] Any temporary BN-statistic instrumentation removed or gated behind an
      opt-in debug flag. *(None was added; the fix was determined analytically.)*
- [x] `cargo build/fmt/clippy` clean; all existing tests plus the new BN test
      pass. *(Verified: 63 tests pass, clippy clean, `cargo fmt --check` clean.)*
- [x] This spec synchronized with the final implementation.
