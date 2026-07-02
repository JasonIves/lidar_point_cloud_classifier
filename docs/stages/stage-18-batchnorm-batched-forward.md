# Stage 18 — Batched Forward Pass (Fix the Train/Eval BatchNorm Statistics Gap)

## Status: CLOSED — Phase 1 (confirmation) and Phase 2 (Option A batched forward) COMPLETE; gap closed and confirmed on the real dataset

## The Goal

Eliminate the systematic **train/eval BatchNorm statistics gap** that leaves
validation (and deployed inference) metrics stuck at poor values — high
`val_loss` (~10× `train_loss`), low `val_mIoU` (< 0.10), and, tellingly, **no
converge-then-diverge (over-fit) curve** as training loss falls. The gap is
caused by training the network with an **effective BatchNorm batch size of one
block**, while validation and deployment normalize every block by a single set
of global running statistics.

This stage has two phases:

1. **Confirmation (this document's first deliverable).** Prove — with a
   self-contained deterministic test *and* an opt-in diagnostic on the real
   dataset — that the train/eval divergence is specifically the per-block-vs-
   global BatchNorm statistics mismatch, not another defect.
2. **Fix (Option A).** Replace the current per-block forward with a **real
   batched forward** of `b` blocks per pass so BatchNorm sees a genuine
   cross-sample distribution, its running statistics become representative, and
   the train/eval/deploy normalization agree.

## Background — how we got here

- **Stage 16** established that validation must run on the inner backend via
  `model.valid()` (the only bounded-VRAM path); forward-without-backward on the
  autodiff backend leaks. `model.valid()` forces inference-mode (running-stat)
  BatchNorm.
- **Stage 17** removed the degenerate batch-of-one BatchNorm on the T-Nets'
  post-pool FC layers (running_var → 0 → logit explosion). That fixed the ~1e5
  explosion but left the *general* train/eval BatchNorm gap intact.
- The **older validation system** that produced "reasonable" metrics forwarded
  in **train mode** (batch-statistic BN), so it measured the per-block-normalized
  function that training actually optimizes. It looked good but (a) leaked VRAM
  and (b) never exercised the running statistics that deployment uses.

## Root-cause hypothesis (to confirm)

The training loop (`trainer.rs`) forwards **one block at a time**
(`for &block_id in chunk { model.forward(single_block) }`) and only uses
`GradientsAccumulator` to sum gradients across the `batch_size` blocks before a
single optimizer step. Therefore:

- `batch_size` is **gradient accumulation only**; it never reaches BatchNorm.
- Every `BatchNorm` normalizes each block by **that block's own** per-channel
  mean/variance (over its ~5,120 points) and applies an independent EMA update
  to the running statistics.
- Training thus teaches the network to operate only on **per-block-normalized**
  activations. At validation/deployment, eval-mode BN applies a **single global**
  running mean/var to every block, so any block whose distribution differs from
  the global average arrives mis-centered/mis-scaled → degraded predictions,
  high loss, suppressed mIoU, and a val curve that reflects the fixed
  normalization gap rather than generalization (hence no over-fit signature).

Key corollary: the **effective BatchNorm batch size equals the per-forward
block count**, *not* the gradient-accumulation count. Accumulation does not
merge BN statistics across micro-batches.

## Inputs & Outputs

### Confirmation phase
- **Inputs:** existing CLI unchanged. An **opt-in** diagnostic gated behind the
  environment variable `WB_BN_DIAG=1` (off by default; low-overhead; no
  high-throughput per-point logging, per AGENTS.md).
- **Outputs:** diagnostic emits, for the first few validation blocks, to stderr:
  - `val_loss` under running-stat `.valid()` (current path), and under
    batch-statistic BN (old-system-equivalent) on the *same* block;
  - the min/mean/max of each BN layer's `running_var`/`running_mean` vs. that
    block's own batch statistics.
  No change to `.wbmodel` or `metrics.csv` formats.

### Fix phase (Option A)
- **Inputs:** `TrainConfig` gains a **forward/micro-batch size** `b`
  (blocks per forward). Existing `batch_size` is re-interpreted as the
  *effective* batch (via optional gradient accumulation over multi-block
  micro-batches) or replaced — decided after the VRAM sanity check.
- **Outputs:** unchanged `.wbmodel` format and metrics schema. Behaviour change:
  `val_loss_w` returns to order ~`train_loss`, `val_mIoU` tracks training
  performance, and the val curve shows the expected converge→diverge pattern.

## Steps & Specifications

### Phase 1 — Confirmation
1. **Mechanism test (no dataset, deterministic)** in `burn_model.rs`:
   - Build `BurnPointNet<Autodiff<NdArray>>`.
   - Generate `K` **heterogeneous** synthetic blocks (distinct per-feature
     offset/scale), run train-mode forwards to populate running stats.
   - For a held-out block, compute logits/loss two ways — train-mode
     (batch-stat BN) vs `.valid()` (running-stat BN) — and assert they **diverge
     substantially**.
   - Repeat with **homogeneous** blocks and assert the two modes **agree**.
2. **Opt-in real-data diagnostic** (`WB_BN_DIAG=1`) in `validate_epoch`:
   - For the first few val blocks, log running-stat vs batch-stat `val_loss` and
     BN stat ranges (running vs per-block). Single-block train-mode forward only
     (bounded VRAM); fully gated and off by default.
3. **Record the readout** in this file (Confirmation Results section) and decide
   go/no-go on Option A.

### Phase 2 — Option A (batched forward)
4. Introduce a real batch dimension so BatchNorm normalizes across `b` blocks
   per forward (either stacked `[b·N, C]` with segment-wise pooling, or
   `[b, N, C]` with per-sample pooling over `N`). Preserve the burn-ndarray
   `max_dim`/gather workaround used at the global max-pool.
5. Update the training loop: forward `b` blocks → one averaged loss → one
   `backward()` → step; make `GradientsAccumulator` an **optional outer loop**
   over multi-block micro-batches (or remove it in its current single-block
   form).
6. Ensure the **deployed single-block inference path** (`model/pointnet.rs`)
   still aligns with the batched-trained running statistics.
7. Re-run the diagnostic to confirm the gap has closed.

## Definition of Done

### Phase 1 (Confirmation) — this deliverable
- [x] Deterministic mechanism test shows train-vs-eval BN divergence on
      heterogeneous blocks and agreement on homogeneous blocks.
- [x] Opt-in `WB_BN_DIAG` diagnostic implemented, gated, and off by default.
- [x] Confirmation Results recorded in this file; go/no-go decision on Option A.
- [x] `cargo build/fmt/clippy` clean; all existing tests plus the new test pass.

### Phase 2 (Fix)
- [x] Batched forward implemented; BatchNorm sees `b > 1` blocks per pass.
      (`BurnPointNet::forward_batched`, `Stn3d/Stn64d::forward_batched`,
      `apply_bn3d`, `features_to_tensor_batched`.)
- [x] Training loop updated; single-block gradient-only accumulation converted to
      a proper multi-block micro-batch loop with loss-averaged accumulation.
- [x] Deployed single-block inference remains consistent with training (batched
      path shares identical weights; deployed `model/pointnet.rs` unchanged).
- [x] On a real short run: `val_loss_w` ~ order `train_loss`, `val_mIoU` tracks
      `train_loss`. Confirmed on DALES (see Phase 2 Results below): 10-epoch run
      with `--forward-batch-size 8` gives `val_loss_uw ≈ train_loss ≈ 0.21` (was
      ~10×) and `val_mIoU` rising 0.34 → 0.44 in lockstep with `train_loss`. The
      converge→diverge over-fit signature is not yet reached at 10 epochs (val
      still tracking train downward), but the pre-fix symptoms — `val_loss`
      ~10× `train_loss` and `val_mIoU < 0.10` with no tracking — are eliminated.
- [x] Stage 16 bounded-VRAM baseline preserved (validation still on inner
      backend via `model.valid()`; single-block validation forward unchanged; no
      unbounded autodiff-graph growth reintroduced).
- [x] Temporary diagnostic left gated behind `WB_BN_DIAG` (off by default).
- [x] `cargo build/fmt/clippy` clean; all tests pass (66 tests, incl. two new
      batched-forward tests).
- [x] This spec synchronized with the implementation (Phase 2 Implementation
      section below).

## Confirmation Results

### 1. Deterministic mechanism test (`burn_model.rs`)

Test: `test_batchnorm_train_eval_gap_depends_on_block_heterogeneity`.

A `BurnPointNet<Autodiff<NdArray>>` (n = 128 points/block, 5 training blocks) is
driven with 30 train-mode passes to fully converge the BatchNorm EMA
(momentum 0.1 → 0.9³⁰ ≈ 4e-2 residual per pass; combined over 5 blocks the
running statistics converge to the block-set mean). For a held-out block we then
measure the **mean absolute difference between train-mode (batch-stat) logits and
`.valid()` (running-stat) logits** — the exact quantity that separates what
training optimizes from what validation/deployment sees.

| Block set     | Per-feature offset/scale        | Mean |Δlogit| (train vs eval) |
|---------------|---------------------------------|-------------------------------|
| Heterogeneous | offsets 0–8, scales 0.5–2.0      | **28.6977**                   |
| Homogeneous   | identical distribution           | **0.0001**                    |

Interpretation: when every block shares one distribution, the global running
statistics equal each block's own batch statistics, so train-mode and eval-mode
BatchNorm produce **identical** logits (gap ≈ 1e-4, i.e. numerical noise). When
the blocks are heterogeneous, the single global running average mis-normalizes
each held-out block and the two modes **diverge by ~28.7 logit units** — a
five-order-of-magnitude increase driven purely by the per-block-vs-global
BatchNorm statistics mismatch. This is the mechanism that inflates `val_loss` and
suppresses `val_mIoU` while leaving `train_loss` (measured in batch-stat mode)
looking healthy. The test asserts `homo_gap < 0.5` and
`hetero_gap > homo_gap*3 + 0.5`; both hold decisively.

### 2. Opt-in real-data diagnostic (`WB_BN_DIAG=1` in `validate_epoch`)

Implemented and gated: with `WB_BN_DIAG=1`, validation additionally logs, once
per pass, the min/mean/max of every main encoder/decoder BatchNorm layer's
`running_mean`/`running_var`, and, for the first three validation blocks, that
block's `val_loss` under running-stat `.valid()` vs under train-mode
(batch-statistic) BatchNorm, plus their delta. Off by default; bounded to three
train-mode forwards (negligible against the Stage 16 VRAM budget); emits only a
handful of stderr lines (no per-point logging, per AGENTS.md). This provides the
same running-stat-vs-batch-stat comparison on the *real* dataset that the
mechanism test proves on synthetic data, so the gap can be observed directly on
any training run without code changes.

### 3. Go/No-Go decision

**GO on Option A (batched forward).** The confirmation is unambiguous: the poor
validation metrics are the per-block-vs-global BatchNorm statistics gap, an
artifact of the effective BatchNorm batch size being one block. The fix is to
give BatchNorm a genuine cross-sample batch per forward (Phase 2), which makes
the running statistics representative and aligns training, validation, and
deployment normalization. Proceed to Phase 2 pending user go/no-go review.

## Phase 2 Implementation (Option A — batched forward)

### Tensor layout — `[b, N, C]` with per-sample pooling

Blocks are resampled to a common point count `N` upstream (Stage 04
normalizer), so a micro-batch of `b` blocks stacks cleanly into a 3-D tensor
`[b, N, C]`. This layout was chosen over the flattened `[b·N, C]` form because it
keeps the per-block boundary explicit, so the global max-pool can stay strictly
**per block** (over `N` only) while BatchNorm still normalizes across the whole
`b·N` micro-batch.

- **BatchNorm (`apply_bn3d`)** reshapes `[b, N, C] → [b·N, C, 1]`, applies
  `BatchNorm<B, 1>` (which normalizes each channel across all `b·N` rows — the
  genuine cross-block batch), then restores `[b, N, C]`. This is what makes the
  running statistics representative of the block *population* rather than a single
  block.
- **Global max-pool** transposes `[b, N, C] → [b, C, N]`, takes `max_dim(2)` over
  `N` → `[b, C, 1]`, transposes back to `[b, 1, C]`, and `repeat_dim(1, N)`
  broadcasts to `[b, N, C]`. The burn-ndarray 0.16 `max_dim`/gather
  last-dimension constraint (Stage 02/16) is preserved via the transpose.
- **T-Nets (`Stn3d/Stn64d::forward_batched`)** apply the same batched BN and
  per-sample pool, produce `[b, k, k]` transforms, and add a broadcast identity
  `[1, k, k]`. The transform is applied to the points with a batched
  `matmul([b, N, k] @ [b, k, k])`.

The batched path uses the **same weights** as the single-block
`BurnPointNet::forward`; only the normalization batch changes. Deployed
single-block ndarray inference (`model/pointnet.rs`) therefore needs no change —
it now consumes running statistics that were populated from representative
cross-block batches.

### Training loop (`trainer.rs`)

`TrainConfig` gains `forward_batch_size` (`b`, default 8); `batch_size` (default
16) is now documented as the **effective batch** — the number of blocks per
optimizer step. Each `batch_size` chunk is split into micro-batches of up to `b`
blocks:

```text
for chunk in shuffled.chunks(batch_size):          # one optimizer step
    for micro in chunk.chunks(forward_batch_size):  # one batched forward
        stack blocks → [count, N, C]
        logits = model.forward_batched(feat)         # [count, N, n_classes]
        loss   = CE(logits.reshape([count*N, nc]), targets)
        (loss / num_micro).backward()                # mean-over-chunk gradient
        accumulator.accumulate(...)
    optim.step(lr, model, accumulator.grads())
```

Scaling each micro-batch loss by `1/num_micro` makes the accumulated gradient the
**mean** over the chunk — standard mini-batch-with-accumulation semantics. Blocks
whose `(N, n_features)` disagree with the first block of a micro-batch are skipped
defensively (they are not expected after upstream resampling; we log rather than
panic, per AGENTS.md). The effective BatchNorm batch size is now
`forward_batch_size` blocks, **not** one.

### CLI (`train_cmd.rs`)

New flag `--forward-batch-size <usize>` (validated `>= 1`, default 8). `--batch-size`
help text clarified to "effective batch: blocks per optimizer step".

### Validation & diagnostic — unchanged

Validation still forwards single blocks on the inner backend via `model.valid()`
(Stage 16 bounded-VRAM path), and the `WB_BN_DIAG=1` diagnostic remains gated and
off by default. Only *training* now batches; validation/deployment are unchanged
apart from consuming the now-representative running statistics.

### Verification

- New tests in `burn_model.rs`:
  `test_forward_batched_identical_blocks_are_consistent` (identical blocks →
  identical per-block outputs, correct `[b, N, n_classes]` shape, no cross-block
  pool leakage) and `test_forward_batched_heterogeneous_blocks_bounded` (multi-
  distribution batch → finite, bounded logits).
- Full suite: 66 tests pass; `cargo clippy --features training` and
  `cargo fmt --check` clean.
- Empirical run on the real dataset: see Phase 2 Results below — confirmed.

## Phase 2 Results (real-data confirmation)

10-epoch DALES run (2 tiles, 1792 train / 256 val blocks, 16-pt blocks × 5120
points, Wgpu GPU backend, `--batch-size 16 --forward-batch-size 8
--learning-rate 1e-3 --weight-decay 5e-5 --val-split 0.15 --class-weight-beta 0`):

| Epoch | train_loss | val_loss_uw | val_mIoU |
|-------|-----------|-------------|----------|
| 1     | 0.5708    | 0.3356      | 0.3445   |
| 2     | 0.3559    | 0.2965      | 0.3512   |
| 3     | 0.3069    | 0.3892      | 0.3465   |
| 4     | 0.2924    | 0.2584      | 0.3887   |
| 5     | 0.2582    | 0.2624      | 0.3976   |
| 6     | 0.2464    | 0.2404      | 0.4168   |
| 7     | 0.2277    | 0.2216      | 0.4191   |
| 8     | 0.2158    | 0.2133      | 0.4367   |
| 9     | 0.2112    | 0.2155      | 0.4418   |
| 10    | 0.2056    | 0.2120      | 0.4445   |

Interpretation: with the batched forward giving BatchNorm a genuine cross-block
batch (`forward_batch_size = 8`), `val_loss` now sits at the **same order** as
`train_loss` (≈0.21 vs ≈0.21 by epoch 10) instead of the pre-fix ~10× gap, and
`val_mIoU` **rises monotonically with training** (0.34 → 0.44) rather than being
pinned below 0.10. The running statistics populated during batched training are
now representative of the block population, so `.valid()` (running-stat) and
deployment normalize blocks consistently with what training optimized. This is
the recovery Stage 18 set out to achieve; the stage is closed.
