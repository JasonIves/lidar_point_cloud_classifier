# Stage 16 — GPU Training VRAM Exhaustion (Validation-Pass Autodiff Leak)

## Status: RESOLVED (bounded-VRAM baseline restored via `model.valid()`)

## The Goal

Eliminate the GPU out-of-memory failure that aborts `wb_lidar_train` GPU
training runs. Under `--device gpu` (and after the automatic CPU fallback under
`--device auto`), training crashed with:

```
wgpu error: Validation Error
  In Device::create_buffer
    Not enough memory left.
```

The tool must train to completion on the GPU and continue to fall back
gracefully to CPU when no usable adapter is present (AGENTS.md hardware
independence).

## Diagnosis (evidence-based)

This stage supersedes two earlier misdiagnoses:

1. The original "single oversized allocation / VRAM budget" theory that drove
   the speculative allocator-tuning of Stages 10–14 (reverted in Stage 15).
2. The intermediate "per-step training leak (~4.5 MB/step)" theory recorded in
   an earlier revision of this document — **retired**. It was a misread of a
   coarse step counter.

A breadcrumb-driven investigation (temporary `[breadcrumb]` diagnostics in
`backend.rs` and `trainer.rs`, since removed) established the true root cause.

### Findings

1. **Correct adapter, unlimited per-buffer limit.** wgpu binds the discrete
   NVIDIA RTX 2070 SUPER (Vulkan) with `max_buffer_size = u64::MAX`. Not a
   software rasterizer, not an iGPU. A single allocation cannot be "too large".

2. **Training is memory-flat.** With per-step tracing, the training loop
   completes **all 1792 optimizer steps** of epoch 1 (batch-size 1) with no
   monotonic VRAM growth. `loss.backward()` consumes the autodiff graph every
   step and deterministically frees its retained activation buffers. The
   training loop needs **no** explicit `sync` or cleanup.

3. **The OOM is in `validate_epoch`, not training.** The crash occurs
   consistently at **~block 50 of 256** *after* training completes and
   validation begins.

4. **Root cause: autodiff forward-without-backward.** The previous validation
   implementation forwarded on the **autodiff backend** (to obtain batch-
   statistic BatchNorm) but never called `.backward()`. The autodiff graph plus
   BatchNorm running-state update nodes accumulate GPU buffers that burn 0.16
   never reclaims.

5. **The leak is inherent to burn-0.16 autodiff forward-without-backward** —
   not fixable at the validation-loop level. Three independent approaches all
   OOM'd at *exactly* block 50:
   - Persistent `val_model = model.clone()` reused for all blocks (original).
   - Adding `B::sync(device)` per validation block (sync flushes the queue but
     cannot free live buffers).
   - Per-block `model.clone()` inside the loop (clone does not isolate the
     accumulation).

6. **`model.valid()` is the only approach that bounds VRAM.** Converting the
   module to its `B::InnerBackend` (inference backend) via `AutodiffModule::valid()`
   allocates **no autodiff graph** per forward. Validation then completes all
   256 blocks with no slowdown, and training continues into subsequent epochs.

## Resolution

`validate_epoch` now calls `let val_model = model.valid();` once, then forwards
each block on `B::InnerBackend` (`features_to_tensor::<B::InnerBackend>`). This
keeps VRAM bounded across all validation blocks and all epochs. The CPU
(NdArray) path is unaffected — `.valid()` is a no-op-cost conversion there and
the loop was already memory-stable on CPU.

## Known limitation (deferred to Stage 17)

`model.valid()` uses **running-statistic (inference-mode) BatchNorm**. This
model's trained running statistics are currently pathological and produce a
logit explosion at eval time (`val_loss_uw ≈ 1.6e5`, `val_mIoU ≈ 0.04` —
near-random for 8 classes — while `train_loss ≈ 0.27` is healthy).

This is a pre-existing BatchNorm defect that was previously *masked* by
batch-statistic validation; the memory fix merely surfaces it. Because the same
running statistics drive **deployed inference** (`model/pointnet.rs`), the defect
must be root-caused independently of the memory work. That investigation is
specified in **`stage-17-batchnorm-running-stats.md`**.

Stage 16 is considered complete for its stated goal: the GPU OOM is eliminated
and VRAM stays bounded across a full multi-epoch run. Restoring correct
validation *metrics* is Stage 17's responsibility.

## Inputs & Outputs

- Inputs: unchanged CLI (`wb_lidar_train ... --device {auto|cpu|gpu}`).
- Outputs: unchanged (`.wbmodel`, metrics CSV). Behaviour change: GPU training
  completes instead of OOMing; VRAM stays bounded. Validation metrics remain
  affected by the Stage 17 BatchNorm limitation.

## Definition of Done

- [x] GPU training completes a full multi-epoch run on the RTX 2070 SUPER
      (8 GB) at batch-size 1 without the `create_buffer` OOM; VRAM stays bounded
      across epochs and validation.
- [x] CPU fallback (`--device auto` with no GPU, and `--device cpu`) unchanged.
- [x] Temporary `[breadcrumb]` diagnostics removed.
- [x] Known-limitation (running-stat BatchNorm metrics) documented and handed to
      Stage 17.
- [ ] `cargo build/fmt/clippy` clean; existing 61 tests pass. *(verify after
      this edit)*
- [x] This spec synchronized with the final implementation.