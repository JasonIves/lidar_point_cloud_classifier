# Stage 15 — GPU Default-Device Cleanup: Removing a Misanalyzed VRAM "Fix" Stack

**Status:** COMPLETE  
**Approved:** 2026-07-01  
**Implemented:** 2026-07-01  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier  
**Lead Architect:** AI Collaborator

---

## Goal

Trace a stack of escalating GPU "memory allocation" fixes (Stages 10–14) back to
their source, verify whether they were built on a **misdiagnosed** problem, and —
if so — rip out the extraneous code, comments, and documentation while keeping
**minimal, working GPU support**.

Outcome: the diagnosis was confirmed as a misanalysis. Stages 10–14 were reverted.
GPU support now uses burn's stock `WgpuDevice::default()` with graceful CPU
fallback, exactly as a normal burn user would configure it.

---

## Background: the reported problem

The user reported "a tricky issue with memory allocation when training the model
on a GPU," while noting they had *"used burn before and didn't have this
problem."* That skepticism was the key signal: if stock burn works elsewhere,
the workload — not burn's defaults — is the thing to examine first.

Stages 10–14 had progressively layered increasingly invasive wgpu/cubecl
allocator tuning on top of Stage 09's GPU support:

| Stage | What it added | Claimed cause |
|---|---|---|
| 10 | "Conservative" GPU device init; VRAM budget reservation | VRAM exhaustion |
| 11 | Memory-pool chunk-size cap (`--gpu-max-page-mb`, `max_storage_buffer_binding_size`) | Pool allocations too large |
| 12 | Allocator diagnostics + memory hints; adapter probing | Fragmentation / hint mismatch |
| 13 | Activation-based "dedicated allocation" threshold | Wrong dedicated/pooled split |
| 14 | cubecl runtime memory config (`ExclusivePages`, `tasks_max: 1`) | Pool strategy defeating reuse |

Each stage's own notes admitted the previous one had not worked ("did not
resolve", "SUPERSEDED", "proved insufficient").

---

## Investigation & Evidence

### 1. Git history — none of the GPU work was ever committed or verified

- `HEAD` = `cc08141` corresponds to **Stage 04** and is a clean, **CPU-only**
  tree.
- **All** GPU work (Stages 09–14) was **uncommitted** in the working tree.
- `src/training/backend.rs` was **untracked** — it had never been part of a
  known-good commit.

This means there was never a "before it broke" GPU commit to regress against:
the entire fix stack was speculative, applied on top of never-verified code.

### 2. The verification DoDs were never actually run

Every GPU stage (09–14) had its GPU-execution Definition-of-Done marked
"⏳ Pending — requires user's physical GPU." The fixes were written, documented
as effective, and stacked — but the actual GPU runs that would confirm or refute
them were never performed.

### 3. The physics disproves "VRAM exhaustion"

The model is a PointNet operating on spatial blocks resampled to a fixed point
count. The largest single activation tensor is the STN3d 1024-wide feature map
at ≈ 5120 points:

```
5120 points × 1024 features × 4 bytes ≈ 20 MB
```

Even accounting for autodiff saved-activations, gradients, and optimizer state,
the total is on the order of a few hundred MB — nowhere near enough to exhaust
an 8 GB (or even 4 GB) GPU. A genuine VRAM-exhaustion diagnosis is not
physically consistent with this workload.

### 4. The "fixes" actively fought burn/cubecl's defaults

The stacked tuning replaced burn's stock configuration with:
- `downlevel_defaults()` limits,
- a 25 MiB `max_storage_buffer_binding_size` clamp,
- `ExclusivePages` + `tasks_max: 1`,

which **defeat cubecl's memory pooling and reuse** — the opposite of what a
small, well-behaved workload needs. This is consistent with the fixes *causing*
or *reshaping* failures rather than resolving a real capacity limit.

### Conclusion

The "GPU memory allocation" problem was a **misanalysis**. There was no VRAM
capacity problem to solve. The correct configuration is burn's default device —
the same configuration under which the user's prior burn projects worked.

---

## What Was Removed

### Code

- `src/training/backend.rs` — rewritten from ~938 lines of speculative allocator
  tuning to a minimal ~280-line module. Removed:
  `init_conservative_gpu_device`, `required_storage_binding_bytes`,
  `estimate_vram_bytes`, `probe_buffer_allocations`,
  `gpu_points_per_block_hint`, `AdapterInfo`, all custom wgpu limits / memory
  hints / cubecl pool config, and `pollster` usage.  
  **Kept:** `DevicePreference` (Auto/Cpu/Gpu) + parse/default, `gpu_is_available`,
  `select_and_train` dispatch, `train_cpu`, `train_gpu_or_fallback`
  (`catch_unwind` + CPU fallback), and `train_gpu_inner` using
  `WgpuDevice::default()`.
- `src/cli/train_cmd.rs` — removed the `--vram-budget-mb` and `--gpu-max-page-mb`
  flags, their validation blocks, and their help text. **Kept** `--device`.
- `src/training/trainer.rs` — removed the `vram_budget_mb` and `gpu_max_page_mb`
  fields from `TrainConfig` (+ defaults) and the Stage-10 VRAM-reservation
  comment in the training loop.
- `Cargo.toml` — removed the `pollster` optional dependency and its feature
  reference. **Kept** `panic = "unwind"` (still required for the `catch_unwind`
  CPU fallback).

### Documentation

- Deleted `stage-10-gpu-vram-fix.md`, `stage-11-gpu-memory-pool-chunk-fix.md`,
  `stage-12-gpu-allocator-diagnostics-and-memory-hints.md`,
  `stage-13-activation-based-dedicated-threshold-fix.md`,
  `stage-14-cubecl-runtime-memory-config.md`, and a stray empty `stage-` file.
- Rewrote `stage-09-gpu-support.md` and `stage-09-results.md` to describe the
  final minimal default-device design (removing the pre-flight VRAM estimator
  and large-batch VRAM warning that were also part of the misanalysis).

---

## What Was Kept (Minimal GPU Support)

- `--device <auto|cpu|gpu>` flag with `auto` default.
- Runtime GPU detection via `wgpu::Instance::enumerate_adapters`.
- GPU training on `Autodiff<Wgpu>` using **burn's stock `WgpuDevice::default()`**.
- Graceful CPU fallback: `catch_unwind` around GPU training, falling back to
  `Autodiff<NdArray>` under `--device auto` (or a clear error under
  `--device gpu`).
- `panic = "unwind"` in the release profile (required for the fallback).

---

## Definition of Done

| # | Criterion | Verification |
|---|---|---|
| 1 | Stages 10–14 docs and stray `stage-` file deleted | Directory listing |
| 2 | `backend.rs` contains no custom wgpu limits / memory hints / cubecl pool config | Code review |
| 3 | `train_gpu_inner` uses `WgpuDevice::default()` | Code review |
| 4 | `--vram-budget-mb` / `--gpu-max-page-mb` flags removed; `--device` retained | `train --help` |
| 5 | `TrainConfig` has no `vram_budget_mb` / `gpu_max_page_mb` fields | Code review |
| 6 | `pollster` removed from `Cargo.toml`; `panic = "unwind"` retained | Cargo.toml |
| 7 | Stage 09 spec + results rewritten to the minimal design | Doc review |
| 8 | `cargo build --release --features training` — zero errors | Build gate |
| 9 | `cargo test --features training` — all pass | Test gate |
| 10 | `cargo fmt --check` passes | fmt gate |
| 11 | `cargo clippy --features training -- -D warnings` — zero warnings | Clippy gate |

---

## Lessons / Guardrail Reinforcement

- **Trust the user's prior-art signal.** "Stock burn worked before" pointed
  directly at the workload/config, not at burn's defaults.
- **Verify before stacking.** Each fix should have been GPU-verified before the
  next was layered on. Unverified fixes stacked into a fragile, misleading
  edifice.
- **Check the physics.** A ≈ 20 MB peak activation cannot exhaust multi-GB VRAM;
  a back-of-envelope sanity check would have refuted the diagnosis at Stage 10.
- **Prefer defaults.** Per AGENTS.md (lean, minimal, thoughtful), fighting a
  framework's tested defaults should be a last resort backed by measurement, not
  a first response to an unverified symptom.

---

*This document is the authoritative specification for Stage 15. Per the AGENTS.md
synchronization rule, the code and this spec must remain in sync.*