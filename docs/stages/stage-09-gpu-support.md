# Stage 09 — GPU Support: Wgpu Backend with Runtime Detection

**Status:** COMPLETE  
**Approved:** 2026-07-01  
**Implemented:** 2026-07-01  
**Revised:** 2026-07-01 (see Stage 15 — reverted to burn's default device)  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier  
**Lead Architect:** AI Collaborator

---

## Goal

Add GPU acceleration support to the training pipeline via burn's `wgpu` backend,
satisfying the AGENTS.md mandate: *"If a GPU is available, it may be utilized for
acceleration; if it is absent, the tool must fallback gracefully and run efficiently
on the CPU."*

This stage addresses audit finding **1.1 (No GPU Support — CRITICAL)**.

> **Note (2026-07-01):** The original Stage 09 implementation additionally
> shipped a pre-flight VRAM estimator and a large-batch VRAM warning. Those were
> speculative additions built around a *misdiagnosed* "VRAM exhaustion" problem
> and were removed in **Stage 15**. This spec now describes the final, minimal
> design: GPU support using burn's stock `WgpuDevice::default()` with graceful
> CPU fallback. See `stage-15-gpu-default-device-cleanup.md`.

---

## Inputs & Outputs

### CLI Changes

New `--device` flag for the `train` sub-command:

```
--device <auto|cpu|gpu>   Compute device selection (default: auto)
```

| Value | Behaviour |
|---|---|
| `auto` | Detect GPU at runtime. If available, use `Autodiff<Wgpu>` with burn's default device. If no GPU (or GPU init panics), fall back to `Autodiff<NdArray>` (CPU). |
| `cpu` | Force `Autodiff<NdArray>` (CPU). Always available. |
| `gpu` | Force `Autodiff<Wgpu>`. Requires `--features training` at compile time. If no GPU is found at runtime, returns an error. |

### Cargo Feature Changes

```toml
[features]
# `training` always includes GPU support (burn wgpu backend) with runtime
# fallback to CPU NdArray when no GPU adapter is found.  Per AGENTS.md:
# "If a GPU is available, it may be utilized for acceleration; if it is
# absent, the tool must fallback gracefully and run efficiently on the CPU."
training = ["dep:burn", "burn/wgpu", "dep:wgpu"]
# `gpu` is an alias for `training` (kept for backwards compatibility).
gpu = ["training"]
```

- `burn/wgpu` enables burn's wgpu compute backend (Vulkan/Metal/DX12).
- `wgpu` (direct, optional) is used only for GPU adapter enumeration at runtime.
  It is already a transitive dependency of `burn-wgpu`, so there is **zero
  additional compile cost**.

### Build Commands

| Command | Backends Available |
|---|---|
| `cargo build --release` | NdArray (CPU only, inference only) |
| `cargo build --release --features training` | NdArray + Wgpu (GPU + CPU fallback) |
| `cargo build --release --features gpu` | Same as `--features training` (alias) |

---

## Steps & Specifications

### 1. Cargo.toml Changes

- Add `wgpu = { version = "23", optional = true }` to `[dependencies]`.
- Change `training = ["dep:burn"]` to `training = ["dep:burn", "burn/wgpu", "dep:wgpu"]`.
- Add `gpu = ["training"]` as an alias.
- The existing `burn` dependency remains unchanged (`features = ["ndarray", "autodiff"]`).
- **Critical:** Set `[profile.release]` to `panic = "unwind"` (not `"abort"`).
  This is required so that `std::panic::catch_unwind` in `backend.rs` can
  intercept a wgpu initialization/runtime panic and fall back to the CPU
  backend under `--device auto`. With `panic = "abort"`, the process aborts
  immediately on any panic, bypassing `catch_unwind` entirely.

### 2. New Module: `src/training/backend.rs`

This module encapsulates device selection logic:

```rust
pub enum DevicePreference {
    Auto,
    Cpu,
    Gpu,
}
```

**GPU Detection (when `training` feature is compiled in):**

```rust
#[cfg(feature = "training")]
fn gpu_is_available() -> bool {
    let instance = wgpu::Instance::default();
    !instance.enumerate_adapters(wgpu::Backends::all()).is_empty()
}
```

`wgpu::Instance::enumerate_adapters` is synchronous and returns all
discoverable graphics/compute adapters. If the list is non-empty, at least one
GPU (or software rasterizer) is available.

**GPU Device Initialization:**

GPU training uses burn's stock `WgpuDevice::default()` with the default cubecl
runtime — the configuration burn is designed and tested around. No custom wgpu
limits, memory hints, or cubecl pool tuning are applied. The PointNet training
workload here is small (largest single activation ≈ 20 MB), so nothing more is
required.

```rust
#[cfg(feature = "training")]
fn train_gpu_inner(dataset: &LabeledBlockDataset, config: &TrainConfig) -> Result<PathBuf> {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::{Autodiff, Wgpu};
    type GpuBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    train::<GpuBackend>(dataset, config, &device)
}
```

**Dispatch Function:**

```rust
pub fn select_and_train(
    dataset: &LabeledBlockDataset,
    config: &TrainConfig,
    preference: DevicePreference,
) -> Result<PathBuf>
```

This function:
1. Resolves the preference to a concrete backend + device.
2. Calls `train::<B>(dataset, config, &device)` with the appropriate generic type.
3. Logs which backend/device was selected.

**Resolution Logic:**

| `preference` | `training` feature | GPU available | Backend used |
|---|---|---|---|
| `Auto` | yes | yes | `Autodiff<Wgpu>` (CPU fallback if GPU panics) |
| `Auto` | yes | no | `Autodiff<NdArray>` |
| `Auto` | no | — | `Autodiff<NdArray>` |
| `Cpu` | any | — | `Autodiff<NdArray>` |
| `Gpu` | yes | yes | `Autodiff<Wgpu>` |
| `Gpu` | yes | no | Error |
| `Gpu` | no | — | Error |

**GPU Panic Catching (graceful fallback):**

`train_gpu_or_fallback` wraps the GPU training call in
`std::panic::catch_unwind`. If a panic is caught:
- When `allow_fallback` is `true` (`--device auto`): falls back to CPU with a warning.
- When `allow_fallback` is `false` (`--device gpu`): converts the panic to a
  `ClassifierError::Pipeline` with a clear message.

A custom panic hook is installed during the `catch_unwind` block to suppress the
default wgpu panic output and capture the message cleanly. The original hook is
restored afterward. The `Mutex` used to capture the message is touched only in
the panic path (never a hot loop), so it does not violate the AGENTS.md
lock-free hot-path rule.

### 3. Changes to `src/cli/train_cmd.rs`

- Add `--device` flag parsing.
- Remove hardcoded `type TrainBackend = Autodiff<NdArray>;`.
- Replace direct `train::<TrainBackend>(...)` call with `backend::select_and_train(...)`.
- Add `--device` to help text.

### 4. Changes to `src/training/mod.rs`

- Add `pub mod backend;`.

### 5. No Changes to Existing Training Code

The `train<B>()` function in `trainer.rs` is already generic over
`AutodiffBackend`. The `BurnPointNet<B>`, `bridge.rs`, and all other training
modules use generic `B: Backend` / `B: AutodiffBackend` bounds. **No changes to
these files are required.**

---

## Definition of Done

| # | Criterion | Verification |
|---|---|---|
| 1 | `cargo build --release --features training` — zero errors | Build gate |
| 2 | `cargo build --release --features gpu` — zero errors (alias for training) | Build gate |
| 3 | `cargo build --release` (no features) — zero errors; inference binary unaffected | Regression gate |
| 4 | `cargo clippy --features training -- -D warnings` — zero warnings | Clippy gate |
| 5 | `cargo fmt --check` passes | fmt gate |
| 6 | `--device cpu` always selects NdArray backend regardless of GPU availability | Manual / unit test |
| 7 | `--device auto` with no GPU selects NdArray and logs fallback | Manual |
| 8 | `--device auto` with GPU selects Wgpu (default device) and logs GPU usage | Manual (requires GPU) |
| 9 | `--device gpu` without `training` feature returns clear error | Manual |
| 10 | `--device gpu` with no GPU returns clear error | Manual |
| 11 | `--device` appears in `train --help` output | Manual |
| 12 | All existing tests pass with `--features training` | `cargo test --features training` |
| 13 | `panic = "unwind"` in release profile (required for `catch_unwind`) | Cargo.toml inspection |

---

*This document is the authoritative specification for Stage 09. Per the AGENTS.md
synchronization rule, any deviation between this spec and the implementation must
be reconciled immediately. The Stage 15 cleanup superseded the original
pre-flight VRAM estimator; this spec reflects the final minimal design.*