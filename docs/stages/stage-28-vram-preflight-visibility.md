# Stage 28 — GPU VRAM Pre-Flight Visibility (Informational Only)

**Status:** COMPLETE
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator
**Follows on from:** `stage-15-gpu-default-device-cleanup.md`,
`stage-16-gpu-memory-growth.md`, `stage-18-batchnorm-batched-forward.md`

---

## Goal

Close a genuine "informative logging" gap (AGENTS.md §4.3) that this stage's
own diagnostic journey exposed: when GPU training runs dramatically slower
than expected due to VRAM oversubscription, the tool gives the user **no
signal whatsoever** — no adapter identity, no estimate of the batched
workload size, nothing. The user is left to independently correlate Task
Manager VRAM graphs, GPU utilization telemetry, and thermal/fan behavior to
self-diagnose a problem the tool could have flagged directly.

This stage adds two purely **informational** pieces of visibility to the GPU
training path:

1. Log the selected/enumerated GPU adapter's identity (name, backend,
   device type) once, at GPU training start.
2. Log a one-time, best-effort **VRAM pre-flight estimate** when
   `--forward-batch-size × max block point count` exceeds an
   empirically-calibrated informational threshold, explaining the specific
   failure mode (WDDM VRAM spillover) and suggesting a remedy.

**Hard constraint (per Stage 15's hard-won lesson, and explicitly agreed with
the user before implementation):** this stage must **never** auto-adjust,
clamp, retry, or block training based on the estimate. No cubecl memory-pool
tuning, no custom wgpu limits/memory hints, no revival of anything resembling
the reverted Stages 10–14 allocator-tuning stack. The estimate is a `eprintln!`
message and nothing else — training proceeds completely unmodified regardless
of what it says.

---

## Background: the diagnostic journey that motivated this stage

The user reported: *"The GPU seems to not be engaging for training. It did
when I did a test run last night, but today it isn't."*

Two hypotheses were investigated and **ruled out** by direct evidence before
the true root cause was found:

1. **Silent CPU fallback under `--device auto`.** Ruled out: the user's run
   log showed `"[device] GPU detected — using Wgpu backend (--device auto)"` —
   the Wgpu path was in fact selected.
2. **Wrong/software adapter silently bound.** Investigated via `nvidia-smi`
   (RTX 2070 SUPER present, healthy, driver current, 8192 MiB VRAM) and Windows
   adapter enumeration. Ruled out once the user reported GPU utilization
   spiking 50–99% (genuine compute, not an idle/software adapter).

**Confirmed root cause: VRAM oversubscription.** The user reported Task
Manager showing "8GB dedicated memory... pretty much full, the 8GB shared
memory hovering around 2.5 GB" — Windows' combined Dedicated+Shared VRAM
display. The RTX 2070 SUPER has 8 GB of *dedicated* VRAM (confirmed via
`nvidia-smi`; this GPU model has never shipped with 16 GB). The "shared"
portion is VRAM overflow silently spilled by WDDM into system RAM, reachable
only over the much slower PCIe bus — Windows does not error on this, it just
gets very slow.

The workload driving this was `--forward-batch-size 32` with
`target_points = 5120`: 32 × 5,120 = 163,840 points stacked into one batched
`[B, N, C]` tensor pushed through the PointNet encoder/`Stn3d` T-Net (up to
1024-wide layers), with every intermediate activation retained for the
autodiff backward pass. This is a workload an order of magnitude larger than
the single-block (`B=1`) "~20 MB peak activation" estimate in Stage 15/16's
notes, which never anticipated Stage 18's later batched-forward mechanism
being pushed to `forward_batch_size = 32`.

**Empirical confirmation.** After the user manually lowered
`--forward-batch-size` from `32` to `16` in `lidar_classifier_script.ps1` and
re-ran:

> "I dropped it to batches of 16 and it's sitting at 7.6 GB, maintaining
> 55-70% GPU utilization, and has already finished 2 epochs. Also the fan has
> ramped up as the temp is up about 15 degrees."

This is a clean, textbook confirmation: VRAM now fits under the 8 GB dedicated
ceiling (no spillover), *sustained* (not bursty) utilization confirms
continuous real compute, and the thermal/fan ramp confirms the GPU is finally
being kept fed continuously rather than stalling on PCIe-bound spillover
traffic. **Diagnosis closed.**

This stage exists to make that diagnostic path visible to the tool's user
directly next time, instead of requiring an interactive back-and-forth
debugging session with GPU telemetry.

---

## Why an exact VRAM estimate is not attempted

`wgpu::AdapterInfo` (queried via `Adapter::get_info()`) exposes `name`,
`vendor`, `device`, `device_type`, `backend`, driver strings, and subgroup
size — **it does not expose total or available VRAM**. `wgpu::Limits` exposes
buffer/binding size *ceilings* (and Stage 16 found `max_buffer_size =
u64::MAX` on this hardware — an unbounded single-allocation limit, not a
capacity figure). There is no reliable, portable, dependency-free way to query
"how many bytes of VRAM does this adapter have" through the wgpu API surface
available here.

Rather than adding a heavy OS-specific VRAM-query dependency (violating
AGENTS.md's Minimal Dependencies and Platform Agnostic tenets) or inventing a
precise byte-accounting model of autodiff-retained activations (repeating
Stage 15's mistake of over-engineering a speculative fix around estimated
numbers that were never GPU-verified), this stage uses a **simple, transparent,
empirically-calibrated proxy metric**: total points fed into one batched
forward pass, i.e. `forward_batch_size × max_sampled_points_per_block`. The
two confirmed real-world data points bound the threshold directly:

| Points per batch | Observed VRAM behavior (RTX 2070 SUPER, 8 GB) |
|---|---|
| 81,920 (`fb=16 × 5,120`)  | 7.6 GB, sustained 55–70% utilization — safe |
| 163,840 (`fb=32 × 5,120`) | Oversubscription — dedicated VRAM full + shared spillover |

A warning threshold of **120,000 points per batched forward** is chosen: above
the confirmed-safe value, below the confirmed-bad value. This is a rough,
documented heuristic — not a physical VRAM model — and is presented to the
user as such.

---

## Inputs & Outputs

No CLI surface changes. Behavior changes only in `stderr` logging on the GPU
training path (`--device auto` with a GPU present, or `--device gpu`):

1. **Adapter identity line**, logged once at GPU training start:
   ```
   [device] GPU adapter: NVIDIA GeForce RTX 2070 SUPER (Vulkan, DiscreteGpu)
   ```
   If more than one adapter is enumerated, an additional note clarifies that
   the logged adapter is the first enumerated one and may not exactly match
   whichever adapter burn's `WgpuDevice::default()` internally binds (wgpu
   does not expose which adapter a `Device::default()` call resolved to; see
   "Known limitation" below).

2. **VRAM pre-flight warning**, logged once at GPU training start, only when
   `forward_batch_size × max_sampled_points_per_block > 120,000`:
   ```
   [device] VRAM pre-flight: --forward-batch-size 32 × max block size 5120 points
   = 163840 points per batched forward pass, above the 120000-point informational
   threshold. On 8 GB-class GPUs this configuration has been observed to
   oversubscribe VRAM (Windows WDDM silently spills into slower shared system
   memory rather than erroring), causing a severe training slowdown without a
   crash. This is informational only — training will proceed unmodified.
   Consider lowering --forward-batch-size if you observe reduced GPU
   utilization or dedicated VRAM saturation in Task Manager / nvidia-smi.
   ```

Both lines are silent on the CPU path (irrelevant there) and silent on the GPU
path when the estimate is under threshold.

### Known limitation

`wgpu::Instance::enumerate_adapters` is a separate call from whatever internal
adapter resolution `WgpuDevice::default()` performs inside `burn-wgpu`/cubecl.
On a single-discrete-GPU system (the common case, and the user's own
hardware) there is no ambiguity. On a multi-adapter system (e.g. a laptop with
an integrated + discrete GPU), the logged adapter is the first one enumerated
and is presented with an explicit caveat rather than a false claim of
certainty — consistent with AGENTS.md's "informative logging" requirement
without overclaiming precision the API cannot actually provide.

---

## Steps & Specifications

1. **`src/training/backend.rs`**
   - Add `fn log_gpu_adapter_info()` (`#[cfg(feature = "training")]`): calls
     `wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all())`
     (the same call already used by `gpu_is_available()` — no new API
     surface), logs the first adapter's `AdapterInfo { name, backend,
     device_type, .. }` via `eprintln!`, and appends the multi-adapter caveat
     when `adapters.len() > 1`. Handles the empty case defensively (should be
     unreachable given `gpu_is_available()` gated the call, but must not
     panic).
   - Add `const VRAM_PREFLIGHT_POINTS_PER_BATCH_WARN_THRESHOLD: usize =
     120_000;` with a doc comment citing the two empirical data points above.
   - Add `fn vram_preflight_check(dataset: &LabeledBlockDataset, config:
     &TrainConfig)` (`#[cfg(feature = "training")]`): computes
     `forward_batch_size.max(1).saturating_mul(dataset.max_sampled_points_per_block())`
     and emits the warning `eprintln!` when it exceeds the threshold; silent
     otherwise. Uses `saturating_mul` defensively (no panics on pathological
     config per AGENTS.md error-handling guardrails).
   - Call both new functions once at the top of `train_gpu_inner`, before
     device/model construction — this only runs on the actual GPU path (never
     on `train_cpu`), and runs exactly once per training invocation (not
     per-epoch or per-block, keeping stdout/stderr uncluttered per AGENTS.md
     "Informative Logging" guidance on high-throughput loops — this call site
     is invoked once per process run, not in a hot loop).

2. **`src/cli/train_cmd.rs`**
   - Extend the `--forward-batch-size` line in `print_usage()` with a short
     safe-value guidance note referencing the 120,000-point envelope, so users
     have this context without needing to hit the runtime warning first.

3. **`lidar_classifier_script.ps1`** (user-owned sweep-driver script, not part
   of the crate, but directly implicated in this stage's diagnosis)
   - Replace the single hardcoded `"--forward-batch-size", "16"` (safe only
     for the first sweep config's `target_points = 5120`) with a value
     computed per sweep config from the same 81,920-point safe anchor used
     above: `forward_batch_size = floor(81920 / target_points)`, giving `16`,
     `8`, `4`, `2` for the four configs (`5120`, `10240`, `20480`, `40960`
     target points respectively) — keeping every config's batched-forward
     workload in the same empirically-confirmed-safe envelope instead of
     only the first one.

---

## Definition of Done

- [x] `log_gpu_adapter_info()` added to `backend.rs`, logs adapter name/
      backend/device_type once at GPU training start; degrades gracefully
      (no panic) if enumeration is empty.
- [x] `vram_preflight_check()` added to `backend.rs`; fires exactly one
      `eprintln!` warning when `forward_batch_size × max_sampled_points_per_block
      > 120,000`; completely silent otherwise; never modifies `config` or
      training behavior in any way.
- [x] Both functions called exactly once, only on the GPU path
      (`train_gpu_inner`), never on `train_cpu`.
- [x] No new dependency added; no cubecl/wgpu allocator, memory-hint, or pool
      configuration introduced anywhere.
- [x] `print_usage()` in `train_cmd.rs` documents the safe
      `--forward-batch-size` envelope.
- [x] `lidar_classifier_script.ps1` computes `--forward-batch-size` per sweep
      config from the shared safe-points-per-batch anchor instead of a single
      hardcoded value.
- [x] `cargo build --features training` — zero errors.
- [x] `cargo test --features training` — all 92 existing unit tests + 1
      integration test pass (no new automated tests added this stage — see
      note below on why the pre-flight logic was verified by code review
      rather than a new unit test).
- [x] `cargo clippy --all-targets --features training -- -D warnings` —
      **the two files touched by this stage (`backend.rs`, `train_cmd.rs`)
      are fully clippy-clean.** The repo-wide clippy run surfaces ~111-121
      pre-existing warnings entirely inside `trainer.rs` (doc-markdown
      backticks, `usize`-to-`f32` cast-precision-loss, `cloned` vs `copied`,
      etc.) — confirmed via `git stash` to already exist on `main` before
      this stage's changes, i.e. unrelated pre-existing debt outside this
      stage's scope, not introduced or worsened here.
- [x] `cargo fmt --check` — clean (one pre-existing blank-line/EOF-newline
      formatting issue introduced by this stage's own edits to `backend.rs`
      was caught and fixed via `cargo fmt` before this checklist was closed).
- [x] The pre-flight threshold logic (`forward_batch_size.max(1).saturating_mul(max_sampled_points_per_block())
      > 120_000`) was verified by direct code review rather than an actual
      GPU run in this session (no GPU/dataset available in the
      implementation environment): at `--forward-batch-size 32` with
      `target_points=5,120`, `32 × 5,120 = 163,840 > 120,000` → warning
      fires; at `--forward-batch-size 16`, `16 × 5,120 = 81,920 < 120,000` →
      silent. This exactly reproduces the user's own empirically-confirmed
      real-world safe/unsafe boundary that motivated the threshold's
      calibration in the first place (see "Background" above).

## Results

Implemented exactly as specified above. `log_gpu_adapter_info()` and
`vram_preflight_check()` were added to `backend.rs` and are invoked once at
the top of `train_gpu_inner()`, strictly informational, with no effect on
training behavior regardless of what they report. `print_usage()` was
extended with the safe-envelope guidance. `lidar_classifier_script.ps1` now
computes `--forward-batch-size` per sweep config (`16`, `8`, `4`, `2` for
`5120`/`10240`/`20480`/`40960` target points respectively) from the shared
81,920-point empirically-safe anchor rather than a single hardcoded value
that was only correct for the first config.

Full verification was run from `lidar_point_cloud_classifier/`:
`cargo build --features training` (clean), `cargo test --features training`
(92 unit + 1 integration test, all pass), `cargo clippy --all-targets
--features training -- -D warnings` (clean on this stage's two touched
files; pre-existing unrelated `trainer.rs` warnings confirmed via
`git stash` to predate this stage), and `cargo fmt --check` (clean, after
`cargo fmt` fixed a blank-line/EOF-newline issue this stage's edits had
introduced into `backend.rs`).

---

*This document is the authoritative specification for Stage 28. Per the
AGENTS.md synchronization rule, the code and this spec must remain in sync.*
