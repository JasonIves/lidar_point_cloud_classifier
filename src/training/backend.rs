//! Backend selection — runtime GPU detection and dispatch.
//!
//! When compiled with the `training` feature, this module selects between
//! `Autodiff<Wgpu>` (GPU) and `Autodiff<NdArray>` (CPU) at runtime based on
//! hardware availability and the user's `--device` preference. Without the
//! `training` feature, only the CPU path is available.
//!
//! See `docs/stages/stage-09-gpu-support.md` for the GPU support spec and
//! `docs/stages/stage-15-gpu-default-device-cleanup.md` for the rationale
//! behind using burn's stock device configuration.
//!
//! # Design: use burn's default device
//!
//! GPU initialization intentionally uses burn's stock
//! [`WgpuDevice::default`] with the default cubecl runtime. This is the
//! configuration burn is designed and tested around; the PointNet training
//! workload here is small (largest single activation ≈ 20 MB), so no custom
//! wgpu limits, memory hints, or cubecl pool tuning are required.
//!
//! # Implementation note: `panic = "unwind"`
//!
//! The release profile in `Cargo.toml` **must** use `panic = "unwind"` (not
//! `"abort"`) so that [`std::panic::catch_unwind`] in
//! [`train_gpu_or_fallback`] can intercept a wgpu initialization/runtime panic
//! and fall back to the CPU backend under `--device auto`. With
//! `panic = "abort"` the process terminates immediately on any panic,
//! bypassing `catch_unwind` entirely.

#![allow(
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::unnecessary_wraps
)]

use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::training::dataset::LabeledBlockDataset;
use crate::training::trainer::{train, TrainConfig};

#[cfg(feature = "training")]
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "training")]
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Device preference
// ─────────────────────────────────────────────────────────────────────────────

/// User's preferred compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Auto-detect: use GPU if available, otherwise CPU.
    Auto,
    /// Force CPU (NdArray backend).
    Cpu,
    /// Force GPU (Wgpu backend). Errors if no GPU or `training` feature not compiled.
    Gpu,
}

impl DevicePreference {
    /// Parse from a CLI string (`auto`, `cpu`, `gpu`).
    ///
    /// # Errors
    /// Returns `ClassifierError::Pipeline` for unrecognized values.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(ClassifierError::Pipeline(format!(
                "train: --device '{other}' is invalid (expected auto, cpu, or gpu)"
            ))),
        }
    }

    /// Default is `Auto`.
    #[must_use]
    pub const fn default() -> Self {
        Self::Auto
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU detection (only when `training` feature is compiled)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect whether at least one GPU (or GPU-like adapter) is available.
///
/// Uses `wgpu::Instance::enumerate_adapters` which is synchronous and returns
/// all discoverable graphics/compute adapters on the system.
#[cfg(feature = "training")]
fn gpu_is_available() -> bool {
    let instance = wgpu::Instance::default();
    !instance
        .enumerate_adapters(wgpu::Backends::all())
        .is_empty()
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 28: VRAM pre-flight visibility (informational only)
// ─────────────────────────────────────────────────────────────────────────────
//
// See docs/stages/stage-28-vram-preflight-visibility.md for the full
// diagnostic journey and rationale. Both functions below are strictly
// informational: they never alter `config`, never retry, never clamp, and
// never block training. They exist solely to close an "informative logging"
// gap (AGENTS.md) identified while diagnosing a real GPU VRAM-oversubscription
// slowdown.

/// Log the identity of the first enumerated GPU adapter (name, backend,
/// device type) once at GPU training start.
///
/// `wgpu` provides no way to know which adapter `WgpuDevice::default()` will
/// actually bind internally, so on multi-adapter systems this is presented
/// with an explicit caveat rather than a false claim of certainty. Never
/// panics: an empty adapter list (unreachable in practice, since callers only
/// reach this after `gpu_is_available()` returned true) is handled gracefully.
#[cfg(feature = "training")]
fn log_gpu_adapter_info() {
    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    let Some(first) = adapters.first() else {
        eprintln!("[device] GPU adapter: none enumerated (unexpected — proceeding anyway)");
        return;
    };

    let info = first.get_info();
    eprintln!(
        "[device] GPU adapter: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    if adapters.len() > 1 {
        eprintln!(
            "[device] Note: {} adapters were enumerated; the logged adapter is the first one \
             found and may not be the one burn's WgpuDevice::default() actually binds.",
            adapters.len()
        );
    }
}

/// Informational threshold for total points fed into one batched forward pass
/// (`forward_batch_size × max_sampled_points_per_block`), above which a
/// one-time VRAM-oversubscription advisory is logged.
///
/// Calibrated from two empirically-confirmed data points on an 8 GB-class GPU
/// (RTX 2070 SUPER), gathered while diagnosing a real training slowdown (see
/// `docs/stages/stage-28-vram-preflight-visibility.md`):
/// - 81,920 points/batch (`forward_batch_size=16 × 5,120` target points):
///   confirmed safe — 7.6 GB VRAM, sustained 55-70% GPU utilization.
/// - 163,840 points/batch (`forward_batch_size=32 × 5,120` target points):
///   confirmed oversubscribed — dedicated VRAM full, spilling into slower
///   shared system memory via WDDM (no crash, just a severe slowdown).
///
/// This is a rough, transparent proxy metric, not a physical VRAM byte model:
/// wgpu exposes no portable way to query total adapter VRAM (see the stage
/// doc's "Why an exact VRAM estimate is not attempted" section).
#[cfg(feature = "training")]
const VRAM_PREFLIGHT_POINTS_PER_BATCH_WARN_THRESHOLD: usize = 120_000;

/// Log a one-time informational warning when the configured
/// `forward_batch_size` combined with the dataset's largest block size would
/// push the batched-forward workload above the empirically-calibrated
/// [`VRAM_PREFLIGHT_POINTS_PER_BATCH_WARN_THRESHOLD`].
///
/// Purely informational: never modifies `config`, never blocks or delays
/// training. Uses `saturating_mul` so a pathological configuration cannot
/// panic via overflow (AGENTS.md error-handling guardrails).
#[cfg(feature = "training")]
fn vram_preflight_check(dataset: &LabeledBlockDataset, config: &TrainConfig) {
    let fb = config.forward_batch_size.max(1);
    let max_points = dataset.max_sampled_points_per_block();
    let points_per_batch = fb.saturating_mul(max_points);

    if points_per_batch > VRAM_PREFLIGHT_POINTS_PER_BATCH_WARN_THRESHOLD {
        eprintln!(
            "[device] VRAM pre-flight: --forward-batch-size {fb} × max block size {max_points} \
             points = {points_per_batch} points per batched forward pass, above the \
             {VRAM_PREFLIGHT_POINTS_PER_BATCH_WARN_THRESHOLD}-point informational threshold. \
             On 8 GB-class GPUs this configuration has been observed to oversubscribe VRAM \
             (Windows WDDM silently spills into slower shared system memory rather than \
             erroring), causing a severe training slowdown without a crash. This is \
             informational only — training will proceed unmodified. Consider lowering \
             --forward-batch-size if you observe reduced GPU utilization or dedicated VRAM \
             saturation in Task Manager / nvidia-smi."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the device preference and dispatch to the appropriate training backend.
///
/// Selects between GPU (`Autodiff<Wgpu>`) and CPU (`Autodiff<NdArray>`) at
/// runtime, then calls `train::<B>()`.
///
/// # Errors
/// - `--device gpu` without the `training` Cargo feature compiled in.
/// - `--device gpu` when no GPU adapter is found at runtime.
/// - Any error propagated from the training loop itself.
pub fn select_and_train(
    dataset: &LabeledBlockDataset,
    config: &TrainConfig,
    preference: DevicePreference,
) -> Result<PathBuf> {
    match preference {
        DevicePreference::Cpu => {
            eprintln!("[device] CPU selected (user-specified --device cpu)");
            train_cpu(dataset, config)
        }
        DevicePreference::Gpu => {
            #[cfg(feature = "training")]
            {
                if gpu_is_available() {
                    eprintln!("[device] GPU detected and selected (--device gpu)");
                    train_gpu_or_fallback(dataset, config, false)
                } else {
                    Err(ClassifierError::Pipeline(
                        "train: --device gpu was requested but no GPU adapter was found. \
                         Use --device auto for graceful CPU fallback, or --device cpu."
                            .into(),
                    ))
                }
            }
            #[cfg(not(feature = "training"))]
            {
                let _ = (dataset, config);
                Err(ClassifierError::Pipeline(
                    "train: --device gpu was requested but the binary was not compiled \
                     with GPU support. Rebuild with: cargo build --features training"
                        .into(),
                ))
            }
        }
        DevicePreference::Auto => {
            #[cfg(feature = "training")]
            {
                if gpu_is_available() {
                    eprintln!("[device] GPU detected — using Wgpu backend (--device auto)");
                    train_gpu_or_fallback(dataset, config, true)
                } else {
                    eprintln!(
                        "[device] No GPU detected — falling back to CPU NdArray backend \
                         (--device auto)"
                    );
                    train_cpu(dataset, config)
                }
            }
            #[cfg(not(feature = "training"))]
            {
                let _ = (dataset, config);
                eprintln!(
                    "[device] GPU support not compiled in — using CPU NdArray backend \
                     (--device auto)"
                );
                train_cpu(dataset, config)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU training path (always available)
// ─────────────────────────────────────────────────────────────────────────────

fn train_cpu(dataset: &LabeledBlockDataset, config: &TrainConfig) -> Result<PathBuf> {
    use burn::backend::{Autodiff, NdArray};

    type CpuBackend = Autodiff<NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();
    train::<CpuBackend>(dataset, config, &device)
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU training path (only when `training` feature is compiled)
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt GPU training, catching a wgpu panic and falling back to CPU when
/// `allow_fallback` is true.
///
/// When `allow_fallback` is `false` (i.e. `--device gpu` was explicit), a
/// panic is converted to an error rather than silently falling back.
///
/// # Requirement: `panic = "unwind"`
///
/// This function relies on [`std::panic::catch_unwind`] to intercept wgpu
/// panics. The crate's release profile **must** set `panic = "unwind"` (not
/// `"abort"`); otherwise `catch_unwind` is a no-op and the process will abort
/// on the first panic.
#[cfg(feature = "training")]
fn train_gpu_or_fallback(
    dataset: &LabeledBlockDataset,
    config: &TrainConfig,
    allow_fallback: bool,
) -> Result<PathBuf> {
    // Install a temporary panic hook that captures the panic message,
    // suppressing the default (scary) output. The original hook is restored
    // after the catch_unwind block. This Mutex is only touched in the panic
    // path (not a hot loop), so it does not violate the AGENTS.md
    // "lock-free hot path" rule.
    let panic_msg = Arc::new(Mutex::new(None::<String>));
    let panic_msg_for_hook = Arc::clone(&panic_msg);

    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        if let Ok(mut guard) = panic_msg_for_hook.lock() {
            *guard = Some(msg);
        }
    }));

    let result = panic::catch_unwind(AssertUnwindSafe(|| train_gpu_inner(dataset, config)));

    // Restore the previous panic hook.
    panic::set_hook(prev_hook);

    match result {
        Ok(inner_result) => inner_result,
        Err(panic_payload) => {
            let msg = panic_msg
                .lock()
                .ok()
                .and_then(|guard| guard.clone())
                .unwrap_or_else(|| {
                    if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown wgpu panic".to_string()
                    }
                });

            if allow_fallback {
                eprintln!(
                    "[device] GPU training failed: {msg}\n\
                     [device] Falling back to CPU NdArray backend (--device auto)."
                );
                train_cpu(dataset, config)
            } else {
                Err(ClassifierError::Pipeline(format!(
                    "GPU training panicked: {msg}. \
                     Use --device auto for automatic CPU fallback, or --device cpu."
                )))
            }
        }
    }
}

/// Run training on the GPU using burn's stock [`WgpuDevice::default`].
///
/// No custom wgpu limits, memory hints, or cubecl runtime tuning are applied:
/// the default device is the configuration burn is designed and tested around,
/// and the training workload here is small enough that it needs nothing more.
#[cfg(feature = "training")]
fn train_gpu_inner(dataset: &LabeledBlockDataset, config: &TrainConfig) -> Result<PathBuf> {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::{Autodiff, Wgpu};

    type GpuBackend = Autodiff<Wgpu>;

    // Stage 28 (informational only — see docs/stages/stage-28-vram-preflight-visibility.md):
    // log the bound adapter's identity and, if the configured batched-forward
    // workload is large enough to risk VRAM oversubscription on 8 GB-class
    // GPUs, a one-time advisory. Neither call alters `config` or training
    // behavior in any way.
    log_gpu_adapter_info();
    vram_preflight_check(dataset, config);

    let device = WgpuDevice::default();
    train::<GpuBackend>(dataset, config, &device)
}
