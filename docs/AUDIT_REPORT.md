# Comprehensive Audit Report — LiDAR Point Cloud Classifier

**Date:** 2026-07-01  
**Auditor:** Lead Architect (AI)  
**Scope:** Full codebase — `lidar_point_cloud_classifier/` and integration with `whitebox_next_gen/`

---

## Executive Summary

The codebase is well-structured, follows the AGENTS.md spec-driven development model, and demonstrates solid engineering practices (lock-free parallelism, graceful error handling, spatially-disjoint train/val splits). However, the audit identified **one critical violation** of the AGENTS.md GPU-utilization mandate, several **high-severity** performance and security gaps, and numerous **medium/low** improvement opportunities. This report is organized by category with severity ratings, evidence, and actionable recommendations.

---

## 1. GPU Utilization & Training Efficiency

### 1.1 No GPU Support — Direct AGENTS.md Violation (CRITICAL) ✅ RESOLVED

**Status:** Resolved in Stage 09. See `docs/stages/stage-09-gpu-support.md`.

**Finding:** Training is hardcoded to CPU-only `Autodiff<NdArray>`. There is zero GPU detection, no GPU backend feature flags, and no fallback mechanism.

**Evidence:**
- `Cargo.toml` line 56: `burn = { version = "0.16", features = ["ndarray", "autodiff"], optional = true }` — only the NdArray (CPU) backend is enabled.
- `src/cli/train_cmd.rs` line 20: `type TrainBackend = Autodiff<NdArray>;` — hardcoded.
- `src/cli/train_cmd.rs` line 204: `let device = burn::backend::ndarray::NdArrayDevice::default();` — hardcoded CPU device.
- No `--device` or `--backend` CLI flag exists.

**AGENTS.md violation:** *"The Rule: If a GPU is available, it may be utilized for acceleration; if it is absent, the tool must fallback gracefully and run efficiently on the CPU."*

**Recommendation:**
- Add burn's `wgpu` feature (cross-platform GPU via Vulkan/Metal/DX12) as an optional Cargo feature: `burn = { version = "0.16", features = ["wgpu", "autodiff"], optional = true }`.
- Create a `BackendSelector` that detects GPU availability at runtime and selects `Autodiff<Wgpu>` vs `Autodiff<NdArray>`.
- Add `--device` CLI flag (`auto`, `cpu`, `gpu`).
- Use burn's `B::Device` abstraction so the trainer already accepts any `AutodiffBackend` — the generic `train::<B>()` signature is already correct; only the CLI hardcodes the type.

**Effort:** Medium  
**Priority:** P0

---

### 1.2 No Mixed-Precision Training (HIGH) ⏸️ DEFERRED

**Status:** Researched in Stage 23. See `docs/stages/stage-23-mixed-precision-inference-batching.md`.

**Finding:** No fp16/bf16 support. Burn supports mixed-precision via `Autodiff<Wgpu>` with `f16` tensor types. For PointNet with ~2M parameters, mixed precision could yield 2-3× training speedup on GPU.

**Recommendation:** Add `--mixed-precision` flag that casts tensors to `f16` for forward/backward and keeps `f32` master weights.

**Deferral rationale (Stage 23, 2026-07-02):** This project is pinned to
`burn = 0.16.1`, whose `wgpu` backend has **no f16 tensor support** — this
was an open upstream enhancement request (tracel-ai/burn#597), not
implemented until burn 0.17/0.18 (two major version bumps away). Upgrading
burn is a large, risky change touching every burn-consuming file
(`burn_model.rs`, `trainer.rs`, `bridge.rs`, `backend.rs`) and risks
regressing Stages 09/16/17/18/22's hard-won GPU-memory/BatchNorm fixes, for
a speculative, GPU-only, unverified-for-this-model speedup. Deferred pending
a deliberate, standalone burn-version-upgrade decision — not implemented in
Stage 23.

**Effort:** Medium (originally estimated; actual effort is High once the
required burn version bump is accounted for)  
**Priority:** P1 — deferred, not actionable without a separate burn-upgrade decision

---

### 1.3 No Parallel Data Loading (HIGH) ✅ RESOLVED

**Status:** Resolved in Stage 22. See `docs/stages/stage-22-training-loop-enhancements.md`.

**Finding:** The training loop loads blocks sequentially within each batch chunk (`for &block_id in chunk { dataset.load_block(block_id) }`). Each `load_block` does synchronous file I/O. There is no prefetching or async loading.

**Recommendation:** Implement a background prefetch thread (or use `rayon::spawn` with a bounded channel) to load the next batch's blocks while the current batch computes forward/backward.

**Resolution:** All blocks in a micro-batch are now loaded concurrently via `rayon::prelude::*`'s `par_iter()` (the same pattern already used for Stage 21 item 2.3), instead of a background-prefetch-thread/channel architecture — see the stage spec's Background section for the explicit AGENTS.md "Lightweight"/"Lock-Free Progress" tradeoff analysis behind this design choice. Batch assembly (dims validation, `batch_flat`/`batch_labels` mutation) remains single-threaded.

**Effort:** Medium  
**Priority:** P1 — closed, no longer actionable


### 1.4 Full Model Clone Every Validation Epoch (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 16. See `docs/stages/stage-16-gpu-memory-growth.md`.

**Annotation (2026-07-02):** This finding is **stale** relative to the codebase at
audit time. Stage 16 (root-causing the GPU OOM-during-validation bug) replaced
the full `model.clone()` with `model.valid()` — burn's `AutodiffModule::valid()`
conversion to `B::InnerBackend` — exactly the fix this finding recommends. This
predates the 2026-07-01 audit date but the report was apparently drafted without
cross-referencing the Stage 16 diff. Current `trainer.rs::validate_epoch()`
calls `model.valid()`, not `model.clone()`; no further action needed here.

**Original finding (superseded):** `validate_epoch()` uses `B: AutodiffBackend` and constructs autodiff tensors, wasting memory and time even though `.backward()` is never called during validation.

**Original recommendation (already implemented):** Use burn's `model.valid()` mode (eval mode) which switches BN to use running statistics without cloning.

**Effort:** Medium  
**Priority:** P2 — closed, no longer actionable

---


### 1.5 No Early Stopping (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 22. See `docs/stages/stage-22-training-loop-enhancements.md`.

**Finding:** Training runs all epochs unconditionally. No patience-based early stopping when val mIoU plateaus.

**Recommendation:** Add `--early-stopping-patience <N>` flag; stop if val mIoU doesn't improve for N epochs.

**Resolution:** Added `TrainConfig::early_stopping_patience: Option<usize>` (default `None`, disabled) and a new pure helper `early_stopping_step(...)` that tracks a best-mIoU/no-improvement counter independently of the checkpoint-cadence-gated `best_miou`, called once per epoch after validation; the epoch loop `break`s once `patience` consecutive epochs pass without improvement. Wired to a new `--early-stopping-patience <usize>` CLI flag.

**Effort:** Low  
**Priority:** P2 — closed, no longer actionable

---

### 1.6 No Learning Rate Warmup (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 22. See `docs/stages/stage-22-training-loop-enhancements.md`.

**Finding:** Cosine annealing starts at full LR immediately. For PointNet with T-Net sub-networks, warmup (linear ramp over first N steps) stabilizes early training.

**Recommendation:** Add `--warmup-steps <N>` flag; linearly ramp LR from 0 to `learning_rate` over the first N steps, then cosine anneal.

**Resolution:** `CosineScheduler` gained a `with_warmup(lr_max, lr_min, total_steps, warmup_steps)` constructor; `lr(t)` ramps linearly during `t < warmup_steps`, then runs the existing cosine formula re-based over the post-warmup remainder. `CosineScheduler::new(...)` now delegates to `with_warmup(..., 0)`, preserving prior behavior exactly. Wired to a new `--warmup-steps <usize>` CLI flag (default `0`, disabled).

**Effort:** Low  
**Priority:** P3 — closed, no longer actionable

---

### 1.7 No Gradient Clipping (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 22. See `docs/stages/stage-22-training-loop-enhancements.md`.

**Finding:** No gradient norm clipping. T-Net sub-networks can produce large gradients causing instability.

**Recommendation:** Add `--grad-clip-norm <f32>` flag; clip gradients before optimizer step.

**Resolution:** Added `TrainConfig::grad_clip_norm: Option<f32>` (default `None`, disabled); the AdamW optimizer is now constructed with `.with_grad_clipping(config.grad_clip_norm.map(burn::grad_clipping::GradientClippingConfig::Norm))`, using burn's own built-in per-parameter-tensor L2-norm clipping applied internally at optimizer-step time rather than a hand-rolled global-norm implementation. Wired to a new `--grad-clip-norm <f32>` CLI flag with `> 0.0`/finite validation.

**Effort:** Low  
**Priority:** P2 — closed, no longer actionable

---

## 2. Performance

### 2.1 O(n) Linear Scan in `load_block` (HIGH) ✅ RESOLVED

**Status:** Resolved in Stage 21. See `docs/stages/stage-21-load-path-performance.md`.

**Finding:** `dataset.rs` line 362-364: `entry.manifest.blocks.iter().find(|b| b.meta.id == local_id)` — linear scan through all blocks for every block load. During training, this is called thousands of times per epoch.

**Recommendation:** Build a `HashMap<u64, usize>` (local_id → block index) per directory at load time.

**Resolution:** Added a `block_index: HashMap<u64, usize>` field to `DirEntry`, built once per directory in `load()`. `load_block()` now resolves the local block ID via `entry.block_index.get(&local_id).and_then(|&i| entry.manifest.blocks.get(i))` instead of a linear scan. Covered by `test_block_index_hit_and_miss`.

**Effort:** Low  
**Priority:** P1

---

### 2.2 `class_counts_train()` Rebuilds HashSet Every Call (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 21. See `docs/stages/stage-21-load-path-performance.md`.

**Finding:** `dataset.rs` line 321: `let train_set: HashSet<u64> = self.train_ids.iter().copied().collect();` — allocates a HashSet and iterates all blocks in all directories. Called once during training setup, but wasteful.

**Recommendation:** Cache the HashSet or compute counts during `load()`.

**Resolution:** Added a `train_set: HashSet<u64>` field to `LabeledBlockDataset`, built once in `load()` from `train_ids`. `class_counts_train()` now references `&self.train_set` instead of rebuilding it on every call.

**Effort:** Low  
**Priority:** P2

---

### 2.3 No Parallelism in Feature Extraction (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 21. See `docs/stages/stage-21-load-path-performance.md`.

**Finding:** `feature_extractor.rs` `extract_features()` processes points sequentially. The eigenvalue decomposition per point is independent.

**Recommendation:** Use `rayon::par_iter()` over points. The AGENTS.md explicitly recommends Rayon for embarrassingly parallel tasks.

**Resolution:** `extract_features()`'s per-point loop was converted to `scalar.into_par_iter().zip(pts.par_iter()).map(...).collect()` via `rayon::prelude::*`. `BlockSpatialIndex` and `all_pts` are shared read-only by reference across worker threads. Output row order/values confirmed unchanged by `test_extract_features_output_shape_and_range`, `test_extract_features_multi_scale_width`, and `test_extract_features_single_radius_matches_legacy_width`.

**Effort:** Low  
**Priority:** P2

---

### 2.4 Byte-by-Byte f32 Conversion (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 21. See `docs/stages/stage-21-load-path-performance.md`.

**Finding:** `inference.rs` lines 220-223 and `dataset.rs` lines 493-496: `buf.chunks_exact(4).map(|b| f32::from_le_bytes(...))` — the project already depends on `bytemuck` but doesn't use it here.

**Recommendation:** Use `bytemuck::cast_slice::<u8, f32>(&buf)` for zero-copy conversion (after alignment check).

**Resolution:** Both `dataset.rs::load_feat_file()` and `inference.rs::process_block()` now use `bytemuck::try_cast_slice::<u8, f32>(&buf)` (the non-panicking variant, preserving the no-panics rule), with a fallback to the original manual per-chunk conversion on the (unreachable in practice) misalignment case.

**Effort:** Low  
**Priority:** P3

---

### 2.5 SWA Loads All Checkpoints Into Memory Simultaneously (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 22. See `docs/stages/stage-22-training-loop-enhancements.md`.

**Finding:** `trainer.rs` `apply_swa()` loads all retained checkpoint models into a `Vec` at once. With 5 checkpoints × ~2M parameters, this is ~40 MB — acceptable for 5, but doesn't scale.

**Recommendation:** Stream-accumulate: load one model at a time, add its weights to the running sum, drop it.

**Resolution:** `apply_swa()` refactored to load only the first checkpoint into `base`, then stream-load and accumulate each remaining checkpoint into a per-iteration local `m` (dropped at the end of each loop iteration) across all layers (encoder, decoder, class projection, both T-Nets), instead of collecting every retained checkpoint into a `Vec` up front. Memory footprint is now O(2 models) instead of O(keep_best_n + 1); the existing `test_swa_averages_tnet_weights` test passes unmodified, confirming numerically identical averaged output.

**Effort:** Low  
**Priority:** P2 — closed, no longer actionable


### 2.6 No Inference Batching (MEDIUM) ⏸️ DEFERRED

**Status:** Researched in Stage 23. See `docs/stages/stage-23-mixed-precision-inference-batching.md`.

**Finding:** `inference.rs` processes one block at a time. Multiple blocks could be batched into a single forward pass (with padding/masking) for better GPU utilization.

**Deferral rationale (Stage 23, 2026-07-02):** The deployed inference engine
(`PointNetClassifier` in `model/pointnet.rs`/`model/layers.rs`) is a
separate, pure-`ndarray`, **CPU-only** implementation with no GPU code path
at all — the "better GPU utilization" motivation in the original finding
does not apply to it (GPU acceleration exists only on the training side, via
the unrelated burn-based `BurnPointNet<B>`). Blocks are already resampled to
a fixed `target_points` count, so padding/masking would not actually be
needed as originally assumed. However, `run_inference()` already
parallelizes across blocks via Rayon (`manifest.blocks.par_iter()`), so all
CPU cores are already utilized; a genuinely correct batched forward pass
would additionally require new segment-aware (per-block) global-max-pool
and T-Net handling — a nontrivial rewrite of the same forward-pass code that
caused the Stage 17 BatchNorm logit-explosion regression — for a modest,
unverified CPU-only throughput gain. Deferred pending real-world profiling
evidence that per-block Rayon task overhead is an actual bottleneck.

**Effort:** Medium (originally estimated; actual effort is higher once
per-block segment-aware pooling/T-Net correctness is accounted for)  
**Priority:** P2 — deferred, not currently scheduled

---

### 2.7 `kdtree` Crate Not Cache-Friendly (LOW) ⏸️ DEFERRED

**Status:** Triaged in Stage 26. See `docs/stages/stage-26-remaining-findings-triage.md`.

**Finding:** The `kdtree` crate uses heap-allocated nodes with poor cache locality. For the 2-D nearest-neighbor queries in inference, a simple sorted-grid or R-tree would be more cache-friendly.

**Recommendation:** Consider a flat sorted-array + binary search for 2-D nearest neighbor, or use the existing spatial index from preprocessing.

**Deferral rationale (Stage 26, 2026-07-02):** `kdtree` is used in three
separate call sites (`preprocessing/spatial_index.rs`, `preprocessing/
outlier_filter.rs`, `model/inference.rs`), each with its own existing test
coverage that would need re-validating after a swap. There is no profiling
evidence in this repository that any of the three is an actual throughput
bottleneck — this finding is a code-quality observation, not a measured
regression. Deferred pending real profiling evidence identifying one of
these call sites as a hot path.

**Effort:** Medium  
**Priority:** P3 — deferred, not currently scheduled

---

## 3. Security & Robustness

### 3.1 `assert!` in Production Code (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** `bridge.rs` line 165: `assert!(d_in > 0 && d_out > 0, ...)` — violates AGENTS.md "No Panics in Production" rule.

**Recommendation:** Replace with `if d_in == 0 || d_out == 0 { return Err(ClassifierError::Pipeline(...)); }`.

**Resolution:** Implemented exactly as recommended in `bridge.rs::extract_linear()`.

**Effort:** Low  
**Priority:** P2

---

### 3.2 No File Size Validation (HIGH) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** `dataset.rs` `load_feat_file()` and `inference.rs` `process_block()` allocate `vec![0u8; n_f32 * 4]` based on the header's `n_points * n_features` without any size cap. A malicious or corrupted `.feat` file with `n_points = u32::MAX` would attempt to allocate ~16 GB.

**Recommendation:** Add a maximum block size constant (e.g., 1M points × 100 features = 400 MB cap) and validate before allocation.

**Resolution:** Added `MAX_FEAT_PAYLOAD_BYTES` (512 MB) constant in `preprocessing/mod.rs`; both load paths reject any payload exceeding it before allocating. Covered by `test_load_feat_file_rejects_oversized_header_before_allocating`.

**Effort:** Low  
**Priority:** P1

---

### 3.3 Integer Overflow in Size Computation (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** `dataset.rs` line 485: `let n_f32 = n_points * n_features;` — on 32-bit targets, this can overflow `usize`. On 64-bit it's safe but the multiplication should use checked arithmetic.

**Recommendation:** `n_points.checked_mul(n_features).ok_or_else(|| ...)?`.

**Resolution:** Implemented exactly as recommended in both `dataset.rs::load_feat_file()` and `inference.rs::process_block()`.

**Effort:** Low  
**Priority:** P2

---

### 3.4 No Path Traversal Validation (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** Block file names from manifests are joined directly to directory paths. A malicious manifest with `file: "../../../etc/passwd"` could read arbitrary files.

**Recommendation:** Validate that block file names don't contain path separators or `..`.

**Resolution:** Added a single canonical `pub fn preprocessing::validate_block_filename()`, called from both `dataset.rs::load_block()` and `inference.rs::process_block()` before any path join. Covered by four dedicated tests.

**Effort:** Low  
**Priority:** P3

---

### 3.5 `.lbl` File Size Not Validated Against Header (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** `load_lbl_file()` reads exactly `n_points` bytes from the file but doesn't verify the file is at least that large. A truncated file would produce a misleading I/O error rather than a clear validation message.

**Resolution:** `load_lbl_file()` now checks `f.metadata()?.len()` against the expected byte count before `read_exact`, returning a clear "is truncated" error. Covered by `test_load_lbl_file_rejects_truncated_file`.

**Effort:** Low  
**Priority:** P3

---


## 4. Code Quality & Maintainability

### 4.1 Excessive Clippy Suppressions (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 24. See `docs/stages/stage-24-code-quality-cleanup.md`.

**Finding:** Multiple files begin with large `#![allow(...)]` blocks suppressing 10+ lints. This masks potential issues.

**Evidence:** `trainer.rs` suppresses 13 lints, `dataset.rs` suppresses 8, `burn_model.rs` suppresses 6.

**Recommendation:** Address the underlying issues where feasible; reduce `allow` scopes to specific functions rather than entire modules.

**Resolution:** Module-level `#![allow(...)]` blocks removed from all three files. Trivial lints (`doc_markdown` backticks, `must_use_candidate`, `missing_panics_doc`, `manual_is_multiple_of`, a too-similar binding name) were fixed directly instead of suppressed. Lints pervasive within one large, tightly-coupled function (`too_many_lines`, `cast_precision_loss`/`cast_possible_truncation` on bounded numeric conversions, `too_many_arguments`/`unnecessary_wraps` on `validate_epoch()`, `struct_excessive_bools` on the public `TrainConfig`) were demoted to function-level `#[allow(...)]` with an inline justification comment each. `burn_model.rs` had no module-level suppression by the time of final verification. `cargo clippy --features training` (lib-only, production code) now reports zero warnings anywhere in `lidar_point_cloud_classifier`'s own lib target (a bonus, same-class fix was also applied to an unrelated pre-existing `empty_line_after_doc_comments` warning in `preprocessing/mod.rs`, discovered during the same verification pass).

**Effort:** Medium  
**Priority:** P2

---

### 4.2 Duplicate T-Net Extraction Functions (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 24. See `docs/stages/stage-24-code-quality-cleanup.md`.

**Finding:** `bridge.rs` `extract_tnet3d()` and `extract_tnet64d()` are nearly identical (differ only in `k` value and struct field names).

**Recommendation:** Generic `extract_tnet<B>(stn: &dyn TNetLike, k: usize, use_bn: bool)` or a trait.

**Resolution:** Added a private `extract_tnet_generic()` helper taking each `Stn3d`/`Stn64d` field by reference plus `k`/`use_bn`; `extract_tnet3d()`/`extract_tnet64d()` are now thin one-call wrappers. A `dyn`/trait-object approach was deliberately avoided (heavier than needed for 11 shared-name fields, against AGENTS.md's "Lightweight"/"Minimal dependencies" tenets). Covered by `test_weight_bridge_round_trip` (passing, confirming byte-for-byte identical behavior).

**Effort:** Low  
**Priority:** P3

---

### 4.3 SWA Macros Are Complex and Fragile (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 24. See `docs/stages/stage-24-code-quality-cleanup.md`.

**Finding:** `trainer.rs` `apply_swa()` uses 4 macros (`accum_linear!`, `divide_linear!`, `accum_bn!`, `divide_bn!`) with extensive manual field-by-field accumulation. This is hard to maintain and error-prone.

**Recommendation:** Implement a `WeightAveraging` trait on `Linear` and `BatchNorm1d` types.

**Resolution:** Implemented exactly as recommended — `model/layers.rs` gained an additive `WeightAveraging` trait (`accumulate`/`finalize`) implemented for `Linear` and `BatchNorm1d`, with no existing method touched (zero behavioral risk to the deployed CPU-only inference engine). `apply_swa()`'s four `macro_rules!` macros were removed and replaced with direct `.accumulate(...)`/`.finalize(n)` trait calls in the same per-checkpoint/per-layer traversal order, so the floating-point accumulation order — and therefore the averaged output — is numerically unchanged. Covered by `test_swa_averages_tnet_weights` and `test_swa_averaging` (both passing).

**Effort:** Medium  
**Priority:** P3

---


### 4.4 Manual CLI Argument Parsing (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 20. See `docs/stages/stage-20-security-hardening.md`.

**Finding:** `train_cmd.rs` uses hand-rolled argument parsing with index-based access (`args[i]`). Missing value for a flag causes an index-out-of-bounds panic.

**Evidence:** Line 37: `data_dirs.push(PathBuf::from(&args[i]));` — if `--data-dir` is the last argument, `args[i]` panics.

**Recommendation:** Add bounds checking: `if i + 1 >= args.len() { return Err(...); }` for every value-taking flag. Or use a lightweight argument parser.

**Resolution:** Both `train_cmd.rs` and `preprocess_labeled_cmd.rs` rewritten to use a bounds-checked `next_value()` helper (mirroring the existing `preprocess_cmd.rs` pattern) for every value-taking flag. Covered by dedicated trailing-flag-without-value tests in both modules.

**Effort:** Low  
**Priority:** P2

---


## 5. Architecture & Design

### 5.1 No Model Quantization for Inference (MEDIUM) ⏸️ DEFERRED

**Status:** Triaged in Stage 26. See `docs/stages/stage-26-remaining-findings-triage.md`.

**Finding:** Inference uses f32 weights. For deployment, int8 quantization would reduce model size by 4× and speed up CPU inference.

**Deferral rationale (Stage 26, 2026-07-02):** Quantization would require a
new post-training calibration step, a quantization-aware (or
dequantize-on-the-fly) forward pass, and accuracy-regression validation
(mIoU/OA) against the f32 baseline — a substantial standalone feature
touching the same forward-pass code implicated in the Stage 17 BatchNorm
logit-explosion regression. There is no currently-observed model-size or
CPU-inference-latency constraint driving this. Deferred pending a concrete
deployment requirement (e.g., a measured model-size limit or inference
latency budget).

**Effort:** High  
**Priority:** P3 — deferred, not currently scheduled

---

### 5.2 No Block Caching During Training (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 27. See `docs/stages/stage-27-block-caching.md`.

**Finding:** Each epoch re-reads all `.feat` and `.lbl` files from disk. For datasets that fit in memory, caching blocks would eliminate redundant I/O.

**Recommendation:** Add `--cache-blocks` flag for in-memory caching with LRU eviction.

**Investigation (Stage 26, 2026-07-02):** Checked how `whitebox_next_gen`
itself implements in-memory storage (`wblidar::memory_store`,
`wbraster::memory_store`) to determine the idiomatic "AGENTS.md-lightweight"
caching approach for this codebase — no caching crate (`moka`, `dashmap`,
`quick_cache`, `lru`) appears anywhere in `whitebox_next_gen`'s dependency
tree. Both modules use an identical minimal pattern: a single stdlib
`std::sync::Mutex<HashMap<String, Arc<T>>>` behind a `OnceLock`, with no
eviction policy, size cap, or TTL — entries persist for the process
lifetime. This is a different use case (a session-scoped intermediate-
result registry for chaining tool outputs, vs. repeatedly re-reading a
fixed, bounded set of training blocks across epochs), but it directly
demonstrates this codebase's established idiom: **prefer a plain
`Mutex<HashMap<K, Arc<V>>>` with no eviction over a caching-crate
dependency**, since training datasets are typically sized to comfortably
fit in RAM. See the Stage 26 spec for the full investigation.

**Resolution (Stage 27, 2026-07-02):** Implemented exactly per the Stage 26
recommendation, with one refinement the user explicitly requested: an
opt-in `--cache-blocks-max-mb <usize>` flag gates a per-training-run-scoped
(not process-`static`) `Mutex<HashMap<u64, Arc<LoadedBlock>>>` on
`LabeledBlockDataset` via a new `.with_block_cache(Option<usize>)` builder
method. Rather than the originally-envisioned *unbounded* cache, the user
chose a **byte-budget cap with a one-time informative warning**: once the
configured MB budget would be exceeded by caching another block, further
blocks are silently left uncached (falling back to disk on every
subsequent request) and exactly one `eprintln!("[cache] ...")` warning is
logged the first time this happens per run — never an error. `load_block()`
transparently checks the cache first (no disk I/O on a hit) and performs a
best-effort insert after every disk load; omitting the flag preserves
pre-Stage-27 behavior exactly (`with_block_cache(None)` is a no-op). Three
new unit tests cover the cache-hit, budget-exceeded, and caching-disabled
paths. See the stage spec for full design rationale and verification
details.

**Effort:** Medium
**Priority:** P2 — closed, no longer actionable


---

### 5.3 No Streaming For Very Large Datasets (MEDIUM) ⏸️ DEFERRED

**Status:** Triaged in Stage 26. See `docs/stages/stage-26-remaining-findings-triage.md`.

**Finding:** AGENTS.md requires "smart streaming, spatial partitioning, or chunking mechanisms" for massive datasets. The training pipeline loads blocks on-demand (good), but there's no mechanism to handle datasets larger than available disk-backed I/O throughput.

**Deferral rationale (Stage 26, 2026-07-02):** This finding is architectural
and forward-looking rather than a scoped, actionable task. Training already
loads blocks on-demand per-batch (not a full in-memory load), satisfying
the core AGENTS.md streaming requirement for the common case. A genuinely
useful enhancement (e.g., a background prefetch thread/bounded channel
loading batch N+1 while batch N computes) substantially overlaps with 5.2's
block-caching design space, so this should be re-evaluated as part of a
future 5.2 implementation rather than as an independent stage. There is no
concrete dataset size or storage-environment currently driving an
independent effort here.

**Effort:** High  
**Priority:** P3 — deferred, not currently scheduled

---

### 5.4 Validation Runs on Autodiff Backend (MEDIUM) ✅ RESOLVED

**Status:** Resolved in Stage 16. See `docs/stages/stage-16-gpu-memory-growth.md`.

**Annotation (2026-07-02):** Same Stage 16 fix as 1.4 above — `validate_epoch()`
now explicitly forwards on `B::InnerBackend` via `model.valid()`, with an
extensive in-code comment explaining why (this was literally the root cause of
the GPU-OOM-during-validation bug Stage 16 fixed: autodiff tensors built
computation graphs during validation that were never freed). Confirmed against
current `trainer.rs` source. No further action needed.

**Original finding (superseded):** `validate_epoch()` uses `B: AutodiffBackend` and constructs autodiff tensors, wasting memory and time even though `.backward()` is never called during validation.

**Original recommendation (already implemented):** Run validation with `B::InnerBackend` (no autodiff) to avoid graph construction overhead.

**Effort:** Medium  
**Priority:** P2 — closed, no longer actionable

---


## 6. Testing Gaps

### 6.1 No Integration Tests for Training Loop (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 25. See `docs/stages/stage-25-testing-gaps.md`.

**Finding:** The training loop has unit tests for `compute_class_weights` and `CheckpointManifest`, but no end-to-end training test that verifies loss decreases on a synthetic dataset.

**Resolution:** Added `tests/training_integration.rs`, a Cargo integration test (gated `#![cfg(feature = "training")]`) that synthesizes an on-disk labeled-block dataset (`.feat`/`.lbl`/`labeled_blocks.json`, matching the real `preprocess-labeled` on-disk contract exactly), loads it via the real `LabeledBlockDataset::load()`, and trains via the real `training::trainer::train::<Autodiff<NdArray>>()` end-to-end on the CPU backend for 15 epochs. The test parses the real `metrics.csv` and asserts `train_loss` at the final epoch is strictly lower than at the first — verified over 4 consecutive runs with no flakiness (representative run: `train_loss` 0.3634 → 0.0113, `val_mIoU` reaching 1.0 by epoch 5).

**Effort:** Medium  
**Priority:** P3

---

### 6.2 No Tests for Error Paths (LOW) ✅ RESOLVED

**Status:** Resolved in Stage 25. See `docs/stages/stage-25-testing-gaps.md`.

**Finding:** Error paths (corrupt headers, mismatched n_classes, missing files) lack test coverage.

**Resolution:** Added nine new unit tests to `training/dataset.rs`'s existing test module, covering: empty `--data-dir` list, missing manifest directory, corrupt/unparsable `labeled_blocks.json`, non-contiguous/non-zero-based label map values, `n_classes` mismatch across multiple `--data-dir` directories, `.feat` bad magic bytes, `.feat` unsupported version byte, `load_block()` out-of-range composite directory index, and `load_block()` missing local block ID. Each test asserts `.is_err()` and matches a distinctive substring of the error message. Required adding `#[derive(Debug)]` to `LabeledBlockDataset`, `DirEntry`, and `LoadedBlock` (a zero-behavior-change addition needed for `Result::unwrap_err()` in the new tests).

**Effort:** Medium  
**Priority:** P3

---


## Summary Priority Table

| # | Finding | Severity | Effort | Priority |
|---|---------|----------|--------|----------|
| 1.1 | ~~No GPU support (AGENTS.md violation)~~ ✅ RESOLVED | CRITICAL | Medium | P0 |
| 1.2 | No mixed-precision training ⏸️ DEFERRED (Stage 23) | HIGH | Medium | P1 |
| 1.3 | ~~No parallel data loading~~ ✅ RESOLVED (Stage 22) | HIGH | Medium | P1 |
| 3.2 | ~~No file size validation (OOM risk)~~ ✅ RESOLVED (Stage 20) | HIGH | Low | P1 |
| 2.1 | ~~O(n) linear scan in load_block~~ ✅ RESOLVED (Stage 21) | HIGH | Low | P1 |
| 1.4 | ~~Full model clone every validation~~ ✅ RESOLVED (Stage 16) | MEDIUM | Medium | P2 |
| 1.5 | ~~No early stopping~~ ✅ RESOLVED (Stage 22) | MEDIUM | Low | P2 |
| 1.7 | ~~No gradient clipping~~ ✅ RESOLVED (Stage 22) | MEDIUM | Low | P2 |
| 2.2 | ~~class_counts_train rebuilds HashSet~~ ✅ RESOLVED (Stage 21) | MEDIUM | Low | P2 |
| 2.3 | ~~No parallelism in feature extraction~~ ✅ RESOLVED (Stage 21) | MEDIUM | Low | P2 |
| 2.5 | ~~SWA loads all checkpoints into RAM~~ ✅ RESOLVED (Stage 22) | MEDIUM | Low | P2 |
| 2.6 | No inference batching ⏸️ DEFERRED (Stage 23) | MEDIUM | Medium | P2 |
| 3.1 | ~~assert! in production code~~ ✅ RESOLVED (Stage 20) | MEDIUM | Low | P2 |
| 3.3 | ~~Integer overflow in size computation~~ ✅ RESOLVED (Stage 20) | MEDIUM | Low | P2 |
| 4.1 | ~~Excessive clippy suppressions~~ ✅ RESOLVED (Stage 24) | MEDIUM | Medium | P2 |
| 4.4 | ~~CLI arg parsing panics on missing values~~ ✅ RESOLVED (Stage 20) | LOW | Low | P2 |
| 5.2 | ~~No block caching during training~~ ✅ RESOLVED (Stage 27) | MEDIUM | Medium | P2 |
| 5.4 | ~~Validation uses autodiff backend~~ ✅ RESOLVED (Stage 16) | MEDIUM | Medium | P2 |
| 1.6 | ~~No LR warmup~~ ✅ RESOLVED (Stage 22) | LOW | Low | P3 |

| 2.4 | ~~Byte-by-byte f32 conversion~~ ✅ RESOLVED (Stage 21) | LOW | Low | P3 |
| 2.7 | kdtree cache locality ⏸️ DEFERRED (Stage 26) | LOW | Medium | P3 |
| 3.4 | ~~No path traversal validation~~ ✅ RESOLVED (Stage 20) | LOW | Low | P3 |
| 3.5 | ~~.lbl file size not validated~~ ✅ RESOLVED (Stage 20) | LOW | Low | P3 |

| 4.2 | ~~Duplicate T-Net extraction functions~~ ✅ RESOLVED (Stage 24) | LOW | Low | P3 |
| 4.3 | ~~SWA macros complex and fragile~~ ✅ RESOLVED (Stage 24) | LOW | Medium | P3 |
| 5.1 | No model quantization ⏸️ DEFERRED (Stage 26) | MEDIUM | High | P3 |
| 5.3 | No streaming for very large datasets ⏸️ DEFERRED (Stage 26) | MEDIUM | High | P3 |
| 6.1 | ~~No training integration tests~~ ✅ RESOLVED (Stage 25) | LOW | Medium | P3 |
| 6.2 | ~~No tests for error paths~~ ✅ RESOLVED (Stage 25) | LOW | Medium | P3 |

---

## Recommended Implementation Order

### Phase 1 — Critical & High (P0–P1)
1. **1.1** ~~GPU support via burn `wgpu` backend with runtime detection~~ ✅ DONE (Stage 09)
2. ~~**3.2** File size validation (OOM protection)~~ ✅ DONE (Stage 20)
3. ~~**2.1** HashMap index for `load_block`~~ ✅ DONE (Stage 21)
4. ~~**1.3** Parallel data loading / prefetching~~ ✅ DONE (Stage 22)
5. **1.2** Mixed-precision training — ⏸️ DEFERRED (Stage 23; requires a
   separate burn-version-upgrade decision, see stage spec for rationale)

### Phase 2 — Medium (P2)
6. ~~**1.5** Early stopping~~ ✅ DONE (Stage 22)
7. ~~**1.7** Gradient clipping~~ ✅ DONE (Stage 22)
8. ~~**1.4** Eliminate model clone in validation~~ ✅ DONE (Stage 16, predates this audit)
9. ~~**5.4** Validation on non-autodiff backend~~ ✅ DONE (Stage 16, predates this audit)
10. ~~**2.3** Parallel feature extraction~~ ✅ DONE (Stage 21)
11. ~~**2.5** Stream SWA checkpoint accumulation~~ ✅ DONE (Stage 22)
12. ~~**3.1** Replace `assert!` with `Result`~~ ✅ DONE (Stage 20)
13. ~~**3.3** Checked arithmetic for size computation~~ ✅ DONE (Stage 20)
14. ~~**4.4** CLI bounds checking~~ ✅ DONE (Stage 20)
15. ~~**2.2** Cache train_set HashSet~~ ✅ DONE (Stage 21)
16. ~~**5.2** Block caching~~ ✅ DONE (Stage 27)
17. ~~**4.1** Reduce clippy suppressions~~ ✅ DONE (Stage 24)
18. **2.6** Inference batching — ⏸️ DEFERRED (Stage 23; CPU-only engine,
    modest/unverified benefit vs. real rewrite risk, see stage spec)


### Phase 3 — Low / Polish (P3)
19. ~~**3.4** Path traversal validation~~ ✅ DONE (Stage 20)
20. ~~**3.5** `.lbl` file size validation~~ ✅ DONE (Stage 20)
21. ~~**2.4** `bytemuck` byte conversion~~ ✅ DONE (Stage 21)
22. ~~**4.2** Deduplicate T-Net extraction functions~~ ✅ DONE (Stage 24)
23. ~~**4.3** `WeightAveraging` trait replaces SWA macros~~ ✅ DONE (Stage 24)
24. ~~**6.1** Training integration test~~ ✅ DONE (Stage 25)
25. ~~**6.2** Error-path test coverage~~ ✅ DONE (Stage 25)
26. **2.7** `kdtree` cache locality — ⏸️ DEFERRED (Stage 26; no profiling
    evidence of a real bottleneck, see stage spec)
27. **5.1** Model quantization — ⏸️ DEFERRED (Stage 26; no measured
    deployment constraint, High effort/risk, see stage spec)
28. **5.3** Streaming for very large datasets — ⏸️ DEFERRED (Stage 26;
    overlaps with 5.2's design space, no concrete scenario driving it)
29. All other remaining P3 items — unscheduled

## Stage Mapping (2026-07-02 remediation plan)

The following 6-stage plan was approved to close out this audit's remaining
open findings:

- **Stage 20 — Security & Robustness Hardening** ✅ CLOSED — items 3.1, 3.2, 3.3, 3.4, 3.5, 4.4.
- **Stage 21 — Load-Path & Feature-Extraction Performance** ✅ CLOSED — items 2.1, 2.2, 2.3, 2.4.
- **Stage 22 — Training Loop Enhancements** ✅ CLOSED — items 1.3, 1.5, 1.6, 1.7, 2.5.
- **Stage 23 — Mixed Precision & Inference Batching** ✅ CLOSED (research-only) — items 1.2, 2.6 both formally deferred/rejected after cost-benefit analysis; see `docs/stages/stage-23-mixed-precision-inference-batching.md`. No code changed.
- **Stage 24 — Code Quality Cleanup** ✅ CLOSED — items 4.1, 4.2, 4.3; see `docs/stages/stage-24-code-quality-cleanup.md`.
- **Stage 25 — Testing Gaps** ✅ CLOSED — items 6.1, 6.2; see `docs/stages/stage-25-testing-gaps.md`.
- **Stage 26 — Remaining Findings Triage** ✅ CLOSED (research-only) — items 2.7, 5.1, 5.3 formally deferred after cost-benefit analysis; item 5.2 investigated further (not deferred, still unscheduled at the time) via a `whitebox_next_gen` in-memory-storage precedent study; see `docs/stages/stage-26-remaining-findings-triage.md`. No code changed.
- **Stage 27 — Block Caching** ✅ CLOSED — item 5.2 resolved; see `docs/stages/stage-27-block-caching.md`.

Items 1.2 and 2.6 (deferred by Stage 23), and 2.7, 5.1, 5.3 (deferred by
Stage 26), may be revisited in the future as standalone, deliberately-scoped
projects if their preconditions change (a confirmed burn version with a
usable f16 API for 1.2; profiling evidence of a real CPU-bound inference
bottleneck for 2.6/2.7; a concrete deployment constraint for 5.1; a concrete
large-dataset scenario for 5.3) — see the Stage 23 and Stage 26 specs'
Decision/Background sections for full rationale.


---

*This audit report should be treated as a living document. As findings are addressed, mark them as resolved and update the associated stage specification files per the AGENTS.md synchronization contract.*
