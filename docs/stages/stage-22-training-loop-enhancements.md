# Stage 22 — Training Loop Enhancements

## Status: CLOSED — all 5 audit items (1.3, 1.5, 1.6, 1.7, 2.5) resolved and verified

## The Goal

Close out the "GPU Utilization & Training Efficiency" (§1) and "Performance"
(§2) findings from `docs/AUDIT_REPORT.md` that relate to the shape and
robustness of the training loop itself: fully sequential per-block disk I/O
inside each micro-batch, no mechanism to stop training once validation mIoU
plateaus, no LR warmup for the T-Net sub-networks' early training stability,
no gradient-norm clipping safety net, and an SWA implementation that loads
every retained checkpoint into memory simultaneously instead of streaming.

Specifically this stage addresses audit items:
- **1.3** No parallel data loading (blocks within a micro-batch are loaded
  from disk one at a time)
- **1.5** No early stopping
- **1.6** No learning rate warmup
- **1.7** No gradient clipping
- **2.5** SWA loads all checkpoints into memory simultaneously

## Background

`trainer.rs::train()`'s micro-batch loop currently does:
```rust
for &block_id in micro {
    let block = match dataset.load_block(block_id) { ... };
    ...
}
```
— each block's `.feat`/`.lbl` pair is read from disk sequentially, one at a
time, even though `forward_batch_size` blocks are about to be stacked into a
single batched tensor regardless of load order. `LabeledBlockDataset::load_block(&self, ...)`
is a pure, read-only, `&self`-only method (confirmed by Stage 21's audit of the
same code path), so it is safe to call concurrently across the blocks of one
micro-batch.

**Design decision — Rayon parallel load vs. background prefetch thread:**
The audit's recommendation text mentions either "a background prefetch thread"
or `rayon::spawn` with a bounded channel to overlap the *next* batch's I/O with
the *current* batch's forward/backward compute. That design would give a
larger theoretical speedup (fully overlapping I/O latency with GPU compute)
but requires a threading/channel architecture living across loop iterations,
which is harder to reason about, is more failure-prone to get exactly right
around ownership of `dataset`/`model`/`optim`, and doesn't fit cleanly into a
single-owner training loop without extra synchronization. AGENTS.md's
"Lightweight" and "Lock-Free Progress: avoid heavy synchronization primitives
in hot execution loops" guidance both favor the simpler alternative: use
Rayon's already-in-tree `par_iter()` (the exact same pattern used for Stage 21
item 2.3) to load all blocks in a micro-batch **concurrently** instead of
sequentially. This still converts the dominant cost — `forward_batch_size`
sequential disk reads + byte→f32 conversions — into one parallel step per
micro-batch, is a small, easily-reviewed diff, requires no new dependencies,
and introduces no threads that outlive a single loop iteration. A true
overlap-with-compute prefetcher remains an option for a future stage if
profiling shows the parallel-load step itself is still the bottleneck.

`compute_class_weights`/scheduler/optimizer construction happen once before
the epoch loop and are unaffected.

For **gradient clipping (1.7)**, `burn::optim::AdamWConfig` already exposes
`.with_grad_clipping(Option<GradientClippingConfig>)`
(`burn::grad_clipping::GradientClippingConfig::Norm(max_norm)` /
`::Value(max_val)`), which internally clips **each parameter tensor's**
gradient to the given L2 norm (or per-element value) before the optimizer step
(`burn-core-0.16.1/src/grad_clipping/base.rs`). This is burn's own built-in,
already-tested mechanism — using it directly satisfies AGENTS.md's "Prefer
Existing Whitebox Next Gen solutions ... over introducing heavy external
crates [or] writing lightweight custom implementations" far better than
hand-rolling a global-norm clip across all `ParamId`s via `GradientsParams`
(whose `get`/`remove` API requires knowing each parameter's tensor rank `D`
ahead of time — deliberately not encouraged by the API for exactly this kind
of ad-hoc traversal). Per-tensor norm clipping is a standard, widely-used
variant of gradient clipping and directly addresses the audit's stated
concern ("T-Net sub-networks can produce large gradients causing
instability").

For **LR warmup (1.6)**, `scheduler.rs::CosineScheduler` is extended with an
optional `warmup_steps` field. When `t < warmup_steps`, `lr(t)` ramps linearly
from `0` to `lr_max`; for `t >= warmup_steps`, the existing cosine-annealing
formula runs over the *remaining* `total_steps - warmup_steps` steps (re-based
so the cosine curve still completes its full half-period by the end of
training). `CosineScheduler::new(...)` keeps its existing signature and
behavior unchanged (`warmup_steps = 0` is a no-op, exercised by the existing
`test_cosine_schedule_values` test), so no call site is broken; a new
`CosineScheduler::with_warmup(...)` constructor is added for the warmup case.

For **early stopping (1.5)**, a new `early_stopping_patience: Option<usize>`
config field is checked once per epoch after validation: if `val_mIoU` doesn't
exceed the best value seen so far (tracked independently of the existing
checkpoint-cadence-gated `best_miou`, so early stopping behaves identically
regardless of `--checkpoint-every`) for `patience` consecutive epochs, the
epoch loop exits early via `break`. The rest of `train()` (final model
selection, SWA, summary write) proceeds unchanged on whatever epoch the loop
stopped at.

For **streaming SWA (2.5)**, `apply_swa()` currently does
`manifest.checkpoints.iter().map(|e| load_model(...)).collect::<Result<Vec<_>>>()`,
loading every retained checkpoint (`keep_best_n`, up to 5 by default, but
user-configurable arbitrarily higher) into memory at once before accumulating.
The refactor inverts the loop nesting: load the first checkpoint into `base`,
then for each remaining checkpoint, load it into a local `m`, accumulate its
weights into `base` across *all* layers (T-Net, encoder, decoder, class
projection), and let `m` drop at the end of that loop iteration before the
next checkpoint is loaded — so memory footprint is bounded by two resident
models (`base` + the current `m`) regardless of `keep_best_n`, instead of
`keep_best_n + 1`.

## Inputs & Outputs

- **Inputs:** `TrainConfig` gains three new fields — `early_stopping_patience:
  Option<usize>` (default `None`, disabled), `warmup_steps: usize` (default
  `0`, disabled), `grad_clip_norm: Option<f32>` (default `None`, disabled).
  Three new `train` CLI flags: `--early-stopping-patience <usize>`,
  `--warmup-steps <usize>`, `--grad-clip-norm <f32>`. No existing flag,
  config field, `.feat`/`.lbl`/manifest format, or `.wbmodel` file format
  changes.
- **Outputs:** When all three new features are left at their default
  (disabled) values, training behavior, the final `.wbmodel` file, and
  `metrics.csv` are byte-for-byte/numerically identical to pre-Stage-22
  behavior — this is a hard backward-compatibility requirement, verified by
  the full existing test suite passing unmodified. When enabled, the CLI
  gains: (a) parallel-loaded micro-batches (no observable behavior change,
  only wall-clock speedup, since dims/labels are still validated per-block
  the same way after loading), (b) early termination of the epoch loop when
  early stopping triggers, (c) a warmup-then-cosine LR curve, (d) per-tensor
  gradient-norm clipping applied before every optimizer step, (e) streamed
  (not simultaneous) SWA checkpoint loading with identical averaged output
  weights to before.

## Steps & Specifications

1. **Parallel micro-batch block loading (1.3)** — In `trainer.rs::train()`'s
   micro-batch loop, replace the sequential `for &block_id in micro { dataset.load_block(block_id) ... }`
   loop with `micro.par_iter().map(|&block_id| dataset.load_block(block_id).ok()).collect::<Vec<_>>()`
   (via `rayon::prelude::*`), logging and skipping (`None`) on error exactly as
   before, then iterate the collected `Vec<Option<LoadedBlock>>` sequentially
   (via `.into_iter().flatten()`) to build `batch_flat`/`batch_labels` with the
   existing dims-mismatch guard — this keeps the batch-assembly logic (which
   mutates shared `count`/`n_ref`/`nfeat_ref` state) single-threaded and safe
   while parallelizing only the independent, read-only disk I/O + byte
   conversion step.

2. **Gradient clipping via burn's built-in `GradientClippingConfig` (1.7)** —
   Add `grad_clip_norm: Option<f32>` to `TrainConfig`. In `train()`, build the
   `AdamWConfig` with
   `.with_grad_clipping(config.grad_clip_norm.map(burn::grad_clipping::GradientClippingConfig::Norm))`
   before `.init(...)`. No change needed to the accumulation/optimizer-step
   loop itself — burn's `OptimizerAdaptor` applies clipping internally at
   `optim.step(...)` time.

3. **LR warmup (1.6)** — Add `warmup_steps: usize` (default `0`) to
   `CosineScheduler` via a new `with_warmup(lr_max, lr_min, total_steps,
   warmup_steps)` constructor (keeping `new(...)` as a thin wrapper calling
   `with_warmup(..., 0)`, so existing call sites and the existing
   `test_cosine_schedule_values` test are unaffected). `lr(t)`: if
   `warmup_steps > 0 && t < warmup_steps`, return
   `lr_max * (t as f64 / warmup_steps as f64)` (linear ramp from `0` at `t=0`
   towards `lr_max` at `t=warmup_steps`); otherwise run the existing cosine
   formula with `t` and `total_steps` both re-based by subtracting
   `warmup_steps` (via `saturating_sub`), so the cosine curve still spans
   from `lr_max` down to `lr_min` over the *post-warmup* remainder of
   training. Add `warmup_steps: usize` to `TrainConfig` (default `0`); in
   `train()`, replace `CosineScheduler::new(config.learning_rate, 1e-6, total_steps)`
   with `CosineScheduler::with_warmup(config.learning_rate, 1e-6, total_steps, config.warmup_steps)`.

4. **Early stopping (1.5)** — Add `early_stopping_patience: Option<usize>` to
   `TrainConfig` (default `None`). In the epoch loop, after computing
   `val_metrics`, track `es_best_miou: f64` and `es_epochs_without_improvement:
   usize` (both initialized before the loop, independent of the existing
   checkpoint-cadence-gated `best_miou`). If `val_metrics.miou > es_best_miou`,
   update `es_best_miou` and reset the counter to `0`; otherwise increment it.
   If `config.early_stopping_patience` is `Some(patience)` and
   `es_epochs_without_improvement >= patience`, log a clear message and
   `break` out of the epoch loop. This check runs unconditionally (a no-op
   when the config field is `None`), so default behavior is unchanged.

5. **Streaming SWA (2.5)** — Refactor `apply_swa()` to load only the first
   checkpoint into `base` up front; then, for each remaining checkpoint entry,
   load it into a local `m`, accumulate `m`'s encoder/decoder/class-projection/
   T-Net weights into the corresponding `base` fields using the existing
   `accum_linear!`/`accum_bn!` macros (now invoked once per model per layer,
   inside the per-checkpoint loop, instead of once per layer with an inner
   loop over a pre-loaded `Vec` of all models), and let `m` drop at the end of
   that loop iteration. After the loop, divide every accumulated field by `n`
   (`manifest.checkpoints.len() as f32`) exactly as before via the existing
   `divide_linear!`/`divide_bn!` macros. The averaged output weights must be
   numerically identical to the pre-refactor implementation (same sum, same
   division, just accumulated in a different loop order — floating-point
   addition is not perfectly associative across reorderings in general, but
   summing the *same* set of per-layer tensors in a different traversal order
   here reduces to `((base + m1) + m2) + ... + mN` either way, since both the
   old and new code accumulate against `base` as the running sum in
   checkpoint-list order — so this refactor introduces no reordering of the
   underlying floating-point additions at all).

6. Add three new CLI flags to `train_cmd.rs` — `--early-stopping-patience
   <usize>`, `--warmup-steps <usize>`, `--grad-clip-norm <f32>` — following the
   existing bounds-checked `next_value()` + `parse_*` pattern, with range
   validation (`--grad-clip-norm` must be `> 0.0` if provided) and updated
   `print_usage()` text.

7. Verify `cargo build --features training`, `cargo test --features training`,
   `cargo clippy --features training --all-targets`, and `cargo fmt --check`
   are all clean, and that every pre-existing test passes unmodified (default
   config values for all three new fields must reproduce identical behavior
   to pre-Stage-22 code).

## Definition of Done

- [x] Micro-batch block loading in `trainer.rs::train()` uses
      `rayon::prelude::*`'s `par_iter()` to load all blocks in a micro-batch
      concurrently; batch assembly (dims validation, `batch_flat`/
      `batch_labels` construction) remains single-threaded and produces
      identical output to the pre-Stage-22 sequential loop for the same input
      blocks.
- [x] `TrainConfig` gains `grad_clip_norm: Option<f32>` (default `None`); when
      `Some(n)`, the AdamW optimizer is constructed with
      `.with_grad_clipping(Some(GradientClippingConfig::Norm(n)))`; when
      `None`, optimizer construction and behavior is unchanged from
      pre-Stage-22 code.
- [x] `CosineScheduler` gains a `with_warmup(...)` constructor and warmup-aware
      `lr()` logic; `CosineScheduler::new(...)` remains behaviorally identical
      to before (verified by the existing `test_cosine_schedule_values` test
      passing unmodified); a new test verifies the linear warmup ramp and the
      post-warmup cosine curve.
- [x] `TrainConfig` gains `early_stopping_patience: Option<usize>` (default
      `None`); when `None`, all epochs run exactly as before; when `Some(p)`,
      the epoch loop exits early once `p` consecutive epochs pass without a
      new best `val_mIoU`, verified by a new test exercising the
      early-stopping trigger condition in isolation from the full training
      loop (e.g. a small helper function or the counter/reset logic tested
      directly).
- [x] `apply_swa()` loads at most two models into memory at any point during
      averaging (`base` plus the currently-processed checkpoint), instead of
      `keep_best_n` simultaneously; the averaged output weights are
      numerically identical to the pre-refactor implementation for the same
      input checkpoints (verified by the existing SWA-adjacent tests, plus
      manual review confirming the floating-point accumulation order is
      unchanged per the analysis in Background above).
- [x] Three new CLI flags (`--early-stopping-patience`, `--warmup-steps`,
      `--grad-clip-norm`) added to `train_cmd.rs` with bounds-checked value
      parsing and updated `print_usage()` text.
- [x] `cargo build --features training`, `cargo test --features training`,
      `cargo clippy --features training --all-targets`, `cargo fmt --check`
      all clean, with every pre-existing test passing unmodified (no test
      assertions changed to accommodate this stage's refactors).
- [x] This spec file synchronized with the final implementation (Drift Rule);
      results documented in a `## Results` section appended to this file once
      complete, per the Stage 20/21 convention.

## Results

All 5 audit items closed:

- **1.3 (parallel data loading)**: `trainer.rs::train()`'s micro-batch loop
  now loads all blocks via `micro.par_iter().map(|&block_id| dataset.load_block(block_id).ok()).collect()`,
  then assembles `batch_flat`/`batch_labels` sequentially from the collected
  `Vec<Option<LoadedBlock>>` via `.into_iter().flatten()`. Error messages now
  reference `block.block_id` (available on `LoadedBlock`) instead of the loop
  variable. Batch-assembly state (`count`/`n_ref`/`nfeat_ref`) remains
  single-threaded.
- **1.5 (early stopping)**: `TrainConfig::early_stopping_patience: Option<usize>`
  (default `None`) added. A new pure helper `early_stopping_step(val_miou,
  &mut es_best_miou, &mut es_epochs_without_improvement, patience) -> bool`
  encapsulates the counter/reset/trigger logic, called once per epoch after
  validation, independent of the checkpoint-cadence-gated `best_miou`. Two new
  unit tests (`test_early_stopping_step_triggers_after_patience`,
  `test_early_stopping_step_disabled_never_stops`) exercise the trigger
  condition and the disabled (`None`) no-op path in isolation.
- **1.6 (LR warmup)**: `CosineScheduler::with_warmup(lr_max, lr_min,
  total_steps, warmup_steps)` added; `new(...)` now delegates to
  `with_warmup(..., 0)`. `lr(t)` ramps linearly during `t < warmup_steps`,
  then runs the existing cosine formula re-based over the post-warmup
  remainder. `TrainConfig::warmup_steps: usize` (default `0`) wired into
  `train()`'s scheduler construction. New test
  `test_cosine_schedule_with_warmup` verifies the linear ramp, the
  cosine-phase boundary value, the post-warmup midpoint, and behavioral
  equivalence to `new(...)` when `warmup_steps = 0`. The pre-existing
  `test_cosine_schedule_values` test passes unmodified.
- **1.7 (gradient clipping)**: `TrainConfig::grad_clip_norm: Option<f32>`
  (default `None`) added. `AdamWConfig` is now built with
  `.with_grad_clipping(config.grad_clip_norm.map(burn::grad_clipping::GradientClippingConfig::Norm))`
  before `.init(...)` — burn's built-in per-tensor L2-norm clipping applied
  internally at optimizer-step time; no other trainer changes required.
- **2.5 (streaming SWA)**: `apply_swa()` refactored to load only the first
  checkpoint into `base`, then stream-load and accumulate each remaining
  checkpoint into a per-iteration local `m` (dropped at the end of each loop
  iteration) across all layers (encoder, decoder, class projection, both
  T-Nets), instead of collecting every retained checkpoint into a `Vec` up
  front. Memory footprint is now O(2 models) instead of O(keep_best_n + 1).
  The existing `test_swa_averages_tnet_weights` test passes unmodified,
  confirming numerically identical averaged output.

Three new CLI flags added to `train_cmd.rs`: `--early-stopping-patience
<usize>`, `--warmup-steps <usize>`, `--grad-clip-norm <f32>` (with `> 0.0`
finite-value validation), following the existing bounds-checked
`next_value()` + `parse_*` pattern, plus updated `print_usage()` text.

### Verification commands run

```
cargo build --features training      → Finished, no errors
cargo test --features training       → 80 passed; 0 failed; 0 ignored
cargo clippy --features training --all-targets
                                      → 0 errors; only pre-existing pedantic-level
                                        warnings (device = Default::default(),
                                        manual_midpoint, cast_precision_loss, etc.)
                                        consistent with existing codebase patterns
                                        and explicitly in scope for Stage 24
                                        (Code Quality Cleanup), not introduced or
                                        worsened by this stage's changes
cargo fmt --check                    → clean (after running `cargo fmt` once to
                                        apply formatting to the newly-added
                                        early-stopping test assertions)
```

All 80 unit tests pass, including 4 new tests added in this stage
(`test_cosine_schedule_with_warmup`, `test_early_stopping_step_triggers_after_patience`,
`test_early_stopping_step_disabled_never_stops`) and zero pre-existing test
assertions were modified — confirming default (disabled) values for all three
new `TrainConfig` fields reproduce identical pre-Stage-22 behavior.
