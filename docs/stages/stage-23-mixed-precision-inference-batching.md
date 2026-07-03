# Stage 23 — Mixed Precision & Inference Batching (Research Stage — Both Items Deferred)

## Status: CLOSED — items 1.2 and 2.6 formally deferred/rejected after research; no code changes made

## The Goal

Investigate audit items **1.2 (No Mixed-Precision Training)** and **2.6 (No
Inference Batching)** from `docs/AUDIT_REPORT.md`, determine a concrete
implementation design for each, and either implement them or — if the
research reveals the audit's original cost/benefit assumptions no longer
hold — formally document why the item is deferred, per AGENTS.md's
spec-driven-development contract ("no development work... may begin without
a dedicated stage specification file").

Unlike Stages 20–22, this stage's Definition of Done is satisfied by
**research + a documented decision**, not by a code change. AGENTS.md does
not mandate that every stage produce new code; it mandates that every stage
be preceded and accompanied by a synchronized spec describing what was done
and why. A stage that concludes "the recommended change should not be made,
and here is the reasoning" is itself a valid, auditable outcome, and prevents
the same speculative work from being silently re-litigated in a future audit
without record of why it was previously rejected.

## Background

### Item 1.2 — Mixed-Precision Training

The audit's original recommendation assumed burn's `Autodiff<Wgpu>` backend
already supported `f16` tensors and that adding a `--mixed-precision` flag
was a `Medium`-effort, self-contained change. Research this stage found:

- The project is pinned to **`burn = 0.16.1`** (confirmed via
  `Cargo.lock`: `name = "burn"` / `version = "0.16.1"`).
- burn 0.16.1's `wgpu` backend has **no f16 tensor support at all**. This was
  tracked as an open GitHub enhancement request
  ([tracel-ai/burn#597](https://github.com/tracel-ai/burn/issues/597)),
  still open as of mid-2025, explicitly blocked on upstream `wgpu`/`gfx-rs`
  f16 buffer support.
- f16-adjacent work only begins appearing in **burn 0.17** (quantized matmul,
  a "flex32" lower-precision matmul fix) and **burn 0.18** ("mixed precision
  accumulation w/ fusion", a quantization-scheme refactor) — i.e. **two major
  version bumps** beyond what is currently vendored.
- It is not confirmed that burn 0.18's "mixed precision accumulation w/
  fusion" is an exposed, user-facing f16 training toggle as opposed to an
  internal kernel-fusion optimization inside CubeCL — the changelog language
  reads like the latter.

### Item 2.6 — Inference Batching

The audit's original recommendation assumed batching would primarily
improve **GPU utilization**, and that padding/masking would be required
since block point counts might differ. Research this stage found:

- The deployed inference engine, `PointNetClassifier` in
  `src/model/pointnet.rs` / `src/model/layers.rs`, is a **separate,
  pure-`ndarray`, CPU-only implementation** with **no GPU code path at all**
  — GPU acceleration exists only on the training side
  (`training::burn_model::BurnPointNet<B>`), which is not used at inference
  time. The audit's "better GPU utilization" motivation for this item
  therefore does not apply to the code that would actually be changed.
- All blocks are resampled to a fixed `target_points` count (default 1024)
  during preprocessing (`preprocessing/normalizer.rs::resample_block()`), so
  the padding/masking concern in the original finding does not apply either
  — blocks could be concatenated directly if a batched forward pass existed.
- However, `run_inference()` (`src/model/inference.rs`) **already
  parallelizes across blocks via Rayon** (`manifest.blocks.par_iter()`), so
  all CPU cores are already utilized under the current one-block-at-a-time
  design. Batching would only pay off in the specific regime where
  individual blocks are small and numerous enough that per-task/per-matmul
  overhead dominates — a regime that has not been profiled or confirmed for
  this codebase's real-world block sizes.
- A *correct* batched forward pass is materially more invasive than it
  first appears: `Linear`/`BatchNorm1d` (inference-mode, row-wise) batch
  trivially by row concatenation, but `global_max_pool` and the per-block
  T-Net (STN3d/STN64d) global-pool + FC-decoder steps are **inherently
  per-block** — each block must get its own pooled global descriptor and its
  own T-Net transform matrix. Naively pooling across a concatenated batch
  would cross-contaminate one block's global feature into another block's
  classification, silently corrupting output. Implementing this correctly
  requires new segment-aware pooling/broadcast logic threaded through
  `pointnet.rs::forward()`, `layers.rs::TNet::forward()`, and
  `inference.rs`'s block-processing loop — a nontrivial rewrite of exactly
  the forward-pass code that produced the Stage 17 BatchNorm logit-explosion
  regression and required careful, hard-won fixing (see
  `stage-17-batchnorm-running-stats.md`, `stage-18-batchnorm-batched-forward.md`).

## Decision (reached with the project owner, 2026-07-02)

Both items are **deferred / rejected as currently scoped**, with the
reasoning above recorded here for any future audit or contributor:

- **Item 1.2 (Mixed-Precision Training): rejected for now.** The
  cost (a 2-major-version burn upgrade touching every burn-consuming file —
  `burn_model.rs`, `trainer.rs`, `bridge.rs`, `backend.rs` — and risking
  regression of Stages 09/16/17/18/22's hard-won GPU-memory and
  BatchNorm-batching fixes) is concrete and large, while the benefit (a
  generic "2-3× GPU speedup" figure with no confirmation the bottleneck for
  this specific, small (~2M-parameter) PointNet model is even compute-bound
  rather than memory-bandwidth-bound) is speculative, GPU-only, and
  unverified. This does not preclude revisiting mixed precision in the
  future as a **deliberate, standalone burn-version-upgrade project** with
  its own spec file, once/if GPU training throughput is confirmed to be a
  real user-facing pain point and a target burn version is confirmed to
  expose a genuine user-facing f16 training API.
- **Item 2.6 (Inference Batching): rejected for now.** The audit's
  GPU-utilization motivation does not apply to the CPU-only inference
  engine that would need to change. The realistic benefit is a modest,
  unverified CPU throughput improvement in a code path that is already
  Rayon-parallelized across blocks, while the cost is a genuine rewrite of
  stable, previously-fragile forward-pass code (`global_max_pool` and
  T-Net segment-aware handling) with real regression risk. This does not
  preclude revisiting inference batching in the future if profiling of a
  real production dataset shows block-level Rayon parallelism leaves
  significant CPU headroom unused (e.g., very large block counts with very
  small point counts per block).

No source code was modified as part of this stage. `docs/AUDIT_REPORT.md`
is updated (see below) to reflect this decision so the audit's living
document accurately tracks the current disposition of both items.

## Inputs & Outputs

- **Inputs:** `docs/AUDIT_REPORT.md` items 1.2 and 2.6; `Cargo.lock`'s pinned
  `burn` version; the existing training (`training/backend.rs`,
  `training/trainer.rs`, `training/burn_model.rs`) and inference
  (`model/inference.rs`, `model/pointnet.rs`, `model/layers.rs`) source;
  public burn release notes/changelogs (0.16.1 through 0.18.0) and the
  tracel-ai/burn GitHub issue tracker.
- **Outputs:** This stage spec file (research + decision record);
  corresponding updates to `docs/AUDIT_REPORT.md`'s items 1.2/2.6, Summary
  Priority Table, Recommended Implementation Order, and Stage Mapping
  sections. No binary, CLI flag, model format, or runtime behavior changes.

## Steps & Specifications

1. Confirm the pinned `burn` version via `Cargo.lock` and research its
   `wgpu`-backend f16/mixed-precision support status, plus the nearest
   subsequent burn versions that introduce such support, via public
   changelogs/release notes and the upstream issue tracker.
2. Re-examine the deployed inference engine (`model/pointnet.rs`,
   `model/layers.rs`, `model/inference.rs`) to confirm whether it has any
   GPU code path (it does not) and to identify exactly which primitives
   (`Linear`, `BatchNorm1d`, `global_max_pool`, `TNet`) would need to change
   to support a genuine batched multi-block forward pass, and whether
   per-block segment-boundary handling is required (it is, for pooling and
   T-Nets).
3. Present a motivation-vs-cost analysis for each item to the project
   owner and obtain an explicit decision before writing any implementation
   code, per AGENTS.md's spec-driven-development and Greenfield-Only
   guardrails.
4. Record the decision and its full reasoning in this spec file (Background
   + Decision sections above).
5. Update `docs/AUDIT_REPORT.md` to reflect the deferred/rejected status of
   items 1.2 and 2.6, including the Summary Priority Table, Recommended
   Implementation Order, and Stage Mapping sections, per the Stage 20/21/22
   documentation-synchronization convention.

## Definition of Done

- [x] `burn` version pinned in this project confirmed, and its f16/mixed
      precision support status (and the nearest version that adds related
      functionality) researched and documented.
- [x] Deployed inference engine's architecture (CPU-only, no GPU path;
      per-block pooling/T-Net semantics) confirmed and documented as it
      relates to batching feasibility.
- [x] Motivation-vs-cost analysis presented for both items; explicit
      decision obtained from the project owner for each, independently.
- [x] Decision and full reasoning recorded in this stage spec file
      (Background + Decision sections).
- [x] `docs/AUDIT_REPORT.md` items 1.2 and 2.6 updated to reflect the
      deferred/rejected disposition, with the Summary Priority Table,
      Recommended Implementation Order, and Stage Mapping sections kept in
      sync (Drift Rule).
- [x] No source code changes were made or required, since this stage's
      deliverable is a research + decision record, not an implementation.

## Results

Both audit items 1.2 (Mixed-Precision Training) and 2.6 (Inference
Batching) were researched in full, presented to the project owner with an
explicit motivation-vs-cost breakdown, and **both were deliberately
deferred/rejected** rather than implemented, for the reasons documented in
the Background and Decision sections above. This is a valid, complete
closure of Stage 23 under AGENTS.md's spec-driven-development model: the
decision — and the reasoning behind it — is now a permanent, auditable part
of the project's documentation, preventing future re-litigation of the same
questions without this context.

`docs/AUDIT_REPORT.md` has been updated in the same session to mark items
1.2 and 2.6 as deferred (not "resolved", since no code changed), with
cross-references to this stage file, and the Summary Priority Table /
Recommended Implementation Order / Stage Mapping sections have been kept in
sync accordingly.

No `cargo build`/`test`/`clippy`/`fmt` verification was required for this
stage, since no source files were modified.
