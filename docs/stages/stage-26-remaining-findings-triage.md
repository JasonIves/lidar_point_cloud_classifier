# Stage 26 — Remaining Findings Triage (Deferral Decision)

## Status: CLOSED (research-only, no code changed)

## The Goal

Following Stage 25's closure of the "Testing Gaps" (§6) findings, `docs/AUDIT_REPORT.md`
still listed four genuinely unscheduled findings: **2.7** (`kdtree` cache
locality), **5.1** (model quantization), **5.2** (block caching during
training), and **5.3** (streaming for very large datasets). This stage
performs the same kind of cost-benefit / motivating-evidence triage Stage 23
performed for items 1.2/2.6, and records an explicit decision for each of
the four items:

- **2.7, 5.1, 5.3** — formally deferred. Each lacks a concrete, currently-
  observed motivating factor (a measured performance bottleneck, a real
  deployment constraint, or a specific dataset scale) to justify scoping a
  full implementation stage right now.
- **5.2** — investigated further (not deferred, not yet scheduled either).
  A quick audit of `whitebox_next_gen`'s own in-memory storage patterns
  (`wblidar::memory_store`, `wbraster::memory_store`) was performed to
  inform what "AGENTS.md-lightweight" caching looks like elsewhere in this
  codebase, which meaningfully simplifies a future implementation, but no
  code was written in this stage.

No production code changes are made in this stage — purely a documentation/
triage exercise, exactly mirroring Stage 23's "research-only" precedent.

## Background — investigation performed

### 2.7 — `kdtree` crate not cache-friendly

`kdtree` (v0.8.0) is used in three places: `preprocessing/spatial_index.rs`
(3-D `BlockSpatialIndex`, used by feature extraction), `preprocessing/
outlier_filter.rs` (2-D XY tree, outlier removal), and `model/inference.rs`
(2-D XY tree, nearest-label lookup for inference). All three are
heap-allocated, pointer-chasing tree structures. The original audit finding
was a code-quality observation about cache locality, not a measured
regression — there is no profiling evidence in this repository that this is
an actual throughput bottleneck for any of the three call sites.

**Decision: deferred.** Replacing `kdtree` with a flat/grid-based structure
in three separate call sites is a nontrivial, correctness-sensitive change
(each site has its own existing test coverage that would need re-validating)
for a purely speculative performance gain. Revisit only if profiling
identifies one of these three call sites as an actual hot path.

### 5.1 — No model quantization for inference

The deployed CPU-only inference engine (`model/layers.rs`'s `Linear`/
`BatchNorm1d`) stores `f32` weights exclusively. Quantizing to int8 would
require: a new post-training calibration/quantization step, a
quantization-aware (or dequantize-on-the-fly) forward pass, and accuracy-
regression validation (mIoU/OA) against the existing f32 baseline — a
substantial, standalone feature touching the same forward-pass code
implicated in the Stage 17 BatchNorm logit-explosion regression.

**Decision: deferred.** There is no currently-observed model-size or
CPU-inference-latency complaint driving this; the effort (High) and
correctness risk are large relative to a speculative benefit. Revisit only
if a concrete deployment constraint (e.g., model size limit, measured
inference latency budget) emerges.

### 5.3 — No streaming for very large datasets

The original finding is architectural and forward-looking rather than a
scoped, actionable task: "no mechanism to handle datasets larger than
available disk-backed I/O throughput." Training already loads blocks
on-demand per-batch (not fully in-memory), which already satisfies the
core AGENTS.md streaming requirement for the common case. A genuinely
useful "streaming" enhancement (e.g., a background prefetch thread/bounded
channel loading batch N+1 while batch N computes) substantially overlaps
with the 5.2 block-caching design space.

**Decision: deferred, to be re-evaluated as part of a future 5.2
implementation** rather than treated as an independent stage. There is no
concrete dataset size or storage-environment (e.g., "network-mounted
storage with N blocks") currently driving an independent 5.3 effort.

### 5.2 — No block caching during training (investigated further)

`training/dataset.rs`'s `load_block()` re-reads `.feat`/`.lbl` files from
disk on every call, once per block per epoch, with no caching across
epochs. Investigated how `whitebox_next_gen` itself implements in-memory
storage, to check whether a heavier caching dependency (`moka`, `dashmap`,
`quick_cache`, `lru`) would be idiomatic here — none of these appear
anywhere in `whitebox_next_gen`'s dependency tree.

**Finding:** both `wblidar::memory_store` and `wbraster::memory_store` use
an identical, minimal pattern:

```rust
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static LIDAR_STORE: OnceLock<Mutex<HashMap<String, Arc<PointCloud>>>> = OnceLock::new();
```

A single stdlib `std::sync::Mutex<HashMap<String, Arc<T>>>`, no external
caching crate, no eviction policy, no size cap or TTL — entries persist for
the life of the process (or test) until explicitly removed. This is a
session-scoped intermediate-result registry for chaining tool outputs
in-process, not a bounded LRU cache — a different use case from repeatedly
re-reading a fixed, bounded set of training blocks across epochs, but it
directly demonstrates the "Minimal Dependencies"/"Lightweight" idiom this
codebase already uses in an equivalent problem space: **prefer a plain
`Mutex<HashMap<K, Arc<V>>>` with no eviction over pulling in a caching
crate**, since typical training datasets are sized to comfortably fit in
RAM once cached.

**Decision: not deferred, not yet scheduled.** This remains the most
concrete, lowest-risk, highest-practical-value of the four items and a
reasonable candidate for a future dedicated stage (tentatively "Stage 27").
A v1 implementation would mirror the `whitebox_next_gen` idiom exactly: an
opt-in `--cache-blocks` flag gating a plain, unbounded
`Mutex<HashMap<u64, Arc<LoadedBlock>>>` scoped to one training run (not a
process-wide `static`, to avoid leaking memory across unrelated CLI
invocations) — no LRU/eviction complexity needed for a first version, since
whitebox's own precedent shows unbounded-for-the-run in-memory storage is
an accepted pattern in this codebase. Not implemented in this stage; no
stage spec has been opened for it yet.

## Inputs & Outputs

- **Inputs:** none (no CLI flags, file formats, or code paths change).
- **Outputs:** updated `docs/AUDIT_REPORT.md` — findings 2.7, 5.1, 5.3
  marked "⏸️ DEFERRED (Stage 26)" with rationale; 5.2 left open/unscheduled
  but annotated with the `whitebox_next_gen` precedent investigation above
  for a future stage to reference; Summary Priority Table and Stage Mapping
  sections updated accordingly.

## Steps & Specifications

1. Investigate `kdtree` usage sites (2.7) — confirm no existing profiling
   evidence of a bottleneck. Done above.
2. Investigate the current inference engine's weight representation (5.1)
   — confirm no existing quantization infrastructure or measured need.
   Done above.
3. Investigate the current training block-loading mechanism (5.3) — confirm
   it already loads on-demand per-batch (not a full in-memory load), and
   that a genuine streaming/prefetch enhancement overlaps with 5.2. Done
   above.
4. Investigate `whitebox_next_gen`'s own in-memory storage/caching pattern
   (5.2) via `wblidar::memory_store` / `wbraster::memory_store` source, and
   confirm no caching crate (`moka`/`dashmap`/`quick_cache`/`lru`) appears
   anywhere in `whitebox_next_gen/Cargo.toml` or any crate's `Cargo.toml`.
   Done above.
5. Update `docs/AUDIT_REPORT.md`: mark 2.7/5.1/5.3 as deferred (finding
   sections + Summary Priority Table + Stage Mapping section), matching the
   Stage 23 deferral style exactly. Leave 5.2 as unscheduled but add a
   cross-reference to this stage's investigation findings.

## Definition of Done

- [x] 2.7, 5.1, 5.3 each have a documented deferral rationale in this spec
      file.
- [x] 5.2 has a documented investigation of the `whitebox_next_gen`
      in-memory storage precedent, informing (but not yet implementing) a
      future stage.
- [x] `docs/AUDIT_REPORT.md` updated: finding sections for 2.7/5.1/5.3 show
      "⏸️ DEFERRED (Stage 26)"; Summary Priority Table rows updated; Stage
      Mapping section includes a Stage 26 entry.
- [x] No production code changed in this stage (research/triage only).
- [x] This spec file synchronized with the final `AUDIT_REPORT.md` state
      (Drift Rule) — Results section below.

## Results

All four items triaged as scoped above. `docs/AUDIT_REPORT.md` updated:

- **2.7** finding section: added "⏸️ DEFERRED (Stage 26)" status line and a
  deferral-rationale paragraph (no profiling evidence of a real bottleneck
  across any of the three `kdtree` call sites).
- **5.1** finding section: added "⏸️ DEFERRED (Stage 26)" status line and a
  deferral-rationale paragraph (no measured model-size/latency constraint;
  High effort/risk touches the Stage 17-sensitive forward pass).
- **5.3** finding section: added "⏸️ DEFERRED (Stage 26)" status line and a
  deferral-rationale paragraph (overlaps with 5.2's design space; no
  concrete large-dataset scenario currently driving it).
- **5.2** finding section: left unscheduled (not deferred), with a new
  paragraph cross-referencing this stage's `whitebox_next_gen::memory_store`
  precedent investigation, to inform a future "Stage 27" implementation.
- Summary Priority Table: 2.7/5.1/5.3 rows updated to show
  "⏸️ DEFERRED (Stage 26)"; 5.2 row left unchanged (still open).
- Stage Mapping section: added a `Stage 26 — Remaining Findings Triage`
  entry mirroring the Stage 23 research-only style.

No code was changed; this was a documentation-only triage stage.
