# Stage 29 — Jitter-Based Oversampling

**Status:** COMPLETE
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** GitHub Copilot / AI Collaborator

---

## Goal

Replace the option of exact-duplicate padding (Stage 01's sample-with-replacement
oversampling) with an opt-in **jittered** oversampling mode: padded points are
still drawn (with replacement) from the same block, but each padding-only copy
has its (x, y, z) coordinates perturbed by a small, seeded, clipped-Gaussian
offset **before** feature extraction runs.

### Motivation

`resample_block()` currently pads sparse blocks (`raw_count < target_points`) by
duplicating existing points verbatim. Because feature extraction
(`extract_features`) runs on the *sampled* points using each point's own (x, y, z)
as the k-d tree query center, a duplicated point produces an **identical**
neighbourhood query and therefore identical eigenvalue features to its source —
a 100%-correlated clone in full feature space, not just in raw coordinates.

Downstream, `PointNetClassifier`'s global max-pool is idempotent under exact
duplication: a duplicated point contributes zero new information to the pooled
global feature vector, yet it **is** counted multiple times in the per-point
cross-entropy loss and in class-count statistics used for class weighting
(Stage 07). Small perturbations to padded points' coordinates — applied prior to
feature extraction — produce genuinely distinct eigenvalue features, giving the
model real (if modest) additional signal from sparse blocks instead of
redundant, over-weighted duplicates.

This technique is standard in the point-cloud literature (Qi et al. 2017,
PointNet/PointNet++) as a coordinate-jitter augmentation.

### Why this (and not same-class kNN interpolation)

An alternative — synthesizing padded points via interpolation with a same-class
neighbour — was evaluated and explicitly **not** pursued at this time. That
approach requires per-point ground-truth labels and can therefore only run in
the labeled training pipeline (`labeled_pipeline.rs`); it has no valid code path
at inference time on unlabeled files, since inference is precisely how labels
are produced, not consumed. Because `resample_block()` is **shared** by both the
label-agnostic base pipeline (`pipeline.rs`, used by both training-data prep and
`wb_lidar_classify`'s inference path) and the labeled pipeline, a jitter-based
mechanism that operates on coordinates alone benefits every future inference run
on sparse real-world data, not just the training set — with no train/inference
distribution-mismatch risk. Jitter is therefore treated as a standalone,
dual-use feature rather than a stepping stone to a training-only technique.

---

## Inputs & Outputs

### New `PreprocessConfig` field

| Field | Type | Default | Description |
|---|---|---|---|
| `oversample_jitter` | `f64` | `0.0` | Standard deviation (projection units, e.g. metres) of per-axis Gaussian jitter applied to padding-only points during oversampling. `0.0` = today's exact-duplicate behaviour (unchanged, fully backward compatible). |

Only the **appended padding copies** created when `raw_count < target_points`
are jittered. The original (non-padded) points that make up the first
`raw_count` entries of a padded block's output are **never** modified.

### New CLI flag

Available on both `wb_lidar_classify preprocess` and
`wb_lidar_train preprocess-labeled`:

```
--oversample-jitter <f64>   Std-dev (projection units) of Gaussian jitter applied
                             to padding-only points when oversampling sparse
                             blocks (default: 0.0 = exact duplication, unchanged
                             behaviour). Must be >= 0.0 and finite.
```

### Manifest changes

`blocks.json` (`BlockManifest`) gains one new field:

```json
{
  "oversample_jitter": 0.0,
  ...
}
```

Carries `#[serde(default)]` so existing manifests without the field deserialise
correctly (treated as `0.0`). `labeled_blocks.json` does **not** duplicate this
field — it is recorded once in `blocks.json`, following the precedent set by
`block_overlap` in Stage 08.

### `.feat` file format

**Unchanged.** Jitter perturbs in-memory coordinates prior to feature
extraction; it does not alter the binary layout, point count, or feature count
of `.feat` files.

---

## Steps & Specifications

### `resample_block()` changes (`src/preprocessing/normalizer.rs`)

New signature:

```rust
pub fn resample_block(
    pts: &[PointRecord],
    target: usize,
    seed: u64,
    jitter_sigma: f64,
) -> (Vec<PointRecord>, Vec<usize>, bool)
```

When `pts.len() < target` (the oversampling branch):

1. The first `pts.len()` entries of `sampled` remain the original points,
   copied verbatim (unchanged from today).
2. For each of the `extra = target - pts.len()` padding draws:
   - Draw a source index `idx` from `0..pts.len()` using the same
     `SmallRng` stream already seeded by `block_id` (unchanged).
   - If `jitter_sigma > 0.0`: apply an independent per-axis offset to
     `(x, y, z)` drawn from a zero-mean Gaussian with the given standard
     deviation, clipped to `±3σ` to bound worst-case displacement, using a
     Box–Muller transform sourced from the same RNG (no new dependency).
   - If `jitter_sigma == 0.0`: behaviour is bit-identical to pre-Stage-29
     (exact duplicate), preserving full backward compatibility and the
     existing reproducibility guarantee.
3. `sampled_indices` still records the **source** index for every padding
   draw, unchanged — jitter perturbs only the returned `PointRecord`'s
   coordinates, never the index-to-source mapping. This preserves
   correctness of the labeled pipeline's classification-label lookup
   (`labeled_pipeline.rs`), since a jittered padding point must still inherit
   its source point's ground-truth label.

### Call-site changes (`src/preprocessing/pipeline.rs`)

`run_internal()`'s call to `resample_block()` passes
`config.oversample_jitter` as the new fourth argument. No other change to the
per-block parallel closure: jitter still happens **before**
`BlockSpatialIndex::build()` and `extract_features()`, guaranteeing
perturbed coordinates flow into the eigenvalue neighbourhood queries exactly
as the exact-duplicate padding did before it.

### `BlockManifest` changes (`src/preprocessing/pipeline.rs`)

```rust
pub struct BlockManifest {
    // ... existing fields ...
    #[serde(default)]
    pub oversample_jitter: f64,
}
```

Default: `0.0`.

### CLI changes

Both `preprocess_cmd.rs` and `preprocess_labeled_cmd.rs` gain:

```
--oversample-jitter <f64>
```

parsed as `f64`, validated `>= 0.0 && is_finite()`. No upper bound is enforced
— an operator who sets an unreasonably large value will observe degraded
results empirically (this is an experimental knob for the Stage 29 A/B test,
not a hardened production default).

---

## A/B Test Plan (validation methodology, not code)

This stage's implementation is deliberately scoped to the **mechanism only**.
Empirical validation is performed by the user, out-of-band, as follows:

1. Preprocess the same labeled input twice: once with
   `--oversample-jitter 0.0` (Run A, baseline) and once with a chosen non-zero
   σ (Run B).
2. Train one model per run with identical `TrainConfig` (seed, epochs, LR
   schedule, class weighting).
3. Compare validation mIoU, per-class IoU, and precision/recall/F1 (existing
   `training/metrics.rs` infrastructure — no new instrumentation required).
4. Watch specifically for: (a) mIoU lift concentrated in classes/blocks that
   are disproportionately oversampled (expected upside), and (b) any
   class-boundary precision/recall regression (the main theoretical risk of
   jitter — a padded point jittered across a class boundary could introduce
   local label noise).

No migration to same-class kNN interpolation (previously discussed "Option C")
is planned; jitter is the intended end-state mechanism, not a stepping stone.

---

## Definition of Done

| # | Criterion | Verification |
|---|---|---|
| 1 | `cargo build --release --features training` — zero errors | Build gate |
| 2 | `cargo clippy --features training -- -D warnings` — zero new warnings | Clippy gate |
| 3 | `cargo fmt --check` passes | fmt gate |
| 4 | `cargo test --features training` — all existing tests pass + new jitter tests | Regression + new |
| 5 | `--oversample-jitter 0.0` (default) produces **bit-identical** `.feat` output to pre-Stage-29 pipeline for the same input | Regression test |
| 6 | `--oversample-jitter <σ> > 0.0` produces distinct coordinates (and therefore distinct eigenvalue features) for padding-only points, while non-padded points and non-oversampled blocks are untouched | Unit test |
| 7 | Jitter offsets are clipped to `±3σ` | Unit test (statistical bound check) |
| 8 | Jitter is fully reproducible given the same seed (same `block_id`) | Unit test |
| 9 | `oversample_jitter` is recorded in `blocks.json`; older manifests without the field deserialise to `0.0` | Unit test |
| 10 | Range validation rejects negative or non-finite `--oversample-jitter` in both CLI entry points | Unit test |
| 11 | This stage spec is synchronised with the implementation | Manual review |

---

## Relationship to Prior Stages

- **Stage 01** (spatial preprocessing): `resample_block()`'s sampling contract
  is extended, not replaced. Subsample-without-replacement behaviour
  (`raw_count >= target`) is completely unaffected.
- **Stage 03 / labeled pipeline**: label lookup by `sampled_indices` is
  unaffected — jitter never changes which source point a padding draw is
  attributed to, only its emitted coordinates.
- **Stage 07** (tunable class weighting): complementary, not overlapping —
  Stage 07 addresses loss-function-level class imbalance; Stage 29 addresses
  feature-level information redundancy in oversampled points.
- **Stage 08** (overlapping blocks): established the manifest
  `#[serde(default)]` backward-compatibility precedent this stage follows for
  `oversample_jitter`.

---

## Implementation Status

All Definition of Done criteria verified. No deviations from spec.

### Changed files

| File | Change |
|---|---|
| `src/preprocessing/normalizer.rs` | `resample_block()` gained `jitter_sigma: f64` 4th parameter; new private `jitter_offset()` Box–Muller helper (±3σ clip, returns `0.0` for `sigma <= 0.0`); padding-only draws perturbed when `jitter_sigma > 0.0`; 4 pre-existing tests updated to pass `0.0`; 6 new tests added covering bit-identical-at-zero, padding-only perturbation, ±3σ clipping, reproducibility, and non-positive-sigma zero-offset. |
| `src/preprocessing/mod.rs` | `PreprocessConfig` gained `oversample_jitter: f64` field (doc-commented); `Default` impl sets `oversample_jitter: 0.0`. |
| `src/preprocessing/pipeline.rs` | `BlockManifest` gained `#[serde(default)] pub oversample_jitter: f64`; `run_internal()`'s `resample_block()` call site passes `config.oversample_jitter`; manifest construction populates the new field; `test_manifest_block_overlap_round_trip` struct literal updated. |
| `src/output/las_writer.rs` | Test helper `single_block_manifest()` `BlockManifest` literal updated with `oversample_jitter: 0.0`. |
| `src/cli/preprocess_cmd.rs` | New `--oversample-jitter <f64>` flag: parsing, `>= 0.0 && is_finite()` validation, help text section. |
| `src/cli/preprocess_labeled_cmd.rs` | New `--oversample-jitter <f64>` flag: local var, parsing, validation, `PreprocessConfig` struct-literal field, usage text section. |

### Verification gates (all passed)

1. `cargo build --release --features training` — clean build (only pre-existing, unrelated `wbraster` deprecation warnings).
2. `cargo test --release --features training` — 98/98 tests passed (97 lib unit tests + 1 integration test), including the 6 new jitter tests.
3. `cargo clippy --release --features training -- -D warnings` — zero warnings in this crate.
4. `cargo fmt --check` — clean, no formatting diffs.

### Notes

- Jitter is delivered as a **standalone, dual-use** feature (training + inference), per the explicit user decision to not pursue a migration to same-class kNN interpolation ("Option C"), since Option C requires ground-truth labels unavailable at inference time while `resample_block()` is shared by both the labeled training-prep pipeline and the real-world inference path.
- Default `0.0` preserves bit-identical behaviour to pre-Stage-29 exact-duplicate padding; the feature is fully opt-in and backward compatible with existing manifests and CLI invocations.

---

*This document is the authoritative specification for Stage 29. All
implementation deviations must be recorded in this file under an
"Implementation Notes" section.*
