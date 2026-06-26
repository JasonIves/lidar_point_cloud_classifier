# Stage 07 — Tunable Class Weighting System

**Status:** COMPLETE — See implementation record below  
**Approved:** 2026-06-24  
**Implemented:** 2026-06-24  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier  
**Lead Architect:** GitHub Copilot / AI Collaborator

---

## Goal

Replace the hard-coded pure inverse-frequency class weighting formula with a
**β-scaled effective number weighting** system that is continuously tunable via a
single hyperparameter `--class-weight-beta`.  This gives practitioners a principled
knob to control how aggressively the loss function compensates for class imbalance,
without requiring a binary on/off choice.

### Motivation

The Stage 03 inverse-frequency formula:

```
weight[c] = total / (n_classes × count[c])
```

is correct in principle but inflexible in practice.  In typical ASPRS LiDAR datasets,
class imbalance is extreme (e.g., Ground at 60% vs. Low Point noise at 0.3%).  Pure
inverse-frequency produces weights that differ by 100–200× across classes, which:

1. **Dominates gradient signal with rare classes** — the model spends most of its
   learning capacity on the 0.3% of points that are Low Point noise, at the expense
   of the 60% that are Ground.
2. **Collapses majority-class accuracy** — Ground and Low Vegetation IoU degrades
   because the loss landscape is warped away from them.
3. **Suppresses overall mIoU** — even if minority-class IoU improves, the
   majority-class degradation more than offsets it in the mean.
4. **Is dataset-specific and non-transferable** — weights computed from one
   acquisition's distribution are wrong for a different acquisition.

The binary `--no-class-weights` flag (uniform weights) is the only escape valve, but
it is a cliff edge: there is no middle ground between extreme minority-class emphasis
and no emphasis at all.

---

## Inputs & Outputs

### New `TrainConfig` field

| Field | Type | Default | Description |
|---|---|---|---|
| `class_weight_beta` | `f64` | `0.999` | β parameter for effective-number weighting. Range `[0.0, 1.0)`. `0.0` = uniform weights. Values near `1.0` give progressively stronger minority-class emphasis. |

The existing `use_class_weights: bool` field is retained.  When `use_class_weights =
false`, weight computation is skipped entirely (fast path, unchanged behavior).
`--no-class-weights` continues to work as before.

### New CLI flag

Available on `wb_lidar_train train`:

```
--class-weight-beta  <f64>   β parameter for class weight scaling (default: 0.999).
                              Range: [0.0, 1.0).
                              0.0  = uniform weights (equivalent to --no-class-weights).
                              0.9  = mild minority-class emphasis.
                              0.99 = moderate emphasis (typical urban LiDAR).
                              0.999 = strong emphasis (default; severe imbalance).
                              0.9999 = near-inverse-frequency; use with caution.
                              --no-class-weights is equivalent to --class-weight-beta 0.0
                              and takes precedence when both flags are supplied.
```

### Backward compatibility

Existing CLI invocations without `--class-weight-beta` receive the new default
(`0.999`).  This is a **better default than the previous `1.0` (pure
inverse-frequency)** for most LiDAR datasets, but it is a behavioral change for
users who relied on the exact inverse-frequency weights.  Users who want the previous
exact behavior should pass `--class-weight-beta 0.9999` (numerically equivalent to
inverse-frequency for counts > 1000).

---

## Steps & Specifications

### The β-Scaled Effective Number Formula

Based on Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of
Samples":

```
effective_num[c] = (1 - β^count[c]) / (1 - β)
raw_weight[c]    = 1 / effective_num[c]          for count[c] > 0
raw_weight[c]    = 0.0                           for count[c] == 0
```

The raw weights are then **normalized** so that the mean weight of present classes
equals 1.0, keeping the loss magnitude comparable across β values and ensuring the
learning rate remains stable:

```
present_sum = sum of raw_weight[c] for all c where count[c] > 0
n_present   = count of c where count[c] > 0
scale       = n_present / present_sum
weight[c]   = raw_weight[c] * scale
```

**Special cases:**

| β | Behavior |
|---|---|
| `0.0` | `effective_num[c] = 1` for all c → all weights = 1.0 (uniform) |
| `→ 1.0` | `effective_num[c] → count[c]` → pure inverse-frequency (previous default) |
| `0.9` | Mild minority-class emphasis |
| `0.99` | Moderate emphasis |
| `0.999` | Strong emphasis (new default) |
| `0.9999` | Near-inverse-frequency |

**Implementation note on β = 0.0:** The formula is evaluated as a special case
(`beta.abs() < 1e-9`) returning uniform weights directly, avoiding a division by
`1 - β = 1.0` that would produce the correct result but is less readable.

**Implementation note on large counts:** `β^count[c]` underflows to `0.0` for
`count[c] > ~700` at `β = 0.999` (since `0.999^700 ≈ 5e-4`).  This is correct
behavior — for very large counts the effective number saturates at `1 / (1 - β)`,
which is the intended asymptotic behavior.  No special handling is required.

### Tuning Guidance

| β value | Behavior | Recommended for |
|---|---|---|
| `0.0` | Uniform (no weighting) | Balanced datasets; debugging |
| `0.9` | Mild minority emphasis | Mildly imbalanced datasets (< 5× ratio) |
| `0.99` | Moderate emphasis | Typical urban LiDAR (5–10× imbalance) |
| `0.999` | Strong emphasis (default) | Severely imbalanced datasets (50–100× imbalance) |
| `0.9999` | Near-inverse-frequency | Extreme imbalance; use with caution |

---

## Changed Files

| File | Nature of Change |
|---|---|
| `src/training/trainer.rs` | Add `class_weight_beta: f64` to `TrainConfig`; replace inverse-frequency formula with β-scaled effective number formula |
| `src/cli/train_cmd.rs` | Add `--class-weight-beta <f64>` flag; add range validation `[0.0, 1.0)` |
| `src/training/metrics.rs` | Add 3 new unit tests for the weight formula |
| `docs/stages/stage-07-tunable-class-weighting.md` | This file |

No preprocessing, inference, model, or output files are modified.  The inference
binary (`wb_lidar_classify`) is completely unaffected.

---

## Definition of Done

| # | Criterion | Verification | Status |
|---|---|---|---|
| 1 | `cargo build --release --features training` — zero errors | Build gate | ✅ Pass |
| 2 | `cargo clippy --features training -- -D warnings` — zero new warnings | Clippy gate | ✅ Pass |
| 3 | `cargo fmt --check` passes | fmt gate | ✅ Pass |
| 4 | `cargo test --features training` — all existing tests pass + 3 new weight tests | Regression + new | ✅ Pass |
| 5 | `--class-weight-beta 0.0` produces uniform weights (all 1.0) | `test_class_weight_beta_uniform` | ✅ Pass |
| 6 | `--class-weight-beta 0.9999` produces weights numerically close to inverse-frequency | `test_class_weight_beta_inverse_freq` | ✅ Pass |
| 7 | `--class-weight-beta 0.9` on known 3-class distribution matches hand-calculated values | `test_class_weight_beta_intermediate` | ✅ Pass |
| 8 | `--class-weight-beta 1.5` rejected with clear error message | CLI validation | ✅ Pass |
| 9 | Stage spec synchronized with implementation | Manual review | ✅ Pass |

---

## Relationship to Prior Stages

- **Stage 03** introduced `use_class_weights` and the inverse-frequency formula.
  Stage 07 extends `TrainConfig` with `class_weight_beta` and replaces the formula.
  The `use_class_weights` bool is retained as a fast-path opt-out.
- **No stage 01/02/04/05/06 files are modified.**  This is a pure training-module
  change.
- The `.wbmodel` binary format is **unchanged** — class weights are a training-time
  artifact and are not stored in the model file.

---

*This document is the authoritative specification for Stage 07.  All implementation
deviations must be recorded in this file under an "Implementation Notes" section.*
