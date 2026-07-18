# Stage 37 — Absolute Height-Above-Ground (HAG) Normalization

**Status:** COMPLETE — implementation landed alongside this spec.
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `PROJECT_SPEC.md §1` (Preprocessing — Height Above Ground),
`src/preprocessing/normalizer.rs`, `src/preprocessing/feature_extractor.rs`

---

## Goal

Fix the HAG (feature index 6 of the 17-feature per-point vector) so that a point
at a given **physical** height above ground always maps to the **same** feature
value, regardless of what else happens to be inside its block.

This is "Option D" from the HAG System Review executive summary. It targets an
observed failure mode in a 30-epoch training run: the model cannot reliably
separate **low / medium / high vegetation**.

### Root cause

`normalise_scalar_features` (pre-Stage-37) normalized raw HAG by the **99th
percentile of each block's own HAG values**:

```text
h_max = percentile_99(block_hag_values)
hag   = clamp(hag_raw / h_max, 0, 1)
```

ASPRS low/medium/high vegetation are defined by **absolute** height bands
(≈ <0.5 m, 0.5–2 m, >2 m). Per-block percentile normalization destroys that
absolute scale: a 1 m shrub in a block whose tallest object is 2 m produces
`hag ≈ 0.5`, while the *same* 1 m shrub in a block containing a 20 m tree
produces `hag ≈ 0.05`. Physically identical points therefore present wildly
different HAG features depending purely on their block neighbours — precisely
the ambiguity that collapses the class-separating signal for vegetation tiers.

---

## Decision

Introduce a normalization **strategy** and make the default a **fixed absolute
reference height** (in projection z-units, i.e. metres for a metric CRS):

```text
hag = clamp(hag_raw / hag_max_meters, 0, 1)      # default: hag_max_meters = 50.0
```

Points at or above `hag_max_meters` saturate at `1.0`; a fixed reference of
50 m comfortably spans tall trees/buildings while preserving fine resolution in
the 0–10 m band where the vegetation tiers live. The legacy per-block
percentile behaviour is retained behind an explicit opt-in flag so prior runs
can be reproduced and A/B-compared.

### Breaking change

This changes the numeric values written into the HAG column of every `.feat`
file (the fixed-width 17-feature layout from Stage 30 is **unchanged** — only
the HAG value semantics change). Consistent with the project's handling of such
changes (see Stage 30), **any model trained against pre-Stage-37 `.feat` files
must be retrained** after adopting the new default. The current effortful run is
a fresh retrain, so this is absorbed at no extra cost.

---

## Choosing `hag_max` (sizing guidance)

The transform is `hag = clamp(raw / hag_max, 0, 1)` — a **linear scale** plus a
**hard ceiling**. These two parts behave very differently, and that split is
what tells you when the value actually matters:

1. **Below the ceiling — the linear scale is largely *not* critical.**
   Dividing every point by the same constant is a scale factor the model can
   mostly absorb on its own: the first `Linear` layer can learn any multiplier,
   and the early BatchNorm neutralizes input scale. Whether a 2 m point reads as
   `0.04` (`hag_max = 50`) or `0.10` (`hag_max = 20`), the model can learn to use
   either. In this regime `hag_max` is an **arbitrary but stable reference** —
   the *stability* (identical value in every block) is the Stage 37 win, not the
   specific number.

2. **The ceiling / saturation — this *is* the part that matters.**
   This is the only non-linear, information-destroying piece: every point at or
   above `hag_max` collapses to exactly `1.0` and becomes indistinguishable.
   Hence the one hard rule:

   > **`hag_max` must be ≥ the tallest object you need to distinguish.**

   Set it to 10 m and all tall trees, buildings, and towers saturate together
   and that signal is lost. The 50 m default is chosen so essentially nothing of
   interest clips.

**Strong reasons to change it:**

- **Raise** it if the scene contains features taller than ~50 m that must be
  told apart (tall conifers, towers, high-rise) — otherwise they saturate.
- **Lower** it (toward ~20–25 m) only if nothing above that height needs
  distinguishing. The benefit is marginal because of point (1), but non-zero.

**Caveat specific to the vegetation-tier problem this stage targets:** the
low/medium/high vegetation tiers live in the 0–2 m band, which at `hag_max = 50`
maps to just `[0, 0.04]` — a thin sliver of the output range. This is fine for
*correctness* (those values are perfectly distinguishable in `f32`, and
BatchNorm rescales them), so it should not block learning. But if, after
retraining, the tiers remain fuzzy, **lowering `hag_max` to ~20–25 m is a cheap,
principled A/B knob**: it stretches the vegetation band across more of `[0, 1]`
while still clearing typical tree/building heights. `--hag-max` is a runtime
flag precisely so this can be swept without recompiling.

**Summary:** `hag_max` is primarily a stable reference, governed by one hard
constraint (never below the tallest object you care about). 50 m is a safe
default; the main lever worth reaching for is *lowering* it toward ~20–25 m as
an experiment to give the vegetation tiers more numeric spread.

---

## Inputs & Outputs

### New public type (`src/preprocessing/normalizer.rs`)

```rust
pub enum HagNormalization {
    /// Divide raw HAG by a fixed absolute reference height (projection
    /// z-units). Preserves absolute vertical scale across blocks. Default.
    FixedMeters(f64),
    /// Legacy (pre-Stage-37): divide by the 99th percentile of the block's
    /// own HAG values. Retained for reproducibility / comparison only.
    BlockPercentile99,
}
```

`DEFAULT_HAG_MAX_METERS: f64 = 50.0`; `HagNormalization::default() = FixedMeters(50.0)`.

### `PreprocessConfig` (`src/preprocessing/mod.rs`)

New field:

```rust
/// Strategy for normalizing raw HAG into the [0,1] feature range.
/// Default: FixedMeters(DEFAULT_HAG_MAX_METERS) — see Stage 37.
pub hag_normalization: HagNormalization,
```

### Function signature changes

- `normalise_scalar_features(pts, origin_x, origin_y, block_size, hag_values, hag_norm)`
  — gains a trailing `hag_norm: HagNormalization` parameter.
- `extract_features(pts, eigen_rows, dtm, origin_x, origin_y, block_size, hag_norm)`
  — gains a trailing `hag_norm: HagNormalization` parameter; forwards it.

### CLI flags (both `preprocess` and `preprocess-labeled`)

- `--hag-max <f64>` — fixed absolute reference height in projection units
  (default 50.0). Selects `FixedMeters`.
- `--hag-norm-percentile` — opt into the legacy block-99th-percentile
  normalization (ignores `--hag-max`).

Validation: `--hag-max` must be positive and finite.

---

## Steps & Specifications

1. Add `HagNormalization` + `DEFAULT_HAG_MAX_METERS` to `normalizer.rs`; re-export
   from `preprocessing::mod`.
2. Thread `hag_norm` through `normalise_scalar_features` and compute:
   `h_max = match hag_norm { FixedMeters(m) => m.max(1e-9), BlockPercentile99 => percentile_99(..).max(1e-9) }`.
3. Thread `hag_norm` through `extract_features`; pipeline passes
   `config.hag_normalization`.
4. Add `hag_normalization` to `PreprocessConfig` + `Default`.
5. Parse `--hag-max` / `--hag-norm-percentile` in both CLI commands; validate;
   document in `--help`.
6. Tests: fixed-scale invariance (same physical height ⇒ same value across
   blocks with different neighbours); explicit percentile opt-in reproduces the
   legacy value; `--hag-max` validation rejects non-positive/non-finite.
7. Update `PROJECT_SPEC.md §1` HAG bullet to describe the new default.

---

## Definition of Done

- [x] `cargo build` and `cargo clippy` clean (no new warnings).
- [x] `cargo test` green, including new normalizer tests (68 passed).
- [x] Default preprocessing uses `FixedMeters(50.0)`; legacy reachable via
      `--hag-norm-percentile`.
- [x] A point at a fixed physical HAG maps to an identical feature value in two
      blocks with different neighbour height distributions (regression test:
      `test_fixed_meters_is_neighbour_invariant`).
- [x] `--help` for both `preprocess` and `preprocess-labeled` documents the new
      flags; `PROJECT_SPEC.md §1` updated.

---

*This document is the authoritative specification for Stage 37. Per the
AGENTS.md synchronization rule, the code and this spec must remain in sync.*
