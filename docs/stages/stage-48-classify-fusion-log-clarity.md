# Stage 48 — `classify` Fusion Status Logging Clarity

**Status:** Implemented
**Date:** 2026-08-08
**Depends on:** Stage 44 (classify-time prediction fusion)

---

## The Goal

Fix a user-reported "stickiness" issue: a user enabled prediction fusion via
`--fusion-radius <f64>` on one `classify` run, then ran `classify` again *without*
that flag expecting fusion to be off — but the terminal log showed fusion was still
active.

Investigation confirmed this is **not** a state-persistence bug — `classify` is a
stateless CLI invocation; nothing can leak between runs. The real cause is an
**ambiguous default source** combined with **incomplete logging**:

`resolve_fusion_radius()` (`src/cli/classify_cmd.rs`, unchanged by this stage) defaults
an omitted `--fusion-radius` to the `blocks.json` manifest's `block_overlap` field
(baked in at `preprocess --block-overlap <f64>` time) whenever that value is `> 0.0`.
This is **intentional, documented Stage 44 behavior** (`block_overlap`/halo reach
composing automatically with fusion — see `stage-44-classify-time-prediction-fusion.md`
and the user guide's "Fusion composes" note) — but the pre-Stage-48 log line:

```
[classify] prediction fusion enabled: radius=12.5, temperature=1.0
```

fires identically whether `radius` came from an explicit `--fusion-radius` flag or
from the manifest's `block_overlap` default, and **prints nothing at all** when fusion
is off. A user comparing two terminal logs side-by-side has no way to tell that the
"off" run's omitted flag silently fell back to a nonzero manifest default, nor any
line to notice is *missing* to confirm fusion was actually disabled.

This stage does not change any fusion decision logic, defaults, or CLI flags —
purely a logging clarity fix, per the user-approved "Option 1" from the investigation
summary.

---

## Inputs & Outputs

No new/changed CLI flags. No change to `resolve_fusion_radius`'s resolution logic,
`validate_fusion_temp`, `FusionConfig`, or any file format. The only observable change
is the `stderr` log line(s) `classify` prints regarding fusion status.

### Before

```
# fusion on (radius > 0, regardless of source):
[classify] prediction fusion enabled: radius=12.5, temperature=1.0
# fusion off: (nothing printed)
```

### After

```
# fusion on, explicit CLI flag:
[classify] prediction fusion: ON (radius=12.5, temperature=1.0, source=--fusion-radius flag)
# fusion on, manifest block_overlap default (no --fusion-radius passed):
[classify] prediction fusion: ON (radius=12.5, temperature=1.0, source=manifest block_overlap default)
# fusion off (whether via explicit `--fusion-radius 0` or manifest block_overlap == 0.0):
[classify] prediction fusion: OFF
```

---

## Steps & Specifications

`src/cli/classify_cmd.rs::run()`: after resolving `fusion_radius`/`fusion_temp`,
replace the single `if fusion_radius > 0.0 { .. }` conditional log with an
always-printed status line that:

1. Always prints exactly one fusion status line per run (previously: zero lines when
   off), so a log diff between an "on" and "off" run always shows the difference.
2. When `fusion_radius > 0.0`, names the **source** of the resolved value —
   `--fusion-radius flag` when `cfg.fusion_radius.is_some()`, else
   `manifest block_overlap default` — using the already-available `cfg.fusion_radius`
   `Option` (no new state; `resolve_fusion_radius`'s own `explicit` parameter *is*
   `cfg.fusion_radius`, so `cfg.fusion_radius.is_some()` exactly answers "did the user
   pass this on the CLI").

No changes to `resolve_fusion_radius`, `validate_fusion_temp`, or any type signature —
this is purely an additional `eprintln!` at the existing call site, keyed off data
already in scope.

---

## Module Touch List

| File | Change |
|---|---|
| `src/cli/classify_cmd.rs` | `run()`: replaced the conditional "enabled" log line with an always-printed ON/OFF status line that names the value's source when ON |

No changes to: `model::fusion`, `output::las_writer`, `evaluate_cmd.rs` (the analogous
`--fused-eval` path already always prints its scored-block summary line unconditionally
via `eprintln!("[evaluate] (fused) scored ... radius={} ... temp={fusion_temp}", ...)`,
so no equivalent ambiguity exists there — `--fusion-radius`/`--fusion-temp` are
hard-rejected without `--fused-eval` and have no manifest-derived default for
`evaluate`, only for `classify`).

---

## Definition of Done

- [x] `classify` prints exactly one fusion status line on every run (on or off).
- [x] When fusion is ON, the line names whether the radius came from the
      `--fusion-radius` flag or the manifest's `block_overlap` default.
- [x] When fusion is OFF (explicit `--fusion-radius 0`, or no flag and
      `manifest.block_overlap <= 0.0`), a clear `OFF` line is printed (previously:
      nothing).
- [x] No change to fusion decision logic, defaults, flags, or output file bytes —
      confirmed by the pre-existing `test_write_classified_fusion_blends_seam_and_preserves_interior`-style
      regression tests in `output/las_writer.rs` still passing unchanged.
- [x] `cargo test --features training` passes with zero regressions.
- [x] `cargo clippy --all-targets --features training -- -D warnings` → zero warnings.
- [x] `cargo fmt` → clean.
- [x] This spec file is synchronized with the implementation (AGENTS.md
      living-synchronization contract).

---

## Verification Log

- `cargo test --features training` — all tests pass (238 baseline + none removed;
  this stage adds no new test cases since the change is a log-message-only diff with
  no new branching logic worth a dedicated unit test — the existing
  `resolve_fusion_radius`/`validate_fusion_temp` unit tests already cover every value
  this log line reads).
- `cargo clippy --all-targets --features training -- -D warnings` — zero warnings.
- `cargo fmt` — clean.

## Alternatives Considered (and rejected)

| Alternative | Rejection rationale |
|---|---|
| Change the default so an omitted `--fusion-radius` never falls back to `block_overlap` (require explicit opt-in) | Rejected for this stage: this reverses a deliberate, documented Stage 44 design decision (fusion composing automatically with halo/overlap preprocessing) and would be a behavior change requiring its own spec/DoD and user-guide rewrite. The user approved the lower-risk logging-only fix ("Option 1") for now; a default-policy change remains available as a future stage if still desired after this fix. |
| Add an explicit `--no-fusion` flag alias for `--fusion-radius 0` | Deferred, not rejected — a small, low-risk complementary addition, but out of scope for the specific "Option 1" the user approved. Can be added in a follow-up stage on request. |
