# Stage 42 — Absolute Elevation (`z_norm`) Normalization

**Status:** COMPLETE — implementation landed alongside this spec.
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `docs/stages/stage-37-absolute-hag-normalization.md` (direct
precedent — same bug class), `src/preprocessing/normalizer.rs`,
`src/preprocessing/feature_extractor.rs`, `src/preprocessing/pipeline.rs`

---

## Goal

Fix the raw-elevation scalar feature `z_norm` (feature index 2 of the
17-feature per-point vector) so that a point at a given **absolute**
elevation always maps to the **same** feature value, regardless of which
block/tile it happens to fall in.

This is a direct follow-up to Stage 37 (HAG normalization) and was raised
during an architectural review of a user-reported "patchwork quilt"
classification artifact: block-shaped discontinuities visible in classified
output that suggested per-tile object classification rather than genuine
semantic segmentation. That review confirmed the PointNet architecture *is*
performing genuine per-point segmentation, and identified `z_norm` as one of
three contributing root causes of the visual tiling artifact (the other two
— the inherent per-tile-only global max-pool, and the vestigial
`block_overlap` mechanism — are architectural/design topics deferred to a
separate "prediction blending" discussion, not addressed by this stage).

### Root cause

`normalise_scalar_features` (pre-Stage-42) normalized raw elevation `z` by
**that block's own local min/max z values**:

```text
z_norm = (z - block_z_min) / (block_z_max - block_z_min)
```

This is the exact same class of bug fixed for HAG in Stage 37: a point at a
fixed absolute elevation (e.g. 143.2 m) produces a different `z_norm` value
depending purely on the elevation range of whatever else happens to be in
its block. Two adjacent blocks with different local elevation spreads (e.g.
one block spanning flat ground, the neighbouring block spanning a slope or
containing a tall structure) assign different `z_norm` values to points at
the *same* absolute height near their shared boundary — a tile-boundary
discontinuity in the feature space itself, independent of any model
weakness. This directly contributes to the visual "seams" observed in
classified output.

---

## Decision

Introduce a normalization **strategy** — mirroring Stage 37's
`HagNormalization` pattern — and make the default the **whole-file absolute
elevation range**, sourced once from the input LAS/LAZ/COPC header's own
`min_z` / `max_z` fields:

```text
z_norm = clamp((z - file_z_min) / (file_z_max - file_z_min), 0, 1)   # default
```

Unlike HAG, raw elevation has no universal fixed reference constant that
would be meaningful across arbitrary datasets (a HAG of "2 m above ground"
means the same thing in any dataset; an absolute elevation of "143 m" does
not — it depends entirely on the site's vertical datum). The correct
"absolute" reference for `z_norm` is therefore **that specific input file's
own elevation range**, not a project-wide constant. This is still an
*absolute* (neighbour-invariant) normalization in the sense that matters:
every block in a given pipeline run uses the identical range, so a point at
a fixed elevation maps to the identical feature value no matter which block
it lands in.

The legacy per-block min/max behaviour is retained behind an explicit
opt-in flag so prior runs can be reproduced and A/B-compared, exactly as
Stage 37 did for HAG.

### Breaking change

This changes the numeric values written into the `z_norm` column (feature
index 2) of every `.feat` file (the fixed-width 17-feature layout from
Stage 30 is **unchanged** — only the `z_norm` value semantics change).
Consistent with the project's handling of such changes (see Stage 30 and
Stage 37), **any model trained against pre-Stage-42 `.feat` files must be
retrained** after adopting the new default — the distribution of the
`z_norm` feature has shifted, and a previously trained model's learned
weights are calibrated against the old (neighbour-dependent) distribution.

---

## Inputs & Outputs

### New public type (`src/preprocessing/normalizer.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZNormalization {
    /// Normalise against the whole input file's absolute elevation range
    /// (LAS/LAZ/COPC header `min_z`/`max_z`, resolved once per pipeline run).
    /// Neighbour-invariant. Default.
    Global { z_min: f64, z_max: f64 },
    /// Legacy (pre-Stage-42): normalise against each block's own local
    /// min/max z. Retained for reproducibility / comparison only.
    /// Neighbour-dependent.
    BlockMinMax,
}
```

Unlike `HagNormalization::FixedMeters`, `ZNormalization::Global`'s bounds
cannot be resolved at CLI-parse time — they depend on the input file's own
header, which is only inspected once the pipeline actually runs. Therefore
`PreprocessConfig` stores only a `bool` (`z_norm_use_block_relative`), and
`pipeline.rs::run_internal()` resolves the concrete `ZNormalization` value
immediately after the header inspection step:

```rust
let z_norm_strategy: ZNormalization = if config.z_norm_use_block_relative {
    ZNormalization::BlockMinMax
} else {
    ZNormalization::Global { z_min, z_max }
};
```

### `PreprocessConfig` (`src/preprocessing/mod.rs`)

New field:

```rust
/// `false` (default) — use ZNormalization::Global (whole-file absolute
/// range, resolved from the header). `true` — opt into the legacy
/// per-block ZNormalization::BlockMinMax.
pub z_norm_use_block_relative: bool,
```

### Function signature changes

- `normalise_scalar_features(pts, ..., z_norm_strategy: ZNormalization)` —
  gains a trailing `z_norm_strategy` parameter; the block-local min/max
  computation is replaced with a match on the strategy.
- `extract_features(pts, eigen_rows, dtm, origin_x, origin_y, block_size,
  hag_normalization, z_norm_strategy)` — gains a trailing `z_norm_strategy`
  parameter; forwards it to `normalise_scalar_features`.
- `inspect_lidar_header(path) -> Result<(f64, f64, f64, f64, f64, f64, u64,
  Option<u32>)>` — gains two additional return tuple elements, `z_min` and
  `z_max`, sourced from the LAS header's own `min_z` / `max_z` fields
  (previously only `x_min, y_min, x_max, y_max, point_count, epsg` were
  returned).

### `BlockManifest` (`src/preprocessing/pipeline.rs`)

New field (recorded for provenance, `#[serde(default)]` for backward
compatibility with older manifests):

```rust
/// `false` (default) means the fixed/global z-normalisation mode was used.
#[serde(default)]
pub z_norm_use_block_relative: bool,
```

### CLI flags (both `preprocess` and `preprocess-labeled`)

- `--z-norm-block-relative` — opt into the legacy per-block `z_norm`
  normalisation. No value required (boolean flag); mirrors the parsing
  convention of other boolean CLI flags in this project
  (`parse_optional_bool`).

No new numeric flag is needed (unlike `--hag-max`) since the "reference"
for `ZNormalization::Global` is always the input file's own header — there
is nothing for the user to size or tune.

---

## Steps & Specifications

1. Add `ZNormalization` to `normalizer.rs`; re-export from
   `preprocessing::mod`.
2. Thread `z_norm_strategy` through `normalise_scalar_features`, replacing
   the local block min/max computation with a match on the strategy.
3. Thread `z_norm_strategy` through `extract_features` as a new trailing
   parameter.
4. Extend `inspect_lidar_header` to also return the header's `min_z` /
   `max_z`; update both match arms (`las`/`laz` and `copc`).
5. In `pipeline.rs::run_internal()`, resolve `z_norm_strategy` once
   immediately after the header inspection call, and pass it into the
   per-block `extract_features` call inside the Step 7 parallel closure
   (`ZNormalization` is `Copy`, so the Rayon closure captures it trivially).
6. Add `z_norm_use_block_relative` to `PreprocessConfig` + `Default`
   (default `false`), and to `BlockManifest` (`#[serde(default)]`,
   recorded from `config.z_norm_use_block_relative`).
7. Parse `--z-norm-block-relative` in both `preprocess_cmd.rs` and
   `preprocess_labeled_cmd.rs`; document in `--help`/usage text.
8. Tests: global-strategy neighbour invariance (same absolute elevation ⇒
   same feature value across blocks with different local z ranges);
   explicit block-relative opt-in reproduces the legacy neighbour-dependent
   value; saturation/clamping at the file's z bounds; CLI default/flag
   tests; manifest round-trip test extended with the new field.

---

## Definition of Done

- [x] `normalizer.rs`: `ZNormalization` enum added; `normalise_scalar_features`
      takes the new strategy parameter; 4 new/updated tests
      (`test_global_z_norm_is_neighbour_invariant`,
      `test_block_min_max_z_norm_is_neighbour_dependent`,
      `test_global_z_norm_saturates_at_bounds`, plus a `global_from_pts`
      test helper).
- [x] `feature_extractor.rs`: `extract_features` threads the strategy through
      to `normalise_scalar_features`; existing test call sites updated.
- [x] `mod.rs`: `PreprocessConfig::z_norm_use_block_relative` field added
      (default `false`); `ZNormalization` re-exported;
      `test_preprocess_config_default_z_norm_uses_global` added.
- [x] `pipeline.rs`: `inspect_lidar_header` returns header z-bounds;
      `z_norm_strategy` resolved once in `run_internal()`; `BlockManifest`
      gains `z_norm_use_block_relative` (`#[serde(default)]`); existing
      manifest round-trip test updated.
- [x] `preprocess_cmd.rs` / `preprocess_labeled_cmd.rs`: `--z-norm-block-relative`
      flag parsed, documented in help/usage text, and (in `preprocess_cmd.rs`)
      covered by default/opt-in tests.
- [x] Cross-codebase sweep confirmed no other callers of `extract_features`,
      `normalise_scalar_features`, or `inspect_lidar_header` exist outside
      the files listed above, and no `tests/` files construct
      `PreprocessConfig` literals requiring updates.
- [x] `cargo build`, `cargo fmt --check`, `cargo clippy --features training --
      -D warnings`, and `cargo test --features training` all clean (171 unit
      tests + 1 integration test passing). One test-only `BlockManifest {
      ... }` literal in `src/output/las_writer.rs`'s `single_block_manifest()`
      helper was missed by the initial cross-codebase sweep and required the
      new `z_norm_use_block_relative: false` field added; a construction bug
      in `test_block_min_max_z_norm_is_neighbour_dependent` (the shared point
      sat exactly at each block's own min, which trivially degenerates to
      `z_norm == 0.0` under `BlockMinMax` regardless of the surrounding
      range, making the test self-defeating) was also caught and fixed by
      repositioning the shared point strictly between each block's local
      min/max.


---

## Retraining Requirement

**Any model trained on `.feat` files produced before this stage must be
retrained.** The default pipeline behaviour now writes different numeric
values into the `z_norm` feature column than it did previously — this is
the same severity of breaking change as Stage 37's HAG fix. Users who need
to reproduce prior results exactly can pass `--z-norm-block-relative` to
restore the legacy per-block behaviour during preprocessing, but the
recommended path forward is to regenerate `.feat` files with the new
default and retrain.

---

## Deferred / Related Findings (not addressed by this stage)

The architectural review that surfaced this bug also identified two other
contributors to the "patchwork quilt" visual artifact, intentionally **not**
addressed here and reserved for a follow-up "prediction blending"
discussion:

1. **Per-tile-only global max-pool.** PointNet's global feature is computed
   independently per block/tile; a point's classification has no visibility
   into points outside its own tile. This is architecturally inherent to
   tiled inference, not a bug, but it means genuinely different local
   context across a tile boundary can legitimately produce different
   classifications for nearby points on either side of the seam.
2. **`block_overlap` is dead/vestigial code.** Border points loaded via the
   `block_overlap` mechanism (Stage 08) are read from disk and then
   immediately dropped without being merged into any feature computation or
   spatial index (since Stage 30 removed the local eigenvalue computation
   that used to consume them) — the mechanism currently has no effect on
   model input. Reviving it as a genuine channel of cross-tile context is
   one candidate direction for the blending discussion.

These are being discussed separately per the user's explicit direction:
*"Go ahead with the fix to z_norm and then let's discuss methods for
reconciling disparate classifications for the same point across blocks."*

---

*This document is the authoritative specification for Stage 42. Per the
AGENTS.md synchronization rule, the code and this spec must remain in sync.*
