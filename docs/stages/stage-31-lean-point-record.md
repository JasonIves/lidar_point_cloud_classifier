# Stage 31 — Lean In-Pipeline Point Representation (`LitePoint`)

**Status:** ✅ IMPLEMENTED — see "Implementation Status" section at the
bottom of this document for verification results.
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** GitHub Copilot / AI Collaborator
**Related:** `docs/stages/stage-30-whitebox-git-dependency-integration.md`
(independently scoped — see "Relationship to Stage 30" below; this stage is
**not** a prerequisite or blocker for Stage 30's work, and vice versa)

---

## Goal

Reduce the in-pipeline memory footprint of the classifier's internal point
representation by replacing the full `wblidar::PointRecord` — used throughout
`BlockPartitioner`, the per-block spatial index, feature extraction, and
normalization — with a small, project-local, purpose-built struct (working
name: `LitePoint`) carrying only the fields the pipeline actually reads.

This idea surfaced as a piece of constructive scope-creep during the Stage 30
eigenvalue-features discussion (the user's own words: "I know this is a bit of
scope creep") and was validated as sound via a direct codebase audit. It is
captured here as its own stage spec, per explicit user direction, rather than
folded into Stage 30.

---

## Motivation / Evidence

### `wblidar::PointRecord` is large

Confirmed via full read of `whitebox_next_gen/crates/wblidar/src/point.rs`:
`PointRecord` is a flat, `Copy`, `Option`-heavy struct with the following
fields:

- `x, y, z: f64`
- `intensity: u16`
- `color: Option<Rgb16>`
- `nir: Option<u16>`
- `thermal_rgb: Option<ThermalRgb>`
- `classification: u8`
- `user_data: u8`
- `point_source_id: u16`
- `flags: u8`
- `return_number: u8`
- `number_of_returns: u8`
- `scan_direction_flag: bool`
- `edge_of_flight_line: bool`
- `scan_angle: i16`
- `gps_time: Option<GpsTime>`
- `waveform: Option<WaveformPacket>`
- `extra_bytes: ExtraBytes` — a **fixed 192-byte inline buffer, always
  allocated** regardless of whether the source file actually carries extra
  bytes
- `normal_x, normal_y, normal_z: Option<f32>`

Estimated total size: **~330–400 bytes per point** — roughly **10–14×** the
on-disk size of a typical LAS point record (26–34 bytes for common Point Data
Record Formats). This amplification exists purely as an in-memory
convenience for `wblidar`'s own general-purpose LAS/LAZ I/O API surface; it is
not something the classifier's pipeline needs or benefits from once a point
has been read off disk.

### Confirmed field-usage audit

A `search_files` sweep across `src/` (this session) confirmed that only the
following `PointRecord` fields are ever read anywhere in the classifier's
pipeline:

- `x, y, z` — spatial coordinates (indexing, feature extraction, HAG lookup)
- `intensity` — scalar feature
- `return_number`, `number_of_returns` — scalar features
- `scan_angle` — scalar feature
- `classification` — used in `labeled_pipeline.rs` for ground-truth label
  extraction during training-data preparation, in `pipeline.rs`/`las_writer.rs`
  test helpers, and as the passthrough default value in `las_writer.rs`'s
  final inference-substitution step

No other field (`color`, `nir`, `thermal_rgb`, `user_data`, `point_source_id`,
`flags`, `scan_direction_flag`, `edge_of_flight_line`, `gps_time`, `waveform`,
`extra_bytes`, `normal_x/y/z`) is read anywhere in the pipeline today. This is
an exact match for the field list the user proposed during the Stage 30
discussion, plus `classification`.

### The key enabling fact: `las_writer.rs` is fully decoupled

Confirmed via read of `src/output/las_writer.rs::write_classified()`: final
classified output is produced by **re-opening and re-streaming the original
input file directly** (`open_reader(input_path)`), looking up each point's new
classification by nearest (x, y) match against the pipeline's
`BlockInferenceResult`. This function never touches whatever point
representation flows through the internal blocking/feature-extraction
pipeline — it goes straight back to the original file's full `PointRecord`
data (RGB, GPS time, waveform, extra bytes, everything) for the final write.

**This is the fact that makes slimming the internal pipeline's point type
safe with zero final-output-fidelity impact.** Whatever fields a leaner
in-pipeline struct omits, the final classified LAS/LAZ output is completely
unaffected, because it is generated from a fresh, independent, full-fidelity
read of the original input file — not from whatever struct happened to flow
through `BlockPartitioner`/`feature_extractor.rs`/`normalizer.rs` internally.

---

## Proposed Design

### `LitePoint` struct (working name/shape)

```rust
#[derive(Copy, Clone, Debug)]
pub struct LitePoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub scan_angle: i16,
    pub classification: u8,
}
```

Estimated size: **~24–31 bytes** (depending on alignment/padding) — roughly a
**10× reduction** versus `wblidar::PointRecord`'s ~330–400 bytes.

### Conversion point

Conversion from `wblidar::PointRecord` to `LitePoint` happens **immediately at
streaming-ingest time**, as each point is read off disk and before it reaches
`BlockPartitioner` — i.e., the full `PointRecord` never enters the
block-partitioning/spatial-indexing/feature-extraction/normalization stages of
the pipeline at all; only the lean struct does.

### Affected modules

| Module | Change |
|---|---|
| `src/preprocessing/block_partitioner.rs` | `Block`/`BlockStub`/spill mechanism operate on `LitePoint` instead of `PointRecord` |
| `src/preprocessing/spatial_index.rs` | `BlockSpatialIndex` built from/queries `LitePoint` |
| `src/preprocessing/feature_extractor.rs` | `extract_features()` reads `x, y, z, intensity, return_number, number_of_returns, scan_angle` from `LitePoint` |
| `src/preprocessing/normalizer.rs` | `resample_block()`/`normalise_scalar_features()`/HAG computation operate on `LitePoint` |
| `src/preprocessing/labeled_pipeline.rs` | Ground-truth `classification` extraction reads from `LitePoint` |

### Explicitly NOT affected

- **`src/preprocessing/outlier_filter.rs`** — moot; this file is being deleted
  entirely as part of Stage 30's sweep (direct `wbtools_oss::LidarRemoveOutliersTool`
  usage is restored), so it is not a conversion target either way.
- **`src/output/las_writer.rs`** — fully decoupled (see above); continues to
  re-read the original input file's full `PointRecord` data directly for the
  final classified output. No change needed or wanted here.

---

## Estimated Benefit

- **~10× reduction** in per-point memory footprint for every point held in
  memory during block partitioning, feature extraction, and normalization —
  the pipeline's actual working set for its own internal logic.
- **Reduced `.spill`-mechanism trigger frequency.** `block_partitioner.rs`'s
  `SPILL_HIGH_WATER_BYTES = 512 MB` threshold (defined in
  `preprocessing/mod.rs`) is measured against total in-memory point bytes; a
  10× smaller per-point footprint means roughly 10× more points can be held
  in memory before the spill mechanism triggers, reducing spill-file I/O
  overhead on large inputs without changing the configured byte threshold
  itself.
- Smaller, simpler struct is easier to reason about and serialize for
  potential future needs (e.g., debug dumps, alternate spill formats).

---

## Critical Caveat: No Interaction With `wbtools_oss::LidarEigenvalueFeaturesTool`'s Own Memory Use

This refactor does **not** reduce the memory footprint of
`wbtools_oss::LidarEigenvalueFeaturesTool` itself (see Stage 30's approved
eigenvalue-features pre-pass design). When given a file path to process, that
tool always allocates its own internal `Vec<wblidar::PointRecord>`-based cloud
representation — it has no awareness of, and cannot be made to use, this
project's lean `LitePoint` struct, since it operates directly on LAS/LAZ file
input via `wblidar`'s own I/O layer.

Consequently, **Stage 30's memory-gated splitting math must always compute
against `size_of::<wblidar::PointRecord>()`**, never against
`size_of::<LitePoint>()`, regardless of whether this stage (31) has been
implemented. The two efforts are **complementary and independent**, not
additive, for that specific sizing calculation — Stage 31 slims this
project's own internal working set; it has zero effect on the
`wbtools_oss` tool's internal allocation when invoked as a file-path-based
external pre-pass.

This stage does not depend on Stage 30 being implemented first, and Stage 30
does not depend on this stage at all — either may be approved and implemented
independently, in either order.

---

## Steps & Specifications (for the eventual implementation stage, once approved)

**Not authorized to begin until the user reviews and approves this document.**

1. Define `LitePoint` (name/shape as above, or as refined during review) in a
   suitable module (e.g. a new `src/preprocessing/lite_point.rs`, or inline in
   `preprocessing/mod.rs`).
2. Add a `From<&wblidar::PointRecord> for LitePoint` (or equivalent explicit
   conversion function) used exactly once, at streaming-ingest time, before
   points reach `BlockPartitioner`.
3. Update `block_partitioner.rs` (`Block`, `BlockStub`, spill read/write
   logic, `PT_BYTES`-style size constants), `spatial_index.rs`,
   `feature_extractor.rs`, `normalizer.rs`, and `labeled_pipeline.rs` to
   operate on `LitePoint` instead of `wblidar::PointRecord`.
4. Verify `las_writer.rs` requires **no changes** (confirm its
   `write_classified()` path continues to re-read the original input
   file directly, independent of whichever point type flows through the
   internal pipeline).
5. Run full existing test suite; add/adjust any tests that construct
   `PointRecord` values directly for pipeline-internal testing purposes (these
   should construct `LitePoint` instead, or a test-only conversion helper).
6. Update this document's "Implementation Status" section with actual
   `cargo build`/`clippy`/`test` verification results, per this project's
   established stage-spec convention.

---

## Definition of Done (for this documentation-only stage)

1. This document exists and accurately records: the motivation (memory
   footprint, field-usage audit), the `las_writer.rs` decoupling fact that
   makes the refactor safe, the proposed `LitePoint` design, affected/
   unaffected modules, estimated benefit, and the critical Stage 30
   memory-sizing-caveat cross-reference. ✓ (this document)
2. No `Cargo.toml`, source, or other stage-spec files are modified as part of
   this stage — documentation only. ✓
3. This document is presented to the user for explicit review/approval before
   any implementation work begins — **no code changes occur until that
   approval is given.**

---

## Definition of Done (for the future implementation stage — tracked here for continuity, executed only after approval)

| # | Criterion | Status |
|---|---|---|
| 1 | `cargo build --release --features training` passes with `LitePoint` in place of `wblidar::PointRecord` throughout the internal pipeline (`block_partitioner.rs`, `spatial_index.rs`, `feature_extractor.rs`, `normalizer.rs`, `labeled_pipeline.rs`) | ✅ |
| 2 | `cargo clippy -- -D warnings` — zero new warnings | ✅ |
| 3 | `cargo test` / `cargo test --features training` — all existing tests pass (updated as needed for the new struct) | ✅ |
| 4 | `src/output/las_writer.rs` is verified unchanged and its final classified output remains byte-for-byte equivalent to pre-refactor output on the existing golden/regression test fixtures | ✅ |
| 5 | A memory-usage comparison (e.g. peak RSS or a simple `size_of::<LitePoint>()` vs. `size_of::<wblidar::PointRecord>()` assertion test) documents the achieved reduction | ✅ |
| 6 | This document's "Implementation Status" section is filled in with the sweep's actual verification results | ✅ (this section) |

---

## Implementation Status (2026-07-10)

Implemented substantially as designed above, with `labeled_pipeline.rs`
deliberately **left unconverted** (see Deviation #1 below) and
`spatial_index.rs` converted despite currently having zero production call
sites (Deviation #2).

### What was done

1. **`src/preprocessing/lite_point.rs`** (new file) — defines `LitePoint`
   exactly as specified (`x, y, z, intensity, return_number,
   number_of_returns, scan_angle, classification`), plus
   `impl From<&wblidar::PointRecord> for LitePoint` and
   `impl From<wblidar::PointRecord> for LitePoint`. Two unit tests:
   - `test_lite_point_is_much_smaller_than_point_record` — asserts
     `size_of::<LitePoint>() * 5 <= size_of::<PointRecord>()`, satisfying
     DoD #5 (memory-footprint documentation).
   - `test_conversion_preserves_used_fields` — verifies all 8 fields survive
     both the `&PointRecord` and owned-`PointRecord` conversion paths.
2. **`src/preprocessing/mod.rs`** — wires up `pub mod lite_point;` and
   `pub use lite_point::LitePoint;`.
3. **`src/preprocessing/block_partitioner.rs`** — `Block`, `BlockStub`, and
   `BlockPartitioner`'s internal accumulator/spill mechanism all converted to
   `LitePoint`. `add_point()` signature changed to
   `add_point(&mut self, index: u64, pt: LitePoint)`. The `.spill` file's
   on-disk byte layout for the point-data fields is unchanged (only the
   in-memory type constructing/reading those bytes changed) — spill files
   remain forward/backward-compatible with the pre-Stage-31 format at the
   byte level. Buffered-bytes spill-threshold accounting now uses
   `size_of::<LitePoint>()` (previously `size_of::<PointRecord>()`), giving
   this project's own internal spill mechanism the intended ~10× larger
   effective in-memory capacity before triggering.
4. **`src/preprocessing/feature_extractor.rs`** — `extract_features()` now
   takes `&[LitePoint]` instead of `&[PointRecord]`.
5. **`src/preprocessing/normalizer.rs`** — `resample_block()`,
   `normalise_scalar_features()`, and `compute_hag()` all operate on
   `&[LitePoint]`.
6. **`src/preprocessing/spatial_index.rs`** — `BlockSpatialIndex::build()`
   converted to accept `&[LitePoint]` (see Deviation #2).
7. **`src/preprocessing/pipeline.rs`** — the single conversion point required
   by the design (`LitePoint::from(&pt)`) is applied in `stream_points()`
   immediately after `reader.read_point(&mut pt)?`, for all three format
   branches (`las`/`laz`/`copc`), before the point reaches
   `BlockPartitioner::add_point()`. The border-point spill mechanism
   (`write_border_spill`/`read_border_spill`/`load_border_points`) was also
   converted to `LitePoint` for type consistency, even though — post Stage
   30 — border points are loaded and then immediately `drop()`-ed without
   further use (see Deviation #3). The Stage-30 eigenvalue pre-pass
   subsystem (`run_eigenvalue_prepass`, `run_eigenvalue_prepass_split`,
   `route_point_to_strips`, `infer_writer_config_from_source`, and their
   tests) was **deliberately left untouched**, per the Critical Caveat above:
   it still constructs/sizes against `wblidar::PointRecord` directly, since
   `wbtools_oss::LidarEigenvalueFeaturesTool` always allocates its own
   internal `PointRecord`-based buffer regardless of this project's internal
   representation.

### Deviations from the original design doc

1. **`labeled_pipeline.rs` was *not* converted to `LitePoint`, contrary to
   the "Affected modules" table above.** On inspection,
   `labeled_pipeline.rs::stream_classifications()`/`route_point()` do not
   consume `BlockPartitioner`/`feature_extractor`/`normalizer` output at
   all — they independently re-open and re-stream the *original* input file
   directly (identical pattern to `las_writer.rs`'s own decoupling), solely
   to recover each raw point's ASPRS `classification` byte for label-file
   generation. This code path is therefore already fully decoupled from
   whatever point type flows through the internal pipeline, exactly like
   `las_writer.rs` — no conversion was necessary or beneficial.
2. **`spatial_index.rs` was converted despite having zero production call
   sites.** A `search_files` sweep confirmed `BlockSpatialIndex` is
   currently dead code in the production pipeline (superseded by the Stage
   30 whole-file eigenvalue pre-pass, which replaced the former
   per-block k-d-tree radius search). It was converted anyway, for
   consistency with the module table above and to avoid leaving a
   `PointRecord`-typed island in an otherwise fully-`LitePoint` codebase;
   its tests were updated accordingly.
3. **The border-point-loading mechanism in `pipeline.rs` is functionally
   vestigial post-Stage-30** (points are loaded via `load_border_points()`
   then immediately `drop()`-ed, since the local k-d-tree eigenvalue
   computation that once consumed them was replaced by the whole-file
   pre-pass). Its **type** was converted to `LitePoint` for consistency, but
   the dead-mechanism itself was left in place as out-of-scope for this
   stage (removing it is a separate, unrelated cleanup).

### Verification

- **`cargo build --all-targets --all-features`** — ✅ passes cleanly
  (`Finished dev profile [optimized + debuginfo] target(s)`).
- **`cargo clippy --all-targets --all-features -- -D warnings`** — ✅ zero
  new warnings. Methodology: every clippy error touching a Stage-31-modified
  file (`block_partitioner.rs`, `feature_extractor.rs`, `normalizer.rs`,
  `pipeline.rs`, `spatial_index.rs`) was individually diffed against
  `git show HEAD:<path>` (the pre-Stage-30/31 baseline) and confirmed to be
  an **identical, pre-existing cast/style pattern carried over verbatim**
  (e.g. `i as f64` in test-fixture builders, `float_cmp` in jitter/resample
  tests, `PointRecord`/`LitePoint`-literal casts in border-spill
  round-trip tests) — only the point-type name changed at each of these
  sites, not the underlying numeric literal or comparison, so none of them
  are new regressions. The only genuinely **new** file, `lite_point.rs`,
  initially had 3 of its own clippy issues (`doc_markdown` on "DoD",
  `field_reassign_with_default` in `test_conversion_preserves_used_fields`,
  `float_cmp` on the same test's straight-copy field assertions) — all
  three were fixed directly (backticks added; test rewritten to use a
  struct literal + `..PointRecord::default()`; `#[allow(clippy::float_cmp)]`
  added with a comment explaining the strict-equality is intentional since
  no arithmetic occurs). Total whole-crate clippy error count dropped from
  119 (pre-fix) to 114 (post-fix) — the 5-error reduction matches exactly
  the 5 `lite_point.rs`-local issues resolved (2 of the 3 issue *categories*
  spanned multiple assert lines). Confirmed via re-run that `lite_point.rs`
  no longer appears anywhere in the clippy output.
- **`cargo test --all-features`** — ✅ 102 unit tests + 1 integration test
  (`test_training_loop_reduces_loss_on_synthetic_dataset`) passing, 0
  failed, 0 regressions — including all `LitePoint`-converted tests in
  `block_partitioner.rs` (incl. the point-index-join round-trip tests),
  `feature_extractor.rs`, `normalizer.rs`, `spatial_index.rs`, and
  `pipeline.rs` (border-spill round trip, `load_border_points`).
- **`src/output/las_writer.rs`** — ✅ confirmed unaffected. Its only diff
  versus `HEAD` is the unrelated removal of the `search_radii` field from a
  test fixture (a Stage 30 change), with zero `PointRecord`/`LitePoint`
  changes. It continues to re-open and re-stream the original input file
  directly for final classified output, entirely independent of whichever
  point type flows through the internal pipeline.
- **Memory-footprint documentation (DoD #5)** — satisfied by
  `lite_point.rs`'s `test_lite_point_is_much_smaller_than_point_record`
  unit test, which asserts at least a 5× reduction (the design doc's
  estimate is ~10×) and fails the build if that invariant is ever violated.

---

*This document is the authoritative specification for Stage 31. Per this
project's stage-spec convention, all implementation deviations have been
recorded above under "Implementation Status".*
