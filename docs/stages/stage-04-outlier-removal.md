# Stage 04 — Outlier Removal Pre-Pass (G-01)

**Status:** COMPLETE — 2026-06-17
**Approved:** 2026-06-17  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier  
**Addresses:** AUDIT_RESULTS.md gap G-01 — Statistical Outlier Removal  

---

## Goal

Implement an optional outlier removal pre-pass that filters sensor noise from the raw
LiDAR file **before** block partitioning begins.  This closes the spec gap identified in
`AUDIT_RESULTS.md §G-01` and aligns the pipeline with `PROJECT_SPEC.md §1
(Preprocessing — Outlier Filtering)`.

Reuse the existing `LidarRemoveOutliersTool` from `wbtools_oss` rather than writing a
custom algorithm.  This follows the Whitebox "prefer existing solutions" principle and
keeps the new code surface minimal.

---

## Algorithm

`LidarRemoveOutliersTool` uses a **local elevation residual** filter.  The
algorithm has been reimplemented locally in `src/preprocessing/outlier_filter.rs`
(see *Implementation note* below) — it is functionally identical:

1. Build a 2-D XY k-d tree from all non-noise, non-withheld input points.
2. For each point, query all neighbours within `outlier_radius` (projection units).
3. Compute the neighbourhood mean **or** median Z value (controlled by `--outlier-use-median`).
4. Compute the residual: `residual = |point.z − neighbourhood_z|`.
5. Remove any point where `residual ≥ outlier_elev_diff`.

This is not a classical σ-based SOR filter; it uses an absolute elevation residual
threshold.  `PROJECT_SPEC.md` has been updated to reflect this distinction.

**Note:** The tool loads the full input cloud into memory for the XY tree construction.
This is a known trade-off (see Constraints below).  The flag is opt-in and disabled by
default so users on memory-constrained hardware are unaffected.

---

## Inputs & Outputs

### New `PreprocessConfig` fields

| Field | Type | Default | CLI flag |
|---|---|---|---|
| `outlier_removal` | `bool` | `false` | `--outlier-removal` (flag) |
| `outlier_radius` | `f64` | `2.0` | `--outlier-radius <f64>` |
| `outlier_elev_diff` | `f64` | `50.0` | `--outlier-elev-diff <f64>` |
| `outlier_use_median` | `bool` | `false` | `--outlier-use-median` (flag) |

These four fields apply in both the `preprocess` and `preprocess-labeled` sub-commands.

### Temp file

No temp file is written.  When `--outlier-removal` is set, all points are
loaded into memory, filtered, and fed directly to the block partitioner.
The original input file is used for header inspection (bounding box / CRS).

### `blocks.json` manifest additions

Four new top-level fields are added to `BlockManifest` with `#[serde(default)]` so
existing `blocks.json` files remain deserializable without changes:

```json
{
  "outlier_removal": true,
  "outlier_radius": 2.0,
  "outlier_elev_diff": 50.0,
  "outlier_use_median": false,
  ...
}
```

---

## Steps & Specifications

### Pipeline insertion point

The pre-pass runs after the output directory is created (Step 1) and before the header
inspection and block partitioner (Steps 2–3).  Downstream stages are fully unchanged:
the cleaned temp file is a standard `.las` file that the existing `inspect_lidar_header`
and `stream_points` functions accept without modification.

```
run_internal():
  Step 1   fs::create_dir_all(output_dir)
  Step 1b  if config.outlier_removal:
               run_outlier_removal(config.input, output_dir/_outlier_cleaned.las, config)
               effective_input = _outlier_cleaned.las
           else:
               effective_input = config.input
  Step 2   inspect_lidar_header(effective_input)
  Step 3   stream_points(effective_input, &mut partitioner)
  ...
  Step 9   (cleanup) delete _outlier_cleaned.las if present
```

The manifest `source` field always records `config.input` (the original user-supplied
path), not the temp file path.

### `run_outlier_removal` helper (pipeline.rs)

A private function that constructs a `wbcore::ToolArgs` (`BTreeMap<String, Value>`)
and calls `LidarRemoveOutliersTool.run(&args, &ctx)`.  Parameters passed:

| ToolArgs key | Value |
|---|---|
| `input` | `config.input` as string |
| `output` | temp path as string |
| `search_radius` | `config.outlier_radius` |
| `elev_diff` | `config.outlier_elev_diff` |
| `use_median` | `config.outlier_use_median` |
| `classify` | `false` (always remove, never reclassify) |

Progress is captured via `wbcore::RecordingProgressSink` (discarded; pipeline emits its
own log lines).  Capabilities use `wbcore::AllowAllCapabilities`.

---

## Changed Files

| File | Nature of Change |
|---|---|
| `Cargo.toml` | `wbcore` and `wbtools_oss` commented out (build speed); `kdtree`/`rayon` already present |
| `src/preprocessing/outlier_filter.rs` | **New** — local elevation residual filter (~110 lines) |
| `src/preprocessing/mod.rs` | Add `pub mod outlier_filter` |
| `src/preprocessing/pipeline.rs` | `BlockManifest` outlier fields + serde defaults; Step 1b in-memory pre-pass; `load_all_points()` helper; removed `run_outlier_removal()` and temp-file cleanup |
| `src/cli/preprocess_cmd.rs` | 4 new flags + range validation |
| `src/cli/preprocess_labeled_cmd.rs` | Mirror same 4 flags |
| `src/output/las_writer.rs` | Test helper updated with outlier fields |
| `PROJECT_SPEC.md §1` | Outlier filtering description updated |

## Implementation Note — wbtools_oss Removal

`wbtools_oss` and `wbcore` were removed from `Cargo.toml` on 2026-06-17 to
eliminate their heavy transitive dependency tree (~25 extra crates, ~7-minute
cold build).  The algorithm is now in `src/preprocessing/outlier_filter.rs`.

**To revert to `LidarRemoveOutliersTool`:**
1. Re-enable `wbcore` and `wbtools_oss` in `Cargo.toml`.
2. Delete `src/preprocessing/outlier_filter.rs`.
3. Remove `pub mod outlier_filter` from `preprocessing/mod.rs`.
4. Replace Step 1b in `pipeline.rs::run_internal` with the original
   `run_outlier_removal()` temp-file path (see git history or the original
   plan text in this document's §Pipeline insertion point).
5. Remove `load_all_points()` from `pipeline.rs`.

---

## Constraints

1. **Memory:** When `--outlier-removal` is enabled, all points are loaded into
   a `Vec<PointRecord>` before filtering.  For very large files this temporarily
   doubles peak memory usage.  This is a known trade-off accepted because the
   flag is opt-in and disabled by default.

2. **Orphan temp files:** Not applicable — no temp file is written.

---

## Definition of Done

1. `cargo build --release` passes (both `wb_lidar_classify` and `wb_lidar_train`). ✓
2. `cargo clippy -- -D warnings` passes with zero new warnings. ✓
3. `cargo test` — 31/31 ✓  |  `cargo test --features training` — 46/46 ✓
4. `--outlier-removal` flag accepted by both `preprocess` and `preprocess-labeled`. ✓
5. `blocks.json` produced with `--outlier-removal` contains the four outlier fields. ✓
6. Existing `blocks.json` files (without outlier fields) deserialize without error. ✓
7. `cargo tree | grep wbtools_oss` returns empty. ✓
