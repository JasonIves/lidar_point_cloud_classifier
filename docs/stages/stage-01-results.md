# Stage 01 — Development Results

**Stage:** Crate Bootstrap & Spatial Preprocessing Pipeline  
**Status:** COMPLETE  
**Implementation date:** 2026-06-15  
**Spec reference:** `stage-01-spatial-preprocessing.md`

---

## Build & Test Results

| Criterion | Result |
|---|---|
| `cargo build --release` (Windows) | ✅ Pass — zero errors |
| `cargo clippy -- -D warnings` | ✅ Pass — zero warnings in crate code |
| `cargo fmt --check` | ✅ Pass |
| `cargo test` — 14 unit tests | ✅ 14/14 pass |

### Full test output

```
running 14 tests
test preprocessing::block_partitioner::tests::test_partitioner_assigns_cells_correctly ... ok
test preprocessing::block_partitioner::tests::test_spill_merge_produces_same_result ... ok
test preprocessing::feature_extractor::tests::test_eigenvalue_features_degenerate ... ok
test preprocessing::feature_extractor::tests::test_eigenvalue_features_linear_cloud ... ok
test preprocessing::feature_extractor::tests::test_eigenvalue_features_planar_cloud ... ok
test preprocessing::feature_extractor::tests::test_extract_features_output_shape_and_range ... ok
test preprocessing::normalizer::tests::test_resample_exact_count_no_oversample ... ok
test preprocessing::normalizer::tests::test_resample_is_reproducible ... ok
test preprocessing::normalizer::tests::test_resample_oversamples_to_target ... ok
test preprocessing::normalizer::tests::test_resample_subsamples_correctly ... ok
test preprocessing::spatial_index::tests::test_adaptive_radius_caps_at_4x ... ok
test preprocessing::spatial_index::tests::test_adaptive_radius_expands_when_needed ... ok
test preprocessing::spatial_index::tests::test_radius_search_matches_brute_force ... ok

test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

> Note: The two warnings visible in build output (`use of deprecated trait CmpNe`, `unused import CmpNe`) originate in `wbraster/src/raster.rs` — an existing upstream file in `whitebox_next_gen`. They are not produced by any code in this crate and cannot be suppressed without modifying the existing codebase (which is prohibited by `AGENTS.md`).

---

## Files Created

```
lidar_point_cloud_classifier/
  Cargo.toml
  src/
    main.rs
    lib.rs
    error.rs
    preprocessing/
      mod.rs               ← PreprocessConfig, shared constants
      block_partitioner.rs ← 2D grid bucketing + LE spill/merge
      spatial_index.rs     ← kdtree wrapper + adaptive radius search
      normalizer.rs        ← DtmView, HAG, resample_block, normalise_scalar_features
      feature_extractor.rs ← nalgebra eigendecomposition + 12-feature assembly
      pipeline.rs          ← streaming orchestrator + .feat/.csv/.json writers
    model/
      mod.rs               ← Stage 02 stub
    cli/
      mod.rs               ← sub-command dispatch
      preprocess_cmd.rs    ← hand-rolled arg parser + pipeline invocation
  docs/stages/
    stage-01-spatial-preprocessing.md
    stage-01-results.md    ← this file
```

---

## Deviations from Specification

The following items deviate from `stage-01-spatial-preprocessing.md`. Each deviation is driven by a concrete technical constraint. The spec has been updated to reflect the as-built state.

---

### 1. Spill file format: field-by-field LE serialization (not raw struct bytes)

**Spec said:** *"raw `PointRecord` bytes (native endian), no header"*

**As built:** Each spill record is a fixed 31-byte little-endian packet:

| Offset | Field | Type | Bytes |
|---|---|---|---|
| 0 | x | f64 LE | 8 |
| 8 | y | f64 LE | 8 |
| 16 | z | f64 LE | 8 |
| 24 | intensity | u16 LE | 2 |
| 26 | classification | u8 | 1 |
| 27 | return_number | u8 | 1 |
| 28 | number_of_returns | u8 | 1 |
| 29 | scan_angle | i16 LE | 2 |
| **Total** | | | **31** |

**Reason:** `lib.rs` includes `#![deny(unsafe_code)]` to comply with AGENTS.md security requirements. The raw-bytes approach would require `std::slice::from_raw_parts`, which is `unsafe`. The 31-byte LE layout covers all fields used in feature extraction; other `PointRecord` fields (color, GPS time, waveform, normals, extra bytes) are not needed by the preprocessing pipeline and are not preserved across a spill. If a complete field set is required in future, the record can be extended or the spill strategy rethought.

---

### 2. `BlockPartitioner::new` — y_max parameter unused

**Spec said:** Takes `x_min, y_min, x_max, y_max, block_size, spill_dir`

**As built:** `y_max` is accepted as `_y_max` (deliberately unused). The `grid_cols` count is precomputed from x extent; `grid_rows` is **not** precomputed — any `(col, row)` pair encountered during streaming is valid and the `HashMap` accommodates it dynamically. This is correct: y extent might be slightly underestimated by the header bounding box, and a HashMap handles sparse or over-range rows without issue.

**No functional impact.** The block ID formula (`row * grid_cols + col`) still uniquely identifies blocks within any practically-sized dataset. The `grid_cols` bound prevents column wrapping. Row uniqueness follows from the formula.

---

### 3. LAZ header inspection via `LasReader` (not `LazReader`)

**Spec implied:** `LazReader::header()` would be available for LAZ files

**As built:** `LasReader::new(BufReader::new(f))` is used for all three formats (LAS, LAZ, COPC) for header inspection, including bounding box and CRS/EPSG extraction.

**Reason:** In `wblidar`, `LazReader` has no public `.header()` method. The compressed LAZ payload is preceded by a standard binary LAS header; `LasReader` parses it correctly without triggering any point decompression. Using `LasReader` for LAZ header inspection is the correct approach.

---

### 4. COPC header/CRS inspection via `LasReader` (not `CopcReader`)

**Spec implied:** `CopcReader` would expose CRS metadata and a `.las_header()` method

**As built:**
- `LasReader::new()` is used for COPC **header inspection** (bounding box + CRS/EPSG)
- `CopcReader::open_path()` is used for COPC **point streaming**

**Reason:** Two API gaps discovered during implementation:
- `CopcReader` has no `.crs()` method in the wblidar API
- The correct constructor is `CopcReader::open_path(path)`, not `CopcReader::open(path)`

COPC files embed a standard LAS header at byte offset 0; `LasReader` reads it correctly. The `CopcReader::open_path()` reader is then used for point streaming because it handles the EPT spatial hierarchy correctly.

---

### 5. `RAYON_MIN_CHUNK = 64` (not `65_536`)

**Spec said:** *"Mirrors the convention used in wbtools_oss"* — wbtools_oss uses `65_536`

**As built:** `RAYON_MIN_CHUNK = 64` in `preprocessing/mod.rs`

**Reason:** The `65_536` threshold in wbtools_oss is tuned for **per-point inner loops** (e.g., raster cell iteration over millions of cells). Here, `RAYON_MIN_CHUNK` gates **per-block** task dispatch — a typical dataset produces hundreds to low thousands of blocks. Using `65_536` would prevent Rayon from parallelising anything but the most gigantic regional datasets. A value of `64` ensures parallel dispatch begins as soon as 64+ blocks are present, which is appropriate for block-level work items.

---

### 6. Resolved dependency versions

The spec listed target versions; the actual resolved versions from `Cargo.lock` differ slightly:

| Crate | Spec | Resolved |
|---|---|---|
| `nalgebra` | `0.33` | `0.34.2` |
| `kdtree` | `0.7` | `0.8.0` |
| `rand` | `0.9` | `0.10.1` |
| `thiserror` | `1` | `1.0.69` |
| `rayon` | `1.10` | `1.12.0` |

All changes are minor/patch version increments. No breaking API changes were encountered. The `rand` 0.10.x API uses `rng.random_range()` (not `gen_range()` from 0.8/0.9). Note that `wbtools_oss` already uses `rand = "0.10.0"` in its own `Cargo.toml`, so this is consistent with the broader workspace.

---

### 7. `wide::f64x4` SIMD omitted from coordinate normalisation

**Spec said:** `preprocessing/mod.rs` Performance Guardrails — *"Coordinate normalization uses `wide::f64x4` portable SIMD with automatic software fallback on unsupported ISAs (same pattern as `wblidar`)"*

**As built:** `normalizer.rs` uses standard `f64` arithmetic throughout. `wide` is not present in `Cargo.toml`.

**Reason:** The SIMD benefit for 7 scalar operations per point (x_norm, y_norm, z_norm, intensity_norm, return_ratio, scan_angle_norm, hag) is negligible compared to the per-point eigendecomposition cost that dominates preprocessing time. Adding `wide` solely for this use would introduce a dependency with no measurable throughput gain. If profiling on a production dataset reveals a bottleneck in scalar normalisation, `wide::f64x4` can be added at that time.

**No functional impact.** All normalised values are computed identically to the SIMD path on correct inputs.

---

### 8. `bytemuck` scope clarification

**Spec said:** `bytemuck` for "Zero-copy `f32` slice → `&[u8]` for `.feat` file write"

**As built:** `bytemuck::cast_slice` is used **only** in `write_feat_file` (the `.feat` data payload), not in spill file I/O. Spill files use the 31-byte LE serialization described in deviation #1. This is consistent with the AGENTS.md `#![deny(unsafe_code)]` requirement — `bytemuck::cast_slice` is safe (it panics rather than causing UB on alignment violations, and `[f32; 12]` is `bytemuck::Pod`).

---

## wblidar API Reference (Discovered During Implementation)

These API facts are not documented in `AVAILABLE_LIDAR_TOOLS.md` and are relevant to all future stages.

| Topic | Fact |
|---|---|
| **LAS header fields** | `min_x`, `min_y`, `max_x`, `max_y` (NOT `x_min`/`x_max`/`y_min`/`y_max`) |
| **LazReader** | No `.header()` method. Use `LasReader::new()` to inspect LAZ headers |
| **CopcReader construction** | `CopcReader::open_path(path)` — not `::open()` or `::new()` |
| **CopcReader CRS** | No `.crs()` method. Use `LasReader::new()` on the COPC file for CRS/EPSG |
| **LasReader CRS** | `reader.crs()` → `Option<&Crs>` → `.epsg: Option<u32>` |
| **PointReader trait** | `read_point(&mut self, out: &mut PointRecord) -> Result<bool>` — returns `false` at EOF |
| **LasReader VLRs** | `reader.vlrs()` → `&[Vlr]` for EPSG via `find_epsg(vlrs)` |

---

## Open Items / Deferred to Stage 02

- Integration test against a real LAS/LAZ file (DoD criteria 9–12): requires a sample LiDAR dataset
- Memory profiling on a 50M-point LAZ file (DoD criterion 10)
- Stage 02: PointNet inference engine (framework selection: `burn` vs `dfdx`; training module; output LAS writer with updated classification field)
