# Stage 01 — Crate Bootstrap & Spatial Preprocessing Pipeline

**Status:** COMPLETE — See [stage-01-results.md](stage-01-results.md) for full development record and deviations  
**Approved:** 2026-06-13  
**Implemented:** 2026-06-15  
**Retroactive extension:** 2026-06-16 — Stage 03 added `sampled_indices` to `resample_block` return and `BlockProcessResult` / `run_with_indices` to `pipeline.rs`; see Stage 03 results for rationale  
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier

---

## Goal

Establish the `lidar_point_cloud_classifier` crate from scratch: workspace layout, module skeleton, and a fully functional streaming spatial preprocessing pipeline that transforms raw LiDAR point clouds (`.las` / `.laz` / `.copc`) into fixed-size, normalized per-point feature tensors partitioned into 2D spatial blocks. This stage produces no ML model — only the ground-truth-quality input data representation that the PointNet inference engine (Stage 02) will consume.

---

## Crate & Workspace Layout

```
lidar_point_cloud_classifier/
  Cargo.toml
  src/
    lib.rs
    error.rs
    preprocessing/
      mod.rs
      block_partitioner.rs      ← 2D grid blocking + raw-point spill/merge logic
      spatial_index.rs          ← per-block k-d tree, adaptive radius search
      feature_extractor.rs      ← 12-feature per-point extraction
      normalizer.rs             ← coordinate/intensity normalization, density-gated sampling
      pipeline.rs               ← streaming orchestrator
    model/
      mod.rs                    ← stub only; wired in Stage 02
    cli/
      mod.rs
      preprocess_cmd.rs
  docs/
    stages/
      stage-01-spatial-preprocessing.md   ← this file
```

---

## Dependencies (`Cargo.toml`)

| Crate | Source | Justification |
|---|---|---|
| `wblidar` | `path = "../whitebox_next_gen/crates/wblidar"` | LAS/LAZ/COPC streaming I/O |
| `wbcore` | `path = "../whitebox_next_gen/crates/wbcore"` | `Tool` trait, error types |
| `wbraster` | `path = "../whitebox_next_gen/crates/wbraster"` | DTM raster lookup for HAG |
| `rayon` | `1` | Data-parallel block processing |
| `kdtree` | `0.8.0` | 3D k-NN queries |
| `nalgebra` | `0.34.2` | 3×3 covariance matrix + symmetric eigendecomposition |
| `thiserror` | `1` | Ergonomic error enum |
| `serde` + `serde_json` | `1` (workspace) | `blocks.json` metadata serialization |
| `bytemuck` | `1` | Zero-copy `f32` slice → `&[u8]` for `.feat` data payload |
| `rand` | `0.10.x` | Seeded point sampling (reproducible oversample/subsample) |

> **No ML frameworks in this stage.** `burn` / `dfdx` are deferred to Stage 02.

---

## CLI Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--input` | `PathBuf` | *required* | LAS, LAZ, or COPC source file |
| `--output` | `PathBuf` | *required* | Output directory for `.feat` blocks |
| `--block-size` | `f64` | `50.0` | 2D cell edge length in projection units (meters) |
| `--target-points` | `usize` | `1024` | Fixed $N$ points per block after density-gated sampling |
| `--min-density` | `f64` | `1.0` | Minimum pts/m² — blocks below this threshold are discarded |
| `--search-radius` | `f64` | `1.0` | Base radius (in projection units) for eigenvalue neighborhood queries |
| `--min-neighbors` | `usize` | `8` | Minimum neighbors required; radius expands adaptively up to `search-radius × 4` if this count is not met within `search-radius` |
| `--hag-model` | `Option<PathBuf>` | `None` | Optional DTM raster for Height Above Ground; falls back to block-minimum-z proxy if absent |
| `--threads` | `Option<usize>` | system cores | Rayon thread pool size override |
| `--debug-csv` | flag | `false` | Also emit per-block `.csv` files alongside `.feat` files (development inspection only) |

---

## Outputs

### Per-Block Binary Feature File (`.feat`)

Each retained spatial block is serialized to a self-describing binary file:

```
[header — 37 bytes]
  magic:      4 bytes  = b"WBFT"
  version:    u8       = 1
  n_points:   u32      (= target_points N after sampling)
  n_features: u32      (= 12)
  block_id:   u64      (row * grid_cols + col)
  origin_x:   f64
  origin_y:   f64

[data]
  f32[n_points × n_features]  — row-major, point-major order
```

### Block Metadata File (`blocks.json`)

```json
{
  "source": "input.las",
  "block_size": 50.0,
  "target_points": 1024,
  "min_density": 1.0,
  "search_radius": 1.0,
  "min_neighbors": 8,
  "crs_epsg": 32617,
  "blocks": [
    {
      "id": 42,
      "file": "block_00042.feat",
      "origin_x": 450000.0,
      "origin_y": 4850000.0,
      "raw_point_count": 1587,
      "sampled_point_count": 1024,
      "oversampled": false
    }
  ]
}
```

The `"oversampled": true` flag is set on any block whose `raw_point_count` was below `target_points` and was padded via sampling with replacement.

### Debug CSV (optional, `--debug-csv`)

Per-block `block_XXXXX.csv` with one header row and one data row per point:

```
x_norm,y_norm,z_norm,intensity_norm,return_ratio,scan_angle_norm,hag,linearity,planarity,sphericity,omnivariance,curvature
```

This output is for development and validation use only. It is not consumed by the inference pipeline.

---

## Feature Vector — 12 Features Per Point

Eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$ are derived from the 3×3 covariance matrix of the neighbor set found by the adaptive radius search. If fewer than 3 valid neighbors remain after the maximum radius expansion, eigenvalue-derived features (indices 7–11) default to `0.0` rather than panicking.

| Idx | Name | Formula / Source | Range |
|---|---|---|---|
| 0 | `x_norm` | $(x - x_{block\_min})\ /\ \text{block\_size}$ | $[0, 1]$ |
| 1 | `y_norm` | $(y - y_{block\_min})\ /\ \text{block\_size}$ | $[0, 1]$ |
| 2 | `z_norm` | $(z - z_{block\_min})\ /\ (z_{block\_max} - z_{block\_min})$ | $[0, 1]$ |
| 3 | `intensity_norm` | $\text{intensity}\ /\ 65535.0$ | $[0, 1]$ |
| 4 | `return_ratio` | $\text{return\_num}\ /\ \text{num\_returns}$ | $[0, 1]$ |
| 5 | `scan_angle_norm` | $|\text{scan\_angle}|\ /\ 90.0$ | $[0, 1]$ |
| 6 | `hag` | $\text{clamp}((z - z_{ground})\ /\ h_{max},\ 0.0,\ 1.0)$ | $[0, 1]$ |
| 7 | `linearity` | $(\lambda_1 - \lambda_2)\ /\ \lambda_1$ | $[0, 1]$ |
| 8 | `planarity` | $(\lambda_2 - \lambda_3)\ /\ \lambda_1$ | $[0, 1]$ |
| 9 | `sphericity` | $\lambda_3\ /\ \lambda_1$ | $[0, 1]$ |
| 10 | `omnivariance` | $(\lambda_1 \cdot \lambda_2 \cdot \lambda_3)^{1/3}$ | $[0, +\infty)$ |
| 11 | `curvature` | $\lambda_3\ /\ (\lambda_1 + \lambda_2 + \lambda_3)$ | $[0, 1]$ |

**HAG computation detail (feature 6):**
- If `--hag-model` is provided: bilinear-interpolate the DTM raster at each point's (x, y) to obtain `z_ground`.
- Otherwise: `z_ground = z_block_min` (proxy — no external model required).
- `hag_raw = z - z_ground`
- `h_max` = 99th percentile of `hag_raw` values across the sampled block (computed before normalization).
- `hag = clamp(hag_raw / h_max, 0.0, 1.0)`

---

## Algorithmic Steps

### Step 1 — Streaming Ingest

Open the input file via `wblidar` format-detected reader.

- **LAS/LAZ:** `LasReader::new(BufReader::new(f))` is used for **header inspection** (bounding box + CRS/EPSG) for both formats. `LasReader` is also used for LAS point streaming. `LazReader::new()` is used for LAZ point streaming. Note: `LazReader` has no `.header()` method; `LasReader` correctly parses the LAS header embedded in a LAZ file without triggering point decompression.
- **COPC:** `LasReader::new()` is used for **header inspection** (COPC embeds a standard LAS header; `CopcReader` has no `.crs()` method). `CopcReader::open_path(path)` is used for **point streaming** to preserve the EPT spatial hierarchy.

Stream `PointRecord` chunks via the `PointReader::read_point` trait method. No full-cloud memory allocation. Each point is dispatched immediately to the block partitioner.

### Step 2 — 2D Grid Block Partitioner

Determine cloud XY extent from the file header bounding box. Compute `grid_rows × grid_cols` from `ceil((x_max - x_min) / block_size)` and `ceil((y_max - y_min) / block_size)`. Assign each incoming `PointRecord` to its cell via:

```
col = floor((x - x_min) / block_size) as i32
row = floor((y - y_min) / block_size) as i32
```

Accumulate into `HashMap<(i32, i32), Vec<PointRecord>>`.

**Chunk-spanning guarantee:** Streaming chunks are arbitrary *file* segments with no spatial locality guarantee. A single logical block may receive points from many non-contiguous streaming chunks. The `HashMap` accumulator remains open for the full duration of the stream; no block is finalized until `finalize()` is called after EOF. This is the correct approach — chunk boundaries are irrelevant to block assignment.

**Memory-pressure spill path:** When the total in-flight point buffer exceeds the high-water mark (default: 512 MB), the largest occupied cells are flushed to temporary spill files (`block_XXXXX.spill`) in the output directory. Spill files store a fixed **31-byte little-endian record per point** — not raw struct bytes (see deviation #1 in [stage-01-results.md](stage-01-results.md)). After the full stream is exhausted and `finalize()` is called, each block's spill files are merged with any remaining in-memory data for that block before feature extraction begins. This ensures every block's complete point set is available for eigenvalue computation and sampling, regardless of how many streaming chunks contributed to it. Spill files are deleted after successful merge.

### Step 3 — Block Filtering

Compute per-block point density:

$$\rho = \frac{\text{raw\_count}}{\text{block\_size}^2}$$

Discard any block where $\rho < \text{min\_density}$. These blocks are excluded from `blocks.json` and produce no output files.

### Step 4 — Per-Block Parallel Processing (`rayon::par_iter`)

For each retained block, the following sub-steps execute in a Rayon parallel iterator. Each task owns its mutable state exclusively — no `Mutex` or `RwLock` in the hot path.

**(a) Build 3D k-d tree**

Construct from the full raw block point set (post-merge) using the `kdtree` crate. Tree construction happens before sampling so that eigenvalue neighborhoods use the original, unsampled spatial distribution.

**(b) Density-gated point sampling**

| Condition | Strategy |
|---|---|
| `raw_count >= target_points` | Random sample without replacement, seeded by `block_id` |
| `raw_count < target_points` (density passed) | Random oversample with replacement to pad to `target_points`, seeded by `block_id`; `"oversampled": true` in metadata |

Seeding by `block_id` ensures reproducible output across runs with the same input.

**(c) HAG computation**

See feature 6 formula above. The 99th-percentile `h_max` is computed over the sampled block's raw `hag_raw` values. If `h_max == 0.0` (e.g., flat block at ground level), `hag` defaults to `0.0` for all points.

**(d) Coordinate normalization**

Compute `z_block_min` and `z_block_max` from the sampled set. If `z_block_max == z_block_min`, `z_norm` defaults to `0.0`. Apply feature formulas for indices 0–6 using standard `f64` arithmetic (see deviation #7 in [stage-01-results.md](stage-01-results.md) — `wide::f64x4` SIMD was omitted).

**(e) Adaptive radius eigenvalue features**

For each sampled point:
1. Query k-d tree within `search_radius`. Count neighbors.
2. If `neighbor_count < min_neighbors`, expand radius by `search_radius × 0.5` increments.
3. Continue expanding until `neighbor_count >= min_neighbors` or `radius > search_radius × 4.0` (hard cap).
4. Use the neighbor set at the first radius satisfying the count requirement, or the cap-radius set.
5. If `neighbor_count < 3` at the cap: features 7–11 = `0.0` (degenerate neighborhood).
6. Otherwise: construct the 3×3 covariance matrix, decompose via `nalgebra::linalg::SymmetricEigen`, sort eigenvalues descending, compute features 7–11.

**(f) Assemble feature matrix**

Collect into a `Vec<f32>` of length `target_points × 12`, row-major (each row = one point's 12 features).

**(g) Serialize outputs**

Write `.feat` binary file. If `--debug-csv`, write `.csv` alongside it.

### Step 5 — Metadata Flush

Serialize `blocks.json` to the output directory. Delete all `.spill` temp files. Log final summary (blocks processed, blocks discarded, total points written).

---

## Module Responsibilities

| Module | Responsibility | Key Public API |
|---|---|---|
| `block_partitioner` | 2D grid bucketing, high-water spill, post-stream merge | `BlockPartitioner::new(cfg)`, `::add_point(pt)`, `::finalize() -> Vec<Block>` |
| `spatial_index` | k-d tree construction + adaptive radius search | `BlockSpatialIndex::build(pts)`, `::adaptive_radius_search(pt, base_r, min_n, max_r) -> Vec<[f64;3]>` |
| `feature_extractor` | Covariance matrix, eigendecomposition, HAG, feature assembly | `extract_features(pts, index, hag_fn, cfg) -> Vec<[f32;12]>` |
| `normalizer` | Coordinate/intensity normalization, density-gated sampling | `resample_block(pts, target_n, seed) -> (Vec<PointRecord>, Vec<usize>, bool)`, `normalize_coords(pts, cfg) -> Vec<[f32;6]>` |
| `pipeline` | Streaming orchestrator tying all modules together | `PreprocessingPipeline::run(config) -> Result<BlockManifest>`, `::run_with_indices(config) -> Result<(BlockManifest, Vec<BlockProcessResult>)>` |
| `cli/preprocess_cmd` | CLI argument parsing + `Pipeline::run()` invocation | `run_preprocess_cmd(matches) -> Result<()>` |

---

## Performance Guardrails

- **Memory ceiling:** Spill path keeps peak RSS ≤ 1 GB for a 200 M-point regional dataset.
- **Parallel isolation:** No `Mutex`/`RwLock` in per-block Rayon tasks. All shared data is read-only (config, DTM raster wrapped in `Arc`).
- **No panics in production:** All fallible paths return `Result<_, ClassifierError>`. Degenerate eigenvalue and HAG edge cases return `0.0` features, never `unwrap()` or `expect()`.
- **RAYON_MIN_CHUNK = 64:** Gates per-block task dispatch. (Note: wbtools_oss uses 65,536 for per-point inner loops over millions of raster cells; 64 is correct here for block-level work items where a typical run produces hundreds to low thousands of blocks.)
- **SIMD:** `wide::f64x4` SIMD for coordinate normalisation was omitted (see deviation #7 in [stage-01-results.md](stage-01-results.md)). Standard `f64` arithmetic is used. The `wide` crate is not a dependency of this crate.
- **Logging:** Progress reported per-block via coalescing `eprintln!` progress lines (no per-point stdout flood in high-throughput loops).

---

## Definition of Done

| # | Criterion | Verification Method | Status |
|---|---|---|---|
| 1 | `cargo build --release` succeeds | CI matrix: Windows, macOS, Linux | ✅ Pass (Windows 2026-06-15) |
| 2 | `cargo clippy -- -D warnings` passes with zero warnings | CI | ✅ Pass |
| 3 | `cargo fmt --check` passes | CI | ✅ Pass |
| 4 | Unit: block partitioner correctly assigns 10k random (x,y) points into expected `(row, col)` grid cells | `cargo test` | ✅ Pass |
| 5 | Unit: spill/merge path produces identical block point sets as the in-memory path on a 1M-point synthetic dataset | `cargo test` | ✅ Pass (spill round-trip verified) |
| 6 | Unit: adaptive radius expands correctly — base radius yields < `min_neighbors`, verify final neighbor set matches brute-force reference | `cargo test` | ✅ Pass |
| 7 | Unit: all 12 feature values verified against hand-calculated ground truth for a 5-point synthetic cloud | `cargo test` | ✅ Pass (planar + linear + degenerate cases) |
| 8 | Unit: oversampling produces exactly `target_points` with seeded reproducibility across two identical calls | `cargo test` | ✅ Pass |
| 9 | Integration: 100k-point synthetic LAS → full pipeline → `blocks.json` block count matches expected grid cell count, all `.feat` headers parse correctly, `"oversampled"` flags are correct | `cargo test --test integration` | ⏳ Deferred — requires sample LAS dataset |
| 10 | Memory: streaming a 50M-point LAZ file keeps peak RSS ≤ 1 GB | Manual validation on reference dataset | ⏳ Deferred — requires reference dataset |
| 11 | CLI: `--debug-csv` flag emits one `.csv` per `.feat` with correct column count and row count | Manual smoke test | ⏳ Deferred — requires sample LAS dataset |
| 12 | CLI: `preprocess --input test.las --output ./out --block-size 50 --target-points 1024` exits code 0 on reference dataset | Manual smoke test | ⏳ Deferred — requires sample LAS dataset |

---

## Open Items / Deferred Decisions

- **Stage 02:** PointNet-style inference engine (model architecture, `burn`/`dfdx` framework selection, training module).
- **Label support:** This stage has no awareness of ground-truth classification labels. A label-injection mechanism (for training workflows) is a Stage 02 or Stage 03 concern.
- **COPC spatial queries:** COPC's EPT hierarchy allows fetching only the spatial tiles needed. Leveraging COPC-native spatial filtering (instead of streaming all nodes) is a future optimization once the preprocessing pipeline is validated.
