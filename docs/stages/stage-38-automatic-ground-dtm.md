# Stage 38 — Automatic Ground DTM Generation (Height-Above-Ground)

**Status:** COMPLETE — implemented, tested, clippy/fmt-clean.

**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Lead Architect:** AI Collaborator (Cline)
**Relates to:** `PROJECT_SPEC.md §1` (Preprocessing — Height Above Ground),
`docs/stages/stage-37-absolute-hag-normalization.md`,
`src/preprocessing/pipeline.rs`, `src/preprocessing/normalizer.rs`

---

## Goal

Give the classifier a **physically meaningful Height-Above-Ground (HAG)** even
when the user has **not** supplied an external DTM via `--hag-model`.

This is **"Option A"** from the HAG System Review executive summary, and it
composes with Stage 37: Stage 37 fixed *how* raw HAG is normalized into `[0,1]`;
Stage 38 improves *where the raw HAG comes from* in the no-external-DTM case.

### Problem it solves

Today, when `config.hag_model` is `None`, `compute_hag()` falls back to a
**per-block minimum-z proxy** for ground elevation:

```rust
// normalizer.rs — current fallback
let z_min = pts.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
let z_ground = dtm.and_then(|d| d.bilinear_interp(pt.x, pt.y)).unwrap_or(z_min);
```

That proxy has three well-known failure modes:

1. **Sloped terrain** — a single block-minimum z is only correct at the lowest
   corner; every other point in the block reads an inflated HAG proportional to
   the terrain slope across the block.
2. **Block-boundary discontinuities** — adjacent blocks pick different minima,
   so the same physical ground surface produces step artifacts in HAG at block
   seams — noise the model must fight through.
3. **Empty-ground blocks** — a block whose lowest return is a rooftop or dense
   canopy (no bare-earth hit) anchors *everything* to that elevated surface,
   collapsing HAG toward zero for genuinely tall objects.

All three directly degrade the vegetation-tier and building signals Stage 37 is
trying to sharpen. A real interpolated bare-earth DTM removes them.

---

## Decision

When no `--hag-model` is given, **auto-generate a bare-earth DTM from the input
cloud** with a two-stage Whitebox pipeline, then feed that raster into the
existing `DtmView` / `compute_hag` path — no changes to the HAG maths
themselves:

```text
input cloud ─▶ [1] improved_ground_point_filter ─▶ ground-only LAS
            ─▶ [2] lidar_tin_gridding (elevation)  ─▶ bare-earth DTM raster
            ─▶ [existing] DtmView::from_raster ─▶ compute_hag ─▶ HAG feature
```

Both tools are already available in the pinned `wbtools_oss` crate
(`wbtools_oss::tools::{ImprovedGroundPointFilterTool, LidarTinGriddingTool}`) and
are invoked with the exact same `Tool::validate` / `Tool::run` pattern the
pipeline already uses for `LidarRemoveOutliersTool` (see `run_outlier_removal`).

### Reviewer decisions (finalised)

1. **Default ON.** Auto-DTM is the **default** for the no-external-DTM path.
   Disable with `--no-auto-dtm` to fall back to the historical block-min-z proxy.
2. **Streamlined surface.** Only `--dtm-resolution` is exposed. The ground
   filter's other knobs (`max_building_size`, `slope_threshold`, `elev_threshold`)
   are hardwired to the Whitebox tool defaults. Users needing finer control can
   generate their own DTM externally and pass it via `--hag-model`.
3. **Intermediates deleted by default.** Auto-generated `_auto_ground.las` and
   `_auto_dtm.tif` are removed after the run unless `--keep-auto-dtm` is given.

`--hag-model` takes **priority** over auto-DTM: if the user supplies an external
DTM, it is used and auto-DTM is skipped (no error; auto-DTM is the *fallback*,
not a competitor).

### Why the *advanced* ground filter (`improved_ground_point_filter`)

There are two ground-classification tools in the toolset. This stage selects the
**advanced** one. Comparison:

| Aspect | `lidar_ground_point_filter` (basic) | `improved_ground_point_filter` (advanced — **chosen**) |
| :-- | :-- | :-- |
| Algorithm | Single-pass local slope/height threshold vs. neighbours | Multi-stage pipeline: percentile filter → TIN grid → fill pits → **remove off-terrain objects** → reference-surface filter |
| Buildings / large flat roofs | Frequently retained as false "ground" (flat roofs beat the slope test) | Explicitly removed by the off-terrain-object stage (bounded by `max_building_size`) |
| Steep / broken terrain | Prone to shaving real ground or keeping low vegetation | Reference-surface stage re-includes true ground within `elev_threshold` |
| Cost | Lower | Higher (several internal passes) — acceptable: runs once per file, off the hot path |
| Accuracy of resulting DTM | Coarser, more off-terrain contamination | Cleaner bare-earth surface → more reliable absolute HAG |

Because the whole point of Option A is **accurate absolute ground elevation**
(which Stage 37 then converts into an absolute-scale HAG feature), the extra
robustness of the advanced filter is worth its cost.

### Why TIN gridding for the surface (`lidar_tin_gridding`)

The filtered ground points are irregularly spaced. Comparison of interpolators:

| Interpolator | Behaviour on irregular ground returns | Verdict |
| :-- | :-- | :-- |
| `lidar_nearest_neighbour_gridding` | Piecewise-constant (blocky); visible facets | Too coarse |
| `lidar_idw_interpolation` | Smooth but bulls-eye artifacts; needs tuning | Acceptable, more tuning |
| **`lidar_tin_gridding`** | **Delaunay linear interpolation — exact at samples, continuous, no bulls-eyes; leaves nodata gaps over large empty areas rather than hallucinating terrain** | **Chosen** |
| `lidar_sibson` / RBF | Highest smoothness but much more expensive | Overkill |

Over large holes with no ground returns, TIN gridding declines to interpolate
(produces **nodata**). `compute_hag()` already treats nodata / out-of-extent
samples (`NodataPolicy::Strict`) as a graceful fall-through to the block-min-z
proxy, so those rare gaps degrade to *today's* behaviour — a safe worst case.

### Breaking change

Making auto-DTM the default changes the HAG column values for the no-external-DTM
path versus the old proxy default, so **models must be (re)trained on features
produced with the current setting**. Consistent with Stage 30 / Stage 37
handling. The current fresh retrain absorbs this at no extra cost.

---

## Performance & overhead

The block-min-z proxy is **effectively free** (a `min` reduction already inside
`compute_hag()`). `--auto-dtm` adds two whole-file passes. Estimated cost, by
algorithmic reasoning (⚠️ estimates, not measured):

| Pass | Complexity | Est. share |
| :-- | :-- | :-- |
| `improved_ground_point_filter` (percentile → TIN → fill-pits → off-terrain removal → reference-surface) | ~O(n) grid passes + internal Delaunay of a thinned subset | dominant |
| `lidar_tin_gridding` (Delaunay of ground subset, ~10–40 % of n) | O(m log m) | secondary |

Yardstick: the eigenvalue kNN pre-pass the pipeline already runs (over *every*
point) is generally the most expensive step; auto-DTM should cost **less** than
it — roughly **+20–50 %** on top of total preprocessing.

**Ballpark for an ~1 km² / ~10–20 M-point tile:** ~2–5 minutes added
wall-clock (a couple of minutes, not 10+). Very large/dense tiles (≳50 M pts)
could approach ~10 min; small tiles (≤2 M pts) well under a minute. Scales with
point count; largely independent of `--dtm-resolution`.

The auto-DTM step `eprintln!`s per sub-stage so each run self-reports overhead
(consistent with the outlier-removal / eigen pre-pass logging). Per the
reviewer, if real-world performance disappoints, a modification request will
follow; the default stays ON for now.

---

## Choosing `--dtm-resolution`

`--dtm-resolution` is the **edge length of one DTM raster cell, expressed in the
data's own projection units** (metres for a metre-based CRS such as UTM; feet
for a feet-based CRS — it is *not* unitless). It sets both the ground filter's
`block_size` and the TIN-gridding output cell size, i.e. the **spatial grain of
the bare-earth surface** every point's HAG is measured against.

### Recommended default

**Leave it at `1.0`** for typical airborne LiDAR (~5–20 pts/m² over
metre-based projections). Most users should never touch it. Adjust only when
point density or terrain clearly falls outside that middle band.

### The core trade-off

Resolution balances two competing error sources:

- **Too coarse** (e.g. 5–10 m): each cell averages a wide terrain patch, so real
  relief (ridges, ditch banks, slopes) is smoothed away — reintroducing the very
  terrain-shape bias in HAG that Stage 38 exists to remove. *Under-resolves* the
  surface.
- **Too fine** (e.g. 0.1–0.25 m): cells become smaller than the ground-return
  spacing, so many cells have no ground hit and become **nodata gaps** (which
  fall back to the block-min-z proxy per `NodataPolicy::Strict`); residual
  micro-relief/noise also produces a bumpy surface. *Over-resolves* relative to
  the data.

The sweet spot sits **at or slightly above the spacing between *ground*
returns.**

### Calibration rule of thumb

> Set resolution ≈ **1–2× the average spacing between ground returns.**

Ground points are a subset (~10–40 %) of all returns, so ground spacing is
coarser than overall point spacing. Quick estimate:

```text
ground_spacing ≈ 1 / sqrt(total_density × ground_fraction)
```

e.g. 10 pts/m² total, ~25 % ground → ~2.5 ground pts/m² → spacing ≈ 0.63 m →
resolution ~0.6–1.2 m → **1.0 is appropriate.**

| Scenario | Suggested `--dtm-resolution` | Rationale |
| :-- | :-- | :-- |
| Typical airborne (5–20 pts/m²) | `1.0` (default) | Matches ground spacing; balanced |
| High-density / drone (>50 pts/m²), detailed terrain | `0.3–0.5` | Data supports finer relief |
| Sparse / older ALS (1–3 pts/m²) | `2.0–3.0` | Avoids nodata holes |
| Flat terrain (floodplain, playa) | `2.0–5.0` | Little relief to resolve; cleaner & faster |
| Steep / dissected terrain | near `1.0` (don't go coarse) | Coarse cells smooth away needed slope |

### Diagnosing a bad choice

Run with `--keep-auto-dtm` and inspect `_auto_dtm.tif`:

- **Speckled / many nodata holes** → too fine for the ground density → *increase*.
- **Flat "terraced" surface or washed-out slopes/ditches** → too coarse →
  *decrease*.
- Downstream sanity check: HAG for known bare-ground points should sit near 0;
  if vegetation tiers still blur together, the surface grain is likely off.

### Notes

- **Grain & density, not object size.** Resolution is chosen from how finely the
  *ground* can be reconstructed, not from how tall trees/buildings are — HAG
  resolves objects vertically regardless of cell size.
- **Cost is ~independent of resolution.** Runtime scales with point count, not
  cell size, so tune for surface fidelity, not speed.
- **Need fundamentally different control** (hydro-flattening, breaklines, a
  specific product spec)? Generate the DTM externally and pass it via
  `--hag-model`, which bypasses auto-DTM entirely.

---

## Inputs & Outputs


### Constants (`src/preprocessing/normalizer.rs`, re-exported from `mod`)

```rust
/// Default cell size (projection units) for the auto-generated ground DTM.
pub const DEFAULT_DTM_RESOLUTION: f64 = 1.0;
```

### `PreprocessConfig` (`src/preprocessing/mod.rs`)

New fields (all defaulted so existing construction sites still compile):

```rust
/// When `true` (default) and `hag_model` is `None`, auto-generate a
/// bare-earth DTM from the input cloud (Stage 38) rather than using the
/// block-min-z proxy. Disable with `--no-auto-dtm`.
pub auto_dtm: bool,

/// Output raster cell size (projection units) for the auto-generated DTM.
/// Default: DEFAULT_DTM_RESOLUTION (1.0).
pub auto_dtm_resolution: f64,

/// When `true`, retain the intermediate `_auto_ground.las` / `_auto_dtm.tif`
/// artifacts for inspection instead of deleting them after the run.
pub keep_auto_dtm: bool,
```

`Default`: `auto_dtm = true`, `auto_dtm_resolution = DEFAULT_DTM_RESOLUTION`,
`keep_auto_dtm = false`.

### New pipeline helper (`src/preprocessing/pipeline.rs`)

```rust
/// Auto-generate a bare-earth DTM raster from `input` (the possibly
/// outlier-cleaned cloud) into `output_dir`, returning the raster path.
///
/// Stage 1: improved_ground_point_filter (classify=false → ground-only LAS).
/// Stage 2: lidar_tin_gridding (interpolation_parameter="elevation") → DTM.
fn run_auto_dtm(input: &Path, output_dir: &Path, config: &PreprocessConfig)
    -> Result<PathBuf>;
```

Modeled on `run_outlier_removal`. Tool-call argument mapping:

*Stage 1 — `improved_ground_point_filter`:* `input` = effective input path;
`output` = `output_dir/_auto_ground.las`; `block_size` = `auto_dtm_resolution`;
`classify` = `false` (filter mode → ground-only output). Other params left at
tool defaults.

*Stage 2 — `lidar_tin_gridding`:* `input` = `_auto_ground.las`; `output` =
`output_dir/_auto_dtm.tif`; `resolution` = `auto_dtm_resolution`;
`interpolation_parameter` = `"elevation"`. Other params left at tool defaults.

### Pipeline wiring (`run_internal`, step 6)

DTM `Option<Arc<DtmView>>` resolved in priority order:

1. `config.hag_model.is_some()` → load user raster (unchanged path).
2. else if `config.auto_dtm` → `run_auto_dtm(...)`, then `DtmView::from_raster`.
3. else → `None` (block-min-z proxy).

The auto-generated raster path is tracked; on completion (step 9 cleanup),
`_auto_dtm.tif` and `_auto_ground.las` are deleted unless `config.keep_auto_dtm`.

### CLI flags (both `preprocess` and `preprocess-labeled`)

- `--no-auto-dtm` — disable auto-DTM; use the block-min-z proxy.
- `--dtm-resolution <f64>` — auto-DTM cell size (default 1.0; must be
  positive & finite).
- `--keep-auto-dtm` — retain intermediate `_auto_ground.las` / `_auto_dtm.tif`.

When `--hag-model` is supplied, auto-DTM is skipped (external DTM wins); a note
is printed if auto-DTM-tuning flags were also given.

---

## Steps & Specifications

1. Add `DEFAULT_DTM_RESOLUTION` + `auto_dtm` / `auto_dtm_resolution` /
   `keep_auto_dtm` fields to `PreprocessConfig` + `Default` (auto_dtm = true).
2. Implement `run_auto_dtm(input, output_dir, config)` in `pipeline.rs`.
3. Rewire step 6 to the 3-way priority; track the generated path; extend step 9
   cleanup to delete intermediates unless `keep_auto_dtm`.
4. Parse + validate the three CLI flags in both `preprocess_cmd.rs` and
   `preprocess_labeled_cmd.rs`; document in `--help`.
5. Update `PROJECT_SPEC.md §1` HAG bullet to describe the auto-DTM default.
6. Tests:
   - `run_auto_dtm` on a small synthetic cloud produces a readable raster.
   - Priority logic: `hag_model` wins; `auto_dtm=false` ⇒ proxy (None).
   - `PreprocessConfig::default().auto_dtm == true` and resolution == 1.0.
   - CLI: `--no-auto-dtm` clears the flag; bad `--dtm-resolution` rejected.

---

## Definition of Done

- [x] `cargo build` and `cargo clippy` clean (no new warnings).
- [x] `cargo test` green, including new auto-DTM tests.
- [x] Auto-DTM is the default no-external-DTM path (via
      `improved_ground_point_filter` → `lidar_tin_gridding`); `--no-auto-dtm`
      restores the proxy.
- [x] `--hag-model` takes priority over auto-DTM.
- [x] Only `--dtm-resolution` is exposed for tuning; other filter params use
      tool defaults.
- [x] Intermediates deleted unless `--keep-auto-dtm`.
- [x] `--help` for both subcommands documents the new flags; `PROJECT_SPEC.md
      §1` updated.


---

*This document is the authoritative specification for Stage 38. Per the
AGENTS.md synchronization rule, code and this spec must remain in sync.*
