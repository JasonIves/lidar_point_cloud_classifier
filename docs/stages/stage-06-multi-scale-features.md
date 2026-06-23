# Stage 06 — Multi-Scale Geometric Features (G-02)

**Status:** COMPLETE — 2026-06-23
**Approved:** 2026-06-23
**Project:** Whitebox Next Gen: LiDAR Point Cloud Classifier
**Closes:** AUDIT_RESULTS.md gap G-02

---

## Goal

Compute eigenvalue-derived structural features (linearity, planarity, sphericity,
omnivariance, curvature) at **multiple search radii** per point instead of a single
radius.  This gives the PointNet encoder richer multi-scale geometric context:

- **Fine radius (≤1 m):** individual returns, thin vegetation, surface roughness
- **Medium radius (2–4 m):** tree crowns, car rooftops, facade segments
- **Coarse radius (8–15 m):** building footprints, canopy structure, open fields

---

## Inputs & Outputs

### New CLI flag

Available on both `preprocess` and `preprocess-labeled`:

```
--search-radii  <r0,r1,...>   Comma-separated eigenvalue search radii in
                               projection units.  Overrides --search-radius.
                               Default: use --search-radius (single scale).
                               Radii are sorted ascending before use.
```

**Backward compatibility:** omitting `--search-radii` uses the existing
`--search-radius` (default 1.0 m) as a single radius → 12-feature output, identical
to all prior preprocessing.

### Feature vector layout

For N radii `[r₀, r₁, ..., rₙ₋₁]` (sorted ascending):

| Index range | Features |
|---|---|
| 0–6 | Scalar features (unchanged: x_norm … hag) |
| 7–11 | Eigenvalue features at r₀ |
| 12–16 | Eigenvalue features at r₁ |
| … | … |
| 7 + 5(N-1) – 7 + 5N-1 | Eigenvalue features at rₙ₋₁ |

Each 5-element eigenvalue block is: linearity, planarity, sphericity, omnivariance,
curvature — identical to the existing single-radius block.

Total features: `7 + 5 × N`.  Default (N=1): 12 (unchanged).

### Manifest changes

`blocks.json` gains `search_radii: Vec<f64>` (`#[serde(default)]`).
`labeled_blocks.json` gains the same field.

Old manifests (without `search_radii`) deserialize cleanly — the default is `[]`,
which the dataset loader interprets as single-radius (12 features).

---

## Design Decisions

### D1 — Fixed radius per scale in multi-scale mode

In single-radius mode the adaptive radius expansion (up to `radius × 4`) is retained
for robustness on sparse blocks.

In multi-scale mode (`search_radii.len() > 1`) each radius uses a **fixed** search
with no expansion.  Rationale: if a fine-scale radius adaptively expands to the same
size as the coarse radius, the two scales collapse to the same neighbourhood and the
multi-scale benefit is lost.  Sparse-neighbourhood degenerate cases fall back to
`[0.0; 5]` (same as the current single-radius degenerate path).

### D2 — `N_FEATURES` retained as backward-compat alias

`N_FEATURES = 12` is kept as a compile-time constant equal to `n_features_for_radii(1)`.
This prevents breaking any code that uses `N_FEATURES` for single-radius context.
All multi-scale paths use the runtime-computed value.

### D3 — Feature count flows through manifests, not `.feat` headers

The `n_features` stored in every `.feat` header already carries the ground-truth count.
The manifest `search_radii` field provides a second, human-readable source.
`LabeledBlockDataset` uses the manifest value during initialization so the trainer
knows `n_features_in` without opening any `.feat` file.

### D4 — `features_to_tensor` takes explicit n_features

`burn_model::features_to_tensor` gains a `n_features: usize` parameter to shape the
tensor correctly.  This is the single additional parameter propagated from the
`LoadedBlock` to the trainer.

---

## Changed Files

| File | Nature of change |
|---|---|
| `preprocessing/mod.rs` | Add `N_SCALAR_FEATURES`, `N_EIGEN_FEATURES_PER_RADIUS`, `n_features_for_radii()`, `search_radii: Vec<f64>` in `PreprocessConfig`, `search_radii_effective()` method |
| `preprocessing/feature_extractor.rs` | `extract_features`: `search_radii: &[f64]`; returns `Vec<Vec<f32>>`; fixed vs adaptive per mode |
| `preprocessing/pipeline.rs` | `write_feat_file` and `write_debug_csv` use runtime `n_features`; `BlockManifest` gains `search_radii`; call sites updated |
| `preprocessing/labeled_pipeline.rs` | `LabeledBlockManifest` gains `search_radii`; manifest construction updated |
| `training/dataset.rs` | Remove hard `N_FEATURES` check; add `n_features_inner` + `n_features()` to dataset |
| `training/trainer.rs` | Use `dataset.n_features()` for `PointNetConfig.n_features_in` and tensor shaping |
| `training/burn_model.rs` | `features_to_tensor` gains `n_features` param; `LinearConfig::new(cfg.n_features_in, ...)` replaces `N_FEATURES`; `forward()` uses `input.dims()[1]` |
| `cli/preprocess_cmd.rs` | Parse `--search-radii`; retain `--search-radius` |
| `cli/preprocess_labeled_cmd.rs` | Mirror same changes |

---

## Definition of Done

1. `cargo build --release` passes (both binaries). ✓
2. `cargo clippy -- -D warnings` passes. ✓
3. `cargo test --features training` — all 48 existing + 3 new tests pass. ✓
4. `--search-radius 1.0` (no `--search-radii`) → byte-identical `.feat` output to baseline. ✓
5. `--search-radii 0.5,1.0,2.0` → 22-feature `.feat` files that train successfully with matching `n_classes`. ✓
6. `labeled_blocks.json` includes `search_radii` field. ✓
7. Stage spec synchronized with implementation. ✓
