# Stage 25 — Testing Gaps

## Status: CLOSED


## The Goal

Close out the "Testing Gaps" (§6) findings from `docs/AUDIT_REPORT.md`:

- **6.1** No Integration Tests for Training Loop (LOW) — the training loop has
  unit tests for `compute_class_weights` and `CheckpointManifest`, but no
  end-to-end test that runs the real `training::trainer::train()` entry point
  over a synthetic on-disk dataset and verifies the reported loss decreases.
- **6.2** No Tests for Error Paths (LOW) — error paths in the training data
  loader (`training::dataset::LabeledBlockDataset::load()` / `load_block()` /
  `load_feat_file()`) — corrupt headers, mismatched `n_classes`, missing
  files/directories, non-contiguous label maps — lack test coverage.

Both items are purely additive test-authoring work; no production code
behavior changes in this stage.

## Background

Both findings live in the `training` feature-gated code (`src/training/`),
so all new tests are gated the same way: unit tests inside
`#[cfg(feature = "training")]`-gated modules (already the case for the whole
`training` module tree per `src/lib.rs`), and a new integration test file
under `tests/` gated with `#![cfg(feature = "training")]` at the top so it is
a silent no-op when the crate is built without `--features training`.

**Item 6.1 approach:** add `tests/training_integration.rs`, a Cargo
integration test (its own compilation unit, linking against the crate's
public API only — `lidar_point_cloud_classifier::training::dataset`,
`::training::trainer`, `::preprocessing::labeled_pipeline`,
`::preprocessing::pipeline::BlockMeta`, plus the `burn` CPU
(`Autodiff<NdArray>`) backend, which is available to test targets the same
way it is to the library once the package is built with `--features
training`, since all Cargo targets in one package share the same resolved
`[dependencies]` feature set). The test:

1. Synthesizes a tiny on-disk labeled-block dataset — `.feat` binary files,
   `.lbl` raw-byte files, and a `labeled_blocks.json` manifest — using the
   exact same binary/JSON contract the real `preprocess-labeled` pipeline
   produces (verified against `training/dataset.rs::load_feat_file()` and
   `preprocessing/labeled_pipeline.rs`'s manifest types). Each point's first
   three features (the xyz columns the Input T-Net transforms) are set to a
   value strongly correlated with a two-class label (class 1 clustered near
   0.9, class 0 clustered near 0.1) so the segmentation task is trivially
   separable regardless of the (near-identity-initialized) T-Net's learned
   3×3 transform; the remaining feature columns are uncorrelated noise.
2. Loads the dataset via the real `LabeledBlockDataset::load()` (never
   constructing the struct by hand), exercising the actual manifest-parsing
   and spatial train/val split logic.
3. Runs `training::trainer::train::<Autodiff<NdArray>>()` end-to-end for a
   modest number of epochs on the CPU backend (no GPU dependency, keeping the
   test hardware-independent per AGENTS.md), with `use_class_weights: false`
   (labels are already balanced) and no checkpoint directory (keeps the test
   fast and filesystem-light).
4. Parses the real `metrics.csv` written by `append_metrics_csv()` (rather
   than reaching into training internals) and asserts the recorded
   `train_loss` at the final epoch is lower than at the first epoch —
   the audit's literal "loss decreases on a synthetic dataset" criterion.

**Item 6.2 approach:** add unit tests directly inside `training/dataset.rs`'s
existing `#[cfg(test)] mod tests` block (already has access to the private
`load_feat_file`/`load_lbl_file`/`DirEntry` items and the existing
`make_lbm`/`dummy_manifest` fixture helpers), covering every currently-
untested error branch of `LabeledBlockDataset::load()` and `load_block()`:

- empty `data_dirs` slice,
- missing/unreadable manifest directory (`cannot open`),
- corrupt/unparsable `labeled_blocks.json` (`parse error`),
- non-contiguous or non-zero-based label map values,
- `n_classes` mismatch across multiple `--data-dir` directories,
- `load_feat_file()` bad magic bytes,
- `load_feat_file()` unsupported version byte,
- `load_block()` with an out-of-range composite directory index,
- `load_block()` with a local block ID absent from the manifest.

## Inputs & Outputs

- **Inputs:** no CLI flags, file formats, or config fields change in this
  stage. Purely additive tests.
- **Outputs:** a new `tests/training_integration.rs` integration test
  (`cargo test --features training --test training_integration`) proving the
  real training loop reduces loss end-to-end on synthetic data; nine new
  unit tests in `training/dataset.rs` covering the error paths listed above.
  All existing tests continue to pass unmodified.

## Steps & Specifications

1. Create `tests/training_integration.rs`:
   - `#![cfg(feature = "training")]` at the top.
   - A `write_synthetic_block()` helper that writes one `.feat` + `.lbl` pair
     matching the on-disk WBFT format (`FEAT_MAGIC`, `FEAT_VERSION`,
     `N_FEATURES` from `preprocessing`), with per-point labels alternating
     0/1 and the first three feature columns encoding the class as described
     above.
   - `test_training_loop_reduces_loss_on_synthetic_dataset()`: builds an
     8-block synthetic dataset (one macro-tile per block for a clean spatial
     split), loads it via `LabeledBlockDataset::load()`, trains via
     `train::<Autodiff<NdArray>>()` for a fixed small epoch count, then
     parses `metrics.csv` and asserts `train_loss` strictly decreases from
     the first to the last recorded epoch.
2. Append 9 new unit tests to `training/dataset.rs`'s `mod tests` covering
   the error paths enumerated above, each asserting `.is_err()` and matching
   a distinctive substring of the error message (mirroring the style of the
   existing `test_load_feat_file_rejects_oversized_header_before_allocating`
   / `test_load_lbl_file_rejects_truncated_file` tests).
3. Verify `cargo build --features training`, `cargo test --features
   training` (including the new integration test binary), `cargo clippy
   --all-targets --features training`, and `cargo fmt --check` all clean,
   with every pre-existing test still passing unmodified.

## Definition of Done

- [x] `tests/training_integration.rs` exists, compiles only under
      `--features training`, and its test asserts a real end-to-end
      `train_loss` decrease on synthetic data via the actual `train()` entry
      point (not a mocked/simplified stand-in).
- [x] Nine new unit tests in `training/dataset.rs` cover: empty data-dirs,
      missing manifest, corrupt manifest JSON, non-contiguous label map,
      cross-directory `n_classes` mismatch, `.feat` bad magic, `.feat`
      unsupported version, `load_block()` out-of-range directory index, and
      `load_block()` missing local block ID.
- [x] `cargo build --features training`, `cargo test --features training`,
      `cargo clippy --all-targets --features training`, `cargo fmt --check`
      all clean; every pre-existing test passes unmodified.
- [x] This spec file synchronized with the final implementation (Drift
      Rule); results documented in a `## Results` section appended once
      complete.

## Results

Both items closed exactly as scoped, with no production code behavior
changes — purely additive test coverage.

**Item 6.1 — `tests/training_integration.rs`:**

- New Cargo integration test file, gated with `#![cfg(feature = "training")]`
  so it is a no-op build (and invisible to `cargo test` without
  `--features training`).
- `write_synthetic_block()` writes a real on-disk `.feat`/`.lbl` pair using
  the exact `FEAT_MAGIC`/`FEAT_VERSION`/`N_FEATURES` binary header contract
  `load_feat_file()` expects. The class-discriminative signal is placed on
  all three of the first three feature columns (the xyz columns the Input
  T-Net transforms) — class 1 clustered near 0.9-0.95, class 0 clustered
  near 0.0-0.05 — with the remaining feature columns filled with
  deterministic pseudo-random noise (no extra RNG dependency needed).
- `test_training_loop_reduces_loss_on_synthetic_dataset()` builds an
  8-block/32-point-per-block synthetic dataset, loads it via the real
  `LabeledBlockDataset::load()` (exercising real manifest parsing + the real
  spatial train/val macro-tile split — 6 train blocks / 2 val blocks),
  then trains via the real `training::trainer::train::<Autodiff<NdArray>>()`
  for 15 epochs on the CPU backend (`use_class_weights: false`, no
  checkpoint dir), and finally parses the real `metrics.csv` that
  `append_metrics_csv()` writes and asserts the recorded `train_loss` at the
  final epoch is strictly lower than at the first epoch.
- Verified empirically over 4 consecutive runs (`cargo test --features
  training --test training_integration`): consistently passing. A
  representative run: `train_loss` starts at `0.3634` (epoch 1) and ends at
  `0.0113` (epoch 15), with `val_mIoU` reaching `1.0` by epoch 5 — confirming
  the synthetic dataset is trivially learnable end-to-end through the real
  training loop, T-Net included.
- Two clippy lints surfaced during authoring were fixed directly rather than
  suppressed: `&PathBuf` parameter → `&Path` (`ptr_arg`), and
  `LabeledBlockDataset::load(&[dir_path.clone()], …)` →
  `LabeledBlockDataset::load(std::slice::from_ref(&dir_path), …)`
  (`cloned_ref_to_slice_refs`).

**Item 6.2 — nine new `training/dataset.rs` unit tests:**

Added directly to the existing `#[cfg(test)] mod tests` block, reusing the
existing `make_lbm`/`dummy_manifest`/`build_block_index` fixtures:

1. `test_load_rejects_empty_data_dirs` — asserts the `"at least one
   --data-dir"` error.
2. `test_load_rejects_missing_manifest` — a tempdir with no
   `labeled_blocks.json` inside it; asserts the `"cannot open"` error.
3. `test_load_rejects_corrupt_manifest_json` — malformed JSON bytes written
   to `labeled_blocks.json`; asserts the `"parse error"` error.
4. `test_load_rejects_non_contiguous_label_map` — a label map with values
   `{1, 2}` (not 0-based contiguous); asserts the `"non-contiguous"` error.
5. `test_load_rejects_n_classes_mismatch_across_dirs` — two directories, one
   with a 2-class label map and one with a 3-class label map; asserts the
   `"n_classes mismatch"` error.
6. `test_load_feat_file_rejects_bad_magic` — a `.feat` file with a wrong
   4-byte magic; asserts the `"bad magic"` error via the private
   `load_feat_file()` directly.
7. `test_load_feat_file_rejects_unsupported_version` — correct magic, wrong
   version byte; asserts the `"unsupported version"` error.
8. `test_load_block_rejects_out_of_range_dir_index` — a hand-constructed
   single-directory `LabeledBlockDataset` (same pattern as the pre-existing
   `test_max_sampled_points_per_block_uses_manifest_metadata` test);
   `load_block(make_global_id(5, 0))` asserts the `"out of range"` error.
9. `test_load_block_rejects_missing_local_id` — same hand-constructed
   dataset pattern; `load_block()` for a local ID absent from the manifest
   asserts the `"not found in manifest"` error.

Constructing `LabeledBlockDataset`/`DirEntry` by hand in tests 8 and 9
required adding `#[derive(Debug)]` to `LabeledBlockDataset`, `DirEntry`, and
`LoadedBlock` (all were previously non-`Debug`, which `Result::unwrap_err()`
requires); this is a purely additive, zero-behavior-change trait derive.

**Verification (final, `lidar_point_cloud_classifier/` crate root):**

- `cargo build --features training` — clean, no warnings from this crate.
- `cargo test --features training` — **89** lib tests passing (80
  pre-existing + 9 new `dataset.rs` error-path tests) + **1** new
  `training_integration` test passing, 0 failed, 0 doc-test failures.
  Re-ran the integration test 4 additional times back-to-back to rule out
  flakiness from Rayon-parallel float-summation ordering; all passed.
- `cargo clippy --all-targets --features training` — zero warnings from any
  file touched in this stage (`training_integration.rs`, `dataset.rs`); all
  remaining warnings are pre-existing (either in the untouched
  `whitebox_next_gen::wbraster` dependency, or pre-existing
  `cast_precision_loss`-class lints in `dataset.rs`'s pre-existing test
  helpers `make_lbm`/`test_spatial_split_fraction`, unchanged by this stage).
- `cargo fmt --check` — clean.

Both `docs/AUDIT_REPORT.md` findings 6.1 and 6.2 are now resolved; see the
updated Summary Priority Table, Recommended Implementation Order, and Stage
Mapping sections there.


