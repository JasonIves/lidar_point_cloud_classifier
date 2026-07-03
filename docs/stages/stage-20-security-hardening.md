# Stage 20 — Security & Robustness Hardening

## Status: CLOSED — security hardening items 3.1–3.5, 4.4 resolved; build/tests/clippy/fmt clean


## The Goal

Close out the "Security & Robustness" (§3) and CLI-panic (§4.4) findings from
`docs/AUDIT_REPORT.md`. The classifier currently trusts `.feat`/`.lbl` files
and `labeled_blocks.json` / `blocks.json` manifests unconditionally: a
corrupted or maliciously-crafted header can trigger a multi-gigabyte
allocation attempt (denial of service), and several code paths still violate
AGENTS.md's "No Panics in Production" rule via `assert!` or unchecked slice
indexing. This stage hardens every untrusted-input boundary identified by the
audit without changing any on-disk format or CLI-visible behavior for
well-formed inputs.

Specifically this stage addresses audit items:
- **3.1** `assert!` in production code (`bridge.rs::extract_linear`)
- **3.2** No file-size validation before allocation (`dataset.rs`, `inference.rs`)
- **3.3** Unchecked integer multiplication for allocation size (`dataset.rs`)
- **3.4** No path-traversal validation on manifest-supplied file names
- **3.5** `.lbl` file size not validated against the expected point count
- **4.4** Manual CLI argument parsing panics on a missing flag value
  (`train_cmd.rs`, `preprocess_labeled_cmd.rs`)

## Background

`preprocess_cmd.rs` already implements a safe, bounds-checked
`next_value(args, &mut i, flag) -> Result<&str>` helper and uses it for every
flag. `train_cmd.rs` and `preprocess_labeled_cmd.rs` predate that pattern and
still do `i += 1; &args[i]`, which panics with an index-out-of-bounds error if
a value-taking flag is the last CLI argument (e.g. `train --data-dir`).

Similarly, `dataset.rs::load_feat_file()` and `inference.rs::process_block()`
both read a `.feat` header's `n_points`/`n_features` fields and immediately
allocate `vec![0u8; n_points * n_features * 4]` with no upper bound and no
overflow check. A corrupted file (or a hostile one, if this tool is ever
pointed at untrusted input) with `n_points ≈ u32::MAX` can drive an attempted
allocation of tens of gigabytes, aborting the process. `load_lbl_file()` has a
parallel gap: it never confirms the file is at least `n_points` bytes before
issuing a `read_exact`, so truncated `.lbl` files surface as a generic
"unexpected end of file" I/O error instead of a clear validation message.

Finally, `bridge.rs::extract_linear()` uses `assert!(d_in > 0 && d_out > 0, ...)`
guarding against a burn-version layout change — this is exactly the kind of
internal invariant that AGENTS.md requires to be a recoverable `Result` error,
not a panic, even though it is not expected to fire in normal operation.

## Inputs & Outputs

- **Inputs:** `.feat` files, `.lbl` files, `labeled_blocks.json` / `blocks.json`
  manifests (all potentially corrupted or hand-edited), CLI argument vectors
  (potentially missing a trailing value).
- **Outputs:** No change to any on-disk format. No change to CLI flag names or
  semantics for well-formed input. The only user-visible difference is that
  malformed input now produces a clear `ClassifierError::Pipeline` message
  instead of a panic, an OOM abort, or a confusing low-level I/O error.

## Steps & Specifications

1. **CLI bounds-checking (4.4)** — Add a `next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str>` helper (mirroring `preprocess_cmd.rs`) to both `train_cmd.rs` and `preprocess_labeled_cmd.rs`, and rewrite every `i += 1; &args[i]` value-consuming site to route through it. `--debug-csv`/`--outlier-removal`/`--outlier-use-median`-style optional-bool flags (`parse_optional_bool`) are already safe (`args.get`) and are left unchanged.

2. **Block-size validation cap (3.2, 3.3)** — Introduce a shared constant, e.g. `MAX_FEAT_PAYLOAD_BYTES: usize = 512 * 1024 * 1024` (512 MB — comfortably above any realistic block: even 1M points × 100 features × 4 bytes = 400 MB), in `preprocessing/mod.rs` next to the other `.feat` format constants. In both `dataset.rs::load_feat_file()` and `inference.rs::process_block()`/`read_feat_header()`:
   - Replace `n_points * n_features` with `n_points.checked_mul(n_features).ok_or_else(|| ClassifierError::Pipeline(...))?`.
   - Replace the subsequent `* 4` (bytes-per-f32) with another `checked_mul(4)`.
   - Reject (with a descriptive `ClassifierError::Pipeline`) if the resulting byte count exceeds `MAX_FEAT_PAYLOAD_BYTES`, *before* calling `vec![0u8; ...]`.

3. **`.lbl` size validation (3.5)** — In `dataset.rs::load_lbl_file()`, call `f.metadata()?.len()` and compare against `n_points as u64` before the `read_exact`, returning a clear `ClassifierError::Pipeline` (`".lbl file '{path}' is truncated: expected {n_points} bytes, found {actual}"`) rather than letting a short read surface as a generic I/O error.

4. **Path-traversal validation (3.4)** — Add a small helper `fn validate_block_filename(name: &str) -> Result<()>` (in `dataset.rs`, reused from the labeled pipeline's manifest reader, and equivalently in `inference.rs`/wherever `BlockManifest`/`LabeledBlockManifest` file names are joined to a directory) that rejects any file-name component containing `..`, `/`, or `\`. Call it immediately after reading `bm.meta.file` / `bm.lbl_file` from the manifest, before joining to `entry.path`.

5. **Replace `assert!` with `Result` (3.1)** — In `bridge.rs::extract_linear()`, replace the `assert!(d_in > 0 && d_out > 0, ...)` with an `if d_in == 0 || d_out == 0 { return Err(ClassifierError::Pipeline(format!(...))); }` guard, preserving the original diagnostic message content.

6. Verify `cargo build`, `cargo test --features training`, `cargo clippy --features training`, and `cargo fmt --check` are all clean after the change.

## Definition of Done

- [x] `train_cmd.rs` and `preprocess_labeled_cmd.rs` use a bounds-checked `next_value()` helper for every value-taking flag; a trailing flag with no value now returns a `ClassifierError::Pipeline` instead of panicking.
- [x] `dataset.rs::load_feat_file()` and `inference.rs::process_block()`/`read_feat_header()` reject any `.feat` header whose implied payload size exceeds `MAX_FEAT_PAYLOAD_BYTES`, and use `checked_mul` throughout the size computation (no silent overflow on any target width).
- [x] `dataset.rs::load_lbl_file()` validates the file's on-disk size against the expected `n_points` before reading, with a descriptive error on mismatch.
- [x] Manifest-supplied block/label file names are validated against path-traversal (`..`, `/`, `\`) before being joined to a directory path, in both the training dataset loader and the deployed inference loader (consolidated into a single canonical `preprocessing::validate_block_filename()`).
- [x] `bridge.rs::extract_linear()` no longer contains an `assert!`; the zero-dimension case returns a `ClassifierError::Pipeline`.
- [x] `cargo build --features training`, `cargo test --features training`, `cargo clippy --features training`, `cargo fmt --check` all clean.
- [x] New/adjusted unit tests cover: oversized `.feat` header rejected before allocation, truncated `.lbl` file rejected with a clear message, path-traversal file name rejected, missing CLI flag value rejected (not panicking).
- [x] This spec file synchronized with the final implementation (Drift Rule); results documented below (folded into this file rather than a separate `stage-20-results.md`, matching the Stage 19 convention).

## Results

All six audit items (3.1–3.5, 4.4) are resolved:

- **3.1** — `bridge.rs::extract_linear()`'s `assert!` replaced with a `Result`-returning zero-dimension guard. Existing round-trip/SWA tests still pass through the normal (non-zero) path.
- **3.2 / 3.3** — `MAX_FEAT_PAYLOAD_BYTES` (512 MB) constant added to `preprocessing/mod.rs`. Both `dataset.rs::load_feat_file()` and `inference.rs::process_block()` now use `checked_mul` for `n_points × n_features × 4` and reject any payload exceeding the cap *before* allocating. Covered by `training::dataset::tests::test_load_feat_file_rejects_oversized_header_before_allocating`.
- **3.4** — Path-traversal validation consolidated into a single canonical `pub fn preprocessing::validate_block_filename()`, called from both `dataset.rs::load_block()` and `inference.rs::process_block()` before any path join. Covered by four dedicated tests in `preprocessing::tests`.
- **3.5** — `dataset.rs::load_lbl_file()` now checks `f.metadata()?.len()` against the expected byte count before `read_exact`, returning a clear "is truncated" error. Covered by `training::dataset::tests::test_load_lbl_file_rejects_truncated_file`.
- **4.4** — `train_cmd.rs` and `preprocess_labeled_cmd.rs` both rewritten to use a bounds-checked `next_value()` helper (mirroring `preprocess_cmd.rs`) for every value-taking flag. Covered by `cli::train_cmd::tests` and `cli::preprocess_labeled_cmd::tests` (trailing-flag-without-value cases).

**Verification (2026-07-02):**
```
cargo build --features training   → Finished, 0 errors
cargo test  --features training   → 76 passed; 0 failed
cargo clippy --features training  → 0 new warnings introduced by Stage 20 changes
                                     (pre-existing pedantic warnings are tracked under Stage 24)
cargo fmt --check                 → clean (no diff)
```


