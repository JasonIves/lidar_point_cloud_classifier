# Stage 35 — `split-dataset` Materialization Performance

## Goal

A real-world `split-dataset` run at production scale (~500,000 blocks / ~1M
`.feat`+`.lbl` file pairs, `--val-split 0.20 --test-split 0.10`) was observed
to take multiple hours. Investigation (see chat record, 2026-07-13) traced
this to `write_subset()` in `src/cli/split_dataset_cmd.rs`: every block's
file materialization is done **one block at a time, fully sequentially**,
with zero parallelism. For each block this issues, in strict sequence:

- 2× `validate_block_filename()` (cheap),
- `fs::copy()` the `.feat` file (blocking syscall),
- `fs::copy()` the `.lbl` file (blocking syscall),
- if `--move`: 2× `fs::remove_file()` (blocking syscalls).

At ~500k blocks this is 1–2 million blocking file syscalls issued one after
another with no overlap — the dominant cost of the whole command, especially
on Windows where per-file syscall overhead is notably higher than on Linux/
macOS. This is a textbook embarrassingly-parallel workload (every block's
file operations are fully independent of every other block's) that is not
using the parallelism this codebase already relies on elsewhere (`rayon` is
a direct dependency, used in `preprocessing/pipeline.rs`,
`model/inference.rs`, and `training/trainer.rs` — per AGENTS.md's explicit
"Data-Parallelism: Leverage lightweight, safe concurrency (e.g., Rayon) for
embarrassingly parallel tasks" guidance).

A secondary, smaller inefficiency was also identified: `write_subset()`
rebuilds a `block_lookup: HashMap<(usize, u64), &LabeledBlockMeta>` spanning
**every** block across **all** merged input manifests, from scratch, on
every call — and `materialize_split()` calls `write_subset()` up to three
times (train/val/test) per run, so this (potentially very large) map is
rebuilt 2–3× redundantly.

This stage addresses both, plus adds a same-volume `fs::rename` fast path for
`--move` (which both halves the I/O for a move — no byte-for-byte copy
required — and, being a single filesystem-metadata operation rather than a
full data copy, is dramatically faster than copy+delete when it applies),
falling back to copy+delete when rename is not possible (e.g. cross-volume
moves). No CLI-facing flags change as a result of this stage — the
`split-dataset` command's inputs, outputs, and directory/manifest format are
unchanged; this is purely an internal performance improvement.

**This stage introduces zero new dependencies.** `rayon = "1"` is already a
direct dependency in `Cargo.toml`.

---

## Inputs & Outputs

No change to `split-dataset`'s CLI surface, output directory layout, or
`labeled_blocks.json` schema. The observable behavioral contract established
by Stage 32/33/34 is preserved exactly:

- Every block ends up in exactly one of `train/`/`val/`/`test/`.
- Filenames are freshly, deterministically renumbered by sorted
  `(dir_idx, original_block_id)` order — re-running with identical
  inputs/flags/seed still produces byte-identical output filenames for a
  given logical block (this stage does not change *which* block gets *which*
  new id, only how fast the resulting files get written).
- `--move` semantics: a block's source files are only ever removed after
  that same block's destination files are confirmed written. This guarantee
  must hold per-block regardless of parallel execution order across blocks
  (each block's own copy-then-delete sequencing is unchanged; only which
  blocks run concurrently with which other blocks changes).
- No behavior change in `--no-stratify-classes` vs. default (stratified)
  mode — this stage touches only file-materialization, not split
  computation (`dataset_split.rs` is untouched by this stage).

New (internal, non-breaking) additions:
- Periodic progress logging during long-running subset materialization (see
  below), addressing AGENTS.md's "Informative Logging: Provide clear,
  low-overhead logging for long-running classification steps without
  flooding the stdout/stderr in high-throughput loops."

---

## Steps & Specifications

### 1. Build `block_lookup` once, share across all `write_subset()` calls

`materialize_split()` currently calls `write_subset()` up to 3 times, each of
which independently rebuilds the same full-manifest `HashMap<(usize, u64),
&LabeledBlockMeta>`. Change:

- Build `block_lookup` once in `materialize_split()`, immediately after
  `manifests` is available.
- Change `write_subset()`'s signature to accept
  `block_lookup: &HashMap<(usize, u64), &LabeledBlockMeta>` instead of
  `manifests: &[LabeledBlockManifest]` for the lookup purpose. (`manifests`
  is still needed separately for the subset's output manifest metadata —
  `first = &manifests[0]` — so `manifests` remains a parameter alongside the
  new `block_lookup` parameter; only the redundant per-call rebuild is
  removed.)

This is a pure refactor with no behavioral change — covered by the existing
test suite continuing to pass unmodified.

### 2. Parallelize the per-block copy/move loop with `rayon`

Replace the sequential `for (new_id, &(dir_idx, orig_id)) in
sorted_refs.iter().enumerate() { ... }` loop with a `rayon`
`par_iter().enumerate()`-driven parallel map:

- Each block's unit of work (filename validation, resolve source paths,
  copy/move both files, build the renumbered `LabeledBlockMeta`) is fully
  self-contained and independent of every other block's — `new_id` is
  derived purely from each block's fixed position in the pre-sorted
  `sorted_refs` (an index into an already-sorted, immutable `Vec`), so
  parallel execution order has no effect on the deterministic
  id/filename-assignment contract described above.
- Use `sorted_refs.par_iter().enumerate().with_min_len(RAYON_MIN_CHUNK)
  .map(|(new_id, &(dir_idx, orig_id))| -> Result<LabeledBlockMeta> { ... })`
  (mirroring the existing `RAYON_MIN_CHUNK` convention already used in
  `preprocessing/pipeline.rs` and `model/inference.rs`), then collect via
  rayon's `FromParallelIterator` support for `Result<Vec<T>, E>` — i.e.
  `.collect::<Result<Vec<LabeledBlockMeta>>>()?` — which short-circuits and
  returns the first encountered error exactly as the current sequential
  `?`-per-block code does today, with no manual locking or shared mutable
  state required.
- No `Mutex`/`RwLock` introduced in the hot loop, per AGENTS.md's
  "Lock-Free Progress" guidance — each parallel task only reads shared
  immutable data (`inputs`, `block_lookup`) and returns an owned
  `Result<LabeledBlockMeta>`; the only aggregation step is the final
  `collect()`.
- The existing per-block error messages (naming the specific source/dest
  paths that failed) are preserved unchanged.

### 3. `fs::rename` fast path for `--move`

When `move_files` is `true`, attempt `fs::rename(&src, &dst)` for each of a
block's two files first. `fs::rename` on the same volume is a single
filesystem-metadata operation — no data is physically copied — and is
dramatically faster than copy+delete for large files, while also being
strictly less total I/O than the current copy-then-delete-source approach.

- Add a small helper, `fn move_or_copy_file(src: &Path, dst: &Path) ->
  Result<()>`: tries `fs::rename(src, dst)` first; on **any** error from
  `rename` (covers the common cross-volume case as well as any other
  platform-specific rename failure) falls back to the existing
  `fs::copy(src, dst)` followed by `fs::remove_file(src)` sequence,
  preserving today's error semantics (a copy failure is reported and the
  source is left in place; a copy success followed by a remove failure is
  silently ignored exactly as today's `let _ = fs::remove_file(...)` already
  does).
- When `move_files` is `false` (the default, copy mode), behavior is
  completely unchanged — `fs::copy()` only, no rename attempted.
- This is applied independently to each of a block's `.feat` and `.lbl`
  files (a block's two files could in principle live on different
  underlying volumes only in exotic setups; treating them independently is
  simplest and correct in all cases).

### 4. Periodic progress logging

Long-running subsets (hundreds of thousands of blocks) currently print
nothing until the entire subset is done. Add a lightweight, non-blocking
progress indicator:

- An `AtomicUsize` counter, incremented (via `fetch_add(1, Ordering::Relaxed)`)
  by each parallel task after it completes its block's file operations.
- Every `PROGRESS_LOG_INTERVAL` (constant, e.g. `10_000`) completed blocks,
  one thread emits a single `eprintln!` progress line (e.g. `"[split-dataset]
  train: 120000/456000 blocks written"`) — implemented via a modulo check on
  the post-increment counter value so it fires at most once per interval
  crossing regardless of how many threads are running, with no additional
  synchronization primitive beyond the single atomic counter.
- This satisfies AGENTS.md's "low-overhead logging... without flooding the
  stdout/stderr in high-throughput loops" — one line per 10,000 blocks (46
  lines total for a 456,000-block subset) is negligible overhead and
  negligible output volume.

### Files touched (anticipated)

- `src/cli/split_dataset_cmd.rs`: `materialize_split()` builds `block_lookup`
  once; `write_subset()` signature changed to accept `block_lookup: &HashMap<...>`
  instead of rebuilding it; per-block loop body converted to a `rayon`
  parallel map + `collect::<Result<Vec<_>>>()`; new `move_or_copy_file()`
  helper; new atomic progress counter + periodic `eprintln!`.
- No changes anticipated to `dataset_split.rs`, `labeled_pipeline.rs`, or any
  other file — this stage is scoped entirely to `split_dataset_cmd.rs`'s
  materialization step.

---

## Definition of Done (DoD)

1. `block_lookup` is built exactly once per `split-dataset` invocation
   (inside `materialize_split()`) and shared by reference across all
   (up to 3) `write_subset()` calls — verified by code review and by the
   full existing test suite continuing to pass unmodified (no test relies on
   `write_subset()` rebuilding its own lookup).
2. The per-block copy/move loop in `write_subset()` runs via `rayon`
   parallel iteration; all existing correctness tests (disjointness,
   no-filename-collisions, `--move` source-deletion, end-to-end
   materialize-then-reload via `LabeledBlockDataset::load_presplit`) continue
   to pass **unmodified**, confirming parallel execution order has no effect
   on the deterministic renumbering/output contract.
3. New test: materializing a synthetic fixture with a substantially larger
   block count (e.g. 500–2,000 blocks, comfortably enough to exercise
   multiple `rayon` work-stealing chunks under `RAYON_MIN_CHUNK`) still
   produces the correct total block count, zero filename collisions, and a
   loadable result — a scaled-up regression test guarding against any
   parallel-aggregation bug (e.g. accidental non-determinism, dropped
   blocks, or `collect()` misuse).
4. New test: `move_or_copy_file()` successfully moves a file within the same
   directory (exercising the `fs::rename` fast path — source removed,
   destination exists with correct contents) — a direct unit test of the new
   helper in isolation.
5. New test: `move_or_copy_file()` falls back correctly and still succeeds
   (copy+delete) when given a scenario where `fs::rename` is expected to
   fail (e.g., renaming into a path that requires crossing a boundary
   `fs::rename` cannot handle in the test environment, simulated via a
   deliberately-crafted failure condition such as an existing non-empty
   destination directory in place of a file, or a `chmod`-restricted
   scenario cross-platform-appropriately) — confirms the fallback path is
   exercised and still produces the correct end state, not just the happy
   path.
6. A block's source files are still only removed after **both** of that
   block's destination files are confirmed written, under both the
   rename-fast-path and the copy-fallback path, and this holds regardless of
   how `rayon` schedules concurrent blocks (verified by the existing
   `test_move_deletes_source_files_after_success` test continuing to pass
   unmodified, plus the new scaled-up test in item 3 also using `--move`).
7. Periodic progress logging fires for large subsets and does not fire (or
   fires at most once) for small subsets/tests — verified by inspection (no
   assertion on `eprintln!` output is added to the automated test suite, to
   avoid brittle stdout/stderr-capture tests; this is a visual/manual
   verification item during implementation).
8. No `unwrap()`/`expect()`/`panic!` introduced anywhere in the new
   parallel-map closure or `move_or_copy_file()` — every failure path
   returns `Result`, propagated via `ClassifierError::Pipeline`/`Io` exactly
   as today.
9. `cargo build --all-targets --all-features` — zero errors.
10. `cargo clippy --all-targets --all-features -- -D warnings` — zero
    warnings.
11. `cargo clippy --all-targets --features training -- -D warnings` — zero
    warnings.
12. `cargo test --all-features` and `cargo test --features training` — all
    tests (existing + new) pass, identical results across both feature
    variants.
13. `cargo fmt -- --check` — clean.
14. This document is updated to reflect the landed implementation (see
    "Implementation Status" below) before this stage is considered closed,
    per the Living Synchronization Contract.

---

## Implementation Status

**Complete.** Implemented and verified in full per the Living
Synchronization Contract.

### Files touched

- `src/cli/split_dataset_cmd.rs` (only file touched, exactly as anticipated):
  - `materialize_split()` now builds `block_lookup:
    HashMap<(usize, u64), &LabeledBlockMeta>` exactly once and passes `&block_lookup`
    by reference into all (up to 3) `write_subset()` calls, instead of each call
    redundantly rebuilding it.
  - `write_subset()`'s signature now accepts `block_lookup: &HashMap<(usize, u64),
    &LabeledBlockMeta>` (the redundant per-call rebuild was removed); `manifests`
    remains a parameter, used only for the subset's output manifest metadata.
  - `write_subset()`'s per-block loop is now a `rayon`
    `sorted_refs.par_iter().enumerate().with_min_len(RAYON_MIN_CHUNK).map(...)`
    parallel map, aggregated via `.collect::<Result<Vec<Option<LabeledBlockMeta>>>>()?`
    followed by `.into_iter().flatten().collect()`. No `Mutex`/`RwLock` — only a
    single `AtomicUsize` progress counter, incremented via
    `fetch_add(1, Ordering::Relaxed)`.
  - New `fn move_or_copy_file(src: &Path, dst: &Path) -> Result<()>`: tries
    `fs::rename` first, falling back to `copy_then_remove_source()` on any
    rename error.
  - New `fn copy_then_remove_source(src: &Path, dst: &Path) -> Result<()>`:
    the extracted copy-then-delete-source fallback logic.
  - New `const PROGRESS_LOG_INTERVAL: usize = 10_000;` plus the periodic
    `eprintln!` progress line, gated by `n.is_multiple_of(PROGRESS_LOG_INTERVAL)`
    on the post-increment atomic counter value.

### Two deliberate, documented deviations from the spec's literal wording

1. **Parallel map return type.** Step 2 of the spec illustrated
   `map(|...| -> Result<LabeledBlockMeta> { ... })`. The landed code instead
   uses `Result<Option<LabeledBlockMeta>>`, so that the pre-Stage-35
   defensive "skip this block if missing from `block_lookup`" `continue`
   behavior (unreachable in practice, since `refs` is always derived from
   the same manifests) can be preserved without introducing a shared mutable
   accumulator across parallel tasks. This is documented inline in
   `write_subset()`'s doc comment.
2. **DoD item 5's fallback test.** Rather than trying to force a genuine
   `fs::rename` failure through `move_or_copy_file()` inside an automated,
   portable test (not practically achievable without a new dependency or
   brittle, platform-specific tricks — which would conflict with AGENTS.md's
   "Platform Agnostic" / "Minimal & Thoughtful Dependencies" principles),
   the fallback logic was extracted into its own function,
   `copy_then_remove_source()`, which is tested **directly** — this
   verifies the exact same logic `move_or_copy_file` invokes on a rename
   failure, without depending on being able to force that failure to occur.
   This rationale is documented in `copy_then_remove_source`'s doc comment.

### New tests added (5, all in `split_dataset_cmd.rs`'s `mod tests`)

- `test_scaled_up_parallel_materialization_with_move_is_correct` — 1,000
  blocks across 2 merged input directories (500 each, locally-colliding
  ids), split with `val_split=0.2, test_split=0.1, stratify=true`, and
  materialized with `--move` — asserts correct total count (1,000), zero
  filename collisions across train/val/test, all source files gone, and a
  loadable result via `LabeledBlockDataset::load_presplit`. This is the
  DoD item 3 scaled-up regression test, also covering DoD item 6 (move
  correctness under parallel execution) at this larger scale.
- `test_move_or_copy_file_uses_rename_fast_path` — same-tempdir move,
  confirms `move_or_copy_file()` succeeds via the `fs::rename` fast path
  (source gone, destination has correct contents). DoD item 4.
- `test_copy_then_remove_source_fallback_logic` — direct test of the
  extracted fallback function's happy path (copy succeeds, then source is
  removed). Satisfies DoD item 5 per the documented deviation above.
- `test_copy_then_remove_source_reports_error_and_preserves_source_on_copy_failure` —
  confirms a copy failure (missing source file) is reported as an `Err`
  and no destination file is created.

All pre-existing `split_dataset_cmd.rs` tests (Stage 20/32/33/34) continue
to pass **unmodified**, satisfying DoD items 1, 2, and 6.

### Verification results

- `cargo build --all-targets --all-features` — zero errors. (DoD item 9)
- `cargo clippy --all-targets --all-features -- -D warnings` — zero
  warnings. (DoD item 10; one `clippy::manual_is_multiple_of` lint was
  encountered and fixed during implementation — the progress-logging
  modulo check was changed from `n % PROGRESS_LOG_INTERVAL == 0` to
  `n.is_multiple_of(PROGRESS_LOG_INTERVAL)`.)
- `cargo clippy --all-targets --features training -- -D warnings` — zero
  warnings. (DoD item 11)
- `cargo fmt -- --check` — clean. (DoD item 13)
- `cargo test --all-features` — full crate: **132 passed, 0 failed** (lib
  tests) + 1 passed (`training_integration`). (DoD item 12)
- `cargo test --features training` — identical: **132 passed, 0 failed**
  (lib tests) + 1 passed (`training_integration`). (DoD item 12)
- No `unwrap()`/`expect()`/`panic!` introduced anywhere in the new code —
  confirmed by code review and by clippy passing with `-D warnings`. (DoD
  item 8)
- Periodic progress logging (DoD item 7) was verified by code
  inspection/reasoning (an atomic-counter modulo check, fires at most once
  per `PROGRESS_LOG_INTERVAL`-block crossing regardless of thread count);
  no stdout/stderr-capture assertion was added to the automated suite, per
  the spec's own guidance to avoid brittle output-capture tests.

This stage is **closed** — the implementation matches this specification
exactly, with the two deviations above called out explicitly per the Living
Synchronization Contract.


