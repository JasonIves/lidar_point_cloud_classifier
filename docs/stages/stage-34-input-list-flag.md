# Stage 34 — `--input-list` Response File for `split-dataset`

## Goal

Stage 33 generalized `wb_lidar_train split-dataset` to accept multiple
`--input <dir>` flags (Option A: explicit, repeated flags), so that
per-file `preprocess-labeled` output directories can be merged into one
globally-stratified split.

In practice, at real-world dataset sizes (1500+ source `.laz` files), this
runs into a hard **platform limitation that has nothing to do with this
project's code**: every OS process-creation call has a maximum total
command-line length. On Windows, `CreateProcessW` is capped at
approximately 32,767 characters — this applies uniformly to every process
launch mechanism (`cmd.exe`, PowerShell's `&` call operator with an argument
array, `Start-Process`, etc.), because they all eventually construct one
flat command-line string for the OS. 1500 `--input <dir>` pairs, even with
moderately short paths (~60-90 chars each), assembles a command line of
100,000+ characters — **the process fails to launch at all**, independent
of anything `split-dataset`'s own argument parser does.

This stage adds a `--input-list <file>` flag: a simple, additive "response
file" mechanism (the same convention used by `rustc`'s `@file` syntax, GCC's
`@file`, and MSVC's `/OPTIONS:file`) that reads a newline-delimited list of
input directories from a text file instead of the command line, sidestepping
the OS command-line-length ceiling entirely regardless of platform or shell.

This does not replace repeated `--input` flags — both are supported
simultaneously and their results are concatenated (in the order
`--input-list` entries first, then explicit `--input` flags, matching
argument-parsing order), so small ad-hoc invocations can continue using
`--input` directly while large sweep/batch scripts switch to
`--input-list`.

---

## Inputs & Outputs

### CLI change: new `--input-list <file>` flag

```
wb_lidar_train split-dataset
    [--input      <dir>]   Directory produced by `preprocess-labeled`.
                            REPEATABLE. May be combined with --input-list.
    [--input-list <file>]  Text file containing one input directory path
                            per line. Blank lines and lines starting with
                            '#' are ignored (comment convention). REPEATABLE
                            (multiple --input-list files may be supplied;
                            their contents are concatenated in the order
                            given). At least one of --input / --input-list
                            (combined) must resolve to a non-empty input
                            directory list.
    --output      <dir>    Output directory; train/, val/, [test/]
                            subdirectories are created inside it
    ...                    (all other flags unchanged from Stage 33)
```

Example `--input-list` file (`inputs.txt`):
```
# ugs_lidar sweep — block_50_points_1024
D:/data6700_ext/data/labeled/ugs_lidar/block_50_points_1024/ot_BearRiver_000001
D:/data6700_ext/data/labeled/ugs_lidar/block_50_points_1024/ot_BearRiver_000002
D:/data6700_ext/data/labeled/ugs_lidar/block_50_points_1024/ot_BearRiver_000003
```

Called as:
```
wb_lidar_train split-dataset --input-list inputs.txt --output <dir> --val-split 0.20 --test-split 0.10 --move
```

This lets a PowerShell (or any shell) loop write one path per line to a
plain text file — trivially done with `Add-Content`/`Out-File` inside the
same loop that already invokes `preprocess-labeled` once per source file —
instead of accumulating a giant in-memory argument array that is later
splatted onto a single command line.

### Parsing rules

- Each non-blank, non-comment (`#`-prefixed) line is trimmed of leading/
  trailing whitespace and treated as one directory path.
- Blank lines (after trimming) are skipped.
- Lines whose first non-whitespace character is `#` are treated as comments
  and skipped (allows sweep scripts to self-document which parameter
  combination a given list corresponds to, as in the example above).
- No glob expansion, path validation, or existence checking happens at
  parse time — exactly as with repeated `--input` flags today, an invalid
  or missing directory simply fails later with the same
  `"cannot open <dir>/labeled_blocks.json"` error already produced by the
  existing manifest-loading step.
- File reads use `std::fs::read_to_string` (the file lists themselves are
  tiny — a few tens of KB even at 1500+ entries — so no streaming/chunking
  concern applies here, unlike the actual LiDAR point-cloud data this tool
  processes).

### Combination semantics

`inputs = input_list_entries_in_file_order ++ explicit_--input_flags_in_order`

Both sources contribute to the same flattened `Vec<PathBuf>` passed to
`three_way_spatial_split_multi()`/`materialize_split()` — there is no
special-casing between "list-sourced" and "flag-sourced" directories
downstream of argument parsing.

---

## Algorithm

### `src/cli/split_dataset_cmd.rs` changes

- New local `--input-list` accumulator: `Vec<PathBuf>` of list-file paths
  (repeatable, parsed the same way as `--input`).
- After the argument-parsing loop, for each `--input-list` file (in the
  order given), read it via `std::fs::read_to_string`, split on `\n`
  (`.lines()`, which already handles `\r\n` correctly), filter out blank/
  comment lines, and push each resulting path onto the **front** of the
  final `inputs: Vec<PathBuf>` list construction — implemented by building
  `inputs` as `list_derived_inputs` followed by the explicit `--input`
  accumulator, per the ordering rule above.
- The existing `if inputs.is_empty() { ... at least one --input is
  required ... }` check is updated to fire only when the **combined**
  list (list-derived + explicit) is empty, with an updated error message
  mentioning both `--input` and `--input-list`.
- `print_usage()` updated to document `--input-list`.
- No changes to `dataset_split.rs` — this is purely an argument-sourcing
  change; `three_way_spatial_split_multi()`/`materialize_split()` already
  operate on a plain `&[PathBuf]`/`Vec<PathBuf>` regardless of how the
  caller assembled it.

---

## Definition of Done (DoD)

1. `--input-list <file>` reads a newline-delimited directory list, skipping
   blank lines and `#`-comment lines.
2. `--input-list` and `--input` can be combined in a single invocation; all
   resulting directories are merged into one global split exactly as if
   they had all been passed as individual `--input` flags.
3. Multiple `--input-list` files may be supplied; their entries are
   concatenated in file order.
4. An invocation with zero total input directories (no `--input`, no
   `--input-list`, or all supplied `--input-list` files are empty/
   comments-only) produces a clear error.
5. A nonexistent `--input-list` file path produces a clear I/O error
   naming the file, not a panic.
6. End-to-end test: write a temp `--input-list` file referencing 2 synthetic
   `preprocess-labeled`-style directories (mirroring the Stage 33 multi-
   input merge test), invoke the `run()`-equivalent path, and confirm the
   materialized output is identical to the equivalent all-`--input`-flags
   invocation.
7. `cargo build --all-targets --all-features` — zero errors.
8. `cargo clippy --all-targets --all-features -- -D warnings` — zero
   warnings.
9. `cargo clippy --all-targets --features training -- -D warnings` — zero
   warnings.
10. `cargo test --all-features` and `cargo test --features training` — all
    tests (existing + new) pass, identical results across both feature
    variants.
11. `cargo fmt -- --check` — clean.
12. This document accurately reflects the landed implementation (see
    "Implementation Status" below).

---

## Implementation Status

**Complete.** All DoD items implemented and verified.

### Files touched

- `src/cli/split_dataset_cmd.rs`:
  - Renamed the `--input`-only accumulator to `explicit_inputs: Vec<PathBuf>`
    and added a new `input_list_files: Vec<PathBuf>` accumulator, with a new
    `"--input-list"` match arm in the argument-parsing loop (parsed
    identically to `--input`, just collected into the new vector).
  - Added a new `resolve_inputs(input_list_files, explicit_inputs) ->
    Result<Vec<PathBuf>>` helper (kept separate from `run()` to stay under
    Clippy's `too_many_lines` threshold). It reads each `--input-list` file
    via `std::fs::read_to_string` (returning a clear
    `ClassifierError::Pipeline` naming the file on I/O failure — never
    panics), splits on `.lines()` (correctly handles both `\n` and `\r\n`),
    trims each line, skips blank lines and lines starting with `#`, and
    appends the resulting paths (in file order, across all `--input-list`
    files in the order given) followed by `explicit_inputs` (in flag
    order) into the final combined list. Returns an error mentioning both
    `--input` and `--input-list` if the combined result is empty.
  - `run()` now calls `resolve_inputs(&input_list_files, explicit_inputs)?`
    immediately after the argument-parsing loop; all downstream logic
    (manifest loading, `three_way_spatial_split_multi`, `materialize_split`)
    is completely unchanged and operates on the resulting `inputs:
    Vec<PathBuf>` exactly as it did under Stage 33.
  - `print_usage()` updated to document `--input-list <file>` (repeatable,
    response-file semantics, combinable with `--input`, explains the
    Windows command-line-length motivation).
  - Six new unit tests added (see below).
- `docs/stages/stage-34-input-list-flag.md` (this file): authored from
  scratch, now with this Implementation Status section filled in.

### New tests added

1. `test_run_rejects_nonexistent_input_list_file` — a `--input-list` file
   path that does not exist produces a clear `Err`, not a panic (DoD 5).
2. `test_run_rejects_empty_input_list_and_no_input` — a list file
   containing only a comment/blank lines, with no `--input` flags, produces
   a clear `Err` (DoD 4).
3. `test_input_list_parsing_skips_blank_and_comment_lines` — directly
   exercises the parsing rules (trim, skip blank, skip `#`-comment) against
   a list file mixing comments/blank lines with two real paths, confirming
   exact ordering (DoD 1).
4. `test_multiple_input_list_files_concatenate_in_order` — two separate
   list files' contents are concatenated in the order the files are
   supplied (DoD 3).
5. `test_input_list_end_to_end_materializes_identically_to_input_flags` —
   full `run()` invocation via `--input-list` against two synthetic
   `preprocess-labeled`-style directories, confirming materialized output
   (12 total blocks, no filename collisions, loadable via
   `LabeledBlockDataset::load_presplit`) exactly mirrors the equivalent
   Stage 33 all-`--input`-flags test (DoD 6).
6. `test_input_list_combined_with_explicit_input_flag` — one directory
   supplied via `--input-list`, another via an explicit `--input` flag in
   the same invocation; confirms both are merged into a single global split
   (DoD 2).

### Verification results

- `cargo build --all-targets --all-features` — clean, 0 errors (DoD 7).
- `cargo clippy --all-targets --all-features -- -D warnings` — clean, 0
  warnings (DoD 8). (Required extracting `resolve_inputs()` as a standalone
  function to keep `run()` under the `clippy::too_many_lines` threshold.)
- `cargo clippy --all-targets --features training -- -D warnings` — clean,
  0 warnings (DoD 9).
- `cargo test --all-features` — 125 lib tests + 1 integration test, all
  passing (DoD 10).
- `cargo test --features training` — 125 lib tests + 1 integration test,
  all passing, identical results to `--all-features` (DoD 10).
- `cargo fmt -- --check` — clean (DoD 11).
- This document reflects the landed implementation (DoD 12).

### Definition of Done checklist

1. ✅ `--input-list <file>` reads a newline-delimited directory list,
   skipping blank/`#`-comment lines.
2. ✅ `--input-list` and `--input` combine into one global split.
3. ✅ Multiple `--input-list` files concatenate in file order.
4. ✅ Zero total input directories produces a clear error.
5. ✅ A nonexistent `--input-list` file produces a clear I/O error, not a
   panic.
6. ✅ End-to-end test confirms `--input-list`-sourced materialization is
   equivalent to the all-`--input`-flags invocation.
7. ✅ `cargo build --all-targets --all-features` clean.
8. ✅ `cargo clippy --all-targets --all-features -- -D warnings` clean.
9. ✅ `cargo clippy --all-targets --features training -- -D warnings`
   clean.
10. ✅ `cargo test --all-features` and `cargo test --features training`
    pass identically.
11. ✅ `cargo fmt -- --check` clean.
12. ✅ This document reflects the landed implementation.


