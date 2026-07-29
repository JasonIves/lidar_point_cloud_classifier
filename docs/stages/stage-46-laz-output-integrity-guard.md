# Stage 46 — LAZ Output Integrity Guard

**Status:** Implemented
**Date:** 2026-07-28
**Depends on:** Stage 44 (classify-time prediction fusion), Stage 02 (`open_writer` format dispatch)

---

## The Goal

Prevent the `classify` sub-command from silently emitting **corrupt LAZ files**.

A classified marina tile written as `.laz` was rejected by CloudCompare/LASlib with:

```
laszip error: reading point 1596 of 4080355 total points
```

The file "loaded" but rendered as a sparse 3-D dusting of points ("night sky"): only the
~1595 points decoded before the failure carry valid coordinates, and everything after is
garbage. The identical classification run written as `.las` opens cleanly and is verified
correct.

The defect is **not** in this project. It is in the in-house LASzip encoder inside
`whitebox_next_gen/crates/wblidar/src/laz/`, which this repository is **prohibited from
modifying** (AGENTS.md — "Greenfield Only"). See
[`../LAZ_CODEC_DEFECT_REPORT.md`](../LAZ_CODEC_DEFECT_REPORT.md) for the full upstream
analysis and repro.

This stage therefore delivers the only remedy available to us inside our own boundary:
**make the broken path unreachable by default, and impossible to hit by accident.**

### Explicit non-goals

This stage deliberately stops at the guard. It does **not**:

- Fix, patch, wrap, vendor, or work around the LASzip codec itself.
- Add a post-write output verification pass (an independent decoder oracle would be
  required — `wblidar`'s own reader agrees with `wblidar`'s own writer and is therefore
  useless as a check). Deferred; not scheduled.
- Add any new runtime dependency.

---

## Inputs & Outputs

### CLI surface (`wb_lidar_classify classify`)

| Argument | Type | Default | Behaviour change |
|---|---|---|---|
| `--output <path>` | Path | *(required)* | **Changed.** A `.laz` or `.copc` extension is now **redirected to a sibling `.las` path** with a loud stderr warning. A `.las` extension is unaffected. |
| `--allow-laz` | flag | `false` (off) | **New.** Escape hatch. Honours a `.laz` extension as written, after printing the same warning plus an explicit corruption caveat. Provided so the defect stays reproducible for upstream debugging and so a future fixed `wblidar` needs no code change here to be exercised. |

### Redirect rule

| Requested `--output` | `--allow-laz` absent | `--allow-laz` present |
|---|---|---|
| `out.las` | `out.las` (silent) | `out.las` (silent) |
| `out.laz` | `out.las` + warning | `out.laz` + warning + caveat |
| `out.copc` / `out.copc.laz` | `out.las` + warning | error (COPC write unsupported by `wblidar`) |
| `out.txt` / no extension | error (unchanged) | error (unchanged) |

Notes:

- Redirection replaces the **final** extension only, via `Path::with_extension("las")`.
  `area51_classified.laz` → `area51_classified.las`.
- COPC is affected because the COPC payload *is* LASzip-chunked. `wblidar` exposes no COPC
  writer, so `--allow-laz` cannot rescue it; it stays an error, and without the flag it is
  redirected to `.las` rather than failing outright (strictly more useful).
- The redirect is reported through the existing `[classify] output: …` line, which already
  prints the **actual** path written, so the final log line never lies.

### Warning text (stderr, exact)

```text
================================ WARNING ================================
Compressed LAZ output is DISABLED because the LAZ encoder in this build of
Whitebox Next Gen (wblidar) produces files that reference LASzip decoders
(LAStools, laszip, CloudCompare, PDAL) reject mid-stream. Affected files
appear to load but render as a sparse scatter of points; all coordinates
after the failure point are garbage.

  requested: <requested path>
  writing:   <actual path>

Your classified output is being written as uncompressed LAS instead, which
is fully valid. To obtain a .laz, compress the result with a reference
implementation, e.g.:

  laszip -i "<actual path>"

See docs/LAZ_CODEC_DEFECT_REPORT.md. Override with --allow-laz (NOT
recommended: the output will very likely be unreadable).
=========================================================================
```

With `--allow-laz`, the final two lines are replaced by:

```text
--allow-laz was supplied: writing LAZ anyway. THE OUTPUT IS LIKELY CORRUPT
AND UNREADABLE BY OTHER TOOLS. Do not use it for analysis or delivery.
```

### Data outputs

Unchanged. Point records, classification substitution, VLR/CRS handling, and header
generation are all untouched by this stage. Only the *container and path selection*
changes.

---

## Steps & Specifications

### 1. New module: `src/output/format_guard.rs`

Self-contained and pure so it is unit-testable without touching the filesystem.

```rust
/// Outcome of applying the Stage 46 LAZ guard to a requested output path.
pub struct ResolvedOutput {
    /// The path that will actually be written.
    pub path: PathBuf,
    /// True when the requested path's extension was overridden.
    pub redirected: bool,
}

/// Apply the Stage 46 guard. Emits the warning to stderr as a side effect.
pub fn resolve_output_path(requested: &Path, allow_laz: bool) -> Result<ResolvedOutput>;
```

Helper `fn is_compressed_output(path: &Path) -> bool` matches a case-insensitive final
extension of `laz` or `copc`. Extension comparison is lowercased so `OUT.LAZ` is caught
(Windows paths are case-insensitive; `LidarFormat::detect` already lowercases, so this
keeps the guard consistent with dispatch).

`resolve_output_path` never panics and never touches the filesystem — it is pure string/path
logic plus an `eprintln!`. It returns `Ok` for `.las` and for redirected paths; it returns
the existing `ClassifierError::Pipeline` for an unsupported COPC + `--allow-laz` combination
so the message surfaces at argument-resolution time rather than after inference has run.

### 2. Wire into `src/cli/classify_cmd.rs`

- Add `allow_laz: bool` to `ClassifyConfig`; parse the valueless `--allow-laz` flag.
- Call `resolve_output_path(&cfg.output, cfg.allow_laz)` **before** model loading, so a
  user who mistyped the extension learns immediately instead of after a multi-minute
  inference run. Pass `resolved.path` to `write_classified`.
- Update `print_help()` and the module doc-comment usage block.

### 3. Leave `las_writer.rs` alone

`open_writer` keeps its `LidarFormat::Laz` arm intact — that arm is what `--allow-laz`
reaches, and removing it would make the escape hatch impossible. The guard sits strictly
upstream of the writer, at the CLI boundary. This keeps the change additive and keeps
`write_classified` usable as a library function with an explicit caller-chosen path.

### 4. Documentation synchronisation

Per AGENTS.md's Living Synchronization Contract, the same change lands in all three places
in the same commit:

- **This file** — the specification of record.
- `docs/LAZ_CODEC_DEFECT_REPORT.md` — new; the shareable upstream analysis.
- `docs/user/user_guide.md` — §4.2 flag tables and workflow, §9 Classified LAS/LAZ, §11
  File Format Compatibility, Appendix A flag reference.
- `PROJECT_SPEC.md` §4.1 "Recorded Implementation Deviations (Stage 46)" — a new subsection
  recording both deviations (see below).

### 5. Recorded deviation from `PROJECT_SPEC.md` §4

PROJECT_SPEC §4 "Output Serialization" states:

> **Format Flexibility:** Default to the input file format (e.g., maintaining `.laz`
> compression if input was `.laz`) …

Stage 46 **knowingly deviates**. Honouring that clause for a `.laz` input means defaulting
straight into the broken encoder and shipping corrupt deliverables. Correctness outranks
container fidelity, so `.las` wins until `wblidar`'s codec is fixed. A new subsection
**`PROJECT_SPEC.md` §4.1 "Recorded Implementation Deviations (Stage 46)"** records the
deviation at the top-level spec rather than letting it lurk only here, so a future auditor
reading §4 in isolation cannot mistake the requirement for satisfied. When the upstream fix
lands, §4.1 and this stage are reverted together and the original clause resumes unqualified.

A second, pre-existing deviation in the same section is also worth noting here because this
stage documents the output path: **"Lossless Preservation: … preserving all original VLRs"**
is *not* currently honoured — `infer_writer_config` leaves `WriterConfig::vlrs` at default,
so source VLRs are dropped and CRS is re-synthesized as a fresh OGC WKT VLR. That is a
separate defect in our own code, fixable in greenfield, and is **out of scope for Stage 46**.
It is recorded as the second bullet of §4.1 (and in the user guide's §9 note) so it is not
lost, but no Stage 46 code addresses it.

---

## Definition of Done

- [x] `resolve_output_path` redirects `.laz` → `.las` and returns `redirected: true`.
- [x] `resolve_output_path` leaves `.las` untouched and returns `redirected: false`, with
      no warning emitted.
- [x] `.copc` is redirected to `.las` without the flag, and errors with it.
- [x] Case-insensitive: `OUT.LAZ` is caught.
- [x] `--allow-laz` honours `.laz` and still warns.
- [x] Redirect happens before model/manifest loading (fail-fast ordering).
- [x] `--allow-laz` appears in `classify --help`.
- [x] Unit tests cover every row of the redirect-rule table, including the
      multi-dot `area51.classified.laz` → `area51.classified.las` case.
- [x] Existing `write_classified` tests are unaffected (guard is upstream of the writer).
- [x] `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` all clean.
- [x] No new dependencies in `Cargo.toml`.
- [x] User guide, defect report, and `PROJECT_SPEC.md` §4.1 addendum all land with the code.

---

## Exit Criteria for This Stage

Stage 46 is **closed** but explicitly **temporary**. It should be reverted — restoring
`.laz` as a first-class output and deleting the guard, the flag, and `PROJECT_SPEC.md` §4.1
addendum — once `whitebox_next_gen` ships a `wblidar` whose LAZ output round-trips through
a reference LASzip implementation. Until then this guard is the load-bearing reason our
classified outputs are trustworthy.
