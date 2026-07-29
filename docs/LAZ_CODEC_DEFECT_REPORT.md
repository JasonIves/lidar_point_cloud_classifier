# Defect Report — `wblidar` in-house LASzip encoder produces streams that reference decoders reject

**Status:** Open (upstream)
**Reported by:** LiDAR Point Cloud Classifier project (`lidar_point_cloud_classifier`)
**Component:** `whitebox_next_gen/crates/wblidar` — `src/laz/` (in-house LASzip writer)
**Date:** 2026-07-28
**Severity:** High — silently produces corrupt deliverables that appear to open successfully
**Downstream mitigation:** Stage 46 LAZ output integrity guard (see [`stages/stage-46-laz-output-integrity-guard.md`](stages/stage-46-laz-output-integrity-guard.md))

> [!NOTE]
> This document is written to be shared with the Whitebox Next Gen maintainer. It records
> only what we observed and what the code inspection suggests. The specific code branches
> named in §5 are **hypotheses ranked by suspicion**, not confirmed root causes — we did not
> instrument the encoder, and we are barred from modifying `whitebox_next_gen/` under our own
> project's greenfield rule.

---

## 1. Summary

`wblidar`'s LAZ writer emits compressed point streams that its own decoder round-trips
correctly but that **reference LASzip decoders reject part-way through the first chunk**.
The failure is silent at write time: the file has a valid header, a valid LASzip VLR, and a
plausible size. It only surfaces when a third-party tool tries to read it.

Writing the **same point data** to uncompressed `.las` from the same run produces a fully
valid, correctly-rendering file. The defect is therefore isolated to the compression path,
not to the point data, the header, or the classification logic that generated it.

## 2. Observed behaviour

Producing a classified tile of **4 080 355 points** and opening it in CloudCompare:

```text
laszip error: reading point 1596 of 4080355 total points
```

CloudCompare then *continues* and opens the file. The result renders as what the reporting
user described as "a 3D night sky" — a sparse scatter of points distributed through space
rather than a terrain surface. This is consistent with the decoder having produced valid
coordinates only for the ~1595 points decoded before the desync, with everything afterwards
being arithmetic-decoder noise.

### 2.1 The A/B test that isolates it

| Output | Written by | Result |
| --- | --- | --- |
| `tile.las` | `wblidar` uncompressed LAS writer | ✅ Opens correctly, terrain renders normally |
| `tile.laz` | `wblidar` in-house LASzip encoder | ❌ `laszip error: reading point 1596`, renders as sparse scatter |

Identical input file, identical classification run, identical point records — only the
output container differs. This is the strongest single piece of evidence: it excludes the
classifier, the header, the CRS/VLRs, and the scale/offset arithmetic from suspicion.

### 2.2 Why point 1596 matters

The default LASzip chunk size is **50 000 points**, so point 1596 is **inside chunk 0**.
This is important because:

- It is **not** a chunk-boundary bug. The chunk table, chunk offsets, and per-chunk
  re-initialisation are not implicated, because the stream never reaches the end of the
  first chunk.
- It **is** an intra-chunk desync: the encoder and a reference decoder disagree about the
  bit-level meaning of the stream after roughly 1595 successfully-decoded records.
- 1596 is far enough in to be **data-dependent**. A structural error (wrong field order,
  wrong context count, wrong initial state) would almost always fail at or near point 1.
  Failing at 1596 points to a **rare branch** in a predictive coder that is only taken once
  a particular data pattern appears.

## 3. Why the existing test suite does not catch this

This is the most actionable observation in the report, independent of the specific bug.

`crates/wblidar/src/laz/standard_point10_write.rs` (and its siblings) are validated by
**self-round-trip tests**: encode with `wblidar`, decode with `wblidar`, assert equality.
Any mistake that is **symmetric** — implemented identically in both the writer and the
reader — passes such a test forever, while producing output no other LASzip implementation
can read. The observed defect has exactly this signature.

Corroborating evidence already inside the repository:

- `crates/wblidar/README.md` concedes that external interoperability "still benefits from
  broader real-world fixture coverage across toolchains."
- `crates/wblidar/tests/standards_external_validation.rs` — the test that *would* catch
  this — is `#[ignore]`d, because it requires PDAL / `lasinfo` on the machine.
- `docs/internal/LAZ_IN_HOUSE_IMPLEMENTATION_PLAN.md` records a **previously fixed bug of
  the same family**: a "late-stream Point14 scan-angle mismatch" traced to the arithmetic
  coder. That precedent suggests the arithmetic coder / integer compressor layer is where
  this class of defect lives.

**Suggested remedy, regardless of root cause:** add a small number of fixture files
(a few thousand points each, including multi-return, multi-swath GPS time, and full
intensity/scan-angle range) whose *byte-exact LASzip encoding* is produced by the reference
`laszip` implementation and committed to the repository. Assert byte equality against
`wblidar`'s encoder output. A self-round-trip test cannot substitute for this.

## 4. Reproduction

Any sufficiently large real-world tile appears to trigger it; ours was an airborne
multi-swath coastal/marina tile (point format with GPS time, ~4.1 M points).

```bash
# 1. Write both containers from the same source points.
#    (Any wblidar path that writes LAZ will do; ours was our classifier's writer.)
wb_lidar_classify classify --input tile.las --model m.wbmodel \
    --blocks blocks.json --output tile_out.las
wb_lidar_classify classify --input tile.las --model m.wbmodel \
    --blocks blocks.json --output tile_out.laz --allow-laz

# 2. Validate each with a reference decoder.
lasinfo -i tile_out.las   # clean
lasinfo -i tile_out.laz   # reports a read error early in the stream
pdal info tile_out.laz    # same
```

Opening `tile_out.laz` in CloudCompare reproduces the `laszip error: reading point N`
message and the sparse-scatter rendering.

> [!NOTE]
> `--allow-laz` is our downstream flag for deliberately re-enabling the defective path;
> it exists only so this defect stays reproducible after our mitigation landed.

## 5. Candidate root causes (ranked hypotheses)

From reading `crates/wblidar/src/laz/standard_point10_write.rs`. All of these are
**rare, data-dependent branches**, which matches a first failure at point 1596.

### 5.1 GPS-time 4-slot state machine — *most likely*

The Point10 GPS-time coder maintains four rotating prediction slots:

```rust
last: usize,
next: usize,
last_gps_times:        [i64; 4],
last_gps_time_diffs:   [i32; 4],
multi_extreme_counters: [i32; 4],
```

Two behaviours here are unusually easy to get subtly wrong, and both are only exercised
when **multiple interleaved GPS time streams** are present — precisely what a multi-swath
tile looks like, and precisely why it might not fire until well into the file:

1. **Slot search with recursion.** The code selects a candidate slot and then re-enters
   itself:
   ```rust
   self.common.last = candidate_index;
   return self.compress_with(enc, gps_bits);
   ```
   If the reference implementation emits `LASZIP_GPS_TIME_MULTI_CODE_FULL` (or advances
   `next = (next + 1) & 3`) at a point where this implementation instead recurses — or vice
   versa — the two diverge by exactly one symbol and never resynchronise.
2. **The extreme-counter reset.** The `multi_extreme_counters[...] > 3` branch resets slot
   state. Whether the reset happens *before* or *after* the symbol is emitted is a one-line
   difference that is invisible to a self-round-trip test.

### 5.2 `k_bits` feed-forward context saturation

The dx/dy/dz integer compressors select their context from the bit-width of the previous
residual:

```rust
ic_dx = IntegerCompressor::new(32,  2, 8, 0);
ic_dy = IntegerCompressor::new(32, 22, 8, 0);
ic_z  = IntegerCompressor::new(32, 20, 8, 0);
```

with contexts of the form:

```rust
(n == 1) as u32 + if k_bits < 20 { u32_zero_bit(k_bits) } else { 20 }
(n == 1) as u32 + if k_bits < 18 { u32_zero_bit(k_bits) } else { 18 }
```

These saturating thresholds are only reached by **unusually large coordinate jumps** — e.g.
the first flight-line transition or the first large gap in a tile. If the saturation
boundary is off by one relative to the reference, the encoder and decoder pick different
probability models from that point onward.

### 5.3 Related parity defects found while investigating

These are independent of the desync but are real interop/parity bugs in the same area:

| # | Location | Issue |
| --- | --- | --- |
| 1 | `laz/` Point10 packer | `scan_angle_rank: point.scan_angle as i8` is an **unchecked wrapping cast**, whereas `las/writer.rs` **clamps** to ±90. The same source point therefore yields a *different* `scan_angle_rank` in `.las` than in `.laz`. |
| 2 | `laz/writer.rs` `effective_chunk_size()` | For Point14, chunk size is rescaled by `compression_level`, but the **LASzip VLR still advertises `config.chunk_size`**. Any non-default compression level therefore guarantees a decoder desync. Latent for the report above (default level, Point10), but it is a certain bug at other settings. |
| 3 | `laz/` Point10 packer | `classification: (point.classification & 0x1F) \| ((point.flags & 0x07) << 5)` silently truncates ASPRS classification codes above 31 rather than erroring or promoting the point format. |

## 6. Impact

- **Silent data corruption in a delivery format.** Nothing fails at write time; the file
  looks complete. The corruption is discovered by whoever opens it downstream, possibly
  much later.
- **LAZ is the default distribution format** for most public and commercial LiDAR, so any
  `wblidar` tool with LAZ output is affected.
- **COPC shares the codec.** A COPC payload is LASzip-chunked, so any future COPC writer
  built on this encoder inherits the defect.

## 7. What we did downstream (for context, not a fix)

Our project is bound by a greenfield rule that forbids us from editing `whitebox_next_gen/`,
so we could not repair the codec. We instead made the defective path unreachable by default:

- A `.laz` or `.copc` `--output` extension is **redirected to a sibling `.las` path** with a
  loud stderr warning explaining why, plus a suggested `laszip -i …` follow-up command.
- `--allow-laz` restores the old behaviour with a sterner warning, so the defect stays
  reproducible for whoever investigates it.
- Full specification: [`stages/stage-46-laz-output-integrity-guard.md`](stages/stage-46-laz-output-integrity-guard.md).

This mitigation is explicitly temporary. **We would like to revert it** once the encoder is
fixed, or once someone with write access to `wblidar` confirms which of §5's hypotheses is
correct.

## 8. Suggested next steps for the maintainer

1. **Add reference-fixture tests** (§3). This is worth doing whether or not the specific bug
   turns out to be in §5 — it converts a whole class of invisible failures into CI failures.
2. **Bisect the failing tile.** Encode the first *N* points for increasing *N* and decode
   each with reference `laszip`; the smallest failing *N* isolates the triggering record.
   With ~1595 good points, this converges in a handful of iterations.
3. **Diff the triggering record's field values** against the surrounding records. If GPS
   time is discontinuous there, §5.1 is confirmed; if a coordinate delta is unusually large,
   §5.2 is confirmed.
4. **Fix the three parity bugs in §5.3** independently — they are small, self-contained, and
   do not depend on resolving the desync.
5. **Un-`#[ignore]` `standards_external_validation.rs` in CI** with PDAL installed in the CI
   image, so external validation runs at least on one platform.

---

*We are happy to share the failing tile and the exact CLI invocation on request. Contact us
via the `lidar_point_cloud_classifier` project.*
