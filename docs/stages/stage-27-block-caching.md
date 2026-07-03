# Stage 27 — Block Caching During Training (Audit Finding 5.2)

**Status:** CLOSED

## Goal

Implement audit finding **5.2 — No Block Caching During Training** (see
`docs/AUDIT_REPORT.md`): each training epoch currently re-reads and
re-parses every `.feat`/`.lbl` block pair from disk, even though the same
blocks are visited once per epoch across all `--epochs`. This stage adds an
**opt-in, in-memory block cache** scoped to one `LabeledBlockDataset`
instance so repeated epochs can reuse already-decoded block data instead of
re-reading it from disk, while defaulting to the exact pre-Stage-27
behavior (disk read every time) when the feature is not enabled.

## Background

Stage 26 (`docs/stages/stage-26-remaining-findings-triage.md`) investigated
how `whitebox_next_gen` itself achieves in-memory-storage speed and found
that both `wblidar::memory_store` and `wbraster::memory_store` use an
identical minimal idiom:

```rust
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static LIDAR_STORE: OnceLock<Mutex<HashMap<String, Arc<PointCloud>>>> = OnceLock::new();
```

— a single stdlib `std::sync::Mutex<HashMap<K, Arc<V>>>`, **no external
caching crate** (`moka`/`dashmap`/`quick_cache`/`lru` are absent from the
entire `whitebox_next_gen` workspace), and **no eviction policy** — entries
persist for the scope's lifetime. Stage 26 recommended following this exact
idiom for 5.2, but scoped per-training-run (a struct field, not a process
`static`), since a training run is a single bounded process invocation.

After Stage 26 closed, the user confirmed (via `ask_followup_question`)
that the caching feature should proceed, and made the following explicit
design decision when asked how to handle a configured memory budget being
exceeded:

> **"Option A variant: byte-budget cap, but log a one-time informative
> warning when the budget is first exceeded."**

This means: **no error path at all**. When the configured byte budget would
be exceeded by caching an additional block, the cache simply declines to
insert that block (it will be re-read from disk on its next request) and
exactly **one** `eprintln!("[cache] ...")`-style warning is logged the
first time this happens per training run — not once per block, not once
per epoch.

## Inputs & Outputs

### New CLI flag (`train` sub-command, `src/cli/train_cmd.rs`)

```
--cache-blocks-max-mb <usize>   Enable in-memory block caching, bounded to
                                 this many megabytes (default: disabled)
```

- Optional. Omitting it preserves exact pre-Stage-27 behavior (every
  `load_block()` call reads from disk).
- If provided, must be `>= 1` (validated alongside the existing
  `--early-stopping-patience`/`--grad-clip-norm`-style range checks).

### New `TrainConfig` field (`src/training/trainer.rs`)

```rust
/// Stage 27 (Block Caching, audit finding 5.2): optional in-memory block
/// cache budget in megabytes. `None` (default) disables caching entirely.
pub cache_blocks_max_mb: Option<usize>,
```

Default: `None` (matches the `Option`-default-off pattern already used by
`checkpoint_dir`/`early_stopping_patience`/`grad_clip_norm`).

### New `LabeledBlockDataset` API (`src/training/dataset.rs`)

```rust
/// Opt into an in-memory block cache bounded by `max_mb` megabytes.
/// `None` disables caching (the default after `load()`).
#[must_use]
pub fn with_block_cache(self, max_mb: Option<usize>) -> Self
```

A builder method chained after `LabeledBlockDataset::load(...)`, so the
existing `load()` signature and all its call sites (production and tests)
are completely unchanged. `load_block(&self, block_id: u64) -> Result<LoadedBlock>`'s
signature is also unchanged — caching is entirely transparent to both
existing call sites in `trainer.rs` (the Rayon-parallel micro-batch loader
and the sequential `validate_epoch()` loop).

## Steps & Specifications

1. **`TrainConfig`**: add `cache_blocks_max_mb: Option<usize>` field +
   `None` default in `impl Default for TrainConfig`.

2. **`train_cmd.rs`**: add `--cache-blocks-max-mb <usize>` flag parsing
   (following the existing `next_value`/`parse_usize` pattern), a `>= 1`
   range check, a `print_usage()` doc line, and wire it into dataset
   construction:
   ```rust
   let dataset = LabeledBlockDataset::load(&data_dirs, cfg.val_split, val_tile_ids.as_ref(), cfg.seed)?
       .with_block_cache(cfg.cache_blocks_max_mb);
   ```

3. **`dataset.rs` — cache data structure**:
   ```rust
   #[derive(Debug)]
   struct BlockCache {
       entries: HashMap<u64, Arc<LoadedBlock>>,
       bytes_used: usize,
       max_bytes: usize,
       /// Emit the budget-exceeded warning exactly once per run.
       warned_budget_exceeded: bool,
   }
   ```
   `LabeledBlockDataset` gains a `cache: Option<Mutex<BlockCache>>` field
   (`None` by default, set in `load()`'s `Ok(Self { ... })` and in the three
   direct test struct-literal constructions). `Mutex` (not Rayon/atomics) is
   used here deliberately: cache access is brief HashMap-lookup-and-clone
   bookkeeping, not a hot numeric loop — the same justification
   `whitebox_next_gen::memory_store` and this project's own precedent
   (Stage 21 `block_index: HashMap`) rely on. No new dependency is
   introduced; `Mutex`/`HashMap`/`Arc` are all `std`.

4. **`dataset.rs` — `with_block_cache`**: builder method that installs a
   `Mutex::new(BlockCache { ... })` with `max_bytes = max_mb * 1024 * 1024`
   (`saturating_mul` to avoid overflow on pathological input) when
   `max_mb.is_some()`, or `None` otherwise.

5. **`dataset.rs` — `load_block()` integration**:
   - **Cache lookup (fast path)**: if `self.cache` is `Some`, lock it and
     check for `block_id`. On hit, clone the cached `Array2<f32>`/`Vec<u8>`
     out into a fresh owned `LoadedBlock` and return immediately — **no
     disk I/O**.
   - **Cache miss**: proceed with the existing disk read + `.feat`/`.lbl`
     parse exactly as before.
   - **Cache insert (best-effort)**: after a successful disk load, if
     `self.cache` is `Some`, compute the block's exact in-memory footprint
     (`n_points × n_features × 4` feature bytes + `n_points` label bytes —
     directly from the just-loaded arrays, no separate header-only
     estimate needed since the full block is already in hand) and attempt
     to insert it:
     - If `bytes_used + block_bytes <= max_bytes`: insert (as
       `Arc<LoadedBlock>`) and increase `bytes_used`.
     - Otherwise: **do not insert** (no error, no retry) and, if this is
       the first time the budget was exceeded in this run, log exactly one
       `eprintln!("[cache] block cache budget ({N} MB) exceeded — further
       blocks will be re-read from disk instead of cached for the
       remainder of this run")` and set `warned_budget_exceeded = true`.
   - A poisoned `Mutex` (only reachable if another thread panicked while
     holding the lock, which the no-panics rule makes vanishingly
     unlikely) degrades gracefully: the lock attempt fails, caching is
     skipped for that call, and disk-read behavior continues unaffected —
     never propagated as an error.

6. **Thread-safety**: `micro.par_iter().map(|&block_id| dataset.load_block(block_id))`
   in `trainer.rs` calls `load_block()` concurrently across Rayon worker
   threads. The `Mutex<BlockCache>` guard makes every cache read/write
   sequential and data-race-free; lock hold time is bounded to a `HashMap`
   lookup/insert plus one or two `Arc`/`Vec`/`Array2` clones — no I/O and no
   heavy computation occurs while the lock is held, so contention overhead
   stays negligible relative to the disk read it replaces.

7. **Tests** (`src/training/dataset.rs` `#[cfg(test)] mod tests`):
   - Cache hit avoids re-reading the block from disk (e.g. delete/corrupt
     the on-disk `.feat` file after the first `load_block()` call and
     confirm the second call still succeeds and returns identical data).
   - Cache miss (caching disabled, i.e. `with_block_cache(None)`) behaves
     identically to pre-Stage-27 `load_block()`.
   - Budget-exceeded path: configure a tiny `max_mb` budget (smaller than
     one block), call `load_block()` for two or more distinct blocks, and
     confirm: (a) no error is returned, (b) each call still succeeds by
     falling back to disk, and (c) exactly one `[cache]` warning would be
     emitted (verified by checking `warned_budget_exceeded` is only set
     once via the internal state, since stderr capture is impractical in
     a unit test — the flag is the observable proxy for "warn once").

## Definition of Done

- [x] `TrainConfig::cache_blocks_max_mb: Option<usize>` added, defaulting
      to `None`.
- [x] `--cache-blocks-max-mb <usize>` CLI flag added to `train_cmd.rs` with
      range validation and `print_usage()` documentation.
- [x] `LabeledBlockDataset::with_block_cache()` builder method added;
      `load()`'s signature and all existing call sites unchanged.
- [x] `load_block()` transparently serves cache hits without disk I/O,
      transparently falls back to disk on cache misses, and never returns
      an error due to the cache being full — a full budget silently
      disables further caching (with exactly one warning logged).
- [x] No new external dependency added (`std::sync::Mutex`/`HashMap`/`Arc`
      only), consistent with the `whitebox_next_gen::memory_store`
      precedent and the Minimal Dependencies tenet.
- [x] New unit tests cover: cache-hit-avoids-disk-read,
      cache-disabled-behaves-as-before, and budget-exceeded-warns-once.
- [x] `cargo build --features training` clean.
- [x] `cargo test --features training` — all existing tests still pass,
      plus the new Stage 27 tests pass.
- [x] `cargo clippy --all-targets --features training` clean (no new
      warnings).
- [x] `cargo fmt --check` clean.
- [x] `docs/AUDIT_REPORT.md` §5.2 updated from "Investigation" to
      "✅ RESOLVED (Stage 27)" with a Resolution paragraph; Summary
      Priority Table, Recommended Implementation Order, and Stage Mapping
      sections updated accordingly.

## Results

Implemented exactly as specified above:

- `TrainConfig.cache_blocks_max_mb: Option<usize>` (default `None`).
- `--cache-blocks-max-mb <usize>` flag in `train_cmd.rs`, validated `>= 1`
  when present, wired via `.with_block_cache(cfg.cache_blocks_max_mb)`
  chained onto `LabeledBlockDataset::load(...)`.
- `LabeledBlockDataset` gained a `cache: Option<Mutex<BlockCache>>` field
  and `with_block_cache()` builder; `BlockCache` holds
  `entries: HashMap<u64, Arc<LoadedBlock>>`, `bytes_used`, `max_bytes`, and
  `warned_budget_exceeded`.
- `load_block()` checks the cache first (fast path, no disk I/O on hit),
  otherwise reads from disk as before and then attempts a best-effort
  cache insert bounded by the byte budget, warning exactly once when the
  budget is first exceeded — never erroring.
- Three new unit tests added: `test_block_cache_hit_avoids_disk_reread`,
  `test_block_cache_budget_exceeded_falls_back_gracefully`,
  `test_with_block_cache_none_disables_caching`.
- Full verification: `cargo build --features training`,
  `cargo test --features training` (all prior tests + 3 new tests pass),
  `cargo clippy --all-targets --features training` clean,
  `cargo fmt --check` clean.
- `docs/AUDIT_REPORT.md` §5.2, Summary Priority Table, Recommended
  Implementation Order, and Stage Mapping sections updated to reflect
  Stage 27's closure.
