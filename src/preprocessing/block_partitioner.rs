#![allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
//! 2-D grid block partitioner with memory-pressure spill/merge support.
//!
//! Points arriving from any number of streaming chunks are routed into a
//! `HashMap<(i32,i32), Vec<(u64, LitePoint)>>` accumulator that stays open
//! for the entire stream duration.  No block is finalised until `finalize()`
//! is called after EOF, so chunk-spanning blocks are handled correctly.
//!
//! Each point is paired with its **original-file point index** (0-based,
//! matching the order points were streamed from the input LAS/LAZ/COPC file,
//! which is also the ordering used by `wbtools_oss::LidarEigenvalueFeaturesTool`'s
//! own `point_num` field). This index survives the full round trip through
//! spill files so that, once loaded, a block's points can be joined against a
//! precomputed per-point eigenvalue-feature table by index (Stage 30,
//! point-index-join extension — a prerequisite for Step 5e).
//!
//! When the total in-flight buffer exceeds `SPILL_HIGH_WATER_BYTES` the
//! largest cells are spilled to temporary `.spill` files (raw `(u64, LitePoint)`
//! bytes). After the stream ends, `finalize()` merges each block's spill
//! files with any remaining in-memory data before returning complete `Block`
//! structs.
//!
//! **Stage 31 note**: this module operates entirely on the lean, project-local
//! [`LitePoint`] struct rather than `wblidar::PointRecord` — the caller
//! (`pipeline.rs::stream_points()`) converts each point exactly once, at
//! streaming-ingest time, before calling [`BlockPartitioner::add_point`]. See
//! `docs/stages/stage-31-lean-point-record.md`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{ClassifierError, Result};
use crate::preprocessing::lite_point::LitePoint;
use crate::preprocessing::SPILL_HIGH_WATER_BYTES;

/// A raw (unprocessed) point data for a single 2-D grid cell.
#[derive(Debug)]
pub struct Block {
    /// Grid column index.
    pub col: i32,
    /// Grid row index.
    pub row: i32,
    /// Combined block ID used in file names: `row * grid_cols + col`.
    pub id: u64,
    /// X origin of this block (west edge).
    pub origin_x: f64,
    /// Y origin of this block (south edge).
    pub origin_y: f64,
    /// All points belonging to this block (post-merge, pre-sampling).
    pub points: Vec<LitePoint>,
    /// Original-file point index for each entry in `points` (same length,
    /// same order — `point_indices[i]` is the 0-based index of `points[i]`
    /// in the original input stream, matching the ordering used by
    /// `wbtools_oss::LidarEigenvalueFeaturesTool`'s own `point_num` field).
    ///
    /// Added in Stage 30 (point-index-join extension) so per-block feature
    /// extraction can look up precomputed eigenvalue rows for each point
    /// after a round trip through spill files, without relying on point
    /// order being preserved (which it is not, once spilling/merging or
    /// multiple spill files per cell are involved).
    pub point_indices: Vec<u64>,
}

/// Lightweight block descriptor returned by [`BlockPartitioner::finalize_stubs`].
///
/// Holds only metadata and on-disk spill-file paths — **no point data in RAM**.
/// Call [`BlockStub::load`] inside a processing closure to retrieve the full
/// point set on-demand and drop it as soon as processing is complete.
///
/// This is the production path for large files. Peak memory during parallel
/// processing is bounded to `(Rayon thread count) × (largest single block size)`
/// rather than the entire dataset.
#[derive(Debug)]
pub struct BlockStub {
    pub col: i32,
    pub row: i32,
    pub id: u64,
    pub origin_x: f64,
    pub origin_y: f64,
    /// Total point count across all spill files (derived from file sizes,
    /// not by reading data — this is cheap and used for density filtering).
    pub point_count: usize,
    spill_paths: Vec<PathBuf>,
}

impl BlockStub {
    /// Read all spill files for this block into a [`Block`], deleting each
    /// file immediately after reading.
    ///
    /// This is intended to be called once per block, inside a Rayon parallel
    /// closure, so the loaded `Vec<LitePoint>` is dropped at the end of the
    /// closure scope rather than accumulating across all blocks.
    ///
    /// # Errors
    /// Returns [`ClassifierError::SpillCorrupt`] if any spill file is unreadable.
    pub fn load(self) -> Result<Block> {
        let mut points = Vec::with_capacity(self.point_count);
        let mut point_indices = Vec::with_capacity(self.point_count);
        for path in &self.spill_paths {
            let spilled = read_spill_file(path)?;
            for (idx, pt) in spilled {
                point_indices.push(idx);
                points.push(pt);
            }
            let _ = fs::remove_file(path);
        }
        Ok(Block {
            col: self.col,
            row: self.row,
            id: self.id,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            points,
            point_indices,
        })
    }

    /// Read all spill files for this block **without deleting them**,
    /// preserving each point's original-file stream index.
    ///
    /// Used by the Stage 45 halo sampler: halo rows are joined against the
    /// whole-file eigenvalue pre-pass table by original index (the same
    /// join used for canonical points in `pipeline.rs` Step 7d), so the
    /// indices must survive this read.  The neighbour stub retains
    /// ownership of its spill files and will delete them when its own
    /// `load()` is called in the parallel closure.
    ///
    /// # Errors
    /// Returns [`ClassifierError::SpillCorrupt`] if any spill file is unreadable.
    pub fn read_points_indexed(&self) -> Result<Vec<(u64, LitePoint)>> {
        let mut out = Vec::with_capacity(self.point_count);
        for path in &self.spill_paths {
            let spilled = read_spill_file(path)?;
            out.extend(spilled);
        }
        Ok(out)
    }

    /// Read all spill files for this block **without deleting them**.
    ///
    /// Used by the Stage 08 border-point loader to read a neighbour block's
    /// points as context for feature extraction.  The neighbour stub retains
    /// ownership of its spill files and will delete them when its own `load()`
    /// is called in the parallel closure.
    ///
    /// Border points are context-only (never resampled or written to output),
    /// so original-file point indices are discarded here — only `load()`
    /// (used for a block's own canonical points) needs to retain them.
    ///
    /// # Errors
    /// Returns [`ClassifierError::SpillCorrupt`] if any spill file is unreadable.
    pub fn read_points(&self) -> Result<Vec<LitePoint>> {
        let mut points = Vec::with_capacity(self.point_count);
        for path in &self.spill_paths {
            let spilled = read_spill_file(path)?;
            points.extend(spilled.into_iter().map(|(_, pt)| pt));
        }
        Ok(points)
    }
}

/// Accumulates `LitePoint`s into 2-D grid cells with an optional spill path.
pub struct BlockPartitioner {
    /// In-memory per-cell accumulators. Each entry pairs a point with its
    /// original-file index (see `Block::point_indices` doc for rationale).
    cells: HashMap<(i32, i32), Vec<(u64, LitePoint)>>,
    /// Spill files written under memory pressure: key → list of spill paths.
    spill_paths: HashMap<(i32, i32), Vec<PathBuf>>,
    /// Directory used for temporary `.spill` files.
    spill_dir: PathBuf,
    /// Grid geometry.
    x_min: f64,
    y_min: f64,
    block_size: f64,
    /// Number of grid columns (used to compute block ID).
    grid_cols: i32,
    /// Approximate bytes currently held in the in-memory cells.
    buffered_bytes: usize,
}

/// Bytes written to a spill file per point.
/// Layout: `point_index`(8, u64) `x`(8) `y`(8) `z`(8) `intensity`(2)
///         `classification`(1) `return_number`(1) `number_of_returns`(1)
///         `scan_angle`(2) = 39 bytes
const PT_BYTES: usize = 39;

/// Bytes of in-memory accounting per buffered point: the `LitePoint` itself
/// plus its paired `u64` original-file index.
const PT_ACCOUNTING_BYTES: usize = std::mem::size_of::<u64>();

impl BlockPartitioner {
    /// Create a new partitioner.
    ///
    /// * `x_min` / `y_min` — south-west corner of the point-cloud bounding box.
    /// * `x_max` / `y_max` — north-east corner.
    /// * `block_size`       — cell edge length in projection units.
    /// * `spill_dir`        — directory for temporary `.spill` files.
    pub fn new(
        x_min: f64,
        y_min: f64,
        x_max: f64,
        _y_max: f64,
        block_size: f64,
        spill_dir: impl AsRef<Path>,
    ) -> Self {
        // Warn about any stale `.spill` files left by a prior interrupted run.
        // They are not cleaned up automatically (the user may want to inspect
        // them), but making them visible prevents silent disk-space accumulation.
        if let Ok(entries) = std::fs::read_dir(spill_dir.as_ref()) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "spill") {
                    eprintln!(
                        "[warn] stale spill file found (prior interrupted run?): {}",
                        entry.path().display()
                    );
                }
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // f64::ceil can only truncate to a valid i32 for any plausible LiDAR extent.
        let cols = ((x_max - x_min) / block_size).ceil() as i32;
        let cols = cols.max(1);
        Self {
            cells: HashMap::new(),
            spill_paths: HashMap::new(),
            spill_dir: spill_dir.as_ref().to_path_buf(),
            x_min,
            y_min,
            block_size,
            grid_cols: cols,
            buffered_bytes: 0,
        }
    }

    /// Add a single point to its corresponding grid cell.
    ///
    /// `index` is the point's 0-based position in the original input stream
    /// (i.e. the `n`th point read from the LAS/LAZ/COPC file, counting from
    /// zero) — this must match the ordering used by
    /// `wbtools_oss::LidarEigenvalueFeaturesTool`'s own `point_num` field so
    /// that per-block feature extraction can later join against the
    /// pre-pass's eigenvalue table by index (Stage 30, Step 5e).
    ///
    /// Triggers a spill pass when the in-memory high-water mark is exceeded.
    ///
    /// # Errors
    /// Returns an error if a spill file write fails.
    pub fn add_point(&mut self, index: u64, pt: LitePoint) -> Result<()> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // floor() to i32: sign-loss is intentional — points outside the
        // positive grid half-plane are assigned negative col/row and handled
        // by downstream range checks.
        let col = ((pt.x - self.x_min) / self.block_size).floor() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let row = ((pt.y - self.y_min) / self.block_size).floor() as i32;
        let key = (col, row);
        self.cells.entry(key).or_default().push((index, pt));
        // Use the actual in-memory size of LitePoint (+ its paired u64 index)
        // for accounting. This governs this project's own internal-pipeline
        // spill threshold (distinct from, and independent of, the Stage 30
        // eigenvalue pre-pass's own memory-budget calculation, which remains
        // pinned to `size_of::<wblidar::PointRecord>()` regardless of this
        // module's internal representation — see
        // `docs/stages/stage-31-lean-point-record.md`, "Critical Caveat").
        self.buffered_bytes += std::mem::size_of::<LitePoint>() + PT_ACCOUNTING_BYTES;

        if self.buffered_bytes >= SPILL_HIGH_WATER_BYTES {
            self.spill_largest_cells()?;
        }
        Ok(())
    }

    /// Called after the full stream is exhausted.
    ///
    /// Merges spill files with any remaining in-memory data and returns
    /// the complete, raw (unprocessed) block list.
    ///
    /// # Errors
    /// Returns an error if a spill file cannot be read.
    pub fn finalize(mut self) -> Result<Vec<Block>> {
        // Collect all keys touched (in-memory or spilled).
        let mut all_keys: std::collections::HashSet<(i32, i32)> =
            self.cells.keys().copied().collect();
        for k in self.spill_paths.keys() {
            all_keys.insert(*k);
        }

        let mut blocks = Vec::with_capacity(all_keys.len());

        for key @ (col, row) in all_keys {
            // Merge spill files first.
            let mut points: Vec<LitePoint> = Vec::new();
            let mut point_indices: Vec<u64> = Vec::new();
            if let Some(paths) = self.spill_paths.remove(&key) {
                for path in &paths {
                    let spilled = read_spill_file(path)?;
                    for (idx, pt) in spilled {
                        point_indices.push(idx);
                        points.push(pt);
                    }
                    // Clean up the temp file immediately after reading.
                    let _ = fs::remove_file(path);
                }
            }
            // Append whatever remains in memory for this key.
            if let Some(mem_pts) = self.cells.remove(&key) {
                for (idx, pt) in mem_pts {
                    point_indices.push(idx);
                    points.push(pt);
                }
            }

            let id = crate::preprocessing::block_id(row as i64, col as i64, self.grid_cols as i64);
            let origin_x = self.x_min + col as f64 * self.block_size;
            let origin_y = self.y_min + row as f64 * self.block_size;

            blocks.push(Block {
                col,
                row,
                id,
                origin_x,
                origin_y,
                points,
                point_indices,
            });
        }

        Ok(blocks)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Flush the largest in-memory cells to `.spill` files until the buffer
    /// drops below half the high-water mark.
    fn spill_largest_cells(&mut self) -> Result<()> {
        // Collect (key, size) pairs and sort descending by size.
        let mut sizes: Vec<((i32, i32), usize)> =
            self.cells.iter().map(|(k, v)| (*k, v.len())).collect();
        sizes.sort_unstable_by_key(|&(_, len)| std::cmp::Reverse(len));

        let target = SPILL_HIGH_WATER_BYTES / 2;

        for (key, _) in sizes {
            if self.buffered_bytes <= target {
                break;
            }
            if let Some(pts) = self.cells.remove(&key) {
                let freed = pts.len() * (std::mem::size_of::<LitePoint>() + PT_ACCOUNTING_BYTES);
                let path = self.spill_path_for(key);
                write_spill_file(&path, &pts)?;
                self.spill_paths.entry(key).or_default().push(path);
                self.buffered_bytes = self.buffered_bytes.saturating_sub(freed);
            }
        }
        Ok(())
    }

    /// Compute the path for the next spill file for a given cell key.
    fn spill_path_for(&self, (col, row): (i32, i32)) -> PathBuf {
        let existing = self.spill_paths.get(&(col, row)).map_or(0, Vec::len);
        self.spill_dir
            .join(format!("block_{col}_{row}_{existing}.spill"))
    }
}

// ── Spill file I/O ────────────────────────────────────────────────────────────

/// Count the number of points in a spill file from its size alone (no data read).
fn spill_point_count(path: &Path) -> usize {
    fs::metadata(path).map_or(0, |m| m.len() as usize / PT_BYTES)
}

/// Memory-safe finalization for large files.
///
/// Flushes **all** remaining in-memory cells to spill files first, so that no
/// point data remains on the heap when this method returns.  Returns lightweight
/// [`BlockStub`] descriptors (metadata + spill paths only).
///
/// The caller loads each block's data on-demand with [`BlockStub::load`], typically
/// inside a Rayon parallel closure, so point data is dropped as soon as each block
/// is processed.  Peak memory is bounded to:
///
/// ```text
/// (Rayon thread count) × (largest single-block size × size_of::<LitePoint>())
/// ```
///
/// rather than the entire dataset — critical for large files.
///
/// Use this in place of [`BlockPartitioner::finalize`] in all production pipelines.
/// The original `finalize` is retained for unit tests that use small point sets.
///
/// # Errors
/// Returns [`ClassifierError`] if any spill file write fails.
impl BlockPartitioner {
    /// Memory-safe finalization: flushes in-memory cells to disk, returns stubs.
    ///
    /// See module-level doc above for the full description.
    ///
    /// # Errors
    /// Returns [`ClassifierError`] if any spill file write fails.
    pub fn finalize_stubs(mut self) -> Result<Vec<BlockStub>> {
        // Flush all remaining in-memory cells to spill files.
        let remaining: Vec<(i32, i32)> = self.cells.keys().copied().collect();
        for key in remaining {
            if let Some(pts) = self.cells.remove(&key) {
                if !pts.is_empty() {
                    let path = self.spill_path_for(key);
                    write_spill_file(&path, &pts)?;
                    self.spill_paths.entry(key).or_default().push(path);
                }
            }
        }
        self.buffered_bytes = 0;

        // Build stubs — only reads file sizes, no point data loaded.
        let mut stubs = Vec::with_capacity(self.spill_paths.len());
        for ((col, row), paths) in self.spill_paths.drain() {
            let point_count: usize = paths.iter().map(|p| spill_point_count(p)).sum();
            let id = crate::preprocessing::block_id(row as i64, col as i64, self.grid_cols as i64);
            let origin_x = self.x_min + col as f64 * self.block_size;
            let origin_y = self.y_min + row as f64 * self.block_size;
            stubs.push(BlockStub {
                col,
                row,
                id,
                origin_x,
                origin_y,
                point_count,
                spill_paths: paths,
            });
        }
        Ok(stubs)
    }
}

/// Write a slice of `(point_index, LitePoint)` pairs to a `.spill` file.
///
/// Format: per-point little-endian fields — `point_index`(u64) `x`(f64)
/// `y`(f64) `z`(f64) `intensity`(u16) `classification`(u8) `return_number`(u8)
/// `number_of_returns`(u8) `scan_angle`(i16) = 39 bytes per point.
fn write_spill_file(path: &Path, pts: &[(u64, LitePoint)]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut buf = [0u8; PT_BYTES];
    for (idx, pt) in pts {
        let x = pt.x.to_le_bytes();
        let y = pt.y.to_le_bytes();
        let z = pt.z.to_le_bytes();
        let int = pt.intensity.to_le_bytes();
        let sa = pt.scan_angle.to_le_bytes();
        buf[0..8].copy_from_slice(&idx.to_le_bytes());
        buf[8..16].copy_from_slice(&x);
        buf[16..24].copy_from_slice(&y);
        buf[24..32].copy_from_slice(&z);
        buf[32..34].copy_from_slice(&int);
        buf[34] = pt.classification;
        buf[35] = pt.return_number;
        buf[36] = pt.number_of_returns;
        buf[37..39].copy_from_slice(&sa);
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a spill file back into a `Vec<(point_index, LitePoint)>`.
fn read_spill_file(path: &Path) -> Result<Vec<(u64, LitePoint)>> {
    let metadata = fs::metadata(path).map_err(|_| ClassifierError::SpillCorrupt {
        path: path.display().to_string(),
    })?;
    let file_bytes = metadata.len() as usize;
    if !file_bytes.is_multiple_of(PT_BYTES) {
        return Err(ClassifierError::SpillCorrupt {
            path: path.display().to_string(),
        });
    }
    let n = file_bytes / PT_BYTES;
    let mut pts = Vec::with_capacity(n);
    let mut file = File::open(path)?;
    let mut buf = [0u8; PT_BYTES];
    for _ in 0..n {
        file.read_exact(&mut buf)?;
        // Each try_into converts a known-size sub-slice to a fixed-size array.
        // The slices are statically sized so this can never fail; we propagate
        // as SpillCorrupt rather than unwrap() to satisfy the no-panics rule.
        let corrupt = || ClassifierError::SpillCorrupt {
            path: path.display().to_string(),
        };
        let idx = u64::from_le_bytes(buf[0..8].try_into().map_err(|_| corrupt())?);
        let pt = LitePoint {
            x: f64::from_le_bytes(buf[8..16].try_into().map_err(|_| corrupt())?),
            y: f64::from_le_bytes(buf[16..24].try_into().map_err(|_| corrupt())?),
            z: f64::from_le_bytes(buf[24..32].try_into().map_err(|_| corrupt())?),
            intensity: u16::from_le_bytes(buf[32..34].try_into().map_err(|_| corrupt())?),
            classification: buf[34],
            return_number: buf[35],
            number_of_returns: buf[36],
            scan_angle: i16::from_le_bytes(buf[37..39].try_into().map_err(|_| corrupt())?),
        };
        pts.push((idx, pt));
    }
    Ok(pts)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pt(x: f64, y: f64) -> LitePoint {
        LitePoint {
            x,
            y,
            z: 0.0,
            ..LitePoint::default()
        }
    }

    #[test]
    fn test_partitioner_assigns_cells_correctly() {
        // 100×100 area, block_size = 50 → 2×2 grid
        let tmp = tempfile::tempdir().unwrap();
        let mut p = BlockPartitioner::new(0.0, 0.0, 100.0, 100.0, 50.0, tmp.path());

        // Points at the four quadrant centres, tagged with distinct original
        // file indices so we can verify the point-index-join round trip.
        p.add_point(0, make_pt(25.0, 25.0)).unwrap(); // (0,0)
        p.add_point(1, make_pt(75.0, 25.0)).unwrap(); // (1,0)
        p.add_point(2, make_pt(25.0, 75.0)).unwrap(); // (0,1)
        p.add_point(3, make_pt(75.0, 75.0)).unwrap(); // (1,1)

        let mut blocks = p.finalize().unwrap();
        blocks.sort_by_key(|b| (b.col, b.row));

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].col, 0);
        assert_eq!(blocks[0].row, 0);
        assert_eq!(blocks[1].col, 0);
        assert_eq!(blocks[1].row, 1);
        assert_eq!(blocks[2].col, 1);
        assert_eq!(blocks[2].row, 0);
        assert_eq!(blocks[3].col, 1);
        assert_eq!(blocks[3].row, 1);

        for b in &blocks {
            assert_eq!(b.points.len(), 1);
            assert_eq!(b.point_indices.len(), 1);
        }

        // Verify each block retained the correct original-file point index.
        assert_eq!(blocks[0].point_indices, vec![0]);
        assert_eq!(blocks[1].point_indices, vec![2]);
        assert_eq!(blocks[2].point_indices, vec![1]);
        assert_eq!(blocks[3].point_indices, vec![3]);
    }

    // Test fixture point generation from a small index range (n=200);
    // precision loss converting the index to f64 is negligible here.
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn test_spill_merge_produces_same_result() {
        use crate::preprocessing::SPILL_HIGH_WATER_BYTES;

        let tmp = tempfile::tempdir().unwrap();

        // Build a set of points for a single block.
        let n = 200_usize;
        let pts: Vec<LitePoint> = (0..n)
            .map(|i| LitePoint {
                x: 10.0 + i as f64 * 0.1,
                y: 10.0,
                z: i as f64,
                ..LitePoint::default()
            })
            .collect();

        // Add the same points through an in-memory partitioner, tagging each
        // with its position in `pts` as the original-file index.
        let mut p_mem = BlockPartitioner::new(0.0, 0.0, 100.0, 100.0, 50.0, tmp.path());
        for (i, &pt) in pts.iter().enumerate() {
            p_mem.add_point(i as u64, pt).unwrap();
        }
        let mem_blocks = p_mem.finalize().unwrap();

        // Verify the spill write/read round trip directly, including indices.
        let indexed_pts: Vec<(u64, LitePoint)> = pts
            .iter()
            .enumerate()
            .map(|(i, &pt)| (i as u64, pt))
            .collect();
        let spill_path = tmp.path().join("test.spill");
        write_spill_file(&spill_path, &indexed_pts).unwrap();
        let recovered = read_spill_file(&spill_path).unwrap();

        assert_eq!(recovered.len(), pts.len());
        for ((idx, a), (rec_idx, b)) in indexed_pts.iter().zip(recovered.iter()) {
            assert_eq!(idx, rec_idx, "point index must survive spill round trip");
            assert!((a.x - b.x).abs() < 1e-12);
            assert!((a.z - b.z).abs() < 1e-12);
        }

        // Verify the in-memory partitioner put everything in the right cell,
        // and that all original indices (0..n) are present exactly once.
        assert_eq!(mem_blocks.len(), 1);
        assert_eq!(mem_blocks[0].points.len(), n);
        assert_eq!(mem_blocks[0].point_indices.len(), n);
        let mut sorted_indices = mem_blocks[0].point_indices.clone();
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, (0..n as u64).collect::<Vec<_>>());

        // Suppress unused import lint for the constant in test context.
        let _ = SPILL_HIGH_WATER_BYTES;
    }

    /// Verify that point indices survive the full `finalize_stubs()` →
    /// `BlockStub::load()` round trip (the actual production code path,
    /// which always writes through spill files regardless of dataset size),
    /// and that each recovered index is still paired with the correct point.
    #[test]
    fn test_finalize_stubs_preserves_point_indices() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = BlockPartitioner::new(0.0, 0.0, 100.0, 100.0, 50.0, tmp.path());

        // Points spread across two blocks, added with distinct, non-sequential
        // indices to make sure no code path assumes index == position.
        let assignments = [
            (10u64, 25.0, 25.0), // block (0,0)
            (20u64, 30.0, 30.0), // block (0,0)
            (30u64, 75.0, 25.0), // block (1,0)
        ];
        for &(idx, x, y) in &assignments {
            p.add_point(idx, make_pt(x, y)).unwrap();
        }

        let stubs = p.finalize_stubs().unwrap();
        let mut blocks: Vec<Block> = stubs.into_iter().map(|s| s.load().unwrap()).collect();
        blocks.sort_by_key(|b| (b.col, b.row));

        assert_eq!(blocks.len(), 2);
        let mut b0_indices = blocks[0].point_indices.clone();
        b0_indices.sort_unstable();
        assert_eq!(b0_indices, vec![10, 20]);
        assert_eq!(blocks[1].point_indices, vec![30]);

        // Verify each recovered point's coordinates match the point that was
        // originally tagged with that index (not just that the index set is
        // correct, but that index <-> point pairing survived intact).
        for b in &blocks {
            for (idx, pt) in b.point_indices.iter().zip(b.points.iter()) {
                let expected = assignments.iter().find(|(i, _, _)| i == idx).unwrap();
                assert!((pt.x - expected.1).abs() < 1e-9);
                assert!((pt.y - expected.2).abs() < 1e-9);
            }
        }
    }
}
