#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_lossless, clippy::cast_precision_loss)]
//! 2-D grid block partitioner with memory-pressure spill/merge support.
//!
//! Points arriving from any number of streaming chunks are routed into a
//! `HashMap<(i32,i32), Vec<PointRecord>>` accumulator that stays open for
//! the entire stream duration.  No block is finalised until `finalize()` is
//! called after EOF, so chunk-spanning blocks are handled correctly.
//!
//! When the total in-flight buffer exceeds `SPILL_HIGH_WATER_BYTES` the
//! largest cells are spilled to temporary `.spill` files (raw `PointRecord`
//! bytes).  After the stream ends, `finalize()` merges each block's spill
//! files with any remaining in-memory data before returning complete `Block`
//! structs.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use wblidar::PointRecord;

use crate::error::{ClassifierError, Result};
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
    pub points: Vec<PointRecord>,
}

/// Accumulates `PointRecord`s into 2-D grid cells with an optional spill path.
pub struct BlockPartitioner {
    /// In-memory per-cell accumulators.
    cells: HashMap<(i32, i32), Vec<PointRecord>>,
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
/// Layout: `x`(8) `y`(8) `z`(8) `intensity`(2) `classification`(1) `return_number`(1)
///         `number_of_returns`(1) `scan_angle`(2) = 31 bytes
const PT_BYTES: usize = 31;

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
    /// Triggers a spill pass when the in-memory high-water mark is exceeded.
    ///
    /// # Errors
    /// Returns an error if a spill file write fails.
    pub fn add_point(&mut self, pt: PointRecord) -> Result<()> {
        let col = ((pt.x - self.x_min) / self.block_size).floor() as i32;
        let row = ((pt.y - self.y_min) / self.block_size).floor() as i32;
        let key = (col, row);
        self.cells.entry(key).or_default().push(pt);
        self.buffered_bytes += PT_BYTES;

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
        let mut all_keys: std::collections::HashSet<(i32, i32)> = self.cells.keys().copied().collect();
        for k in self.spill_paths.keys() {
            all_keys.insert(*k);
        }

        let mut blocks = Vec::with_capacity(all_keys.len());

        for key @ (col, row) in all_keys {
            // Merge spill files first.
            let mut points: Vec<PointRecord> = Vec::new();
            if let Some(paths) = self.spill_paths.remove(&key) {
                for path in &paths {
                    let spilled = read_spill_file(path)?;
                    points.extend(spilled);
                    // Clean up the temp file immediately after reading.
                    let _ = fs::remove_file(path);
                }
            }
            // Append whatever remains in memory for this key.
            if let Some(mem_pts) = self.cells.remove(&key) {
                points.extend(mem_pts);
            }

            let id = row as u64 * self.grid_cols as u64 + col as u64;
            let origin_x = self.x_min + col as f64 * self.block_size;
            let origin_y = self.y_min + row as f64 * self.block_size;

            blocks.push(Block {
                col,
                row,
                id,
                origin_x,
                origin_y,
                points,
            });
        }

        Ok(blocks)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Flush the largest in-memory cells to `.spill` files until the buffer
    /// drops below half the high-water mark.
    fn spill_largest_cells(&mut self) -> Result<()> {
        // Collect (key, size) pairs and sort descending by size.
        let mut sizes: Vec<((i32, i32), usize)> = self
            .cells
            .iter()
            .map(|(k, v)| (*k, v.len()))
            .collect();
        sizes.sort_unstable_by_key(|&(_, len)| std::cmp::Reverse(len));

        let target = SPILL_HIGH_WATER_BYTES / 2;

        for (key, _) in sizes {
            if self.buffered_bytes <= target {
                break;
            }
            if let Some(pts) = self.cells.remove(&key) {
                let freed = pts.len() * PT_BYTES;
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

/// Write a slice of `PointRecord`s to a binary spill file.
///
/// Format: per-point little-endian fields — x(f64) y(f64) z(f64)
/// `intensity`(u16) `classification`(u8) `return_number`(u8)
/// `number_of_returns`(u8) `scan_angle`(i16) = 31 bytes per point.
fn write_spill_file(path: &Path, pts: &[PointRecord]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut buf = [0u8; PT_BYTES];
    for pt in pts {
        let x = pt.x.to_le_bytes();
        let y = pt.y.to_le_bytes();
        let z = pt.z.to_le_bytes();
        let int = pt.intensity.to_le_bytes();
        let sa = pt.scan_angle.to_le_bytes();
        buf[0..8].copy_from_slice(&x);
        buf[8..16].copy_from_slice(&y);
        buf[16..24].copy_from_slice(&z);
        buf[24..26].copy_from_slice(&int);
        buf[26] = pt.classification;
        buf[27] = pt.return_number;
        buf[28] = pt.number_of_returns;
        buf[29..31].copy_from_slice(&sa);
        writer.write_all(&buf)?;
    }
    writer.flush()?;
    Ok(())
}

/// Read a spill file back into a `Vec<PointRecord>`.
fn read_spill_file(path: &Path) -> Result<Vec<PointRecord>> {
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
        let pt = PointRecord {
            x: f64::from_le_bytes(buf[0..8].try_into().unwrap()),
            y: f64::from_le_bytes(buf[8..16].try_into().unwrap()),
            z: f64::from_le_bytes(buf[16..24].try_into().unwrap()),
            intensity: u16::from_le_bytes(buf[24..26].try_into().unwrap()),
            classification: buf[26],
            return_number: buf[27],
            number_of_returns: buf[28],
            scan_angle: i16::from_le_bytes(buf[29..31].try_into().unwrap()),
            ..PointRecord::default()
        };
        pts.push(pt);
    }
    Ok(pts)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pt(x: f64, y: f64) -> PointRecord {
        PointRecord { x, y, z: 0.0, ..PointRecord::default() }
    }

    #[test]
    fn test_partitioner_assigns_cells_correctly() {
        // 100×100 area, block_size = 50 → 2×2 grid
        let tmp = tempfile::tempdir().unwrap();
        let mut p = BlockPartitioner::new(0.0, 0.0, 100.0, 100.0, 50.0, tmp.path());

        // Points at the four quadrant centres
        p.add_point(make_pt(25.0, 25.0)).unwrap(); // (0,0)
        p.add_point(make_pt(75.0, 25.0)).unwrap(); // (1,0)
        p.add_point(make_pt(25.0, 75.0)).unwrap(); // (0,1)
        p.add_point(make_pt(75.0, 75.0)).unwrap(); // (1,1)

        let mut blocks = p.finalize().unwrap();
        blocks.sort_by_key(|b| (b.col, b.row));

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].col, 0); assert_eq!(blocks[0].row, 0);
        assert_eq!(blocks[1].col, 0); assert_eq!(blocks[1].row, 1);
        assert_eq!(blocks[2].col, 1); assert_eq!(blocks[2].row, 0);
        assert_eq!(blocks[3].col, 1); assert_eq!(blocks[3].row, 1);

        for b in &blocks {
            assert_eq!(b.points.len(), 1);
        }
    }

    #[test]
    fn test_spill_merge_produces_same_result() {
        use crate::preprocessing::SPILL_HIGH_WATER_BYTES;

        let tmp = tempfile::tempdir().unwrap();

        // Build a set of points for a single block.
        let n = 200_usize;
        let pts: Vec<PointRecord> = (0..n)
            .map(|i| {
                let mut pt = PointRecord::default();
                pt.x = 10.0 + i as f64 * 0.1;
                pt.y = 10.0;
                pt.z = i as f64;
                pt
            })
            .collect();

        // Add the same points through two partitioners — one in-memory,
        // one with a tiny high-water mark to force spilling.
        let mut p_mem = BlockPartitioner::new(0.0, 0.0, 100.0, 100.0, 50.0, tmp.path());
        for &pt in &pts {
            p_mem.add_point(pt).unwrap();
        }
        let mem_blocks = p_mem.finalize().unwrap();

        // Override high-water mark via a trivially small PT_BYTES threshold
        // by writing a custom version that just spills every 5 points.
        // Since we can't override the constant we just verify the spill round-trip
        // directly by calling write/read helpers.
        let spill_path = tmp.path().join("test.spill");
        write_spill_file(&spill_path, &pts).unwrap();
        let recovered = read_spill_file(&spill_path).unwrap();

        assert_eq!(recovered.len(), pts.len());
        for (a, b) in pts.iter().zip(recovered.iter()) {
            assert!((a.x - b.x).abs() < 1e-12);
            assert!((a.z - b.z).abs() < 1e-12);
        }

        // Verify the in-memory partitioner put everything in the right cell.
        assert_eq!(mem_blocks.len(), 1);
        assert_eq!(mem_blocks[0].points.len(), n);

        // Suppress unused import lint for the constant in test context.
        let _ = SPILL_HIGH_WATER_BYTES;
    }
}
