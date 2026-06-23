//! Preprocessing module — config types and sub-module declarations.

pub mod block_partitioner;
pub mod feature_extractor;
pub mod labeled_pipeline;
pub mod normalizer;
pub mod outlier_filter;
pub mod pipeline;
pub mod spatial_index;

pub use pipeline::{BlockManifest, BlockMeta, BlockProcessResult, PreprocessingPipeline};

/// Minimum Rayon chunk size before spawning parallel tasks pays off.
/// Mirrors the convention used in `wbtools_oss`.
pub(crate) const RAYON_MIN_CHUNK: usize = 64;

/// Magic bytes written at the start of every `.feat` file.
pub const FEAT_MAGIC: &[u8; 4] = b"WBFT";

/// Current `.feat` format version.
pub const FEAT_VERSION: u8 = 1;

/// Number of scalar (non-eigenvalue) features per point.
pub const N_SCALAR_FEATURES: usize = 7;

/// Number of eigenvalue-derived features computed per search radius.
pub const N_EIGEN_FEATURES_PER_RADIUS: usize = 5;

/// Compute the total feature count for a given number of search radii.
///
/// ```text
/// total = N_SCALAR_FEATURES + N_EIGEN_FEATURES_PER_RADIUS × n_radii
///       =        7          +           5                 × n_radii
/// ```
#[inline]
#[must_use]
pub const fn n_features_for_radii(n_radii: usize) -> usize {
    N_SCALAR_FEATURES + N_EIGEN_FEATURES_PER_RADIUS * n_radii
}

/// Legacy alias: single-radius feature count (12).
/// Retained for backward compatibility with single-scale code paths.
pub const N_FEATURES: usize = n_features_for_radii(1); // = 12

/// High-water mark for the in-flight block accumulator (bytes).
/// When total buffered point data exceeds this threshold, the largest cells
/// are spilled to temporary raw-point files.
pub const SPILL_HIGH_WATER_BYTES: usize = 512 * 1024 * 1024; // 512 MB

/// Compute the flat block ID from grid coordinates.
///
/// This is the single canonical formula used by every pipeline stage that
/// needs to map `(row, col)` to a block ID.  All consumers must call this
/// function; never inline the arithmetic independently.
///
/// `grid_cols` must be the header-derived value stored in [`BlockManifest`],
/// **not** a value re-derived from the retained blocks after density filtering.
#[allow(clippy::cast_sign_loss)]
#[inline]
#[must_use]
pub fn block_id(row: i64, col: i64, grid_cols: i64) -> u64 {
    // row and col are always ≥ 0 at call sites (callers guard this); the cast
    // is safe in practice but we use wrapping arithmetic to avoid UB if a
    // caller ever passes negative values in debug builds.
    (row.wrapping_mul(grid_cols).wrapping_add(col)) as u64
}

/// Configuration for the full preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Path to the input LAS/LAZ/COPC file.
    pub input: std::path::PathBuf,

    /// Directory where `.feat` blocks and `blocks.json` will be written.
    pub output_dir: std::path::PathBuf,

    /// 2-D cell edge length in projection units.
    pub block_size: f64,

    /// Fixed number of points per block after density-gated sampling.
    pub target_points: usize,

    /// Minimum point density (pts/m²) required to retain a block.
    pub min_density: f64,

    /// Base radius for k-NN eigenvalue neighbourhood queries (single-scale shorthand).
    /// If `search_radii` is non-empty this field is ignored.
    pub search_radius: f64,

    /// Explicit list of eigenvalue search radii for multi-scale feature extraction.
    /// When non-empty, overrides `search_radius`.  Radii are sorted ascending
    /// by `search_radii_effective()` before use.
    /// When empty, `[search_radius]` is used (single-scale, backward-compatible).
    pub search_radii: Vec<f64>,

    /// Minimum neighbour count required; in single-scale mode the radius expands
    /// adaptively up to `search_radius × 4` if this is not satisfied.
    /// In multi-scale mode a fixed radius is used per scale and this threshold
    /// only governs the minimum for a valid covariance matrix (3 points).
    pub min_neighbors: usize,

    /// Optional path to a DTM raster for Height Above Ground computation.
    /// When `None`, the block-minimum-z proxy is used.
    pub hag_model: Option<std::path::PathBuf>,

    /// Rayon thread pool size (`None` = use the system default).
    pub threads: Option<usize>,

    /// When `true`, a `.csv` debug file is emitted alongside each `.feat` file.
    pub debug_csv: bool,

    // ── Outlier removal (G-01) ────────────────────────────────────────────
    /// When `true`, run the `lidar_remove_outliers` pre-pass before block partitioning.
    /// Disabled by default to preserve existing pipeline behaviour.
    pub outlier_removal: bool,

    /// Neighbourhood radius (projection units) for the outlier elevation residual
    /// calculation.  Passed as `search_radius` to `LidarRemoveOutliersTool`.
    pub outlier_radius: f64,

    /// Absolute elevation residual threshold.  Points whose Z deviates from the
    /// neighbourhood mean/median by more than this value are removed.
    pub outlier_elev_diff: f64,

    /// Use neighbourhood median instead of mean for the baseline Z.
    pub outlier_use_median: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            input: std::path::PathBuf::new(),
            output_dir: std::path::PathBuf::new(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            search_radii: Vec::new(),
            min_neighbors: 8,
            hag_model: None,
            threads: None,
            debug_csv: false,
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
        }
    }
}

impl PreprocessConfig {
    /// Return the effective list of eigenvalue search radii, sorted ascending.
    ///
    /// If `search_radii` is non-empty, uses that list (multi-scale mode).
    /// Otherwise falls back to `[search_radius]` (single-scale, backward-compatible).
    #[must_use]
    pub fn search_radii_effective(&self) -> Vec<f64> {
        let mut radii = if self.search_radii.is_empty() {
            vec![self.search_radius]
        } else {
            self.search_radii.clone()
        };
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        radii
    }
}
