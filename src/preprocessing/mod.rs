//! Preprocessing module — config types and sub-module declarations.

pub mod block_partitioner;
pub mod dataset_split;
pub mod feature_extractor;
pub mod labeled_pipeline;
pub mod lite_point;
pub mod normalizer;
pub mod pipeline;
pub mod spatial_index;

pub use lite_point::LitePoint;
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

/// Number of eigenvalue-derived structural features per point, produced by a
/// single whole-file `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass
/// (Stage 30, Step 5e+5f+5g): `lambda1, lambda2, lambda3, linearity,
/// planarity, sphericity, omnivariance, eigentropy, slope, residual`.
///
/// Prior to Stage 30 this project computed only 5 eigenvalue features
/// locally, once per configured search radius (multi-scale). That entire
/// concept is now architecturally meaningless: the pre-pass tool yields
/// exactly one 10-value row per point with no "radius" dimension, so
/// eigenvalue features are always a fixed-width block regardless of how
/// many `--search-radii` a user might have once configured.
pub const N_EIGEN_FEATURES: usize = 10;

/// Total feature count per point: `N_SCALAR_FEATURES + N_EIGEN_FEATURES`.
///
/// Fixed-width as of Stage 30 (Step 5e+5f+5g) — there is no longer a
/// variable "N radii" dimension. This is a **breaking change** to the
/// `.feat` binary format (previously `7 + 5×n_radii`, commonly 12 for the
/// single-radius default); all previously-generated `.feat` files and
/// trained models must be regenerated/retrained.
pub const N_FEATURES: usize = N_SCALAR_FEATURES + N_EIGEN_FEATURES; // = 17

/// High-water mark for the in-flight block accumulator (bytes).
/// When total buffered point data exceeds this threshold, the largest cells
/// are spilled to temporary raw-point files.
pub const SPILL_HIGH_WATER_BYTES: usize = 512 * 1024 * 1024; // 512 MB

/// Maximum allowed size (in bytes) of a single `.feat` file's f32 data
/// payload (`n_points * n_features * 4`). Guards against attempting a
/// multi-gigabyte allocation from a corrupted or maliciously-crafted
/// header (Stage 20 — see `docs/stages/stage-20-security-hardening.md`).
///
/// 512 MB comfortably covers any realistic block: even 1M points ×
/// 100 features × 4 bytes/f32 = ~400 MB.
pub const MAX_FEAT_PAYLOAD_BYTES: usize = 512 * 1024 * 1024; // 512 MB

// ─────────────────────────────────────────────────────────────────────────────
// Path-traversal validation (Stage 20 — Security Hardening)
// ─────────────────────────────────────────────────────────────────────────────

/// Reject manifest-supplied file names that could escape the dataset
/// directory via path traversal (`..`) or an embedded path separator.
///
/// Manifests are expected to carry bare file names (e.g. `block_00042.feat`)
/// that are joined directly to a trusted base directory. A manifest that has
/// been hand-edited or corrupted could otherwise smuggle in `../../etc/passwd`
/// or an absolute path, causing arbitrary file reads.
///
/// Shared by `training::dataset` and `model::inference` so both load paths
/// enforce the same rule from a single canonical implementation.
///
/// # Errors
/// Returns an error if `name` contains `..`, `/`, or `\`.
pub fn validate_block_filename(name: &str) -> crate::error::Result<()> {
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err(crate::error::ClassifierError::Pipeline(format!(
            "manifest file name '{name}' is not a valid bare file name \
             (path separators and '..' are rejected — Stage 20 security hardening)"
        )));
    }
    Ok(())
}

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

    /// Neighbourhood radius (projection units) passed to the whole-file
    /// `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass invocation.
    ///
    /// As of Stage 30 (Step 5e+5f+5g), eigenvalue features are always
    /// computed by a single whole-file pre-pass with exactly one radius —
    /// there is no longer a multi-scale (`search_radii: Vec<f64>`) mode.
    pub search_radius: f64,

    /// Retained for potential future use; no longer consumed by eigenvalue
    /// feature extraction (Stage 30, Step 5e+5f+5g — the pre-pass table has
    /// no "minimum neighbours" concept of its own). Harmless as dead config.
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

    // ── Block overlap (Stage 08) ──────────────────────────────────────────
    /// Overlap radius in projection units added to each block's k-d tree context.
    ///
    /// Border points from adjacent blocks that fall within this radius of the
    /// block boundary are included during feature extraction to eliminate the
    /// spatial edge effect at block seams.  They are **never** resampled or
    /// written to `.feat` files — only canonical block points appear in output.
    ///
    /// - `0.0` (default) — disabled; behaviour identical to Stage 01–07.
    /// - Recommended: `block_size / 2` — fully covers any neighbourhood radius
    ///   ≤ `block_size / 2`.
    /// - Constraint: `0.0 ≤ block_overlap < block_size`.
    pub block_overlap: f64,

    // ── Jitter-based oversampling (Stage 29) ───────────────────────────────
    /// Standard deviation (projection units) of per-axis Gaussian jitter
    /// applied to padding-only points when a block is oversampled
    /// (`raw_count < target_points`). Offsets are clipped to `±3σ`.
    ///
    /// - `0.0` (default) — disabled; behaviour identical to pre-Stage-29
    ///   exact-duplicate padding.
    /// - `> 0.0` — each padding-only copy's (x, y, z) is perturbed before
    ///   feature extraction, producing distinct eigenvalue features instead
    ///   of an exact clone of its source point.
    ///
    /// See `docs/stages/stage-29-jitter-oversampling.md`.
    pub oversample_jitter: f64,

    // ── Eigenvalue-feature pre-pass memory gating (Stage 30, Step 5) ───────
    /// Memory budget (bytes) used to decide whether the whole-file
    /// `wbtools_oss::LidarEigenvalueFeaturesTool` pre-pass can be run in a
    /// single whole-file invocation, or must instead be run over
    /// memory-gated spatial splits.
    ///
    /// The decision is made by comparing this budget against
    /// `header_point_count * size_of::<wblidar::PointRecord>()` — an
    /// estimate of how much memory the *tool's own* whole-cloud
    /// point-record buffer would require, not this project's own
    /// (potentially slimmer) in-pipeline point representation. See
    /// `docs/stages/stage-30-whitebox-git-dependency-integration.md`,
    /// "Approved Design" §2 and §6.
    ///
    /// Default: 2 GiB (`2 * 1024 * 1024 * 1024` bytes). Overridable via
    /// `--eigen-memory-budget-mb <usize>`.
    pub eigen_memory_budget_bytes: usize,
}

/// Default eigenvalue-feature pre-pass memory budget: 2 GiB.
pub const DEFAULT_EIGEN_MEMORY_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            input: std::path::PathBuf::new(),
            output_dir: std::path::PathBuf::new(),
            block_size: 50.0,
            target_points: 1024,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,

            hag_model: None,
            threads: None,
            debug_csv: false,
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap: 0.0,
            oversample_jitter: 0.0,
            eigen_memory_budget_bytes: DEFAULT_EIGEN_MEMORY_BUDGET_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Stage 20 (Security Hardening) -- validate_block_filename must reject
    // path-traversal sequences and embedded path separators, and accept
    // ordinary bare file names.
    #[test]
    fn test_validate_block_filename_rejects_parent_traversal() {
        assert!(validate_block_filename("../etc/passwd").is_err());
    }

    #[test]
    fn test_validate_block_filename_rejects_forward_slash() {
        assert!(validate_block_filename("a/b.feat").is_err());
    }

    #[test]
    fn test_validate_block_filename_rejects_backslash() {
        let name = "a".to_string() + "\\" + "b.feat";
        assert!(validate_block_filename(&name).is_err());
    }

    #[test]
    fn test_validate_block_filename_accepts_bare_name() {
        assert!(validate_block_filename("block_00042.feat").is_ok());
    }
}
