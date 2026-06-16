//! Preprocessing module — config types and sub-module declarations.

pub mod block_partitioner;
pub mod feature_extractor;
pub mod normalizer;
pub mod pipeline;
pub mod spatial_index;

pub use pipeline::{BlockManifest, BlockMeta, PreprocessingPipeline};

/// Minimum Rayon chunk size before spawning parallel tasks pays off.
/// Mirrors the convention used in `wbtools_oss`.
pub(crate) const RAYON_MIN_CHUNK: usize = 64;

/// Magic bytes written at the start of every `.feat` file.
pub const FEAT_MAGIC: &[u8; 4] = b"WBFT";

/// Current `.feat` format version.
pub const FEAT_VERSION: u8 = 1;

/// Number of features per point in the feature matrix.
pub const N_FEATURES: usize = 12;

/// High-water mark for the in-flight block accumulator (bytes).
/// When total buffered point data exceeds this threshold, the largest cells
/// are spilled to temporary raw-point files.
pub const SPILL_HIGH_WATER_BYTES: usize = 512 * 1024 * 1024; // 512 MB

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

    /// Base radius for k-NN eigenvalue neighbourhood queries.
    pub search_radius: f64,

    /// Minimum neighbour count required; radius expands adaptively up to
    /// `search_radius × 4` if this is not satisfied at `search_radius`.
    pub min_neighbors: usize,

    /// Optional path to a DTM raster for Height Above Ground computation.
    /// When `None`, the block-minimum-z proxy is used.
    pub hag_model: Option<std::path::PathBuf>,

    /// Rayon thread pool size (`None` = use the system default).
    pub threads: Option<usize>,

    /// When `true`, a `.csv` debug file is emitted alongside each `.feat` file.
    pub debug_csv: bool,
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
            min_neighbors: 8,
            hag_model: None,
            threads: None,
            debug_csv: false,
        }
    }
}
