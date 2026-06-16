//! Error types for the `LiDAR` point-cloud classifier.

use thiserror::Error;

/// All errors that can occur within the classifier pipeline.
#[derive(Debug, Error)]
pub enum ClassifierError {
    /// I/O error reading or writing a file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error originating from the wblidar crate.
    #[error("LiDAR I/O error: {0}")]
    Lidar(#[from] wblidar::Error),

    /// Error originating from the wbraster crate.
    #[error("Raster error: {0}")]
    Raster(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The input `LiDAR` file has an unsupported or undetectable format.
    #[error("Unsupported LiDAR format: {path}")]
    UnsupportedFormat { path: String },

    /// The DTM raster supplied for HAG computation does not cover the `LiDAR` extent.
    #[error("DTM raster does not cover point ({x}, {y})")]
    DtmCoverageGap { x: f64, y: f64 },

    /// A block spill file is missing or corrupt during the merge phase.
    #[error("Spill file missing or corrupt: {path}")]
    SpillCorrupt { path: String },

    /// Generic pipeline error with a descriptive message.
    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

/// Convenient `Result` alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ClassifierError>;
