//! Whitebox Next Gen — `LiDAR` Point Cloud Classifier
//!
//! Stage 01: Spatial Preprocessing Pipeline
//!
//! Transforms raw LAS/LAZ/COPC point clouds into fixed-size, normalised
//! per-point feature tensors partitioned into 2-D spatial blocks, ready
//! for `PointNet`-style inference in Stage 02.

#![deny(unsafe_code)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod output;
pub mod preprocessing;
pub mod model;
pub mod cli;

#[cfg(feature = "training")]
pub mod training;

pub use error::{ClassifierError, Result};
