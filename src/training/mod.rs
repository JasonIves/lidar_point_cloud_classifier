//! Training module — gated behind the `training` Cargo feature.
//!
//! Sub-modules:
//! - [`backend`]   — runtime GPU/CPU backend selection and dispatch
//! - [`burn_model`] — `BurnPointNet<B>` (1:1 mirror of Stage 02 inference model)
//! - [`bridge`]     — weight extraction from burn → Stage 02 `.wbmodel`
//! - [`dataset`]    — labeled block dataset (`.feat` + `.lbl` loader + spatial tile split)
//! - [`metrics`]    — `mIoU`, per-class `IoU`, F1, confusion matrix
//! - [`scheduler`]  — cosine annealing LR scheduler
//! - [`trainer`]    — epoch / batch training loop + checkpoint management

#![cfg(feature = "training")]

pub mod backend;
pub mod bridge;
pub mod burn_model;
pub mod dataset;
pub mod metrics;
pub mod scheduler;
pub mod trainer;
