#![cfg(feature = "training")]
//! Stage 25 (Testing Gaps, item 6.1): end-to-end training-loop integration test.
//!
//! Synthesizes a tiny on-disk labeled-block dataset (`.feat` + `.lbl` + a
//! `labeled_blocks.json` manifest) using the exact on-disk contract the real
//! `preprocess-labeled` pipeline produces, loads it via the real
//! `LabeledBlockDataset::load()`, and trains via the real
//! `training::trainer::train()` entry point on the CPU (`Autodiff<NdArray>`)
//! backend — keeping the test hardware-independent per AGENTS.md. Asserts the
//! recorded `train_loss` in the resulting `metrics.csv` decreases from the
//! first to the last epoch, the audit's literal "loss decreases on a
//! synthetic dataset" criterion.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::backend::{Autodiff, NdArray};

use lidar_point_cloud_classifier::preprocessing::labeled_pipeline::{
    LabeledBlockManifest, LabeledBlockMeta, SpatialTileGrid,
};
use lidar_point_cloud_classifier::preprocessing::pipeline::BlockMeta;
use lidar_point_cloud_classifier::preprocessing::{FEAT_MAGIC, FEAT_VERSION, N_FEATURES};
use lidar_point_cloud_classifier::training::dataset::LabeledBlockDataset;
use lidar_point_cloud_classifier::training::trainer::{train, TrainConfig};

/// A tiny deterministic hash used to generate reproducible pseudo-random
/// noise for the non-discriminative feature columns, without pulling in an
/// extra RNG dependency for a single test file.
fn pseudo_rand(seed: u64) -> f32 {
    let x = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    ((x >> 16) & 0xFFFF) as f32 / 65536.0
}

/// Write one synthetic `.feat` + `.lbl` pair matching the on-disk WBFT
/// format (`FEAT_MAGIC` + `FEAT_VERSION` header, `N_FEATURES` columns).
///
/// Half the points are labeled class 0, half class 1. The discriminative
/// signal is placed on *all three* of the first three feature columns (the
/// xyz columns the PointNet Input T-Net transforms) so that class
/// separability survives regardless of how the T-Net's near-identity-
/// initialized 3x3 transform evolves during training. Remaining feature
/// columns are uncorrelated pseudo-random noise.
fn write_synthetic_block(dir: &Path, block_id: u64, n_points: usize) -> (String, String, Vec<u8>) {
    let feat_name = format!("block_{block_id:05}.feat");
    let lbl_name = format!("block_{block_id:05}.lbl");

    let mut labels = Vec::with_capacity(n_points);
    let mut floats = Vec::with_capacity(n_points * N_FEATURES);
    for i in 0..n_points {
        let class = (i % 2) as u8;
        labels.push(class);
        for f in 0..N_FEATURES {
            let noise_seed = block_id
                .wrapping_mul(104_729)
                .wrapping_add((i * N_FEATURES + f) as u64);
            let noise = pseudo_rand(noise_seed);
            let v = if f < 3 {
                if class == 1 {
                    0.9 + 0.05 * noise
                } else {
                    0.05 * noise
                }
            } else {
                noise
            };
            floats.push(v.clamp(0.0, 1.0));
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(FEAT_MAGIC);
    bytes.push(FEAT_VERSION);
    bytes.extend_from_slice(&(n_points as u32).to_le_bytes());
    bytes.extend_from_slice(&(N_FEATURES as u32).to_le_bytes());
    bytes.extend_from_slice(&block_id.to_le_bytes());
    bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_x
    bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_y
    bytes.extend_from_slice(&0u32.to_le_bytes()); // n_halo (v2 — all-core block)
    for v in &floats {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    fs::write(dir.join(&feat_name), &bytes).expect("write .feat fixture");
    fs::write(dir.join(&lbl_name), &labels).expect("write .lbl fixture");

    (feat_name, lbl_name, labels)
}

#[test]
fn test_training_loop_reduces_loss_on_synthetic_dataset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dir_path = dir.path().to_path_buf();

    const N_BLOCKS: u64 = 8;
    const N_POINTS: usize = 32;

    let mut blocks = Vec::new();
    for id in 0..N_BLOCKS {
        let (feat_name, lbl_name, labels) = write_synthetic_block(&dir_path, id, N_POINTS);
        let mut class_distribution = HashMap::new();
        for &l in &labels {
            *class_distribution.entry(l.to_string()).or_insert(0u64) += 1;
        }
        blocks.push(LabeledBlockMeta {
            meta: BlockMeta {
                id,
                file: feat_name,
                origin_x: 0.0,
                origin_y: 0.0,
                raw_point_count: N_POINTS,
                sampled_point_count: N_POINTS,
                oversampled: false,
                n_halo: 0,
            },
            lbl_file: lbl_name,
            macro_tile_id: id as u32,
            class_distribution,
        });
    }

    // Stage 41 (Model label_map identity bug fix): deliberately non-identity
    // — ASPRS code "0" maps to model index 1, and ASPRS code "1" maps to
    // model index 0 — so a regression to `trainer::train()` hardcoding an
    // identity label_map (`[0, 1, ..., n-1]`) into the saved `.wbmodel`
    // would be caught below rather than silently passing.
    let mut label_map = HashMap::new();
    label_map.insert("0".to_string(), 1u8);
    label_map.insert("1".to_string(), 0u8);

    let manifest = LabeledBlockManifest {
        source: "synthetic.las".to_string(),
        block_size: 50.0,
        target_points: N_POINTS,
        min_density: 1.0,
        search_radius: 1.0,
        min_neighbors: 8,
        crs_epsg: None,
        label_map,
        spatial_tile_grid: SpatialTileGrid {
            cols: 8,
            rows: 1,
            bbox_min_x: 0.0,
            bbox_min_y: 0.0,
            bbox_max_x: 8.0,
            bbox_max_y: 1.0,
        },
        halo_fraction: 0.0,
        grid_cols: 1,
        grid_rows: 1,
        grid_x_min: 0.0,
        grid_y_min: 0.0,
        blocks,
    };

    let manifest_path = dir_path.join("labeled_blocks.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");

    // Exercise the real manifest-parsing + spatial train/val split logic.
    let dataset = LabeledBlockDataset::load(std::slice::from_ref(&dir_path), 0.25, None, 7)
        .expect("dataset load must succeed");

    assert!(
        !dataset.train_ids.is_empty(),
        "expected non-empty train set"
    );
    assert!(!dataset.val_ids.is_empty(), "expected non-empty val set");

    let metrics_path = dir_path.join("metrics.csv");
    let config = TrainConfig {
        n_classes: 2,
        epochs: 15,
        batch_size: 4,
        forward_batch_size: 4,
        learning_rate: 1e-2,
        val_split: 0.25,
        use_feature_tnet: false,
        use_class_weights: false,
        checkpoint_dir: None,
        swa: false,
        metrics_out: metrics_path.clone(),
        output_model: dir_path.join("model.wbmodel"),
        seed: 7,
        ..TrainConfig::default()
    };

    type CpuBackend = Autodiff<NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let output_path = train::<CpuBackend>(&dataset, &config, &device)
        .expect("training run must succeed end-to-end");
    assert!(output_path.exists(), "trained model file must be written");

    // Stage 41 (Model label_map identity bug fix): the saved model's
    // label_map must be the *inverse* of the dataset's real ASPRS<->index
    // mapping (here [1, 0], since ASPRS "0"->idx1 and "1"->idx0) — NOT the
    // hardcoded identity [0, 1] the pre-fix trainer always wrote.
    let saved_model = lidar_point_cloud_classifier::model::weights::load_model(&output_path)
        .expect("saved model must load");
    let expected_label_map = dataset
        .inverse_label_map()
        .expect("inverse_label_map must succeed");
    assert_eq!(
        saved_model.label_map, expected_label_map,
        "trained model's label_map must match the dataset's inverted label map, \
         not a hardcoded identity mapping"
    );

    let csv = fs::read_to_string(&metrics_path).expect("metrics.csv must be written");
    let mut train_losses = Vec::new();
    for line in csv.lines().skip(1) {
        if let Some(loss_str) = line.split(',').nth(1) {
            if let Ok(loss) = loss_str.parse::<f64>() {
                train_losses.push(loss);
            }
        }
    }
    assert!(
        train_losses.len() >= 2,
        "expected at least 2 recorded epochs, got {}",
        train_losses.len()
    );

    let first = train_losses[0];
    let last = *train_losses.last().expect("non-empty train_losses");
    assert!(
        last < first,
        "expected training loss to decrease over training: first={first:.4}, last={last:.4}, all={train_losses:?}"
    );
}
