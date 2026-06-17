//! `BurnPointNet<B>` — training twin of the Stage 02 `PointNetClassifier`.
//!
//! This module mirrors the architecture from `model/pointnet.rs` exactly so that
//! the weight bridge can extract parameters with 1:1 correspondence.  No
//! architectural variations from Stage 02 are permitted here.
//!
//! ## Weight layout note
//!
//! Burn's `nn::Linear<B>` stores `weight` as `[d_input, d_output]` (row-major),
//! while Stage 02's `layers::Linear` stores it as `[d_output, d_input]`.
//! The bridge transposes the extracted tensors to reconcile this difference.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::must_use_candidate, clippy::doc_markdown)]

use burn::{
    module::Module,
    nn::{self, BatchNormConfig, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData},
};

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetConfig;
use crate::preprocessing::N_FEATURES;

// ─────────────────────────────────────────────────────────────────────────────
// Input T-Net (STN3d) — fixed dims [3→64→128→1024] encoder, [1024→512→256→9] FC
// ─────────────────────────────────────────────────────────────────────────────

/// Input spatial transformer network (STN3d).
///
/// Learns a 3×3 transform applied to the xyz coordinates of each point.
#[derive(Module, Debug)]
pub struct Stn3d<B: Backend> {
    // Encoder: 3 → 64 → 128 → 1024
    pub enc0: nn::Linear<B>,
    pub bn_enc0: nn::BatchNorm<B, 1>,
    pub enc1: nn::Linear<B>,
    pub bn_enc1: nn::BatchNorm<B, 1>,
    pub enc2: nn::Linear<B>,
    pub bn_enc2: nn::BatchNorm<B, 1>,
    // FC decoder: 1024 → 512 → 256 → 9
    pub fc0: nn::Linear<B>,
    pub bn_fc0: nn::BatchNorm<B, 1>,
    pub fc1: nn::Linear<B>,
    pub bn_fc1: nn::BatchNorm<B, 1>,
    pub fc2: nn::Linear<B>, // no BN on final layer
}

impl<B: Backend> Stn3d<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            enc0: LinearConfig::new(3, 64).init(device),
            bn_enc0: BatchNormConfig::new(64).init(device),
            enc1: LinearConfig::new(64, 128).init(device),
            bn_enc1: BatchNormConfig::new(128).init(device),
            enc2: LinearConfig::new(128, 1024).init(device),
            bn_enc2: BatchNormConfig::new(1024).init(device),
            fc0: LinearConfig::new(1024, 512).init(device),
            bn_fc0: BatchNormConfig::new(512).init(device),
            fc1: LinearConfig::new(512, 256).init(device),
            bn_fc1: BatchNormConfig::new(256).init(device),
            fc2: LinearConfig::new(256, 9).init(device),
        }
    }

    /// Forward pass: `xyz` shape `[N, 3]` → `T` shape `[3, 3]` (the transform matrix).
    pub fn forward(&self, xyz: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = xyz.device();

        // Mini-encoder: [N, 3] → [N, 64] → [N, 128] → [N, 1024]
        let h = self.enc0.forward(xyz);
        let h = apply_bn2d(h, &self.bn_enc0);
        let h = h.clamp_min(0.0);

        let h = self.enc1.forward(h);
        let h = apply_bn2d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h);
        let h = apply_bn2d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        // Global max pool: [N, 1024] → [1024] by taking max over dim 0, then unsqueeze
        let g = h.max_dim(0);                           // [1, 1024]

        let g = self.fc0.forward(g);
        let g = apply_bn2d(g, &self.bn_fc0);
        let g = g.clamp_min(0.0);

        let g = self.fc1.forward(g);
        let g = apply_bn2d(g, &self.bn_fc1);
        let g = g.clamp_min(0.0);

        let g = self.fc2.forward(g);                    // [1, 9]
        let g = g.reshape([3, 3]);                      // [3, 3]

        let eye = identity_2d::<B>(3, &device);
        g + eye                                          // [3, 3]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature T-Net (STN64d) — fixed dims [64→64→128→1024] encoder, [1024→512→256→4096] FC
// ─────────────────────────────────────────────────────────────────────────────

/// Feature spatial transformer network (STN64d).
///
/// Learns a 64×64 transform applied to the intermediate local features.
#[derive(Module, Debug)]
pub struct Stn64d<B: Backend> {
    pub enc0: nn::Linear<B>,
    pub bn_enc0: nn::BatchNorm<B, 1>,
    pub enc1: nn::Linear<B>,
    pub bn_enc1: nn::BatchNorm<B, 1>,
    pub enc2: nn::Linear<B>,
    pub bn_enc2: nn::BatchNorm<B, 1>,
    pub fc0: nn::Linear<B>,
    pub bn_fc0: nn::BatchNorm<B, 1>,
    pub fc1: nn::Linear<B>,
    pub bn_fc1: nn::BatchNorm<B, 1>,
    pub fc2: nn::Linear<B>,
}

impl<B: Backend> Stn64d<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            enc0: LinearConfig::new(64, 64).init(device),
            bn_enc0: BatchNormConfig::new(64).init(device),
            enc1: LinearConfig::new(64, 128).init(device),
            bn_enc1: BatchNormConfig::new(128).init(device),
            enc2: LinearConfig::new(128, 1024).init(device),
            bn_enc2: BatchNormConfig::new(1024).init(device),
            fc0: LinearConfig::new(1024, 512).init(device),
            bn_fc0: BatchNormConfig::new(512).init(device),
            fc1: LinearConfig::new(512, 256).init(device),
            bn_fc1: BatchNormConfig::new(256).init(device),
            fc2: LinearConfig::new(256, 4096).init(device),
        }
    }

    /// Forward pass: `feat` shape `[N, 64]` → `T` shape `[64, 64]`.
    pub fn forward(&self, feat: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = feat.device();

        let h = self.enc0.forward(feat);
        let h = apply_bn2d(h, &self.bn_enc0);
        let h = h.clamp_min(0.0);

        let h = self.enc1.forward(h);
        let h = apply_bn2d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h);
        let h = apply_bn2d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        let g = h.max_dim(0);                           // [1, 1024]

        let g = self.fc0.forward(g);
        let g = apply_bn2d(g, &self.bn_fc0);
        let g = g.clamp_min(0.0);

        let g = self.fc1.forward(g);
        let g = apply_bn2d(g, &self.bn_fc1);
        let g = g.clamp_min(0.0);

        let g = self.fc2.forward(g);                    // [1, 4096]
        let g = g.reshape([64, 64]);

        let eye = identity_2d::<B>(64, &device);
        g + eye
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main PointNet model
// ─────────────────────────────────────────────────────────────────────────────

/// Training twin of `PointNetClassifier` from Stage 02.
///
/// Architecture: PointNet segmentation backbone (Qi et al. 2017).
#[derive(Module, Debug)]
pub struct BurnPointNet<B: Backend> {
    // Input T-Net (STN3d) — always present for valid training
    pub stn3d: Stn3d<B>,

    // Feature T-Net (STN64d) — optional
    pub stn64d: Option<Stn64d<B>>,

    // Main encoder: [n_features → enc_dims[0] → enc_dims[1] → enc_dims[2]]
    pub enc0: nn::Linear<B>,
    pub bn_enc0: nn::BatchNorm<B, 1>,
    pub enc1: nn::Linear<B>,
    pub bn_enc1: nn::BatchNorm<B, 1>,
    pub enc2: nn::Linear<B>,
    pub bn_enc2: nn::BatchNorm<B, 1>,

    // Main decoder: [concat_dim → dec_dims[0] → dec_dims[1]]
    pub dec0: nn::Linear<B>,
    pub bn_dec0: nn::BatchNorm<B, 1>,
    pub dec1: nn::Linear<B>,
    pub bn_dec1: nn::BatchNorm<B, 1>,

    // Final projection: dec_dims.last() → n_classes (no BN)
    pub proj: nn::Linear<B>,
}

impl<B: Backend> BurnPointNet<B> {
    /// Construct a new model from a `PointNetConfig`.
    ///
    /// # Errors
    /// Returns an error if the config dims are invalid (e.g. fewer than 3 encoder dims).
    pub fn new(cfg: &PointNetConfig, device: &B::Device) -> Result<Self> {
        if cfg.encoder_dims.len() < 3 {
            return Err(ClassifierError::Pipeline(
                "BurnPointNet requires at least 3 encoder_dims".into(),
            ));
        }
        if cfg.decoder_dims.len() < 2 {
            return Err(ClassifierError::Pipeline(
                "BurnPointNet requires at least 2 decoder_dims".into(),
            ));
        }

        let ed = &cfg.encoder_dims;
        let dd = &cfg.decoder_dims;
        let concat_dim = cfg.concat_dim(); // ed[0] + ed.last()

        Ok(Self {
            stn3d: Stn3d::new(device),
            stn64d: if cfg.use_feature_tnet { Some(Stn64d::new(device)) } else { None },

            enc0: LinearConfig::new(N_FEATURES, ed[0]).init(device),
            bn_enc0: BatchNormConfig::new(ed[0]).init(device),
            enc1: LinearConfig::new(ed[0], ed[1]).init(device),
            bn_enc1: BatchNormConfig::new(ed[1]).init(device),
            enc2: LinearConfig::new(ed[1], ed[2]).init(device),
            bn_enc2: BatchNormConfig::new(ed[2]).init(device),

            dec0: LinearConfig::new(concat_dim, dd[0]).init(device),
            bn_dec0: BatchNormConfig::new(dd[0]).init(device),
            dec1: LinearConfig::new(dd[0], dd[1]).init(device),
            bn_dec1: BatchNormConfig::new(dd[1]).init(device),

            proj: LinearConfig::new(*dd.last().unwrap(), cfg.n_classes).init(device),
        })
    }

    /// Forward pass.
    ///
    /// # Shapes
    /// - `input`:  `[N, N_FEATURES]` (N sampled points, 12 features)
    /// - output: `[N, n_classes]`   (raw logits, no softmax)
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = input.dims()[0];

        // ── Input T-Net (STN3d) ────────────────────────────────────────────
        let xyz = input.clone().narrow(1, 0, 3);  // [N, 3]
        let t1 = self.stn3d.forward(xyz.clone()); // [3, 3]
        // Apply transform: xyz_new = xyz @ T1  → [N, 3]
        let xyz_new = xyz.matmul(t1);             // [N, 3]
        // Build transformed input: replace columns 0..3 with xyz_new
        let rest = input.narrow(1, 3, N_FEATURES - 3);    // [N, 9]
        let input = Tensor::cat(vec![xyz_new, rest], 1);   // [N, 12]

        // ── Encoder Layer 0 (save as local_feat) ──────────────────────────
        let local_feat = {
            let h = self.enc0.forward(input);              // [N, 64]
            let h = apply_bn2d(h, &self.bn_enc0);
            h.clamp_min(0.0)
        };

        // ── Feature T-Net (optional) ──────────────────────────────────────
        let local_feat = if let Some(stn64d) = &self.stn64d {
            let t2 = stn64d.forward(local_feat.clone());   // [64, 64]
            local_feat.matmul(t2)
        } else {
            local_feat
        };

        // ── Encoder Layers 1+ ─────────────────────────────────────────────
        let h = self.enc1.forward(local_feat.clone());     // [N, 128]
        let h = apply_bn2d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h);                      // [N, 256]
        let h = apply_bn2d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        // ── Global Max Pool ────────────────────────────────────────────────
        let global = h.max_dim(0).repeat_dim(0, n);        // [N, 256]

        // ── Segmentation Concat ───────────────────────────────────────────
        let combined = Tensor::cat(vec![local_feat, global], 1); // [N, 320]

        // ── Decoder ───────────────────────────────────────────────────────
        let h = self.dec0.forward(combined);               // [N, 256]
        let h = apply_bn2d(h, &self.bn_dec0);
        let h = h.clamp_min(0.0);

        let h = self.dec1.forward(h);                      // [N, 128]
        let h = apply_bn2d(h, &self.bn_dec1);
        let h = h.clamp_min(0.0);

        self.proj.forward(h)                               // [N, n_classes]
    }

    /// Return a `Vec<usize>` of per-point class indices.
    pub fn classify(&self, input: Tensor<B, 2>) -> Vec<usize> {
        let logits = self.forward(input);                  // [N, n_classes]
        let n = logits.dims()[0];
        let nc = logits.dims()[1];
        let flat: Vec<f32> = logits
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        (0..n)
            .map(|i| {
                let row = &flat[i * nc..(i + 1) * nc];
                row.iter()
                    .enumerate()
                    // Guard against NaN logits: treat NaN as equal so the lower
                    // index wins rather than panicking.
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(j, _)| j)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: convert features array to burn tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a flat `Vec<f32>` with shape `[n_points, N_FEATURES]` into a
/// burn `Tensor<B, 2>`.
pub fn features_to_tensor<B: Backend>(
    flat: Vec<f32>,
    n_points: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let td = TensorData::new(flat, vec![n_points, N_FEATURES]);
    Tensor::from_floats(td, device)
}

/// Convert a `Vec<u8>` of labels into a burn int tensor `Tensor<B, 1, Int>`.
pub fn labels_to_tensor<B: Backend>(
    labels: &[u8],
    device: &B::Device,
) -> Tensor<B, 1, burn::tensor::Int> {
    let ints: Vec<i64> = labels.iter().map(|&l| i64::from(l)).collect();
    let td = TensorData::new(ints, vec![labels.len()]);
    Tensor::from_ints(td, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply `BatchNorm<B, 1>` to a 2D tensor `[N, C]` by reshaping to `[N, C, 1]`.
fn apply_bn2d<B: Backend>(x: Tensor<B, 2>, bn: &nn::BatchNorm<B, 1>) -> Tensor<B, 2> {
    let [n, c] = x.dims();
    let x3 = x.reshape([n, c, 1]);      // [N, C, 1]
    let x3 = bn.forward(x3);           // [N, C, 1]
    x3.reshape([n, c])                 // [N, C]
}

/// Create a `k×k` identity matrix as a burn tensor.
fn identity_2d<B: Backend>(k: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0.0f32; k * k];
    for i in 0..k {
        data[i * k + i] = 1.0;
    }
    Tensor::from_floats(TensorData::new(data, vec![k, k]), device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn default_cfg() -> PointNetConfig {
        PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: vec![64, 128, 256],
            decoder_dims: vec![256, 128],
            n_classes: 8,
            use_batch_norm: true,
            use_input_tnet: true,
            use_feature_tnet: false,
        }
    }

    #[test]
    fn test_forward_output_shape_no_feature_tnet() {
        let device = Default::default();
        let cfg = default_cfg();
        let model = BurnPointNet::<B>::new(&cfg, &device).unwrap();

        let n = 32usize;
        let flat: Vec<f32> = (0..(n * N_FEATURES)).map(|i| (i % 100) as f32 / 100.0).collect();
        let input = features_to_tensor::<B>(flat, n, &device);

        let out = model.forward(input);
        assert_eq!(out.dims(), [n, cfg.n_classes]);
    }

    #[test]
    fn test_forward_output_shape_with_feature_tnet() {
        let device = Default::default();
        let cfg = PointNetConfig {
            use_feature_tnet: true,
            ..default_cfg()
        };
        let model = BurnPointNet::<B>::new(&cfg, &device).unwrap();

        let n = 16usize;
        let flat: Vec<f32> = vec![0.5f32; n * N_FEATURES];
        let input = features_to_tensor::<B>(flat, n, &device);

        let out = model.forward(input);
        assert_eq!(out.dims(), [n, cfg.n_classes]);
    }
}
