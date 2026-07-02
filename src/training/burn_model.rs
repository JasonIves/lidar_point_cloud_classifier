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

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown
)]

use burn::{
    module::Module,
    nn::{self, BatchNormConfig, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData},
};

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetConfig;

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

        // Global max pool: [N, 1024] → [1, 1024] by taking max over dim 0.
        // burn-ndarray 0.16 gather constraint: indices can only differ from the
        // source tensor in the LAST dimension.  max_dim(0) on [N, 1024] violates
        // this.  Workaround: transpose to [1024, N], max over last dim → [1024, 1],
        // transpose back → [1, 1024].
        let g = h.transpose().max_dim(1).transpose(); // [1, 1024]

        // Stage 17: BatchNorm is intentionally NOT applied to the post-pool FC
        // layers.  After the global max-pool above, `g` is a single pooled
        // sample of shape [1, C].  burn's batch-statistic BatchNorm then sees
        // batch_size = 1, computes a per-sample variance of 0, and drives
        // `running_var` → 0 via its EMA update.  At inference (`model.valid()`
        // and the deployed ndarray path) that near-zero running variance divides
        // by ~sqrt(eps), producing the logit explosion documented in
        // docs/stages/stage-17-batchnorm-running-stats.md.  BatchNorm on a
        // genuine batch-of-1 descriptor is degenerate; removing it makes
        // train-mode and inference-mode agree.  `bn_fc0`/`bn_fc1` are retained as
        // (unused) fields solely to preserve the `.wbmodel` weight layout.
        let g = self.fc0.forward(g);
        let g = g.clamp_min(0.0);

        let g = self.fc1.forward(g);
        let g = g.clamp_min(0.0);

        let g = self.fc2.forward(g); // [1, 9]
        let g = g.reshape([3, 3]); // [3, 3]

        let eye = identity_2d::<B>(3, &device);
        g + eye // [3, 3]
    }

    /// Batched forward pass: `xyz` shape `[B, N, 3]` → `T` shape `[B, 3, 3]`.
    ///
    /// Identical maths to [`Stn3d::forward`] but with a leading batch dimension so
    /// BatchNorm normalizes across all `B·N` points (a genuine cross-block batch,
    /// Stage 18) while the global max-pool stays *per sample* (over `N` only).
    pub fn forward_batched(&self, xyz: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = xyz.device();
        let b = xyz.dims()[0];

        let h = self.enc0.forward(xyz);
        let h = apply_bn3d(h, &self.bn_enc0);
        let h = h.clamp_min(0.0);

        let h = self.enc1.forward(h);
        let h = apply_bn3d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h);
        let h = apply_bn3d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        // Per-sample global max pool over N: [B, N, 1024] → transpose → [B, 1024, N]
        // → max over last dim → [B, 1024, 1] → transpose → [B, 1, 1024].
        let g = h.transpose().max_dim(2).transpose(); // [B, 1, 1024]

        let g = self.fc0.forward(g);
        let g = g.clamp_min(0.0);
        let g = self.fc1.forward(g);
        let g = g.clamp_min(0.0);
        let g = self.fc2.forward(g); // [B, 1, 9]
        let g = g.reshape([b, 3, 3]); // [B, 3, 3]

        let eye = identity_2d::<B>(3, &device).reshape([1, 3, 3]);
        g + eye // broadcast [1, 3, 3] over [B, 3, 3]
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

        // Same transpose workaround as Stn3d: [N, 1024] → [1024, N] → max over
        // last dim → [1024, 1] → transpose → [1, 1024].
        let g = h.transpose().max_dim(1).transpose(); // [1, 1024]

        // Stage 17: BatchNorm intentionally omitted on the post-pool FC layers
        // (batch-of-1 pooled descriptor → degenerate running stats → inference
        // logit explosion).  See Stn3d::forward and
        // docs/stages/stage-17-batchnorm-running-stats.md.
        let g = self.fc0.forward(g);
        let g = g.clamp_min(0.0);

        let g = self.fc1.forward(g);
        let g = g.clamp_min(0.0);

        let g = self.fc2.forward(g); // [1, 4096]
        let g = g.reshape([64, 64]);

        let eye = identity_2d::<B>(64, &device);
        g + eye
    }

    /// Batched forward pass: `feat` shape `[B, N, 64]` → `T` shape `[B, 64, 64]`.
    ///
    /// Batched analogue of [`Stn64d::forward`]; see [`Stn3d::forward_batched`].
    pub fn forward_batched(&self, feat: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = feat.device();
        let b = feat.dims()[0];

        let h = self.enc0.forward(feat);
        let h = apply_bn3d(h, &self.bn_enc0);
        let h = h.clamp_min(0.0);

        let h = self.enc1.forward(h);
        let h = apply_bn3d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h);
        let h = apply_bn3d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        let g = h.transpose().max_dim(2).transpose(); // [B, 1, 1024]

        let g = self.fc0.forward(g);
        let g = g.clamp_min(0.0);
        let g = self.fc1.forward(g);
        let g = g.clamp_min(0.0);
        let g = self.fc2.forward(g); // [B, 1, 4096]
        let g = g.reshape([b, 64, 64]);

        let eye = identity_2d::<B>(64, &device).reshape([1, 64, 64]);
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
            stn64d: if cfg.use_feature_tnet {
                Some(Stn64d::new(device))
            } else {
                None
            },

            // Use cfg.n_features_in so multi-scale feature counts (Stage 06) are
            // correctly wired into the first encoder layer.
            enc0: LinearConfig::new(cfg.n_features_in, ed[0]).init(device),
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
    /// - `input`:  `[N, n_features_in]` (N sampled points, n_features features)
    /// - output: `[N, n_classes]`   (raw logits, no softmax)
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = input.dims()[0];
        let n_feat = input.dims()[1]; // runtime feature count (supports multi-scale)

        // ── Input T-Net (STN3d) ────────────────────────────────────────────
        // Always extract the first 3 features (x_norm, y_norm, z_norm) for the
        // spatial transform, regardless of total feature count.
        let xyz = input.clone().narrow(1, 0, 3); // [N, 3] — always first 3 features
        let t1 = self.stn3d.forward(xyz.clone()); // [3, 3]
                                                  // Apply transform: xyz_new = xyz @ T1  → [N, 3]
        let xyz_new = xyz.matmul(t1); // [N, 3]
                                      // Build transformed input: replace columns 0..3 with xyz_new, keep the rest
        let rest = input.narrow(1, 3, n_feat - 3); // [N, n_feat-3]
        let input = Tensor::cat(vec![xyz_new, rest], 1); // [N, n_feat]

        // ── Encoder Layer 0 (save as local_feat) ──────────────────────────
        let local_feat = {
            let h = self.enc0.forward(input); // [N, 64]
            let h = apply_bn2d(h, &self.bn_enc0);
            h.clamp_min(0.0)
        };

        // ── Feature T-Net (optional) ──────────────────────────────────────
        let local_feat = if let Some(stn64d) = &self.stn64d {
            let t2 = stn64d.forward(local_feat.clone()); // [64, 64]
            local_feat.matmul(t2)
        } else {
            local_feat
        };

        // ── Encoder Layers 1+ ─────────────────────────────────────────────
        let h = self.enc1.forward(local_feat.clone()); // [N, 128]
        let h = apply_bn2d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h); // [N, 256]
        let h = apply_bn2d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        // ── Global Max Pool ────────────────────────────────────────────────
        // burn-ndarray 0.16 gather constraint: indices can only differ from the
        // source tensor in the LAST dimension.  max_dim(0) on [N, C] (C≠N) violates
        // this.  Workaround: transpose to [C, N], max over last dim (N) → [C, 1],
        // transpose back → [1, C], then broadcast to [N, C] with repeat_dim.
        let global = h.transpose().max_dim(1).transpose().repeat_dim(0, n); // [N, 256]

        // ── Segmentation Concat ───────────────────────────────────────────
        let combined = Tensor::cat(vec![local_feat, global], 1); // [N, 320]

        // ── Decoder ───────────────────────────────────────────────────────
        let h = self.dec0.forward(combined); // [N, 256]
        let h = apply_bn2d(h, &self.bn_dec0);
        let h = h.clamp_min(0.0);

        let h = self.dec1.forward(h); // [N, 128]
        let h = apply_bn2d(h, &self.bn_dec1);
        let h = h.clamp_min(0.0);

        self.proj.forward(h) // [N, n_classes]
    }

    /// Batched forward pass (Stage 18).
    ///
    /// # Shapes
    /// - `input`:  `[B, N, n_features_in]` (B blocks, N sampled points each)
    /// - output: `[B, N, n_classes]`   (raw logits, no softmax)
    ///
    /// Every BatchNorm normalizes across the whole `B·N` micro-batch (so its
    /// running statistics become representative of the block *population*), while
    /// the global max-pool remains strictly per-block (over `N`).  The weights are
    /// identical to the single-block [`BurnPointNet::forward`] path, so the
    /// deployed single-block ndarray inference stays consistent with training.
    pub fn forward_batched(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, n, n_feat] = input.dims();

        // ── Input T-Net (STN3d) ────────────────────────────────────────────
        let xyz = input.clone().narrow(2, 0, 3); // [B, N, 3]
        let t1 = self.stn3d.forward_batched(xyz.clone()); // [B, 3, 3]
        let xyz_new = xyz.matmul(t1); // [B, N, 3] @ [B, 3, 3] → [B, N, 3]
        let rest = input.narrow(2, 3, n_feat - 3); // [B, N, n_feat-3]
        let input = Tensor::cat(vec![xyz_new, rest], 2); // [B, N, n_feat]

        // ── Encoder Layer 0 (save as local_feat) ──────────────────────────
        let local_feat = {
            let h = self.enc0.forward(input); // [B, N, 64]
            let h = apply_bn3d(h, &self.bn_enc0);
            h.clamp_min(0.0)
        };

        // ── Feature T-Net (optional) ──────────────────────────────────────
        let local_feat = if let Some(stn64d) = &self.stn64d {
            let t2 = stn64d.forward_batched(local_feat.clone()); // [B, 64, 64]
            local_feat.matmul(t2) // [B, N, 64] @ [B, 64, 64] → [B, N, 64]
        } else {
            local_feat
        };

        // ── Encoder Layers 1+ ─────────────────────────────────────────────
        let h = self.enc1.forward(local_feat.clone()); // [B, N, 128]
        let h = apply_bn3d(h, &self.bn_enc1);
        let h = h.clamp_min(0.0);

        let h = self.enc2.forward(h); // [B, N, 256]
        let h = apply_bn3d(h, &self.bn_enc2);
        let h = h.clamp_min(0.0);

        // ── Per-sample Global Max Pool ─────────────────────────────────────
        // [B, N, 256] → transpose → [B, 256, N] → max over last dim (N) →
        // [B, 256, 1] → transpose → [B, 1, 256] → broadcast to [B, N, 256].
        let global = h.transpose().max_dim(2).transpose().repeat_dim(1, n); // [B, N, 256]

        // ── Segmentation Concat ───────────────────────────────────────────
        let combined = Tensor::cat(vec![local_feat, global], 2); // [B, N, 320]

        // ── Decoder ───────────────────────────────────────────────────────
        let h = self.dec0.forward(combined); // [B, N, 256]
        let h = apply_bn3d(h, &self.bn_dec0);
        let h = h.clamp_min(0.0);

        let h = self.dec1.forward(h); // [B, N, 128]
        let h = apply_bn3d(h, &self.bn_dec1);
        let h = h.clamp_min(0.0);

        self.proj.forward(h) // [B, N, n_classes]
    }

    /// Return a `Vec<usize>` of per-point class indices.
    pub fn classify(&self, input: Tensor<B, 2>) -> Vec<usize> {
        let logits = self.forward(input); // [N, n_classes]
        let n = logits.dims()[0];
        let nc = logits.dims()[1];
        let flat: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap_or_default();
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

/// Convert a flat `Vec<f32>` with shape `[n_points, n_features]` into a
/// burn `Tensor<B, 2>`.
///
/// `n_features` is passed explicitly to support both the legacy 12-feature
/// single-scale path and the multi-scale paths introduced in Stage 06.
pub fn features_to_tensor<B: Backend>(
    flat: Vec<f32>,
    n_points: usize,
    n_features: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let td = TensorData::new(flat, vec![n_points, n_features]);
    Tensor::from_floats(td, device)
}

/// Convert a flat `Vec<f32>` with shape `[batch, n_points, n_features]` (row
/// major: block-major, then point, then feature) into a burn `Tensor<B, 3>`.
///
/// Used by the Stage 18 batched training forward, which stacks several spatial
/// blocks into one micro-batch so BatchNorm sees a genuine cross-block batch.
pub fn features_to_tensor_batched<B: Backend>(
    flat: Vec<f32>,
    batch: usize,
    n_points: usize,
    n_features: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let td = TensorData::new(flat, vec![batch, n_points, n_features]);
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
    let x3 = x.reshape([n, c, 1]); // [N, C, 1]
    let x3 = bn.forward(x3); // [N, C, 1]
    x3.reshape([n, c]) // [N, C]
}

/// Apply `BatchNorm<B, 1>` to a batched 3D tensor `[B, N, C]` by reshaping to
/// `[B·N, C, 1]`, so BatchNorm normalizes each channel across all `B·N` samples
/// of the micro-batch (Stage 18).  Restores the `[B, N, C]` shape afterwards.
fn apply_bn3d<B: Backend>(x: Tensor<B, 3>, bn: &nn::BatchNorm<B, 1>) -> Tensor<B, 3> {
    let [b, n, c] = x.dims();
    let x3 = x.reshape([b * n, c, 1]); // [B·N, C, 1]
    let x3 = bn.forward(x3); // [B·N, C, 1]
    x3.reshape([b, n, c]) // [B, N, C]
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
    use crate::preprocessing::N_FEATURES;
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
        let flat: Vec<f32> = (0..(n * N_FEATURES))
            .map(|i| (i % 100) as f32 / 100.0)
            .collect();
        let input = features_to_tensor::<B>(flat, n, N_FEATURES, &device);

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
        let input = features_to_tensor::<B>(flat, n, N_FEATURES, &device);

        let out = model.forward(input);
        assert_eq!(out.dims(), [n, cfg.n_classes]);
    }

    // ── Stage 17: BatchNorm running-statistic regression tests ───────────────

    /// Root-cause isolation: a `BatchNorm<B, 1>` fed a single pooled sample
    /// (`[1, C, 1]`) computes a per-sample variance of 0 in training mode, so its
    /// EMA-updated `running_var` decays 1.0 → 0 (momentum 0.1 → ×0.9 per step).
    /// At inference this near-zero variance divides by ~sqrt(eps) and explodes.
    /// This documents *why* Stage 17 omits BatchNorm on the post-pool FC layers.
    #[test]
    fn test_batchnorm_batch1_running_var_decays_toward_zero() {
        use burn::backend::Autodiff;

        type Ad = Autodiff<NdArray>;
        let device = Default::default();
        let bn = BatchNormConfig::new(8).init::<Ad, 1>(&device);

        // Feed 15 distinct single-sample [1, 8, 1] tensors in training mode.
        for step in 0..15 {
            let vals: Vec<f32> = (0..8).map(|c| (step + c) as f32 * 0.1).collect();
            let x = Tensor::<Ad, 3>::from_floats(TensorData::new(vals, vec![1, 8, 1]), &device);
            let _ = bn.forward(x);
        }

        let running_var: Vec<f32> = bn.running_var.value().into_data().to_vec::<f32>().unwrap();
        let max_var = running_var.iter().copied().fold(0.0f32, f32::max);
        // 0.9^15 ≈ 0.206 — far below the initial 1.0, confirming the decay that
        // makes inference-mode BatchNorm blow up on batch-of-1 pooled vectors.
        assert!(
            max_var < 0.5,
            "batch-1 running_var should decay toward 0, got max {max_var}"
        );
    }

    /// Regression guard: after several training-mode forwards populate the
    /// BatchNorm running statistics, the inference-mode (`.valid()`) forward must
    /// produce finite, bounded logits.  Before the Stage 17 fix, the post-pool
    /// T-Net FC BatchNorm layers drove `running_var` → 0, so `.valid()` produced a
    /// logit explosion (values ~1e5+, `val_loss` ~1.6e5).  With those layers'
    /// BatchNorm removed, the output stays sane.
    #[test]
    fn test_valid_inference_logits_bounded_after_training() {
        use burn::backend::Autodiff;
        use burn::module::AutodiffModule;
        use burn::tensor::backend::AutodiffBackend;

        type Ad = Autodiff<NdArray>;
        let device = Default::default();
        let cfg = default_cfg();
        let model = BurnPointNet::<Ad>::new(&cfg, &device).unwrap();

        let n = 256usize;
        // Run several training-mode forwards to populate running statistics.
        for step in 0..8 {
            let flat: Vec<f32> = (0..(n * N_FEATURES))
                .map(|i| ((i + step * 7) % 97) as f32 / 97.0)
                .collect();
            let input = features_to_tensor::<Ad>(flat, n, N_FEATURES, &device);
            let _ = model.forward(input);
        }

        // Inference-mode forward on the inner (non-autodiff) backend.
        let val_model = model.valid();
        let flat: Vec<f32> = (0..(n * N_FEATURES))
            .map(|i| (i % 89) as f32 / 89.0)
            .collect();
        let input = features_to_tensor::<<Ad as AutodiffBackend>::InnerBackend>(
            flat, n, N_FEATURES, &device,
        );
        let logits = val_model.forward(input);
        let out: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();

        assert_eq!(out.len(), n * cfg.n_classes);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "inference logits must all be finite"
        );
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e3,
            "Stage 17: inference logits should be bounded; got max |logit| {max_abs}"
        );
    }

    // ── Stage 18: train/eval BatchNorm statistics-gap mechanism test ─────────

    /// Empirical confirmation for Stage 18.
    ///
    /// Demonstrates that the train-mode (batch-statistic) BatchNorm output and
    /// the inference-mode (`.valid()`, running-statistic) output for the *same*
    /// input **diverge** when the training blocks are heterogeneous (each block
    /// has a different per-feature distribution), and **agree** when the blocks
    /// share one distribution.
    ///
    /// This isolates the cause of the poor validation metrics to exactly the
    /// per-block-vs-global BatchNorm statistics mismatch: with an effective
    /// BatchNorm batch size of one block, training normalizes each block by its
    /// own statistics, while `.valid()` (and deployment) normalize every block by
    /// a single global running average — so heterogeneous blocks are systematically
    /// mis-normalized at inference.
    #[test]
    fn test_batchnorm_train_eval_gap_depends_on_block_heterogeneity() {
        use burn::backend::Autodiff;
        use burn::module::AutodiffModule;
        use burn::tensor::backend::AutodiffBackend;

        type Ad = Autodiff<NdArray>;
        let device = Default::default();
        let n = 128usize;

        // Deterministic block generator: fixed content pattern, shifted by a
        // per-block `offset` and scaled by `scale` so we can control how much
        // the block's feature distribution differs from the others.
        let make_block = |offset: f32, scale: f32| -> Vec<f32> {
            (0..(n * N_FEATURES))
                .map(|i| (((i * 31 + 7) % 97) as f32 / 97.0) * scale + offset)
                .collect()
        };

        // Mean absolute difference between train-mode and eval-mode logits for a
        // held-out block, after populating running stats with `offsets`.
        let train_eval_logit_gap = |offsets: &[f32], scales: &[f32], held_out: f32| -> f32 {
            let cfg = default_cfg();
            let model = BurnPointNet::<Ad>::new(&cfg, &device).unwrap();

            // Populate running statistics with the training blocks.  Enough
            // passes that the EMA (momentum 0.1) fully converges — 0.9^150 ≈ 1e-7 —
            // so that in the *homogeneous* case the running statistics equal the
            // block's own batch statistics and the train/eval gap collapses to ~0.
            // Any residual gap in the heterogeneous case is then attributable
            // purely to per-block-vs-global statistics mismatch, not EMA lag.
            for _ in 0..30 {
                for (&off, &sc) in offsets.iter().zip(scales.iter()) {
                    let input =
                        features_to_tensor::<Ad>(make_block(off, sc), n, N_FEATURES, &device);
                    let _ = model.forward(input);
                }
            }

            // Held-out block (its own distribution).
            let held_flat = make_block(held_out, 1.0);

            // Eval-mode (running-stat) logits FIRST so the running stats are not
            // perturbed by the held-out block before we capture them.
            let val_model = model.valid();
            let eval_in = features_to_tensor::<<Ad as AutodiffBackend>::InnerBackend>(
                held_flat.clone(),
                n,
                N_FEATURES,
                &device,
            );
            let eval_logits: Vec<f32> = val_model.forward(eval_in).into_data().to_vec().unwrap();

            // Train-mode (batch-stat) logits on the same held-out block.
            let train_in = features_to_tensor::<Ad>(held_flat, n, N_FEATURES, &device);
            let train_logits: Vec<f32> = model.forward(train_in).into_data().to_vec().unwrap();

            let sum: f32 = eval_logits
                .iter()
                .zip(train_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            sum / eval_logits.len() as f32
        };

        // Heterogeneous blocks: widely varying offsets; held-out far from the mean.
        let hetero_offsets = [0.0, 2.0, 4.0, 6.0, 8.0];
        let hetero_scales = [1.0, 1.5, 0.5, 2.0, 0.8];
        let hetero_gap = train_eval_logit_gap(&hetero_offsets, &hetero_scales, 0.0);

        // Homogeneous blocks: identical distribution; held-out matches it.
        let homo_offsets = [0.0, 0.0, 0.0, 0.0, 0.0];
        let homo_scales = [1.0, 1.0, 1.0, 1.0, 1.0];
        let homo_gap = train_eval_logit_gap(&homo_offsets, &homo_scales, 0.0);

        eprintln!(
            "[stage18] train/eval BN logit gap — heterogeneous={hetero_gap:.4}  homogeneous={homo_gap:.4}"
        );

        // Homogeneous: train-mode and eval-mode must nearly agree.
        assert!(
            homo_gap < 0.5,
            "homogeneous blocks should give matching train/eval BN outputs, got gap {homo_gap}"
        );
        // Heterogeneous: the gap must be substantially larger — this is the
        // train/eval BatchNorm statistics mismatch driving the bad val metrics.
        assert!(
            hetero_gap > homo_gap * 3.0 + 0.5,
            "heterogeneous blocks should widen the train/eval BN gap (hetero={hetero_gap}, homo={homo_gap})"
        );
    }

    // ── Stage 18: batched forward correctness ────────────────────────────────

    /// The batched forward must (a) produce the correct `[B, N, n_classes]`
    /// shape, and (b) when every block in the batch is identical, produce
    /// identical per-block outputs across the batch dimension.  The latter
    /// confirms the per-sample global max-pool does not leak across blocks and
    /// that the batched BatchNorm path is wired correctly.
    #[test]
    fn test_forward_batched_identical_blocks_are_consistent() {
        use burn::backend::Autodiff;

        type Ad = Autodiff<NdArray>;
        let device = Default::default();
        let cfg = default_cfg();
        let model = BurnPointNet::<Ad>::new(&cfg, &device).unwrap();

        let n = 96usize;
        let b = 4usize;

        // One block's features, replicated across the batch.
        let single: Vec<f32> = (0..(n * N_FEATURES))
            .map(|i| ((i * 17 + 3) % 91) as f32 / 91.0)
            .collect();
        let mut batch_flat = Vec::with_capacity(b * n * N_FEATURES);
        for _ in 0..b {
            batch_flat.extend_from_slice(&single);
        }

        let input = features_to_tensor_batched::<Ad>(batch_flat, b, n, N_FEATURES, &device);
        let out = model.forward_batched(input); // [b, n, n_classes]
        assert_eq!(out.dims(), [b, n, cfg.n_classes]);

        let flat: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "batched outputs must all be finite"
        );

        // Every batch row must equal batch row 0 (identical inputs → identical
        // outputs; no cross-block pooling leakage).
        let per_block = n * cfg.n_classes;
        for bi in 1..b {
            for k in 0..per_block {
                let a = flat[k];
                let c = flat[bi * per_block + k];
                assert!(
                    (a - c).abs() < 1e-4,
                    "batched forward row {bi} elem {k} diverged: {a} vs {c}"
                );
            }
        }
    }

    /// Batched forward over a genuine multi-block batch produces finite, bounded
    /// logits of the right shape when the blocks have *different* distributions —
    /// the real training scenario (BatchNorm normalizes across the whole batch).
    #[test]
    fn test_forward_batched_heterogeneous_blocks_bounded() {
        use burn::backend::Autodiff;

        type Ad = Autodiff<NdArray>;
        let device = Default::default();
        let cfg = default_cfg();
        let model = BurnPointNet::<Ad>::new(&cfg, &device).unwrap();

        let n = 64usize;
        let b = 3usize;
        let offsets = [0.0f32, 3.0, 6.0];
        let mut batch_flat = Vec::with_capacity(b * n * N_FEATURES);
        for &off in &offsets {
            for i in 0..(n * N_FEATURES) {
                batch_flat.push(((i * 13 + 5) % 83) as f32 / 83.0 + off);
            }
        }

        let input = features_to_tensor_batched::<Ad>(batch_flat, b, n, N_FEATURES, &device);
        let out = model.forward_batched(input);
        assert_eq!(out.dims(), [b, n, cfg.n_classes]);

        let flat: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "batched heterogeneous outputs must all be finite"
        );
        let max_abs = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e3,
            "batched logits should be bounded; got {max_abs}"
        );
    }
}
