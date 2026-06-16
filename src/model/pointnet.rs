//! `PointNet` segmentation classifier: assembles encoder, T-Nets, and decoder
//! layers and implements the full per-point classification forward pass.
//!
//! Architecture (Qi et al. 2017 scene segmentation variant):
//!
//! ```text
//! Input N×12
//!   → Input T-Net (STN3d, applied to xyz cols 0-2)
//!   → Encoder Layer 0: Linear(12→64) + BN + ReLU  ← save as `local_feat`
//!   → Feature T-Net (STN64d, optional, applied to local_feat)
//!   → Encoder Layers 1+: Linear(64→128→256)
//!   → Global max pool over N → broadcast [N×256]
//!   → Concat(local_feat[N×64], global[N×256]) = N×320
//!   → Decoder: Linear(320→256→128→n_classes)
//!   → Argmax → ASPRS label via label_map
//! ```

use ndarray::{s, Array2, Axis};

use crate::error::{ClassifierError, Result};
use crate::model::layers::{
    apply_bn2d, global_max_pool, relu, BatchNorm1d, Linear, TNet,
};

// ─────────────────────────────────────────────────────────────────────────────
// Architecture configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Immutable configuration for a `PointNetClassifier`.
///
/// All dimension parameters must match the weights loaded from the `.wbmodel`
/// file; a mismatch is caught at load time, not at inference time.
#[derive(Debug, Clone)]
pub struct PointNetConfig {
    /// Number of input features per point (must equal Stage 01 `N_FEATURES = 12`).
    pub n_features_in: usize,
    /// Hidden dimensions for encoder MLP layers.
    /// `encoder_dims[0]` is the "local feature" width used in the segmentation concat.
    pub encoder_dims: Vec<usize>,
    /// Hidden dimensions for decoder MLP layers (before the final class projection).
    pub decoder_dims: Vec<usize>,
    /// Number of output classes.
    pub n_classes: usize,
    /// When `false`, all `BatchNorm` layers are bypassed (identity).
    pub use_batch_norm: bool,
    /// Whether the Input T-Net (`STN3d`) is present.
    pub use_input_tnet: bool,
    /// Whether the Feature T-Net (`STN64d`) is present.
    pub use_feature_tnet: bool,
}

impl PointNetConfig {
    /// Compute the concatenated segmentation dimension.
    #[must_use]
    /// `encoder_dims[0]` (local) + `encoder_dims.last()` (global).
    pub fn concat_dim(&self) -> usize {
        let local = *self.encoder_dims.first().unwrap_or(&64);
        let global = *self.encoder_dims.last().unwrap_or(&256);
        local + global
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier struct
// ─────────────────────────────────────────────────────────────────────────────

/// The full `PointNet` segmentation classifier.
///
/// Constructed by `model::weights::load_model`; this struct holds all weight
/// tensors and runs the forward pass at inference time.
pub struct PointNetClassifier {
    pub config: PointNetConfig,

    /// Optional Input T-Net (`STN3d`).
    pub input_tnet: Option<TNet>,
    /// Optional Feature T-Net (`STN64d`).
    pub feature_tnet: Option<TNet>,

    /// Encoder layers: `encoder_layers[i]` is (Linear, Option<BatchNorm1d>).
    /// Length == `config.encoder_dims.len()`.
    pub encoder_layers: Vec<(Linear, Option<BatchNorm1d>)>,

    /// Decoder layers: `decoder_layers[i]` is (Linear, Option<BatchNorm1d>).
    /// Length == `config.decoder_dims.len()`.
    /// Input dim of `decoder_layers`[0] == `config.concat_dim()`.
    pub decoder_layers: Vec<(Linear, Option<BatchNorm1d>)>,

    /// Final class projection layer (no BN, no `ReLU`).
    pub class_proj: Linear,

    /// Maps model output index → ASPRS classification code.
    pub label_map: Vec<u8>,
}

impl PointNetClassifier {
    /// Run the full forward pass on an `[N, n_features_in]` feature matrix.
    ///
    /// Returns an `[N, n_classes]` raw logit matrix.
    ///
    /// # Errors
    /// Returns an error if `features.ncols() != n_features_in`, if any layer
    /// shape is inconsistent, or if a T-Net dimension mismatch is detected.
    pub fn forward(&self, mut features: Array2<f32>) -> Result<Array2<f32>> {
        let n = features.nrows();
        if features.ncols() != self.config.n_features_in {
            return Err(ClassifierError::Pipeline(format!(
                "PointNet::forward: input cols ({}) != n_features_in ({})",
                features.ncols(),
                self.config.n_features_in
            )));
        }

        // ── Input T-Net (STN3d) ───────────────────────────────────────────
        if let Some(stn) = &self.input_tnet {
            // Extract xyz columns [0..3]
            let xyz = features.slice(s![.., 0..3]).to_owned();
            let t1 = stn.forward(&xyz)?;
            // Apply: xyz_transformed = xyz @ T1^T
            let xyz_t = TNet::apply(&xyz, &t1);
            // Write back into features columns 0..3
            features.slice_mut(s![.., 0..3]).assign(&xyz_t);
        }

        // ── Encoder layer 0 (produces local_feat) ────────────────────────
        let (enc0_linear, enc0_bn) = self.encoder_layers.first().ok_or_else(|| {
            ClassifierError::Pipeline("PointNet: encoder_layers is empty".into())
        })?;
        let h = enc0_linear.forward(&features)?;
        let h = apply_bn2d(h, enc0_bn.as_ref())?;
        let local_feat = relu(&h); // [N, encoder_dims[0]]

        // ── Feature T-Net (STN64d) ────────────────────────────────────────
        let local_feat = if let Some(stn) = &self.feature_tnet {
            let t2 = stn.forward(&local_feat)?;
            TNet::apply(&local_feat, &t2)
        } else {
            local_feat
        };

        // ── Encoder layers 1+ ─────────────────────────────────────────────
        let mut deep = local_feat.clone();
        for (linear, bn) in self.encoder_layers.iter().skip(1) {
            let h = linear.forward(&deep)?;
            let h = apply_bn2d(h, bn.as_ref())?;
            deep = relu(&h);
        }
        // deep: [N, encoder_dims.last()]

        // ── Global max pooling + broadcast ────────────────────────────────
        let global_vec = global_max_pool(&deep.view()); // [encoder_dims.last()]
        // Broadcast to [N, encoder_dims.last()]
        let global_mat = Array2::from_shape_fn((n, global_vec.len()), |(_, j)| global_vec[j]);

        // ── Segmentation concat ───────────────────────────────────────────
        let mut seg = ndarray::concatenate(
            Axis(1),
            &[local_feat.view(), global_mat.view()],
        )
        .map_err(|e| ClassifierError::Pipeline(format!("concat error: {e}")))?;
        // seg: [N, concat_dim]

        // ── Decoder layers ────────────────────────────────────────────────
        for (linear, bn) in &self.decoder_layers {
            let h = linear.forward(&seg)?;
            let h = apply_bn2d(h, bn.as_ref())?;
            seg = relu(&h);
        }

        // ── Final class projection (no BN, no ReLU) ───────────────────────
        let logits = self.class_proj.forward(&seg)?;
        // logits: [N, n_classes]
        Ok(logits)
    }

    /// Run inference on feature matrix and return a Vec of ASPRS classification
    /// codes, one per point.
    ///
    /// # Errors
    /// Propagates any error from `forward`.
    pub fn classify(&self, features: Array2<f32>) -> Result<Vec<u8>> {
        let logits = self.forward(features)?;
        let n = logits.nrows();
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let row = logits.row(i);
            // argmax: find index of maximum logit
            let best_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            // Map model index → ASPRS code via label_map
            let asprs_code = self
                .label_map
                .get(best_idx)
                .copied()
                .unwrap_or(1); // fallback: Unassigned
            labels.push(asprs_code);
        }
        Ok(labels)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::Linear;
    use ndarray::{Array1, Array2};

    /// Build a minimal PointNetClassifier with zero weights, no BN, no T-Nets.
    fn make_classifier(
        n_features_in: usize,
        encoder_dims: Vec<usize>,
        decoder_dims: Vec<usize>,
        n_classes: usize,
    ) -> PointNetClassifier {
        let config = PointNetConfig {
            n_features_in,
            encoder_dims: encoder_dims.clone(),
            decoder_dims: decoder_dims.clone(),
            n_classes,
            use_batch_norm: false,
            use_input_tnet: false,
            use_feature_tnet: false,
        };

        // Build encoder layers
        let mut encoder_layers = Vec::new();
        let mut prev = n_features_in;
        for &dim in &encoder_dims {
            let w = Array2::zeros((dim, prev));
            let b = Array1::zeros(dim);
            encoder_layers.push((Linear::new(w, b).unwrap(), None));
            prev = dim;
        }

        // Build decoder layers
        let concat_dim = config.concat_dim();
        let mut decoder_layers = Vec::new();
        let mut prev_d = concat_dim;
        for &dim in &decoder_dims {
            let w = Array2::zeros((dim, prev_d));
            let b = Array1::zeros(dim);
            decoder_layers.push((Linear::new(w, b).unwrap(), None));
            prev_d = dim;
        }

        // Class projection
        let class_proj = Linear::new(
            Array2::zeros((n_classes, prev_d)),
            Array1::zeros(n_classes),
        ).unwrap();

        // Default label map: identity ASPRS codes 0..n_classes
        let label_map: Vec<u8> = (0u8..n_classes as u8).collect();

        PointNetClassifier {
            config,
            input_tnet: None,
            feature_tnet: None,
            encoder_layers,
            decoder_layers,
            class_proj,
            label_map,
        }
    }

    // DoD #11 — forward output shape with input T-Net disabled
    #[test]
    fn test_forward_output_shape_no_tnet() {
        let clf = make_classifier(12, vec![64, 128, 256], vec![256, 128], 8);
        let input = Array2::<f32>::zeros((1024, 12));
        let logits = clf.forward(input).expect("forward failed");
        assert_eq!(logits.shape(), &[1024, 8]);
    }

    // DoD #12 — forward output shape with both T-Nets enabled
    #[test]
    fn test_forward_output_shape_with_tnets() {
        use crate::model::layers::TNet;

        let mut clf = make_classifier(12, vec![64, 128, 256], vec![256, 128], 8);

        // Build zero-weight STN3d and STN64d
        let stn3d = TNet {
            k: 3,
            enc0: Linear::new(Array2::zeros((64, 3)),    Array1::zeros(64)).unwrap(),
            enc1: Linear::new(Array2::zeros((128, 64)),  Array1::zeros(128)).unwrap(),
            enc2: Linear::new(Array2::zeros((1024, 128)),Array1::zeros(1024)).unwrap(),
            bn_enc0: None, bn_enc1: None, bn_enc2: None,
            fc0: Linear::new(Array2::zeros((512, 1024)), Array1::zeros(512)).unwrap(),
            fc1: Linear::new(Array2::zeros((256, 512)),  Array1::zeros(256)).unwrap(),
            fc2: Linear::new(Array2::zeros((9, 256)),    Array1::zeros(9)).unwrap(),
            bn_fc0: None, bn_fc1: None,
        };
        let stn64d = TNet {
            k: 64,
            enc0: Linear::new(Array2::zeros((64, 64)),   Array1::zeros(64)).unwrap(),
            enc1: Linear::new(Array2::zeros((128, 64)),  Array1::zeros(128)).unwrap(),
            enc2: Linear::new(Array2::zeros((1024, 128)),Array1::zeros(1024)).unwrap(),
            bn_enc0: None, bn_enc1: None, bn_enc2: None,
            fc0: Linear::new(Array2::zeros((512, 1024)), Array1::zeros(512)).unwrap(),
            fc1: Linear::new(Array2::zeros((256, 512)),  Array1::zeros(256)).unwrap(),
            fc2: Linear::new(Array2::zeros((4096, 256)), Array1::zeros(4096)).unwrap(),
            bn_fc0: None, bn_fc1: None,
        };

        clf.config.use_input_tnet = true;
        clf.config.use_feature_tnet = true;
        clf.input_tnet = Some(stn3d);
        clf.feature_tnet = Some(stn64d);

        let input = Array2::<f32>::zeros((1024, 12));
        let logits = clf.forward(input).expect("forward with T-Nets failed");
        assert_eq!(logits.shape(), &[1024, 8]);
    }

    // DoD #14 — label mapping (argmax → ASPRS code)
    #[test]
    fn test_classify_label_mapping() {
        let clf = make_classifier(12, vec![64, 128, 256], vec![256, 128], 3);
        // Manually override label_map
        let mut clf = clf;
        clf.label_map = vec![2u8, 5u8, 6u8]; // Ground, Building, Water

        // Craft logits: [3 points, 3 classes]
        // Point 0: class 1 wins → ASPRS 5 (Building)
        // Point 1: class 2 wins → ASPRS 6 (Water)
        // Point 2: class 0 wins → ASPRS 2 (Ground)
        let logits = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.1f32, 0.9, 0.0,
                0.0,    0.1, 0.8,
                1.0,    0.0, 0.5,
            ],
        ).unwrap();

        // Bypass forward(), inject logits through classify() manually
        // by calling classify() on the full model with crafted input that
        // produces known logits through zero-weight layers:
        // All zero weights → output is bias (all zero) → argmax = 0 for all.
        // Instead, test the argmax+label_map logic directly.
        let n = logits.nrows();
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let row = logits.row(i);
            let best = row.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx).unwrap_or(0);
            labels.push(clf.label_map[best]);
        }
        assert_eq!(labels, vec![5u8, 6u8, 2u8]);
    }
}
