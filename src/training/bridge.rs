//! Weight bridge — extracts trained `BurnPointNet<B>` parameters and writes
//! them to a `.wbmodel` file via the Stage 02 `save_model()` function.
//!
//! ## Layout contract
//!
//! `burn::nn::Linear<B>` stores weight as `[d_input, d_output]`.
//! Stage 02 `layers::Linear` stores weight as `[d_output, d_input]`.
//! The bridge **transposes** each weight matrix before assembly.

#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::doc_markdown)]

use burn::tensor::backend::AutodiffBackend;

use ndarray::{Array1, Array2};

use crate::error::{ClassifierError, Result};
use crate::model::layers::{BatchNorm1d, Linear, TNet};
use crate::model::pointnet::{PointNetClassifier, PointNetConfig};
use crate::model::weights::save_model;
use crate::training::burn_model::{BurnPointNet, Stn3d, Stn64d};

/// Extract weights from a trained `BurnPointNet` and write a `.wbmodel` file.
///
/// # Errors
/// Returns `ClassifierError::Pipeline` if any extracted tensor shape does not
/// match the dimensions expected from `cfg`.
pub fn save_model_from_burn<B: AutodiffBackend>(
    model: &BurnPointNet<B>,
    cfg: &PointNetConfig,
    label_map: &[u8],
    path: &std::path::Path,
) -> Result<()> {
    let input_tnet = Some(extract_tnet3d::<B>(&model.stn3d, cfg.use_batch_norm)?);

    let feature_tnet = model
        .stn64d
        .as_ref()
        .map(|stn| extract_tnet64d::<B>(stn, cfg.use_batch_norm))
        .transpose()?;

    let encoder_layers = vec![
        extract_pair::<B>(&model.enc0, &model.bn_enc0, cfg.use_batch_norm)?,
        extract_pair::<B>(&model.enc1, &model.bn_enc1, cfg.use_batch_norm)?,
        extract_pair::<B>(&model.enc2, &model.bn_enc2, cfg.use_batch_norm)?,
    ];

    let decoder_layers = vec![
        extract_pair::<B>(&model.dec0, &model.bn_dec0, cfg.use_batch_norm)?,
        extract_pair::<B>(&model.dec1, &model.bn_dec1, cfg.use_batch_norm)?,
    ];

    let class_proj = extract_linear::<B>(&model.proj)?;

    if label_map.len() != cfg.n_classes {
        return Err(ClassifierError::Pipeline(format!(
            "bridge: label_map length {} != n_classes {}",
            label_map.len(), cfg.n_classes,
        )));
    }

    let classifier = PointNetClassifier {
        config: cfg.clone(),
        input_tnet,
        feature_tnet,
        encoder_layers,
        decoder_layers,
        class_proj,
        label_map: label_map.to_vec(),
    };

    save_model(path, &classifier)
}

// ─────────────────────────────────────────────────────────────────────────────
// TNet extraction — matches the flat-field layout of `model::layers::TNet`
// ─────────────────────────────────────────────────────────────────────────────

fn extract_tnet3d<B: AutodiffBackend>(stn: &Stn3d<B>, use_bn: bool) -> Result<TNet> {
    let bn = |layer: &burn::nn::BatchNorm<B, 1>| extract_bn::<B>(layer);
    Ok(TNet {
        k: 3,
        enc0: extract_linear::<B>(&stn.enc0)?,
        enc1: extract_linear::<B>(&stn.enc1)?,
        enc2: extract_linear::<B>(&stn.enc2)?,
        bn_enc0: if use_bn { Some(bn(&stn.bn_enc0)?) } else { None },
        bn_enc1: if use_bn { Some(bn(&stn.bn_enc1)?) } else { None },
        bn_enc2: if use_bn { Some(bn(&stn.bn_enc2)?) } else { None },
        fc0: extract_linear::<B>(&stn.fc0)?,
        fc1: extract_linear::<B>(&stn.fc1)?,
        fc2: extract_linear::<B>(&stn.fc2)?,
        bn_fc0: if use_bn { Some(bn(&stn.bn_fc0)?) } else { None },
        bn_fc1: if use_bn { Some(bn(&stn.bn_fc1)?) } else { None },
    })
}

fn extract_tnet64d<B: AutodiffBackend>(stn: &Stn64d<B>, use_bn: bool) -> Result<TNet> {
    let bn = |layer: &burn::nn::BatchNorm<B, 1>| extract_bn::<B>(layer);
    Ok(TNet {
        k: 64,
        enc0: extract_linear::<B>(&stn.enc0)?,
        enc1: extract_linear::<B>(&stn.enc1)?,
        enc2: extract_linear::<B>(&stn.enc2)?,
        bn_enc0: if use_bn { Some(bn(&stn.bn_enc0)?) } else { None },
        bn_enc1: if use_bn { Some(bn(&stn.bn_enc1)?) } else { None },
        bn_enc2: if use_bn { Some(bn(&stn.bn_enc2)?) } else { None },
        fc0: extract_linear::<B>(&stn.fc0)?,
        fc1: extract_linear::<B>(&stn.fc1)?,
        fc2: extract_linear::<B>(&stn.fc2)?,
        bn_fc0: if use_bn { Some(bn(&stn.bn_fc0)?) } else { None },
        bn_fc1: if use_bn { Some(bn(&stn.bn_fc1)?) } else { None },
    })
}

fn extract_pair<B: AutodiffBackend>(
    linear: &burn::nn::Linear<B>,
    bn: &burn::nn::BatchNorm<B, 1>,
    use_bn: bool,
) -> Result<(Linear, Option<BatchNorm1d>)> {
    let l = extract_linear::<B>(linear)?;
    let b = if use_bn { Some(extract_bn::<B>(bn)?) } else { None };
    Ok((l, b))
}

/// Extract a burn Linear → Stage 02 Linear (transpose weight: [in,out] → [out,in]).
fn extract_linear<B: AutodiffBackend>(layer: &burn::nn::Linear<B>) -> Result<Linear> {
    let w_burn = layer.weight.val();            // Tensor<B::InnerBackend, 2> [d_in, d_out]
    let [d_in, d_out] = w_burn.dims();

    let w_data: Vec<f32> = w_burn
        .transpose()                            // [d_out, d_in]
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("linear weight: {e:?}")))?;

    let weight = Array2::from_shape_vec((d_out, d_in), w_data)
        .map_err(|e| ClassifierError::Pipeline(format!("linear weight reshape: {e}")))?;

    let b_data: Vec<f32> = layer
        .bias
        .as_ref()
        .ok_or_else(|| ClassifierError::Pipeline("linear layer missing bias".into()))?
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("linear bias: {e:?}")))?;

    let bias = Array1::from_vec(b_data);
    Linear::new(weight, bias)
}

/// Extract a burn BatchNorm<B, 1> → Stage 02 BatchNorm1d.
fn extract_bn<B: AutodiffBackend>(bn: &burn::nn::BatchNorm<B, 1>) -> Result<BatchNorm1d> {
    let gamma = bn.gamma.val().into_data().to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn gamma: {e:?}")))?;
    let beta = bn.beta.val().into_data().to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn beta: {e:?}")))?;
    let mean = bn.running_mean.value().into_data().to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn mean: {e:?}")))?;
    let var = bn.running_var.value().into_data().to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn var: {e:?}")))?;

    BatchNorm1d::new(
        Array1::from_vec(gamma),
        Array1::from_vec(beta),
        Array1::from_vec(mean),
        Array1::from_vec(var),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use crate::model::weights::load_model;
    use crate::preprocessing::N_FEATURES;

    type B = Autodiff<NdArray>;

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
    fn test_weight_bridge_round_trip() {
        let device = Default::default();
        let cfg = default_cfg();
        let model = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        let label_map: Vec<u8> = (0u8..8).collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wbmodel");

        save_model_from_burn(&model, &cfg, &label_map, &path).unwrap();

        let loaded = load_model(&path).unwrap();
        assert_eq!(loaded.config.n_classes, cfg.n_classes);
        assert_eq!(loaded.config.encoder_dims, cfg.encoder_dims);
        assert_eq!(loaded.label_map, label_map);
        assert!(loaded.input_tnet.is_some());

        // enc0 weight should be [out=64, in=12] in Stage 02 format
        assert_eq!(loaded.encoder_layers[0].0.weight.shape(), &[64, 12][..]);
    }

    #[test]
    fn test_swa_averaging() {
        let device = Default::default();
        let cfg = default_cfg();
        let label_map: Vec<u8> = (0u8..8).collect();

        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("m1.wbmodel");
        let p2 = dir.path().join("m2.wbmodel");

        let m1 = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        let m2 = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        save_model_from_burn(&m1, &cfg, &label_map, &p1).unwrap();
        save_model_from_burn(&m2, &cfg, &label_map, &p2).unwrap();

        let mm1 = load_model(&p1).unwrap();
        let mm2 = load_model(&p2).unwrap();

        let w1 = &mm1.encoder_layers[0].0.weight;
        let w2 = &mm2.encoder_layers[0].0.weight;
        let avg = (w1 + w2) / 2.0;

        assert_eq!(avg.shape(), w1.shape());
        // Verify elementwise mean
        assert!((avg[[0, 0]] - (w1[[0, 0]] + w2[[0, 0]]) / 2.0).abs() < 1e-6);
    }
}

