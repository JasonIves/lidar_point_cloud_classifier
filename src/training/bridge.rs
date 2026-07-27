//! Weight bridge — extracts trained `BurnPointNet<B>` parameters and writes
//! them to a `.wbmodel` file via the Stage 02 `save_model()` function.
//!
//! ## Layout contract
//!
//! `burn::nn::Linear<B>` stores weight as `[d_input, d_output]`.
//! Stage 02 `layers::Linear` stores weight as `[d_output, d_input]`.
//! The bridge **transposes** each weight matrix before assembly.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown
)]

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

    // Stage 43: encoder_layers/decoder_layers are now dynamic-length Vec
    // fields on BurnPointNet, so extraction loops over them instead of
    // hand-unrolling a fixed 3-encoder/2-decoder layer count.
    let encoder_layers = model
        .encoder_layers
        .iter()
        .map(|(lin, bn)| extract_pair::<B>(lin, bn, cfg.use_batch_norm))
        .collect::<Result<Vec<_>>>()?;

    let decoder_layers = model
        .decoder_layers
        .iter()
        .map(|(lin, bn)| extract_pair::<B>(lin, bn, cfg.use_batch_norm))
        .collect::<Result<Vec<_>>>()?;

    let class_proj = extract_linear::<B>(&model.proj)?;

    if label_map.len() != cfg.n_classes {
        return Err(ClassifierError::Pipeline(format!(
            "bridge: label_map length {} != n_classes {}",
            label_map.len(),
            cfg.n_classes,
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

// Stage 24 (Code Quality Cleanup, item 4.2): `Stn3d<B>` and `Stn64d<B>` share
// identical field names (`enc0`/`enc1`/`enc2`/`bn_enc0..2`/`fc0..2`/`bn_fc0..1`);
// the only differences between the former `extract_tnet3d`/`extract_tnet64d`
// were the `k` constant (3 vs 64) and the struct type. This shared helper
// takes each field by reference so both thin wrappers below can call it
// without duplicating the 11-field extraction logic.
#[allow(clippy::too_many_arguments)]
fn extract_tnet_generic<B: AutodiffBackend>(
    k: usize,
    enc0: &burn::nn::Linear<B>,
    bn_enc0: &burn::nn::BatchNorm<B, 1>,
    enc1: &burn::nn::Linear<B>,
    bn_enc1: &burn::nn::BatchNorm<B, 1>,
    enc2: &burn::nn::Linear<B>,
    bn_enc2: &burn::nn::BatchNorm<B, 1>,
    fc0: &burn::nn::Linear<B>,
    fc1: &burn::nn::Linear<B>,
    fc2: &burn::nn::Linear<B>,
    bn_fc0: &burn::nn::BatchNorm<B, 1>,
    bn_fc1: &burn::nn::BatchNorm<B, 1>,
    use_bn: bool,
) -> Result<TNet> {
    let bn = |layer: &burn::nn::BatchNorm<B, 1>| extract_bn::<B>(layer);
    Ok(TNet {
        k,
        enc0: extract_linear::<B>(enc0)?,
        enc1: extract_linear::<B>(enc1)?,
        enc2: extract_linear::<B>(enc2)?,
        bn_enc0: if use_bn { Some(bn(bn_enc0)?) } else { None },
        bn_enc1: if use_bn { Some(bn(bn_enc1)?) } else { None },
        bn_enc2: if use_bn { Some(bn(bn_enc2)?) } else { None },
        fc0: extract_linear::<B>(fc0)?,
        fc1: extract_linear::<B>(fc1)?,
        fc2: extract_linear::<B>(fc2)?,
        bn_fc0: if use_bn { Some(bn(bn_fc0)?) } else { None },
        bn_fc1: if use_bn { Some(bn(bn_fc1)?) } else { None },
    })
}

fn extract_tnet3d<B: AutodiffBackend>(stn: &Stn3d<B>, use_bn: bool) -> Result<TNet> {
    extract_tnet_generic::<B>(
        3,
        &stn.enc0,
        &stn.bn_enc0,
        &stn.enc1,
        &stn.bn_enc1,
        &stn.enc2,
        &stn.bn_enc2,
        &stn.fc0,
        &stn.fc1,
        &stn.fc2,
        &stn.bn_fc0,
        &stn.bn_fc1,
        use_bn,
    )
}

fn extract_tnet64d<B: AutodiffBackend>(stn: &Stn64d<B>, use_bn: bool) -> Result<TNet> {
    extract_tnet_generic::<B>(
        64,
        &stn.enc0,
        &stn.bn_enc0,
        &stn.enc1,
        &stn.bn_enc1,
        &stn.enc2,
        &stn.bn_enc2,
        &stn.fc0,
        &stn.fc1,
        &stn.fc2,
        &stn.bn_fc0,
        &stn.bn_fc1,
        use_bn,
    )
}

fn extract_pair<B: AutodiffBackend>(
    linear: &burn::nn::Linear<B>,
    bn: &burn::nn::BatchNorm<B, 1>,
    use_bn: bool,
) -> Result<(Linear, Option<BatchNorm1d>)> {
    let l = extract_linear::<B>(linear)?;
    let b = if use_bn {
        Some(extract_bn::<B>(bn)?)
    } else {
        None
    };
    Ok((l, b))
}

/// Extract a burn Linear → Stage 02 Linear (transpose weight: [in,out] → [out,in]).
fn extract_linear<B: AutodiffBackend>(layer: &burn::nn::Linear<B>) -> Result<Linear> {
    let w_burn = layer.weight.val(); // Tensor<B::InnerBackend, 2> [d_in, d_out]
    let [d_in, d_out] = w_burn.dims();
    // Layout contract: burn 0.16 stores Linear weights as [d_input, d_output].
    // We transpose to [d_output, d_input] to match Stage 02's convention.
    // Stage 20 (Security Hardening): this was previously an assert!(), which
    // would panic in production if a burn version bump ever changed the
    // weight layout convention. A malformed/zero-dimension weight is now
    // surfaced as a normal Result error instead of crashing the process.
    if d_in == 0 || d_out == 0 {
        return Err(ClassifierError::Pipeline(format!(
            "burn Linear weight has unexpected zero dimension: [{d_in}, {d_out}] — \
             verify burn weight layout convention has not changed"
        )));
    }

    let w_data: Vec<f32> = w_burn
        .transpose() // [d_out, d_in]
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
    let gamma = bn
        .gamma
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn gamma: {e:?}")))?;
    let beta = bn
        .beta
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn beta: {e:?}")))?;
    let mean = bn
        .running_mean
        .value()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| ClassifierError::Pipeline(format!("bn mean: {e:?}")))?;
    let var = bn
        .running_var
        .value()
        .into_data()
        .to_vec::<f32>()
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
    use crate::model::weights::load_model;
    use crate::preprocessing::N_FEATURES;
    use burn::backend::{Autodiff, NdArray};

    type B = Autodiff<NdArray>;

    fn default_cfg() -> PointNetConfig {
        PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: crate::model::pointnet::CANONICAL_ENCODER_DIMS.to_vec(),
            decoder_dims: crate::model::pointnet::CANONICAL_DECODER_DIMS.to_vec(),
            n_classes: 8,
            use_batch_norm: true,
            use_input_tnet: true,
            use_feature_tnet: false,
        }
    }

    #[test]
    fn test_weight_bridge_round_trip() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
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

        // enc0 weight should be [out=64, in=N_FEATURES] in Stage 02 format
        assert_eq!(
            loaded.encoder_layers[0].0.weight.shape(),
            &[64, N_FEATURES][..]
        );
    }

    #[test]
    fn test_swa_averaging() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
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
        assert!((avg[[0, 0]] - f32::midpoint(w1[[0, 0]], w2[[0, 0]])).abs() < 1e-6);
    }

    // ── Stage 40: Burn ↔ ndarray forward-equivalence regression test ─────────

    /// Cross-framework forward-equivalence regression test (Stage 40).
    ///
    /// Bridges a freshly-constructed `BurnPointNet` (untrained, `use_input_tnet:
    /// true`) to a `.wbmodel` and asserts that `BurnPointNet::valid().forward()`
    /// (the training-time validation path, `trainer::validate_epoch`) and the
    /// bridged `PointNetClassifier::forward()` (the deployed ndarray inference
    /// path) produce numerically equivalent logits on the same input.
    ///
    /// Burn's `Linear` layers use non-zero random initialization by default, so
    /// the Input T-Net's learned transform `T = I + learned_residual` is already
    /// asymmetric (`T != T^T`) on a freshly constructed model — no training
    /// steps are needed to exercise the Stage 40 transpose bug. Before the
    /// Stage 40 fix (`TNet::apply` computing `input @ T^T` instead of
    /// `input @ T`), this test would fail because the ndarray path applied a
    /// systematically different spatial transform to the input xyz coordinates
    /// than the burn path, exactly reproducing the mIoU 0.60 (training
    /// validation) vs 0.02 (deployed `evaluate`) collapse observed on
    /// bit-identical model weights and data. See
    /// `docs/stages/stage-40-tnet-transpose-fix.md`.
    // Test fixture flat feature array from a small deterministic index
    // range; precision loss converting the index to f32 is negligible and
    // irrelevant to the burn/ndarray forward-output-agreement behaviour this
    // test verifies.
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn test_burn_and_ndarray_forward_outputs_agree_after_bridge() {
        use crate::training::burn_model::features_to_tensor;
        use burn::module::AutodiffModule;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let cfg = default_cfg();
        assert!(
            cfg.use_input_tnet,
            "this test only exercises the Stage 40 bug when the Input T-Net is enabled"
        );

        let model = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        let label_map: Vec<u8> = (0u8..8).collect();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equiv.wbmodel");
        save_model_from_burn(&model, &cfg, &label_map, &path).unwrap();
        let loaded = load_model(&path).unwrap();

        // Deterministic, non-trivial input (values spread across [-0.5, 0.5)).
        let n = 16usize;
        let flat: Vec<f32> = (0..(n * N_FEATURES))
            .map(|i| (i % 37) as f32 / 37.0 - 0.5)
            .collect();

        // Burn-side: inference-mode forward pass, matching trainer.rs's
        // `validate_epoch` path (`model.valid()` on the inner backend).
        let val_model = model.valid();
        let inner_input = features_to_tensor::<<B as AutodiffBackend>::InnerBackend>(
            flat.clone(),
            n,
            N_FEATURES,
            &device,
        );
        let burn_logits: Vec<f32> = val_model
            .forward(inner_input)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // ndarray-side: deployed inference path via the bridged `.wbmodel`.
        let features = Array2::from_shape_vec((n, N_FEATURES), flat).unwrap();
        let ndarray_logits = loaded.forward(features).unwrap();
        let ndarray_flat: Vec<f32> = ndarray_logits.iter().copied().collect();

        assert_eq!(
            burn_logits.len(),
            ndarray_flat.len(),
            "burn and ndarray logit vectors must have the same length"
        );
        for (i, (b, nd)) in burn_logits.iter().zip(ndarray_flat.iter()).enumerate() {
            assert!(
                (b - nd).abs() < 1e-3,
                "Burn vs ndarray forward mismatch at flat index {i}: burn={b}, ndarray={nd} \
                 (this indicates the training-time and deployed-inference forward passes have \
                 diverged — see the Stage 40 T-Net transpose bug)"
            );
        }
    }
}
