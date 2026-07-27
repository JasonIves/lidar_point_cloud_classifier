//! `.wbmodel` binary format — serialisation and deserialisation.
//!
//! ## File layout (all multi-byte fields little-endian)
//!
//! ```text
//! [Header]
//!   magic:              4 bytes  = b"WBML"
//!   version:            u8       = 1
//!   n_features_in:      u8
//!   n_encoder_layers:   u8
//!   n_decoder_layers:   u8
//!   n_classes:          u8
//!   use_batch_norm:     u8   (0/1)
//!   use_input_tnet:     u8   (0/1)
//!   use_feature_tnet:   u8   (0/1)
//!   reserved:           u8   = 0x00
//!   encoder_dims:       u16 × n_encoder_layers
//!   decoder_dims:       u16 × n_decoder_layers
//!   label_map:          u8  × n_classes
//!
//! [T-Net blocks — only when flag = 1]
//!   STN3d  (k=3,  dims fixed [64,128,1024,512,256,9])
//!   STN64d (k=64, dims fixed [64,128,1024,512,256,4096])
//!   Each T-Net = 6 layer blocks (3 encoder + 3 FC).
//!   Final FC layer block never has BN.
//!
//! [Main encoder blocks — n_encoder_layers layer blocks]
//! [Main decoder blocks — n_decoder_layers + 1 layer blocks (incl. class proj)]
//!
//! A layer block is:
//!   weight:  f32[dim_out × dim_in]
//!   bias:    f32[dim_out]
//!   [if layer has BN:]
//!     bn_gamma, bn_beta, bn_mean, bn_var: f32[dim_out] each
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::io::{self, Read, Write};
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::{ClassifierError, Result};
use crate::model::layers::{BatchNorm1d, Linear, TNet};
use crate::model::pointnet::{PointNetClassifier, PointNetConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"WBML";
const VERSION: u8 = 1;

/// Fixed T-Net mini-encoder hidden dims: [64, 128, 1024].
const TNET_ENC_DIMS: [usize; 3] = [64, 128, 1024];
/// Fixed T-Net FC decoder hidden dims (before final projection): [512, 256].
const TNET_FC_DIMS: [usize; 2] = [512, 256];

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise a trained `PointNetClassifier` to a `.wbmodel` file at `path`.
///
/// # Errors
/// Returns an error if the file cannot be created or any write operation fails.
pub fn save_model(path: &Path, model: &PointNetClassifier) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    let cfg = &model.config;

    // ── Header ────────────────────────────────────────────────────────────
    w.write_all(MAGIC)?;
    write_u8(&mut w, VERSION)?;
    write_u8(
        &mut w,
        u8::try_from(cfg.n_features_in)
            .map_err(|_| ClassifierError::Pipeline("n_features_in exceeds u8".into()))?,
    )?;
    write_u8(
        &mut w,
        u8::try_from(cfg.encoder_dims.len())
            .map_err(|_| ClassifierError::Pipeline("encoder_dims.len() exceeds u8".into()))?,
    )?;
    write_u8(
        &mut w,
        u8::try_from(cfg.decoder_dims.len())
            .map_err(|_| ClassifierError::Pipeline("decoder_dims.len() exceeds u8".into()))?,
    )?;
    write_u8(
        &mut w,
        u8::try_from(cfg.n_classes)
            .map_err(|_| ClassifierError::Pipeline("n_classes exceeds u8".into()))?,
    )?;
    write_u8(&mut w, u8::from(cfg.use_batch_norm))?;
    write_u8(&mut w, u8::from(cfg.use_input_tnet))?;
    write_u8(&mut w, u8::from(cfg.use_feature_tnet))?;
    write_u8(&mut w, 0x00)?; // reserved

    for &dim in &cfg.encoder_dims {
        write_u16(
            &mut w,
            u16::try_from(dim)
                .map_err(|_| ClassifierError::Pipeline("encoder dim exceeds u16".into()))?,
        )?;
    }
    for &dim in &cfg.decoder_dims {
        write_u16(
            &mut w,
            u16::try_from(dim)
                .map_err(|_| ClassifierError::Pipeline("decoder dim exceeds u16".into()))?,
        )?;
    }
    w.write_all(&model.label_map)?;

    // ── T-Net blocks ──────────────────────────────────────────────────────
    if let Some(stn) = &model.input_tnet {
        write_tnet(&mut w, stn, cfg.use_batch_norm)?;
    }
    if let Some(stn) = &model.feature_tnet {
        write_tnet(&mut w, stn, cfg.use_batch_norm)?;
    }

    // ── Main encoder blocks ───────────────────────────────────────────────
    for (i, (linear, bn)) in model.encoder_layers.iter().enumerate() {
        let has_bn = cfg.use_batch_norm && bn.is_some();
        write_layer_block(&mut w, linear, bn.as_ref(), has_bn)
            .map_err(|e| ClassifierError::Pipeline(format!("encoder[{i}] write: {e}")))?;
    }

    // ── Main decoder blocks ───────────────────────────────────────────────
    let n_dec = model.decoder_layers.len();
    for (i, (linear, bn)) in model.decoder_layers.iter().enumerate() {
        let has_bn = cfg.use_batch_norm && bn.is_some() && i < n_dec;
        write_layer_block(&mut w, linear, bn.as_ref(), has_bn)
            .map_err(|e| ClassifierError::Pipeline(format!("decoder[{i}] write: {e}")))?;
    }

    // ── Class projection (no BN) ──────────────────────────────────────────
    write_layer_block(&mut w, &model.class_proj, None, false)
        .map_err(|e| ClassifierError::Pipeline(format!("class_proj write: {e}")))?;

    w.flush()?;
    Ok(())
}

/// Deserialise a `.wbmodel` file from `path` into a `PointNetClassifier`.
///
/// # Errors
/// Returns an error if the file cannot be read, the magic or version does not
/// match, or any tensor block is truncated or has an inconsistent shape.
pub fn load_model(path: &Path) -> Result<PointNetClassifier> {
    let mut f = File::open(path)?;

    // ── Header ────────────────────────────────────────────────────────────
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(ClassifierError::Pipeline(format!(
            "wbmodel: bad magic {magic:?} (expected {MAGIC:?})"
        )));
    }

    let version = read_u8(&mut f)?;
    if version != VERSION {
        return Err(ClassifierError::Pipeline(format!(
            "wbmodel: unsupported version {version} (expected {VERSION})"
        )));
    }

    let n_features_in = read_u8(&mut f)? as usize;
    let n_encoder_layers = read_u8(&mut f)? as usize;
    let n_decoder_layers = read_u8(&mut f)? as usize;
    let n_classes = read_u8(&mut f)? as usize;
    let use_batch_norm = read_u8(&mut f)? != 0;
    let use_input_tnet = read_u8(&mut f)? != 0;
    let use_feature_tnet = read_u8(&mut f)? != 0;
    let _reserved = read_u8(&mut f)?;

    let mut encoder_dims = Vec::with_capacity(n_encoder_layers);
    for _ in 0..n_encoder_layers {
        encoder_dims.push(read_u16(&mut f)? as usize);
    }
    let mut decoder_dims = Vec::with_capacity(n_decoder_layers);
    for _ in 0..n_decoder_layers {
        decoder_dims.push(read_u16(&mut f)? as usize);
    }

    let mut label_map = vec![0u8; n_classes];
    f.read_exact(&mut label_map)?;

    let config = PointNetConfig {
        n_features_in,
        encoder_dims: encoder_dims.clone(),
        decoder_dims: decoder_dims.clone(),
        n_classes,
        use_batch_norm,
        use_input_tnet,
        use_feature_tnet,
    };

    // ── T-Net blocks ──────────────────────────────────────────────────────
    let input_tnet = if use_input_tnet {
        Some(read_tnet(&mut f, 3, use_batch_norm)?)
    } else {
        None
    };
    let feature_tnet = if use_feature_tnet {
        Some(read_tnet(&mut f, 64, use_batch_norm)?)
    } else {
        None
    };

    // ── Main encoder blocks ───────────────────────────────────────────────
    let mut encoder_layers = Vec::with_capacity(n_encoder_layers);
    let mut prev_dim = n_features_in;
    for (i, &out_dim) in encoder_dims.iter().enumerate() {
        let (linear, bn) = read_layer_block(&mut f, prev_dim, out_dim, use_batch_norm)
            .map_err(|e| ClassifierError::Pipeline(format!("encoder[{i}] read: {e}")))?;
        encoder_layers.push((linear, bn));
        prev_dim = out_dim;
    }

    // ── Main decoder blocks ───────────────────────────────────────────────
    let concat_dim = config.concat_dim();
    let mut decoder_layers = Vec::with_capacity(n_decoder_layers);
    let mut prev_dim = concat_dim;
    for (i, &out_dim) in decoder_dims.iter().enumerate() {
        let (linear, bn) = read_layer_block(&mut f, prev_dim, out_dim, use_batch_norm)
            .map_err(|e| ClassifierError::Pipeline(format!("decoder[{i}] read: {e}")))?;
        decoder_layers.push((linear, bn));
        prev_dim = out_dim;
    }

    // ── Class projection (no BN) ──────────────────────────────────────────
    let (class_proj, _) = read_layer_block(&mut f, prev_dim, n_classes, false)
        .map_err(|e| ClassifierError::Pipeline(format!("class_proj read: {e}")))?;

    Ok(PointNetClassifier {
        config,
        input_tnet,
        feature_tnet,
        encoder_layers,
        decoder_layers,
        class_proj,
        label_map,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// T-Net serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_tnet<W: Write>(w: &mut W, stn: &TNet, use_bn: bool) -> Result<()> {
    // 3 encoder layers + 3 FC layers; last FC has no BN
    let enc_layers = [
        (&stn.enc0, &stn.bn_enc0),
        (&stn.enc1, &stn.bn_enc1),
        (&stn.enc2, &stn.bn_enc2),
    ];
    for (i, (linear, bn)) in enc_layers.iter().enumerate() {
        write_layer_block(w, linear, bn.as_ref(), use_bn)
            .map_err(|e| ClassifierError::Pipeline(format!("tnet enc[{i}] write: {e}")))?;
    }
    let fc_layers = [(&stn.fc0, &stn.bn_fc0), (&stn.fc1, &stn.bn_fc1)];
    for (i, (linear, bn)) in fc_layers.iter().enumerate() {
        write_layer_block(w, linear, bn.as_ref(), use_bn)
            .map_err(|e| ClassifierError::Pipeline(format!("tnet fc[{i}] write: {e}")))?;
    }
    // fc2 never has BN
    write_layer_block(w, &stn.fc2, None, false)
        .map_err(|e| ClassifierError::Pipeline(format!("tnet fc2 write: {e}")))?;
    Ok(())
}

fn read_tnet<R: Read>(r: &mut R, k: usize, use_bn: bool) -> Result<TNet> {
    // Encoder: Linear(k→64), Linear(64→128), Linear(128→1024)
    let enc_dims = [TNET_ENC_DIMS[0], TNET_ENC_DIMS[1], TNET_ENC_DIMS[2]];
    let enc_in = [k, enc_dims[0], enc_dims[1]];
    let mut enc_layers = Vec::with_capacity(3);
    for i in 0..3 {
        let (linear, bn) = read_layer_block(r, enc_in[i], enc_dims[i], use_bn)
            .map_err(|e| ClassifierError::Pipeline(format!("tnet enc[{i}] read: {e}")))?;
        enc_layers.push((linear, bn));
    }

    // FC: Linear(1024→512), Linear(512→256), Linear(256→k²)  [last has no BN]
    let fc_in = [TNET_ENC_DIMS[2], TNET_FC_DIMS[0], TNET_FC_DIMS[1]];
    let fc_out = [TNET_FC_DIMS[0], TNET_FC_DIMS[1], k * k];
    let mut fc_layers = Vec::with_capacity(3);
    for i in 0..3 {
        let has_bn_here = use_bn && i < 2; // no BN on last FC
        let (linear, bn) = read_layer_block(r, fc_in[i], fc_out[i], has_bn_here)
            .map_err(|e| ClassifierError::Pipeline(format!("tnet fc[{i}] read: {e}")))?;
        fc_layers.push((linear, bn));
    }

    let [(enc0, bn_enc0), (enc1, bn_enc1), (enc2, bn_enc2)] = <[_; 3]>::try_from(enc_layers)
        .map_err(|_| ClassifierError::Pipeline("tnet enc layer count mismatch".into()))?;
    let [(fc0, bn_fc0), (fc1, bn_fc1), (fc2, _)] = <[_; 3]>::try_from(fc_layers)
        .map_err(|_| ClassifierError::Pipeline("tnet fc layer count mismatch".into()))?;

    Ok(TNet {
        k,
        enc0,
        enc1,
        enc2,
        bn_enc0,
        bn_enc1,
        bn_enc2,
        fc0,
        fc1,
        fc2,
        bn_fc0,
        bn_fc1,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer-block serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_layer_block<W: Write>(
    w: &mut W,
    linear: &Linear,
    bn: Option<&BatchNorm1d>,
    write_bn: bool,
) -> io::Result<()> {
    let weight_flat: &[f32] = linear.weight.as_slice().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "weight array is not contiguous")
    })?;
    write_f32_slice(w, weight_flat)?;
    write_f32_slice(w, linear.bias.as_slice().unwrap())?;
    if write_bn {
        let b = bn.expect("write_bn=true but bn is None");
        write_f32_slice(w, b.gamma.as_slice().unwrap())?;
        write_f32_slice(w, b.beta.as_slice().unwrap())?;
        write_f32_slice(w, b.mean.as_slice().unwrap())?;
        write_f32_slice(w, b.var.as_slice().unwrap())?;
    }
    Ok(())
}

fn read_layer_block<R: Read>(
    r: &mut R,
    dim_in: usize,
    dim_out: usize,
    read_bn: bool,
) -> Result<(Linear, Option<BatchNorm1d>)> {
    let weight_data = read_f32_vec(r, dim_out * dim_in)?;
    let weight = Array2::from_shape_vec((dim_out, dim_in), weight_data)
        .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;

    let bias_data = read_f32_vec(r, dim_out)?;
    let bias = Array1::from_vec(bias_data);

    let linear = Linear::new(weight, bias)?;

    let bn = if read_bn {
        let gamma = Array1::from_vec(read_f32_vec(r, dim_out)?);
        let beta = Array1::from_vec(read_f32_vec(r, dim_out)?);
        let mean = Array1::from_vec(read_f32_vec(r, dim_out)?);
        let var = Array1::from_vec(read_f32_vec(r, dim_out)?);
        Some(BatchNorm1d::new(gamma, beta, mean, var)?)
    } else {
        None
    };

    Ok((linear, bn))
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive I/O
// ─────────────────────────────────────────────────────────────────────────────

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32_slice<W: Write>(w: &mut W, data: &[f32]) -> io::Result<()> {
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        v.push(f32::from_le_bytes(buf));
    }
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pointnet::PointNetClassifier;
    use ndarray::{Array1, Array2};
    use tempfile::NamedTempFile;

    /// Build a deterministically initialised small classifier (no T-Nets, no BN).
    // Test fixture weight arrays are generated from small index ranges (<=768
    // elements); precision loss converting the loop index to f32 is negligible
    // and irrelevant to the round-trip correctness this test verifies.
    #[allow(clippy::cast_precision_loss)]
    fn make_small_classifier() -> PointNetClassifier {
        let config = PointNetConfig {
            n_features_in: 12,
            encoder_dims: vec![16, 32],
            decoder_dims: vec![24],
            n_classes: 4,
            use_batch_norm: false,
            use_input_tnet: false,
            use_feature_tnet: false,
        };

        // Encoder layer 0: Linear(12→16)
        let enc0_w: Vec<f32> = (0..16 * 12).map(|i| i as f32 * 0.001).collect();
        let enc0 = (
            Linear::new(
                Array2::from_shape_vec((16, 12), enc0_w).unwrap(),
                Array1::from_vec(vec![0.1f32; 16]),
            )
            .unwrap(),
            None,
        );
        // Encoder layer 1: Linear(16→32)
        let enc1_w: Vec<f32> = (0..32 * 16).map(|i| i as f32 * 0.001).collect();
        let enc1 = (
            Linear::new(
                Array2::from_shape_vec((32, 16), enc1_w).unwrap(),
                Array1::from_vec(vec![-0.1f32; 32]),
            )
            .unwrap(),
            None,
        );
        // Decoder layer: Linear(16+32=48 → 24)
        let dec0_w: Vec<f32> = (0..24 * 48).map(|i| i as f32 * 0.001).collect();
        let dec0 = (
            Linear::new(
                Array2::from_shape_vec((24, 48), dec0_w).unwrap(),
                Array1::zeros(24),
            )
            .unwrap(),
            None,
        );
        // Class projection: Linear(24→4)
        let proj_w: Vec<f32> = (0..4 * 24).map(|i| i as f32 * 0.01).collect();
        let class_proj = Linear::new(
            Array2::from_shape_vec((4, 24), proj_w).unwrap(),
            Array1::zeros(4),
        )
        .unwrap();

        PointNetClassifier {
            config,
            input_tnet: None,
            feature_tnet: None,
            encoder_layers: vec![enc0, enc1],
            decoder_layers: vec![dec0],
            class_proj,
            label_map: vec![1u8, 2u8, 5u8, 6u8],
        }
    }

    // DoD #13 — .wbmodel round-trip: save, reload, run identical input, bit-identical output
    // Test fixture input generation from small index ranges (8x12=96 elements);
    // precision loss converting the index to f32 is negligible here.
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn test_wbmodel_round_trip() -> crate::error::Result<()> {
        let model = make_small_classifier();

        // Run forward before save
        let input = Array2::<f32>::from_shape_fn((8, 12), |(i, j)| (i * 12 + j) as f32 * 0.01);
        let logits_before = model.forward(input.clone())?;

        // Save to temp file
        let tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;

        save_model(tmp.path(), &model)?;

        // Reload
        let loaded = load_model(tmp.path())?;

        // Run forward after load
        let logits_after = loaded.forward(input)?;

        // Verify bit-identical output
        assert_eq!(logits_before.shape(), logits_after.shape());
        for i in 0..logits_before.nrows() {
            for j in 0..logits_before.ncols() {
                assert_eq!(
                    logits_before[[i, j]].to_bits(),
                    logits_after[[i, j]].to_bits(),
                    "logit[{i},{j}] mismatch after round-trip"
                );
            }
        }
        Ok(())
    }

    /// Build a deterministically initialised classifier using the Stage 43
    /// canonical encoder/decoder dims (`CANONICAL_ENCODER_DIMS`,
    /// `CANONICAL_DECODER_DIMS`) and `N_FEATURES` input width.
    // Weight arrays are generated from small index ranges scaled by a tiny
    // constant; precision loss converting the index to f32 is negligible and
    // irrelevant to the round-trip correctness this test verifies.
    #[allow(clippy::cast_precision_loss)]
    fn make_canonical_classifier() -> PointNetClassifier {
        use crate::model::pointnet::{CANONICAL_DECODER_DIMS, CANONICAL_ENCODER_DIMS};
        use crate::preprocessing::N_FEATURES;

        let config = PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: CANONICAL_ENCODER_DIMS.to_vec(),
            decoder_dims: CANONICAL_DECODER_DIMS.to_vec(),
            n_classes: 4,
            use_batch_norm: false,
            use_input_tnet: false,
            use_feature_tnet: false,
        };

        let mut encoder_layers = Vec::with_capacity(CANONICAL_ENCODER_DIMS.len());
        let mut prev = N_FEATURES;
        for &dim in &CANONICAL_ENCODER_DIMS {
            let w: Vec<f32> = (0..dim * prev).map(|i| (i % 97) as f32 * 0.0001).collect();
            let b = vec![0.01f32; dim];
            encoder_layers.push((
                Linear::new(
                    Array2::from_shape_vec((dim, prev), w).unwrap(),
                    Array1::from_vec(b),
                )
                .unwrap(),
                None,
            ));
            prev = dim;
        }

        let concat_dim = config.concat_dim();
        let mut decoder_layers = Vec::with_capacity(CANONICAL_DECODER_DIMS.len());
        let mut prev_d = concat_dim;
        for &dim in &CANONICAL_DECODER_DIMS {
            let w: Vec<f32> = (0..dim * prev_d)
                .map(|i| (i % 97) as f32 * 0.0001)
                .collect();
            let b = vec![-0.01f32; dim];
            decoder_layers.push((
                Linear::new(
                    Array2::from_shape_vec((dim, prev_d), w).unwrap(),
                    Array1::from_vec(b),
                )
                .unwrap(),
                None,
            ));
            prev_d = dim;
        }

        let n_classes = config.n_classes;
        let proj_w: Vec<f32> = (0..n_classes * prev_d)
            .map(|i| (i % 97) as f32 * 0.001)
            .collect();
        let class_proj = Linear::new(
            Array2::from_shape_vec((n_classes, prev_d), proj_w).unwrap(),
            Array1::zeros(n_classes),
        )
        .unwrap();

        PointNetClassifier {
            config,
            input_tnet: None,
            feature_tnet: None,
            encoder_layers,
            decoder_layers,
            class_proj,
            label_map: vec![1u8, 2u8, 5u8, 6u8],
        }
    }

    // DoD #11 (Stage 43) — .wbmodel round-trip with the canonical encoder/
    // decoder dims: save, reload, run identical input, bit-identical output.
    // Test fixture input generation from small index ranges (16x17=272
    // elements); precision loss converting the index to f32 is negligible and
    // irrelevant to the round-trip correctness this test verifies.
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn test_wbmodel_round_trip_canonical_dims() -> crate::error::Result<()> {
        use crate::preprocessing::N_FEATURES;

        let model = make_canonical_classifier();

        let input = Array2::<f32>::from_shape_fn((16, N_FEATURES), |(i, j)| {
            ((i * N_FEATURES + j) % 97) as f32 * 0.01
        });
        let logits_before = model.forward(input.clone())?;

        let tmp = NamedTempFile::new().map_err(ClassifierError::Io)?;
        save_model(tmp.path(), &model)?;
        let loaded = load_model(tmp.path())?;
        let logits_after = loaded.forward(input)?;

        assert_eq!(logits_before.shape(), logits_after.shape());
        for i in 0..logits_before.nrows() {
            for j in 0..logits_before.ncols() {
                assert_eq!(
                    logits_before[[i, j]].to_bits(),
                    logits_after[[i, j]].to_bits(),
                    "logit[{i},{j}] mismatch after round-trip (canonical dims)"
                );
            }
        }
        Ok(())
    }
}
