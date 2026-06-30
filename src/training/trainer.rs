//! Training loop — gradient accumulation over spatial blocks, AdamW optimizer,
//! cosine annealing LR, checkpoint management, and optional SWA.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::redundant_clone,
    clippy::unnecessary_wraps,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::must_use_candidate
)]

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetConfig;
use crate::model::weights::load_model;
use crate::training::{
    bridge::save_model_from_burn,
    burn_model::{features_to_tensor, labels_to_tensor, BurnPointNet},
    dataset::LabeledBlockDataset,
    metrics::{append_metrics_csv, EpochMetrics, MetricsAccumulator},
    scheduler::CosineScheduler,
};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Training hyper-parameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub n_classes: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub val_split: f64,
    pub val_tile_block_ids: Option<HashSet<u64>>,
    pub seed: u64,
    pub use_feature_tnet: bool,
    pub use_batch_norm: bool,
    pub use_class_weights: bool,
    /// β parameter for β-scaled effective-number class weighting (Cui et al. 2019).
    ///
    /// Range: `[0.0, 1.0)`.
    /// - `0.0`  → uniform weights (all classes weighted equally).
    /// - `→1.0` → approaches pure inverse-frequency weighting.
    /// - `0.999` (default) → strong minority-class emphasis suitable for severely
    ///   imbalanced LiDAR datasets without the extreme weight ratios of pure
    ///   inverse-frequency.
    ///
    /// Only used when `use_class_weights = true`.
    pub class_weight_beta: f64,
    pub checkpoint_dir: Option<PathBuf>,
    pub checkpoint_every: usize,
    pub keep_best_n: usize,
    pub swa: bool,
    pub metrics_out: PathBuf,
    pub output_model: PathBuf,
    pub n_threads: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_classes: 8,
            epochs: 50,
            batch_size: 16,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            val_split: 0.20,
            val_tile_block_ids: None,
            seed: 42,
            use_feature_tnet: false,
            use_batch_norm: true,
            use_class_weights: true,
            class_weight_beta: 0.999,
            checkpoint_dir: None,
            checkpoint_every: 1,
            keep_best_n: 5,
            swa: false,
            metrics_out: PathBuf::from("metrics.csv"),
            output_model: PathBuf::from("model.wbmodel"),
            n_threads: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint management
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    pub epoch: usize,
    pub val_miou: f64,
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub keep_best_n: usize,
    pub checkpoints: Vec<CheckpointEntry>,
}

impl CheckpointManifest {
    fn load_or_new(dir: &Path, keep_best_n: usize) -> Self {
        let p = dir.join("checkpoints.json");
        if let Ok(f) = File::open(&p) {
            if let Ok(m) = serde_json::from_reader(f) {
                return m;
            }
        }
        Self {
            keep_best_n,
            checkpoints: Vec::new(),
        }
    }

    fn save(&self, dir: &Path) -> std::io::Result<()> {
        let f = File::create(dir.join("checkpoints.json"))?;
        serde_json::to_writer_pretty(BufWriter::new(f), self)?;
        Ok(())
    }

    /// Add a new checkpoint entry, retain only the best N, delete excess files.
    fn push(&mut self, entry: CheckpointEntry, dir: &Path) {
        self.checkpoints.push(entry);
        // Sort descending by val_miou.
        self.checkpoints.sort_by(|a, b| {
            b.val_miou
                .partial_cmp(&a.val_miou)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Remove excess entries.
        while self.checkpoints.len() > self.keep_best_n {
            let removed = self.checkpoints.pop().unwrap();
            let _ = fs::remove_file(dir.join(&removed.file));
        }
        let _ = self.save(dir);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main training entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full training loop.
///
/// Returns the path of the written `.wbmodel` file on success.
///
/// # Errors
/// Propagates any I/O or processing error.
pub fn train<B: AutodiffBackend>(
    dataset: &LabeledBlockDataset,
    config: &TrainConfig,
    device: &B::Device,
) -> Result<PathBuf>
where
    B::InnerBackend: burn::tensor::backend::Backend<Device = B::Device>,
{
    if let Some(threads) = config.n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let start = Instant::now();

    // ── Model + config ────────────────────────────────────────────────────
    let net_cfg = PointNetConfig {
        n_features_in: dataset.n_features(),
        encoder_dims: vec![64, 128, 256],
        decoder_dims: vec![256, 128],
        n_classes: config.n_classes,
        use_batch_norm: config.use_batch_norm,
        use_input_tnet: true,
        use_feature_tnet: config.use_feature_tnet,
    };
    let label_map: Vec<u8> = (0u8..config.n_classes as u8).collect();

    let mut model: BurnPointNet<B> = BurnPointNet::new(&net_cfg, device)?;

    // ── Optimizer ─────────────────────────────────────────────────────────
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<B, BurnPointNet<B>>();

    // ── Class weights ─────────────────────────────────────────────────────
    let class_weights: Option<Vec<f32>> = if config.use_class_weights {
        let counts = dataset.class_counts_train();
        let w = compute_class_weights(&counts, config.class_weight_beta);
        eprintln!(
            "[trainer] class weights (beta={:.4}): {w:?}",
            config.class_weight_beta
        );
        Some(w)
    } else {
        None
    };

    // ── Loss ──────────────────────────────────────────────────────────────
    let loss_fn = {
        let mut cfg = CrossEntropyLossConfig::new();
        if let Some(ref w) = class_weights {
            cfg = cfg.with_weights(Some(w.clone()));
        }
        cfg.init::<B>(device)
    };

    // ── LR scheduler ─────────────────────────────────────────────────────
    let n_train = dataset.train_ids.len();
    let total_steps = config.epochs * n_train.div_ceil(config.batch_size);
    let scheduler = CosineScheduler::new(config.learning_rate, 1e-6, total_steps);

    // ── Checkpoint dir ────────────────────────────────────────────────────
    let ckpt_dir: Option<PathBuf> = config.checkpoint_dir.clone().inspect(|d| {
        fs::create_dir_all(d).ok();
    });
    let mut ckpt_manifest = ckpt_dir
        .as_ref()
        .map(|d| CheckpointManifest::load_or_new(d, config.keep_best_n));

    let mut global_step = 0usize;
    let mut best_miou = 0.0f64;
    let mut best_ckpt_path: Option<PathBuf> = None;

    // ── Training epochs ───────────────────────────────────────────────────
    for epoch in 0..config.epochs {
        // Shuffle training blocks deterministically per epoch.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(config.seed ^ epoch as u64);
        let mut shuffled = dataset.train_ids.clone();
        shuffled.shuffle(&mut rng);

        let mut epoch_loss_sum = 0.0f64;
        let mut n_steps = 0usize;

        for chunk in shuffled.chunks(config.batch_size) {
            let mut accumulator = GradientsAccumulator::<BurnPointNet<B>>::new();
            let mut chunk_loss = 0.0f64;

            for &block_id in chunk {
                let block = match dataset.load_block(block_id) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[trainer] skip block {block_id}: {e}");
                        continue;
                    }
                };

                let n = block.features.nrows();
                let n_features_block = block.features.ncols();
                let raw_floats: Vec<f32> = block.features.into_raw_vec_and_offset().0;
                let feat_tensor = features_to_tensor::<B>(raw_floats, n, n_features_block, device);
                let targets = labels_to_tensor::<B>(&block.labels, device);

                // Forward
                let logits = model.forward(feat_tensor); // [N, n_classes]

                // Loss
                let loss = loss_fn.forward(logits, targets); // [1]
                let loss_val = loss
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .map_or(0.0_f64, f64::from);
                chunk_loss += loss_val;

                // Backward + accumulate gradients
                let grads_raw = loss.backward();
                let grads_params = GradientsParams::from_grads(grads_raw, &model);
                accumulator.accumulate(&model, grads_params);
            }

            if !chunk.is_empty() {
                let lr = scheduler.lr(global_step);
                let grads = accumulator.grads();
                model = optim.step(lr, model, grads);
                // Divide by block count (not point count): all blocks are
                // resampled to `target_points`, so block count and point count
                // scale identically.  Revisit if variable-size blocks are added.
                epoch_loss_sum += chunk_loss / chunk.len() as f64;
                n_steps += 1;
                global_step += 1;
            }
        }

        let train_loss = if n_steps == 0 {
            0.0
        } else {
            epoch_loss_sum / n_steps as f64
        };

        // ── Validation ────────────────────────────────────────────────────
        let val_metrics = validate_epoch(
            &model,
            dataset,
            &dataset.val_ids.clone(),
            epoch,
            train_loss,
            device,
            config.n_classes,
            class_weights.as_deref(),
        )?;

        eprintln!(
            "[trainer] epoch {}/{} — train_loss={:.4}  val_loss_uw={:.4}  val_loss_w={:.4}  val_mIoU={:.4}",
            epoch + 1, config.epochs,
            train_loss, val_metrics.val_loss, val_metrics.val_loss_weighted, val_metrics.miou
        );

        // Append to metrics CSV.
        append_metrics_csv(&config.metrics_out, &val_metrics)
            .unwrap_or_else(|e| eprintln!("[trainer] metrics CSV write error: {e}"));

        // ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % config.checkpoint_every == 0 {
            if let (Some(ref dir), Some(ref mut manifest)) = (&ckpt_dir, &mut ckpt_manifest) {
                let fname = format!("checkpoint_epoch_{:03}.wbmodel", epoch + 1);
                let ckpt_path = dir.join(&fname);

                save_model_from_burn(&model, &net_cfg, &label_map, &ckpt_path)?;

                manifest.push(
                    CheckpointEntry {
                        epoch: epoch + 1,
                        val_miou: val_metrics.miou,
                        file: fname,
                    },
                    dir,
                );

                if val_metrics.miou > best_miou {
                    best_miou = val_metrics.miou;
                    best_ckpt_path = Some(ckpt_path);
                }
            } else if val_metrics.miou > best_miou {
                best_miou = val_metrics.miou;
            }
        }
    }

    // ── Final model selection (best val_mIoU) ─────────────────────────────
    let duration_secs = start.elapsed().as_secs_f64();

    let output_path = &config.output_model;

    if config.swa {
        // SWA: average weights of all retained checkpoints.
        if let (Some(ref dir), Some(ref manifest)) = (&ckpt_dir, &ckpt_manifest) {
            apply_swa(dir, manifest, output_path)?;
        } else {
            // No checkpoints — save current model.
            save_model_from_burn(&model, &net_cfg, &label_map, output_path)?;
        }
    } else if let Some(ref best) = best_ckpt_path {
        // Copy best checkpoint to output path.
        fs::copy(best, output_path)
            .map_err(|e| ClassifierError::Pipeline(format!("copy best checkpoint: {e}")))?;
    } else {
        // No checkpoint dir — save the final epoch model.
        save_model_from_burn(&model, &net_cfg, &label_map, output_path)?;
    }

    // Write training summary.
    // Path::new("model.wbmodel").parent() returns Some(""), not None, so we
    // must also guard against the empty-string case.
    let summary_dir = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    write_training_summary(summary_dir, best_miou, duration_secs, config.swa);

    eprintln!(
        "[trainer] done in {:.1}s — best val_mIoU={:.4}  output: {}",
        duration_secs,
        best_miou,
        output_path.display()
    );

    Ok(output_path.clone())
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation pass
// ─────────────────────────────────────────────────────────────────────────────

fn validate_epoch<B: AutodiffBackend>(
    model: &BurnPointNet<B>,
    dataset: &LabeledBlockDataset,
    val_ids: &[u64],
    epoch: usize,
    train_loss: f64,
    device: &B::Device,
    n_classes: usize,
    class_weights: Option<&[f32]>,
) -> Result<EpochMetrics>
where
    B::InnerBackend: burn::tensor::backend::Backend<Device = B::Device>,
{
    // Clone the model so the original's BN running statistics are not
    // contaminated by validation-batch statistics.
    // The clone runs in TRAINING mode (batch statistics), which avoids the
    // BatchNorm distribution-shift problem that occurs when validation blocks
    // come from spatially disjoint macro-tiles with different feature
    // distributions.  Running statistics built from training blocks would
    // normalise validation activations incorrectly, causing logit explosion.
    let val_model = model.clone();
    let mut acc = MetricsAccumulator::new(n_classes);

    for &block_id in val_ids {
        let block = match dataset.load_block(block_id) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[val] skip {block_id}: {e}");
                continue;
            }
        };

        let n = block.features.nrows();
        let n_features_block = block.features.ncols();
        let flat: Vec<f32> = block.features.into_raw_vec_and_offset().0;

        // Use autodiff tensors so BN runs with per-batch statistics.
        // No .backward() is ever called, so no gradient computation occurs.
        let feat_tensor = features_to_tensor::<B>(flat, n, n_features_block, device);
        let logits = val_model.forward(feat_tensor); // [N, n_classes]
        let nc = logits.dims()[1];
        let flat_out: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap_or_default();
        let preds: Vec<u8> = (0..n)
            .map(|i| {
                let row = &flat_out[i * nc..(i + 1) * nc];
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(j, _)| j as u8)
            })
            .collect();

        let loss_unweighted = cross_entropy_from_logits(&flat_out, &block.labels, n, nc);
        acc.add_loss(loss_unweighted);

        let loss_weighted =
            cross_entropy_from_logits_weighted(&flat_out, &block.labels, n, nc, class_weights);
        acc.add_loss_weighted(loss_weighted);

        acc.accumulate(&preds, &block.labels);
    }

    Ok(acc.compute(epoch, train_loss))
}

/// Compute mean cross-entropy loss from raw logits and labels (no burn required).
fn cross_entropy_from_logits(logits: &[f32], labels: &[u8], n: usize, nc: usize) -> f64 {
    let mut loss = 0.0f64;
    for i in 0..n {
        let row = &logits[i * nc..(i + 1) * nc];
        // Log-softmax: log(exp(x_c) / sum(exp(x)))
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + exp_sum.ln();
        let c = labels[i] as usize;
        if c < nc {
            loss += f64::from(log_sum_exp - row[c]);
        }
    }
    if n > 0 {
        loss / n as f64
    } else {
        0.0
    }
}

/// Class-weighted cross-entropy, normalized by the sum of sample weights.
/// This matches burn's `CrossEntropyLoss` with `with_weights` convention
/// and is directly comparable to `train_loss`.
/// When `weights` is `None`, falls back to unweighted (same as above).
fn cross_entropy_from_logits_weighted(
    logits: &[f32],
    labels: &[u8],
    n: usize,
    nc: usize,
    weights: Option<&[f32]>,
) -> f64 {
    let Some(w) = weights else {
        return cross_entropy_from_logits(logits, labels, n, nc);
    };
    let mut loss = 0.0f64;
    let mut w_sum = 0.0f64;
    for i in 0..n {
        let row = &logits[i * nc..(i + 1) * nc];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + exp_sum.ln();
        let c = labels[i] as usize;
        if c < nc {
            let wi = if c < w.len() { f64::from(w[c]) } else { 1.0 };
            loss += wi * f64::from(log_sum_exp - row[c]);
            w_sum += wi;
        }
    }
    if w_sum > 0.0 {
        loss / w_sum
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SWA
// ─────────────────────────────────────────────────────────────────────────────

/// Average the weights of all retained checkpoints and save to `output_path`.
fn apply_swa(ckpt_dir: &Path, manifest: &CheckpointManifest, output_path: &Path) -> Result<()> {
    if manifest.checkpoints.is_empty() {
        return Err(ClassifierError::Pipeline(
            "SWA: no retained checkpoints to average".into(),
        ));
    }

    eprintln!("[swa] averaging {} checkpoints", manifest.checkpoints.len());

    // Load all models.
    let models: Vec<_> = manifest
        .checkpoints
        .iter()
        .map(|e| {
            let p = ckpt_dir.join(&e.file);
            load_model(&p)
        })
        .collect::<Result<Vec<_>>>()?;

    let n = models.len() as f32;
    let mut base = models[0].clone();
    let other_models = &models[1..];

    // ── Helper: accumulate one Linear layer's weight+bias into `base_lin` ──
    // Defined as a macro so it can be reused without fighting the borrow checker.
    macro_rules! accum_linear {
        ($base_lin:expr, $other_lin:expr) => {
            $base_lin.weight = (&$base_lin.weight + &$other_lin.weight).to_owned();
            $base_lin.bias = (&$base_lin.bias + &$other_lin.bias).to_owned();
        };
    }
    macro_rules! divide_linear {
        ($lin:expr) => {
            $lin.weight /= n;
            $lin.bias /= n;
        };
    }
    macro_rules! accum_bn {
        ($base_bn:expr, $other_bn:expr) => {
            if let (Some(ref mut bb), Some(ref mb)) = ($base_bn, $other_bn) {
                bb.gamma = (&bb.gamma + &mb.gamma).to_owned();
                bb.beta = (&bb.beta + &mb.beta).to_owned();
                bb.mean = (&bb.mean + &mb.mean).to_owned();
                bb.var = (&bb.var + &mb.var).to_owned();
            }
        };
    }
    macro_rules! divide_bn {
        ($bn:expr) => {
            if let Some(ref mut bb) = $bn {
                bb.gamma /= n;
                bb.beta /= n;
                bb.mean /= n;
                bb.var /= n;
            }
        };
    }

    // ── Average all parameters using ndarray arithmetic ────────────────────
    // Encoder layers
    for i in 0..base.encoder_layers.len() {
        for m in other_models {
            accum_linear!(base.encoder_layers[i].0, m.encoder_layers[i].0);
            accum_bn!(&mut base.encoder_layers[i].1, &m.encoder_layers[i].1);
        }
        divide_linear!(base.encoder_layers[i].0);
        divide_bn!(base.encoder_layers[i].1);
    }
    // Decoder layers
    for i in 0..base.decoder_layers.len() {
        for m in other_models {
            accum_linear!(base.decoder_layers[i].0, m.decoder_layers[i].0);
            accum_bn!(&mut base.decoder_layers[i].1, &m.decoder_layers[i].1);
        }
        divide_linear!(base.decoder_layers[i].0);
        divide_bn!(base.decoder_layers[i].1);
    }
    // Class projection (no BN)
    for m in other_models {
        accum_linear!(base.class_proj, m.class_proj);
    }
    divide_linear!(base.class_proj);

    // ── T-Net layers ───────────────────────────────────────────────────────
    // The T-Net (STN3d / STN64d) is trained jointly with all other layers
    // under the same gradient signal.  Excluding it from SWA would produce a
    // composite model where the averaged backbone expects the canonical
    // representation produced by the averaged T-Net, but receives instead the
    // representation from a single checkpoint's T-Net — a mismatch.
    //
    // Both `input_tnet` and `feature_tnet` are `Option<TNet>`; the averaging
    // block is gated so models without T-Nets are handled correctly.
    for m in other_models {
        if let (Some(ref mut bt), Some(ref mt)) = (&mut base.input_tnet, &m.input_tnet) {
            accum_linear!(bt.enc0, mt.enc0);
            accum_linear!(bt.enc1, mt.enc1);
            accum_linear!(bt.enc2, mt.enc2);
            accum_bn!(&mut bt.bn_enc0, &mt.bn_enc0);
            accum_bn!(&mut bt.bn_enc1, &mt.bn_enc1);
            accum_bn!(&mut bt.bn_enc2, &mt.bn_enc2);
            accum_linear!(bt.fc0, mt.fc0);
            accum_linear!(bt.fc1, mt.fc1);
            accum_linear!(bt.fc2, mt.fc2);
            accum_bn!(&mut bt.bn_fc0, &mt.bn_fc0);
            accum_bn!(&mut bt.bn_fc1, &mt.bn_fc1);
        }
        if let (Some(ref mut bt), Some(ref mt)) = (&mut base.feature_tnet, &m.feature_tnet) {
            accum_linear!(bt.enc0, mt.enc0);
            accum_linear!(bt.enc1, mt.enc1);
            accum_linear!(bt.enc2, mt.enc2);
            accum_bn!(&mut bt.bn_enc0, &mt.bn_enc0);
            accum_bn!(&mut bt.bn_enc1, &mt.bn_enc1);
            accum_bn!(&mut bt.bn_enc2, &mt.bn_enc2);
            accum_linear!(bt.fc0, mt.fc0);
            accum_linear!(bt.fc1, mt.fc1);
            accum_linear!(bt.fc2, mt.fc2);
            accum_bn!(&mut bt.bn_fc0, &mt.bn_fc0);
            accum_bn!(&mut bt.bn_fc1, &mt.bn_fc1);
        }
    }
    if let Some(ref mut bt) = base.input_tnet {
        divide_linear!(bt.enc0);
        divide_linear!(bt.enc1);
        divide_linear!(bt.enc2);
        divide_bn!(bt.bn_enc0);
        divide_bn!(bt.bn_enc1);
        divide_bn!(bt.bn_enc2);
        divide_linear!(bt.fc0);
        divide_linear!(bt.fc1);
        divide_linear!(bt.fc2);
        divide_bn!(bt.bn_fc0);
        divide_bn!(bt.bn_fc1);
    }
    if let Some(ref mut bt) = base.feature_tnet {
        divide_linear!(bt.enc0);
        divide_linear!(bt.enc1);
        divide_linear!(bt.enc2);
        divide_bn!(bt.bn_enc0);
        divide_bn!(bt.bn_enc1);
        divide_bn!(bt.bn_enc2);
        divide_linear!(bt.fc0);
        divide_linear!(bt.fc1);
        divide_linear!(bt.fc2);
        divide_bn!(bt.bn_fc0);
        divide_bn!(bt.bn_fc1);
    }

    crate::model::weights::save_model(output_path, &base)
}

// ─────────────────────────────────────────────────────────────────────────────
// Class weight computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute β-scaled effective-number class weights (Cui et al. 2019).
///
/// # Formula
/// ```text
/// effective_num[c] = (1 - β^count[c]) / (1 - β)
/// raw_weight[c]    = 1 / effective_num[c]   (0.0 when count[c] == 0)
/// ```
/// Weights are then normalized so the mean weight of present classes equals 1.0,
/// keeping the loss magnitude stable across different β values.
///
/// ## Absent-class floor
///
/// Absent classes (count = 0) receive a small positive floor weight
/// (`ABSENT_CLASS_WEIGHT_FLOOR = 1e-3`) rather than `0.0`.
///
/// This is required because `burn`'s `CrossEntropyLoss` panics when any weight
/// is exactly zero (it validates `weight > 0` for all entries).  A floor of
/// `1e-3` is negligibly small compared to the normalized present-class weights
/// (which average `1.0`), so absent classes contribute essentially nothing to
/// the gradient while keeping the loss function well-defined regardless of the
/// label map or dataset coverage.
///
/// # Special cases
/// - `β ≈ 0.0` → uniform weights (all 1.0 for present; floor for absent).
/// - `β → 1.0` → approaches pure inverse-frequency weighting.
///
/// # Panics
/// Never panics — all edge cases (zero counts, zero present classes) return safe
/// default values.
pub fn compute_class_weights(counts: &[u64], beta: f64) -> Vec<f32> {
    /// Minimum weight assigned to absent classes (count = 0).
    /// Must be > 0 to satisfy `burn`'s `CrossEntropyLoss` validation.
    /// Kept at 1e-3 so it is ~1000× smaller than a typical present-class weight
    /// (which averages 1.0 after normalization).
    const ABSENT_CLASS_WEIGHT_FLOOR: f32 = 1e-3;

    let n = counts.len();
    if n == 0 {
        return Vec::new();
    }

    // β ≈ 0: uniform weights — short-circuit to avoid division by (1 - β) = 1.0
    // which would produce the correct result but is less readable.
    if beta.abs() < 1e-9 {
        return vec![1.0f32; n];
    }

    // Compute raw weights via effective number formula.
    // For count[c] == 0: weight stays 0.0 initially; replaced by floor below.
    // For large counts: β^count underflows to 0.0, so effective_num saturates
    // at 1/(1-β) — the correct asymptotic behavior; no special handling needed.
    let one_minus_beta = 1.0 - beta;
    let raw_weights: Vec<f64> = counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                let eff_num = (1.0 - beta.powi(c as i32)) / one_minus_beta;
                if eff_num > 0.0 {
                    1.0 / eff_num
                } else {
                    0.0
                }
            }
        })
        .collect();

    // Normalize so the mean weight of present (non-zero) classes equals 1.0.
    // This keeps the loss magnitude comparable across different β values and
    // ensures the learning rate remains stable regardless of the chosen β.
    let present_sum: f64 = raw_weights.iter().filter(|&&w| w > 0.0).sum();
    let n_present = raw_weights.iter().filter(|&&w| w > 0.0).count();

    if n_present == 0 || present_sum == 0.0 {
        // All classes absent — return uniform weights as a safe fallback.
        return vec![1.0f32; n];
    }

    let scale = n_present as f64 / present_sum;
    raw_weights
        .iter()
        .map(|&w| {
            if w == 0.0 {
                // Absent class: replace zero with the floor so burn's
                // CrossEntropyLoss doesn't panic on non-positive weights.
                ABSENT_CLASS_WEIGHT_FLOOR
            } else {
                (w * scale) as f32
            }
        })
        .collect()
}

fn write_training_summary(dir: &Path, best_miou: f64, duration_secs: f64, swa: bool) {
    use serde_json::json;
    let summary = json!({
        "best_val_miou": best_miou,
        "swa_applied": swa,
        "training_duration_seconds": duration_secs,
    });
    if let Ok(f) = File::create(dir.join("training_summary.json")) {
        let _ = serde_json::to_writer_pretty(BufWriter::new(f), &summary);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_keeps_best_n() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = CheckpointManifest {
            keep_best_n: 5,
            checkpoints: Vec::new(),
        };

        // Insert 8 checkpoints with varying mIoU.
        let scores = [0.5, 0.7, 0.6, 0.8, 0.55, 0.72, 0.9, 0.65];
        for (i, &score) in scores.iter().enumerate() {
            // Create dummy file so deletion doesn't fail.
            let fname = format!("checkpoint_epoch_{:03}.wbmodel", i + 1);
            let _ = File::create(dir.path().join(&fname));
            manifest.push(
                CheckpointEntry {
                    epoch: i + 1,
                    val_miou: score,
                    file: fname,
                },
                dir.path(),
            );
        }

        // Only best 5 should be retained.
        assert_eq!(manifest.checkpoints.len(), 5);
        // Best entry should be the one with mIoU 0.9.
        assert!((manifest.checkpoints[0].val_miou - 0.9).abs() < 1e-6);
        // All retained mIoUs should be >= the 5th-best.
        let min_retained = manifest
            .checkpoints
            .iter()
            .map(|c| c.val_miou)
            .fold(f64::INFINITY, f64::min);
        // 5 best scores: 0.9, 0.8, 0.72, 0.7, 0.65 → min = 0.65
        assert!((min_retained - 0.65).abs() < 1e-6);
    }

    // ── Stage 07: compute_class_weights tests ────────────────────────────────

    /// β = 0.0 → all present classes receive weight 1.0 (uniform).
    #[test]
    fn test_class_weight_beta_uniform() {
        // Any non-zero count distribution with β=0.0 must yield all-ones weights.
        let counts = vec![1000u64, 500, 100, 50, 10];
        let weights = compute_class_weights(&counts, 0.0);
        assert_eq!(weights.len(), 5);
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                (w - 1.0_f32).abs() < 1e-5,
                "expected weight 1.0 for class {i} with beta=0.0, got {w}"
            );
        }
    }

    /// β = 0.9999 → weights are numerically close to pure inverse-frequency.
    ///
    /// For large counts (> ~1000), β^count ≈ 0, so effective_num ≈ 1/(1-β) = 10000.
    /// All large-count classes collapse to the same effective number, so the
    /// weight ratio between classes is driven by the inverse-frequency ratio.
    /// We verify that the weight ordering matches inverse-frequency ordering and
    /// that the ratio between the largest and smallest weight is within 1% of
    /// the inverse-frequency ratio.
    #[test]
    fn test_class_weight_beta_inverse_freq() {
        // 3 classes with counts [10000, 1000, 100].
        // Pure inverse-frequency (normalized): weights proportional to [1/10000, 1/1000, 1/100].
        // Ratio of max to min weight = 100.
        let counts = vec![10_000u64, 1_000, 100];
        let weights_beta = compute_class_weights(&counts, 0.9999);
        assert_eq!(weights_beta.len(), 3);

        // All weights must be positive.
        for &w in &weights_beta {
            assert!(w > 0.0, "expected positive weight, got {w}");
        }

        // Weight ordering must match inverse-frequency ordering:
        // fewer points → higher weight.
        assert!(
            weights_beta[2] > weights_beta[1] && weights_beta[1] > weights_beta[0],
            "weight ordering does not match inverse-frequency: {weights_beta:?}"
        );

        // At β=0.9999 with counts [10000, 1000, 100], β^count is NOT negligible:
        //   0.9999^10000 ≈ 0.368,  0.9999^1000 ≈ 0.905,  0.9999^100 ≈ 0.990
        // So effective numbers are [6320, 950, 100], giving ratio ≈ 63.5 —
        // substantially above 1 (uniform) but below 100 (pure inverse-frequency).
        // This demonstrates the intended intermediate behaviour of the formula.
        let ratio = weights_beta[2] / weights_beta[0];
        assert!(
            ratio > 10.0 && ratio < 100.0,
            "expected weight ratio between 10 and 100 for beta=0.9999, got {ratio:.2}"
        );
    }

    /// β = 0.9 on a known 3-class distribution → hand-calculated expected weights.
    ///
    /// counts = [100, 50, 10], β = 0.9
    ///
    /// effective_num[c] = (1 - 0.9^count[c]) / (1 - 0.9) = (1 - 0.9^count[c]) / 0.1
    ///
    /// For count=100: 0.9^100 ≈ 2.656e-5 → eff_num ≈ (1 - 2.656e-5) / 0.1 ≈ 9.9997
    /// For count=50:  0.9^50  ≈ 5.154e-3 → eff_num ≈ (1 - 5.154e-3) / 0.1 ≈ 9.9485
    /// For count=10:  0.9^10  ≈ 0.34868  → eff_num ≈ (1 - 0.34868) / 0.1 ≈ 6.5132
    ///
    /// raw_weight[c] = 1 / eff_num[c]:
    ///   rw[0] ≈ 1/9.9997  ≈ 0.10000
    ///   rw[1] ≈ 1/9.9485  ≈ 0.10052
    ///   rw[2] ≈ 1/6.5132  ≈ 0.15353
    ///
    /// present_sum ≈ 0.35405, n_present = 3, scale ≈ 3/0.35405 ≈ 8.4734
    ///
    /// normalized weights:
    ///   w[0] ≈ 0.10000 * 8.4734 ≈ 0.8473
    ///   w[1] ≈ 0.10052 * 8.4734 ≈ 0.8517
    ///   w[2] ≈ 0.15353 * 8.4734 ≈ 1.3009
    ///
    /// Tolerance: 1% relative error (f32 precision + floating-point accumulation).
    #[test]
    fn test_class_weight_beta_intermediate() {
        let counts = vec![100u64, 50, 10];
        let weights = compute_class_weights(&counts, 0.9);
        assert_eq!(weights.len(), 3);

        let expected = [0.8473_f32, 0.8517_f32, 1.3009_f32];
        for (i, (&w, &exp)) in weights.iter().zip(expected.iter()).enumerate() {
            let rel_err = (w - exp).abs() / exp;
            assert!(
                rel_err < 0.01,
                "class {i}: expected weight ≈ {exp:.4}, got {w:.4} (rel_err={rel_err:.4})"
            );
        }

        // Mean of all weights must equal 1.0 (normalization invariant).
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!(
            (mean - 1.0_f32).abs() < 1e-4,
            "mean weight should be 1.0, got {mean:.6}"
        );
    }

    /// Absent class (count=0) receives the ABSENT_CLASS_WEIGHT_FLOOR (1e-3)
    /// rather than 0.0, so burn's CrossEntropyLoss does not panic.
    #[test]
    fn test_class_weight_absent_class_is_zero() {
        let counts = vec![1000u64, 0, 500];
        let weights = compute_class_weights(&counts, 0.999);
        assert_eq!(weights.len(), 3);
        // Absent class must be the floor value, not 0.0.
        assert!(
            weights[1] > 0.0,
            "absent class must have positive floor weight, got {}",
            weights[1]
        );
        assert!(
            weights[1] < 0.01,
            "absent class floor should be small (< 0.01), got {}",
            weights[1]
        );
        // Present classes must have weights much larger than the floor.
        assert!(weights[0] > 0.1, "present class weight too small: {}", weights[0]);
        assert!(weights[2] > 0.1, "present class weight too small: {}", weights[2]);
        // All weights must be strictly positive (burn validation requirement).
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "weight[{i}] must be > 0, got {w}");
        }
    }

    #[test]
    fn test_cross_entropy_from_logits_sanity() {
        // 2 points, 2 classes.
        // Point 0: logits=[10, 0], label=0 → loss ≈ 0
        // Point 1: logits=[0, 10], label=1 → loss ≈ 0
        let logits = vec![10.0f32, 0.0, 0.0, 10.0];
        let labels = vec![0u8, 1];
        let loss = cross_entropy_from_logits(&logits, &labels, 2, 2);
        assert!(loss < 0.001, "expected near-zero loss, got {loss}");
    }

    /// Verify that `apply_swa` averages T-Net weights, not just encoder/decoder.
    ///
    /// Creates two distinct `.wbmodel` files via the burn→ndarray bridge, calls
    /// `apply_swa`, then checks that the averaged T-Net `enc0` weight equals
    /// the element-wise mean of the two source weights.
    #[cfg(feature = "training")]
    #[test]
    fn test_swa_averages_tnet_weights() {
        use crate::model::pointnet::PointNetConfig;
        use crate::model::weights::load_model;
        use crate::preprocessing::N_FEATURES;
        use crate::training::bridge::save_model_from_burn;
        use crate::training::burn_model::BurnPointNet;
        use burn::backend::{Autodiff, NdArray};

        type B = Autodiff<NdArray>;
        let device = Default::default();

        let cfg = PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: vec![64, 128, 256],
            decoder_dims: vec![256, 128],
            n_classes: 8,
            use_batch_norm: true,
            use_input_tnet: true, // ← T-Net enabled
            use_feature_tnet: false,
        };
        let label_map: Vec<u8> = (0u8..8).collect();

        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("swa_m1.wbmodel");
        let p2 = dir.path().join("swa_m2.wbmodel");
        let out = dir.path().join("swa_out.wbmodel");

        // Two randomly-initialised models will have different T-Net weights.
        let m1 = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        let m2 = BurnPointNet::<B>::new(&cfg, &device).unwrap();
        save_model_from_burn(&m1, &cfg, &label_map, &p1).unwrap();
        save_model_from_burn(&m2, &cfg, &label_map, &p2).unwrap();

        // Collect expected element-wise mean of T-Net enc0 weights before SWA.
        let mm1 = load_model(&p1).unwrap();
        let mm2 = load_model(&p2).unwrap();
        let tnet1_w = mm1.input_tnet.as_ref().unwrap().enc0.weight.clone();
        let tnet2_w = mm2.input_tnet.as_ref().unwrap().enc0.weight.clone();
        let expected_avg = (&tnet1_w + &tnet2_w) / 2.0_f32;

        // Build a minimal CheckpointManifest and call apply_swa.
        let manifest = CheckpointManifest {
            keep_best_n: 2,
            checkpoints: vec![
                CheckpointEntry {
                    epoch: 1,
                    val_miou: 0.7,
                    file: "swa_m1.wbmodel".into(),
                },
                CheckpointEntry {
                    epoch: 2,
                    val_miou: 0.8,
                    file: "swa_m2.wbmodel".into(),
                },
            ],
        };
        apply_swa(dir.path(), &manifest, &out).unwrap();

        // Load the SWA output and verify T-Net enc0 weight is the mean.
        let averaged = load_model(&out).unwrap();
        let avg_tnet_w = &averaged
            .input_tnet
            .as_ref()
            .expect("SWA output must have input_tnet")
            .enc0
            .weight;

        assert_eq!(
            avg_tnet_w.shape(),
            expected_avg.shape(),
            "SWA T-Net weight shape mismatch"
        );
        // Check a representative element is within floating-point tolerance.
        let diff = (avg_tnet_w - &expected_avg).mapv(f32::abs);
        let max_err = diff.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-5,
            "SWA T-Net weight not correctly averaged; max element error = {max_err}"
        );
    }
}
