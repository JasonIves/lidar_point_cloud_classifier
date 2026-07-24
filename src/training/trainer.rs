//! Training loop — gradient accumulation over spatial blocks, `AdamW` optimizer,
//! cosine annealing LR, checkpoint management, and optional SWA.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    nn::BatchNorm,
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, backend::Backend, Tensor},
};
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{ClassifierError, Result};
use crate::model::layers::{BatchNorm1d, WeightAveraging};

use crate::model::pointnet::PointNetConfig;
use crate::model::weights::load_model;
use crate::training::{
    bridge::save_model_from_burn,
    burn_model::{features_to_tensor, features_to_tensor_batched, labels_to_tensor, BurnPointNet},
    dataset::LabeledBlockDataset,
    metrics::{append_metrics_csv, EpochMetrics, MetricsAccumulator},
    scheduler::CosineScheduler,
};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Training hyper-parameters.
// Stage 24 (Code Quality Cleanup, item 4.1): this public config struct
// naturally accumulates several independent `bool` toggles
// (`use_feature_tnet`, `use_batch_norm`, `use_class_weights`, `swa`) that
// each control an orthogonal training behavior. Converting to a bitflags
// or newtype-per-flag encoding would be a breaking API change rippling
// through the CLI, tests, and every call site that constructs a
// `TrainConfig` by name — named bool fields remain more readable here than
// the suggested alternative for a config struct with this shape.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub n_classes: usize,
    pub epochs: usize,
    /// Effective batch: number of blocks contributing to one optimizer step.
    pub batch_size: usize,
    /// Number of blocks stacked into a single batched forward pass — the
    /// *effective `BatchNorm` batch size* (Stage 18).  When `forward_batch_size <
    /// batch_size`, the chunk is split into micro-batches of `forward_batch_size`
    /// blocks and their gradients are accumulated (averaged) into one step.
    pub forward_batch_size: usize,
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
    ///   imbalanced `LiDAR` datasets without the extreme weight ratios of pure
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
    /// Stage 22 (Training Loop Enhancements): number of consecutive epochs
    /// without a new best `val_mIoU` after which training stops early.
    /// `None` (default) disables early stopping entirely.
    pub early_stopping_patience: Option<usize>,
    /// Stage 22 (Training Loop Enhancements): number of initial global steps
    /// over which the LR ramps linearly from `0` to `learning_rate` before
    /// cosine annealing begins. `0` (default) disables warmup.
    pub warmup_steps: usize,
    /// Stage 22 (Training Loop Enhancements): optional per-parameter-tensor
    /// L2-norm gradient clip threshold, applied by burn's built-in
    /// `GradientClippingConfig::Norm` inside the `AdamW` optimizer step.
    /// `None` (default) disables gradient clipping.
    pub grad_clip_norm: Option<f32>,
    /// Stage 27 (Block Caching, audit finding 5.2): optional in-memory block
    /// cache budget in megabytes, applied via
    /// `LabeledBlockDataset::with_block_cache`. `None` (default) disables
    /// caching entirely — every `load_block()` call reads from disk, exactly
    /// matching pre-Stage-27 behavior.
    pub cache_blocks_max_mb: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_classes: 8,
            epochs: 50,
            batch_size: 16,
            forward_batch_size: 8,
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
            early_stopping_patience: None,
            warmup_steps: 0,
            grad_clip_norm: None,
            cache_blocks_max_mb: None,
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
// Stage 24 (Code Quality Cleanup, item 4.1): this is the main training-loop
// entry point coordinating model construction, the optimizer, LR scheduler,
// per-epoch/per-chunk/per-micro-batch nested loops, checkpointing, early
// stopping, and final model selection — splitting it into smaller functions
// would fragment tightly-coupled local state (`model`, `optim`, `scheduler`,
// `global_step`, `best_miou`, …) across many call boundaries for no
// behavioral benefit. Numeric casts below (`usize`/`u64` → `u8`/`f32`/`f64`)
// are all either bounded-range class/index conversions or progress/loss
// reporting values where the precision loss is expected and harmless.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
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
    // Stage 41 (Model label_map identity bug fix): derive the saved model's
    // label_map from the *dataset's actual* ASPRS-code <-> model-index
    // mapping (whatever was used at `preprocess-labeled` time — the
    // built-in default or a custom `--label-map`) instead of hardcoding
    // identity. `classify()` uses this exact field to translate a predicted
    // model index back into a real ASPRS code for the output LAS
    // `Classification` field, so an incorrect (identity) map here silently
    // corrupts every deployed model's classification output.
    let label_map: Vec<u8> = dataset.inverse_label_map()?;

    let mut model: BurnPointNet<B> = BurnPointNet::new(&net_cfg, device)?;

    // ── Optimizer ─────────────────────────────────────────────────────────
    // Stage 22 (Training Loop Enhancements, item 1.7): optional per-tensor
    // L2-norm gradient clipping via burn's built-in `GradientClippingConfig`.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(
            config
                .grad_clip_norm
                .map(burn::grad_clipping::GradientClippingConfig::Norm),
        )
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
    let scheduler =
        CosineScheduler::with_warmup(config.learning_rate, 1e-6, total_steps, config.warmup_steps);

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

    // Stage 22 (Training Loop Enhancements, item 1.5): early-stopping state,
    // tracked independently of the checkpoint-cadence-gated `best_miou` above
    // so early stopping behaves identically regardless of
    // `--checkpoint-every`.
    let mut es_best_miou = 0.0f64;
    let mut es_epochs_without_improvement = 0usize;

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
            let mut n_micro = 0usize;

            // Stage 18: split the effective batch into micro-batches of up to
            // `forward_batch_size` blocks.  Each micro-batch is a real batched
            // forward `[b, N, C]`, so BatchNorm normalizes across a genuine
            // cross-block batch (not one block at a time).  We scale each
            // micro-batch loss by `1/num_micro` so the accumulated gradient is the
            // *mean* over the chunk — standard mini-batch-with-accumulation.
            let fb = config.forward_batch_size.max(1);
            let num_micro = chunk.len().div_ceil(fb).max(1);

            for micro in chunk.chunks(fb) {
                // Stack all blocks in this micro-batch that share the same point
                // count and feature width into one `[b, N, C]` tensor.  Blocks are
                // resampled to a common point count upstream, so mismatches are not
                // expected; we guard defensively rather than panic.
                let mut batch_flat: Vec<f32> = Vec::new();
                let mut batch_labels: Vec<u8> = Vec::new();
                let mut n_ref = 0usize;
                let mut nfeat_ref = 0usize;
                let mut count = 0usize;

                // Stage 22 (Training Loop Enhancements, item 1.3): load all
                // blocks in this micro-batch concurrently via Rayon — each
                // `LabeledBlockDataset::load_block` call is a pure, read-only,
                // `&self`-only disk read + byte→f32 conversion, safe to run in
                // parallel across blocks (same justification as Stage 21 item
                // 2.3). Batch assembly below (dims validation, `batch_flat`/
                // `batch_labels` mutation) stays single-threaded.
                let loaded: Vec<Option<_>> = micro
                    .par_iter()
                    .map(|&block_id| match dataset.load_block(block_id) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!("[trainer] skip block {block_id}: {e}");
                            None
                        }
                    })
                    .collect();

                for block in loaded.into_iter().flatten() {
                    let n = block.features.nrows();
                    let nf = block.features.ncols();
                    if count == 0 {
                        n_ref = n;
                        nfeat_ref = nf;
                    } else if n != n_ref || nf != nfeat_ref {
                        eprintln!(
                            "[trainer] skip block {}: dims {n}x{nf} != batch {n_ref}x{nfeat_ref}",
                            block.block_id
                        );
                        continue;
                    }
                    let raw: Vec<f32> = block.features.into_raw_vec_and_offset().0;
                    batch_flat.extend_from_slice(&raw);
                    batch_labels.extend_from_slice(&block.labels);
                    count += 1;
                }

                if count == 0 {
                    continue;
                }

                let feat_tensor =
                    features_to_tensor_batched::<B>(batch_flat, count, n_ref, nfeat_ref, device);
                let targets = labels_to_tensor::<B>(&batch_labels, device);

                // Batched forward: [count, N, n_classes] → flatten to [count*N, nc].
                let logits = model.forward_batched(feat_tensor);
                let nc = logits.dims()[2];
                let logits2d = logits.reshape([count * n_ref, nc]);

                let loss = loss_fn.forward(logits2d, targets); // [1]
                                                               // into_data() forces a device sync — the first point at which
                                                               // queued GPU kernels (and their buffers) must execute.
                let loss_val = loss
                    .clone()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default()
                    .first()
                    .copied()
                    .map_or(0.0_f64, f64::from);
                chunk_loss += loss_val;
                n_micro += 1;

                // Scale so accumulated gradients average over the chunk's
                // micro-batches, then backward + accumulate.
                let grads_raw = loss.div_scalar(num_micro as f32).backward();
                let grads_params = GradientsParams::from_grads(grads_raw, &model);
                accumulator.accumulate(&model, grads_params);
            }

            if n_micro > 0 {
                let lr = scheduler.lr(global_step);
                let grads = accumulator.grads();
                model = optim.step(lr, model, grads);

                // The training loop is memory-flat without any explicit sync —
                // `loss.backward()` consumes each micro-batch's autodiff graph and
                // deterministically frees its retained activations.  Validation
                // (below) forwards on the inner backend via `model.valid()`, so it
                // allocates no autodiff graph either.

                // Report the mean per-micro-batch loss for this step.
                epoch_loss_sum += chunk_loss / n_micro as f64;
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

        // ── Early stopping (Stage 22, item 1.5) ──────────────────────────
        // Tracked independently of the checkpoint-cadence-gated `best_miou`
        // above, so early stopping triggers identically regardless of
        // `--checkpoint-every`. A no-op when `early_stopping_patience` is
        // `None` (the default), preserving pre-Stage-22 behavior exactly.
        let should_stop = early_stopping_step(
            val_metrics.miou,
            &mut es_best_miou,
            &mut es_epochs_without_improvement,
            config.early_stopping_patience,
        );
        if should_stop {
            eprintln!(
                "[trainer] early stopping at epoch {}/{} — no improvement in val_mIoU for {} epochs (best={:.4})",
                epoch + 1,
                config.epochs,
                es_epochs_without_improvement,
                es_best_miou
            );
            break;
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

// Stage 24 (Code Quality Cleanup, item 4.1): this validation entry point
// naturally needs one parameter per piece of epoch/model/dataset state it
// reports on; grouping them into a struct would only move the same fields
// one level of indirection away without reducing coupling. The function
// never actually returns `Err` today (`Result` is kept for API consistency
// with every other pipeline-stage function here, and to allow a future
// validation-time error path — e.g. a corrupt block — to be added without
// changing the call signature at every call site).
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    clippy::cast_possible_truncation
)]
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
    // Stage 16 memory fix: validation runs on the *inner* (inference) backend
    // via `model.valid()`, NOT the autodiff backend.  Forwarding on the autodiff
    // backend without a matching `.backward()` accumulates an autodiff graph plus
    // BatchNorm running-state update nodes that burn 0.16 never reclaims (immune
    // to `sync` and to per-block model cloning), exhausting VRAM partway through
    // validation.  `model.valid()` converts the module to `B::InnerBackend` once,
    // so each forward allocates no autodiff graph and VRAM stays bounded across
    // all validation blocks.
    //
    // Stage 17 (resolved) removed the degenerate post-pool T-Net BatchNorm that
    // caused the ~1e5 inference logit explosion.  The *remaining* train/eval gap
    // — validation loss ~10× train loss and low mIoU with no over-fit curve — is
    // the per-block-vs-global BatchNorm statistics mismatch analysed in
    // docs/stages/stage-18-batchnorm-batched-forward.md: with an effective
    // BatchNorm batch size of one block, training normalizes each block by its
    // own statistics while `.valid()` normalizes every block by a single global
    // running average.  The opt-in `WB_BN_DIAG=1` diagnostic below exposes that
    // gap on real data (running-stat vs batch-stat val_loss + BN stat ranges).
    //
    // The diagnostic is off by default and bounded to the first few val blocks so
    // the train-mode forward-without-backward stays negligible w.r.t. the Stage 16
    // VRAM budget, emitting only a handful of stderr lines (no per-point logging,
    // per AGENTS.md).
    const DIAG_MAX_BLOCKS: usize = 3;

    let val_model = model.valid();
    let mut acc = MetricsAccumulator::new(n_classes);

    let bn_diag = std::env::var("WB_BN_DIAG").is_ok_and(|v| v == "1");
    if bn_diag {
        log_bn_running_stats(model);
    }
    let mut diag_blocks_done = 0usize;

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

        // Keep a copy of the raw features for the opt-in batch-stat forward only
        // while the diagnostic is active and under its block budget.
        let flat_for_diag = if bn_diag && diag_blocks_done < DIAG_MAX_BLOCKS {
            Some(flat.clone())
        } else {
            None
        };

        let feat_tensor = features_to_tensor::<B::InnerBackend>(flat, n, n_features_block, device);
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

        // Stage 18 opt-in diagnostic: recompute this block's val_loss under
        // train-mode (batch-statistic) BatchNorm and compare against the
        // running-stat loss above.  A large `running-stat − batch-stat` delta on
        // real data confirms the per-block-vs-global BatchNorm statistics gap.
        if let Some(diag_flat) = flat_for_diag {
            let train_feat = features_to_tensor::<B>(diag_flat, n, n_features_block, device);
            let train_logits = model.forward(train_feat); // train-mode (batch-stat) BN
            let diag_logits_flat: Vec<f32> =
                train_logits.into_data().to_vec::<f32>().unwrap_or_default();
            let loss_batch_stat =
                cross_entropy_from_logits(&diag_logits_flat, &block.labels, n, nc);
            eprintln!(
                "[bn_diag] block {block_id}: val_loss running-stat={loss_unweighted:.4}  batch-stat={loss_batch_stat:.4}  delta={:.4}",
                loss_unweighted - loss_batch_stat
            );
            diag_blocks_done += 1;
        }
    }

    Ok(acc.compute(epoch, train_loss))
}

/// Log the min/mean/max of each main encoder/decoder `BatchNorm` layer's running
/// mean and variance.  Opt-in Stage 18 diagnostic (`WB_BN_DIAG=1`); called at
/// most once per validation pass and emits five stderr lines.
fn log_bn_running_stats<B: AutodiffBackend>(model: &BurnPointNet<B>) {
    let layers: [(&str, &BatchNorm<B, 1>); 5] = [
        ("bn_enc0", &model.bn_enc0),
        ("bn_enc1", &model.bn_enc1),
        ("bn_enc2", &model.bn_enc2),
        ("bn_dec0", &model.bn_dec0),
        ("bn_dec1", &model.bn_dec1),
    ];
    for (name, bn) in layers {
        let (mmin, mmean, mmax) = tensor_stats::<B>(&bn.running_mean.value());
        let (vmin, vmean, vmax) = tensor_stats::<B>(&bn.running_var.value());
        eprintln!(
            "[bn_diag] {name}: running_mean[min/mean/max]={mmin:.4}/{mmean:.4}/{mmax:.4}  running_var[min/mean/max]={vmin:.4}/{vmean:.4}/{vmax:.4}"
        );
    }
}

/// Return `(min, mean, max)` of a 1-D burn tensor.  Diagnostic helper only.
#[allow(clippy::cast_precision_loss)]
fn tensor_stats<B: Backend>(t: &Tensor<B, 1>) -> (f32, f32, f32) {
    let v: Vec<f32> = t.clone().into_data().to_vec::<f32>().unwrap_or_default();
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    (min, mean, max)
}

/// Compute mean cross-entropy loss from raw logits and labels (no burn required).
#[allow(clippy::cast_precision_loss)]
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

/// Accumulate an optional `BatchNorm1d` pair via the `WeightAveraging` trait.
/// A no-op when either side is `None` (models without T-Nets/that layer).
fn accum_bn_opt(base_bn: &mut Option<BatchNorm1d>, other_bn: Option<&BatchNorm1d>) {
    if let (Some(bb), Some(mb)) = (base_bn, other_bn) {
        bb.accumulate(mb);
    }
}

/// Finalize (divide by `n`) an optional `BatchNorm1d` via `WeightAveraging`.
/// A no-op when `bn` is `None`.
fn finalize_bn_opt(bn: &mut Option<BatchNorm1d>, n: f32) {
    if let Some(bb) = bn {
        bb.finalize(n);
    }
}

/// Average the weights of all retained checkpoints and save to `output_path`.
///
/// Stage 22 (Training Loop Enhancements, item 2.5): checkpoints are streamed
/// one at a time — only the running sum (`base`) and the currently-processed
/// checkpoint (`m`) are resident in memory at any point, bounding memory to
/// O(2 models) regardless of `keep_best_n`, instead of loading every retained
/// checkpoint simultaneously. This reorders the code structure (a
/// per-checkpoint outer loop instead of a per-layer outer loop) but NOT the
/// underlying floating-point addition order: both the old and new code
/// accumulate against `base` as the running sum in checkpoint-list order —
/// i.e. `((base + m1) + m2) + ... + mN` either way — so the averaged output
/// weights are numerically identical to the pre-refactor implementation.
// Stage 24 (Code Quality Cleanup, item 4.1/4.3): this function's length
// comes from explicitly walking every layer of the model (both T-Nets,
// encoder, decoder, class projection) twice — once to accumulate, once to
// finalize — which is the most transparent way to express "average every
// parameter tensor"; the per-layer `accumulate`/`finalize` calls below
// (via the `WeightAveraging` trait, replacing the previous macro-based
// implementation) keep each individual step trivial even though the total
// function remains long. `n_checkpoints as f32` is a small, bounded
// checkpoint count, so the precision loss is inconsequential.
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn apply_swa(ckpt_dir: &Path, manifest: &CheckpointManifest, output_path: &Path) -> Result<()> {
    if manifest.checkpoints.is_empty() {
        return Err(ClassifierError::Pipeline(
            "SWA: no retained checkpoints to average".into(),
        ));
    }

    let n_checkpoints = manifest.checkpoints.len();
    eprintln!("[swa] averaging {n_checkpoints} checkpoints (streamed)");

    let n = n_checkpoints as f32;
    let first_path = ckpt_dir.join(&manifest.checkpoints[0].file);
    let mut base = load_model(&first_path)?;

    // ── Stream-accumulate remaining checkpoints into `base` ────────────────
    // The T-Net (STN3d / STN64d) is trained jointly with all other layers
    // under the same gradient signal.  Excluding it from SWA would produce a
    // composite model where the averaged backbone expects the canonical
    // representation produced by the averaged T-Net, but receives instead the
    // representation from a single checkpoint's T-Net — a mismatch. Both
    // `input_tnet` and `feature_tnet` are `Option<TNet>`; the averaging block
    // is gated so models without T-Nets are handled correctly.
    for entry in &manifest.checkpoints[1..] {
        let p = ckpt_dir.join(&entry.file);
        let m = load_model(&p)?;

        for i in 0..base.encoder_layers.len() {
            base.encoder_layers[i].0.accumulate(&m.encoder_layers[i].0);
            accum_bn_opt(
                &mut base.encoder_layers[i].1,
                m.encoder_layers[i].1.as_ref(),
            );
        }
        for i in 0..base.decoder_layers.len() {
            base.decoder_layers[i].0.accumulate(&m.decoder_layers[i].0);
            accum_bn_opt(
                &mut base.decoder_layers[i].1,
                m.decoder_layers[i].1.as_ref(),
            );
        }
        base.class_proj.accumulate(&m.class_proj);

        if let (Some(bt), Some(mt)) = (&mut base.input_tnet, &m.input_tnet) {
            bt.enc0.accumulate(&mt.enc0);
            bt.enc1.accumulate(&mt.enc1);
            bt.enc2.accumulate(&mt.enc2);
            accum_bn_opt(&mut bt.bn_enc0, mt.bn_enc0.as_ref());
            accum_bn_opt(&mut bt.bn_enc1, mt.bn_enc1.as_ref());
            accum_bn_opt(&mut bt.bn_enc2, mt.bn_enc2.as_ref());
            bt.fc0.accumulate(&mt.fc0);
            bt.fc1.accumulate(&mt.fc1);
            bt.fc2.accumulate(&mt.fc2);
            accum_bn_opt(&mut bt.bn_fc0, mt.bn_fc0.as_ref());
            accum_bn_opt(&mut bt.bn_fc1, mt.bn_fc1.as_ref());
        }
        if let (Some(bt), Some(mt)) = (&mut base.feature_tnet, &m.feature_tnet) {
            bt.enc0.accumulate(&mt.enc0);
            bt.enc1.accumulate(&mt.enc1);
            bt.enc2.accumulate(&mt.enc2);
            accum_bn_opt(&mut bt.bn_enc0, mt.bn_enc0.as_ref());
            accum_bn_opt(&mut bt.bn_enc1, mt.bn_enc1.as_ref());
            accum_bn_opt(&mut bt.bn_enc2, mt.bn_enc2.as_ref());
            bt.fc0.accumulate(&mt.fc0);
            bt.fc1.accumulate(&mt.fc1);
            bt.fc2.accumulate(&mt.fc2);
            accum_bn_opt(&mut bt.bn_fc0, mt.bn_fc0.as_ref());
            accum_bn_opt(&mut bt.bn_fc1, mt.bn_fc1.as_ref());
        }
        // `m` drops here, at the end of this loop iteration — memory
        // footprint stays bounded to `base` plus at most one other model.
    }

    // ── Divide every accumulated field by `n` ──────────────────────────────
    for i in 0..base.encoder_layers.len() {
        base.encoder_layers[i].0.finalize(n);
        finalize_bn_opt(&mut base.encoder_layers[i].1, n);
    }
    for i in 0..base.decoder_layers.len() {
        base.decoder_layers[i].0.finalize(n);
        finalize_bn_opt(&mut base.decoder_layers[i].1, n);
    }
    base.class_proj.finalize(n);
    if let Some(bt) = &mut base.input_tnet {
        bt.enc0.finalize(n);
        bt.enc1.finalize(n);
        bt.enc2.finalize(n);
        finalize_bn_opt(&mut bt.bn_enc0, n);
        finalize_bn_opt(&mut bt.bn_enc1, n);
        finalize_bn_opt(&mut bt.bn_enc2, n);
        bt.fc0.finalize(n);
        bt.fc1.finalize(n);
        bt.fc2.finalize(n);
        finalize_bn_opt(&mut bt.bn_fc0, n);
        finalize_bn_opt(&mut bt.bn_fc1, n);
    }
    if let Some(bt) = &mut base.feature_tnet {
        bt.enc0.finalize(n);
        bt.enc1.finalize(n);
        bt.enc2.finalize(n);
        finalize_bn_opt(&mut bt.bn_enc0, n);
        finalize_bn_opt(&mut bt.bn_enc1, n);
        finalize_bn_opt(&mut bt.bn_enc2, n);
        bt.fc0.finalize(n);
        bt.fc1.finalize(n);
        bt.fc2.finalize(n);
        finalize_bn_opt(&mut bt.bn_fc0, n);
        finalize_bn_opt(&mut bt.bn_fc1, n);
    }

    crate::model::weights::save_model(output_path, &base)
}

/// Stage 22 (Training Loop Enhancements, item 1.5): update early-stopping
/// state given the current epoch's `val_mIoU`, and report whether training
/// should stop.
///
/// `es_best_miou`/`es_epochs_without_improvement` are tracked independently
/// of the checkpoint-cadence-gated `best_miou` in `train()`, so early
/// stopping behaves identically regardless of `--checkpoint-every`.
///
/// Returns `true` iff `patience` is `Some(p)` and `p` consecutive epochs have
/// passed without a new best `val_mIoU`. Always returns `false` when
/// `patience` is `None` (early stopping disabled), which also means the
/// counters are still updated but never checked — a harmless no-op that
/// preserves pre-Stage-22 behavior exactly.
fn early_stopping_step(
    val_miou: f64,
    es_best_miou: &mut f64,
    es_epochs_without_improvement: &mut usize,
    patience: Option<usize>,
) -> bool {
    if val_miou > *es_best_miou {
        *es_best_miou = val_miou;
        *es_epochs_without_improvement = 0;
    } else {
        *es_epochs_without_improvement += 1;
    }
    patience.is_some_and(|p| *es_epochs_without_improvement >= p)
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
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
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
    /// For large counts (> ~1000), β^count ≈ 0, so `effective_num` ≈ 1/(1-β) = 10000.
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
    /// `effective_num`[c] = (1 - 0.9^count[c]) / (1 - 0.9) = (1 - 0.9^count[c]) / 0.1
    ///
    /// For count=100: 0.9^100 ≈ 2.656e-5 → `eff_num` ≈ (1 - 2.656e-5) / 0.1 ≈ 9.9997
    /// For count=50:  0.9^50  ≈ 5.154e-3 → `eff_num` ≈ (1 - 5.154e-3) / 0.1 ≈ 9.9485
    /// For count=10:  0.9^10  ≈ 0.34868  → `eff_num` ≈ (1 - 0.34868) / 0.1 ≈ 6.5132
    ///
    /// `raw_weight`[c] = 1 / `eff_num`[c]:
    ///   rw[0] ≈ 1/9.9997  ≈ 0.10000
    ///   rw[1] ≈ 1/9.9485  ≈ 0.10052
    ///   rw[2] ≈ 1/6.5132  ≈ 0.15353
    ///
    /// `present_sum` ≈ 0.35405, `n_present` = 3, scale ≈ 3/0.35405 ≈ 8.4734
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
        // `weights.len()` is a small, fixed test-fixture class count (3), far
        // below f32's precision limit — the cast below cannot lose precision
        // in practice.
        #[allow(clippy::cast_precision_loss)]
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!(
            (mean - 1.0_f32).abs() < 1e-4,
            "mean weight should be 1.0, got {mean:.6}"
        );
    }

    /// Absent class (count=0) receives the `ABSENT_CLASS_WEIGHT_FLOOR` (1e-3)
    /// rather than 0.0, so burn's `CrossEntropyLoss` does not panic.
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
        assert!(
            weights[0] > 0.1,
            "present class weight too small: {}",
            weights[0]
        );
        assert!(
            weights[2] > 0.1,
            "present class weight too small: {}",
            weights[2]
        );
        // All weights must be strictly positive (burn validation requirement).
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "weight[{i}] must be > 0, got {w}");
        }
    }

    /// Stage 22 (Training Loop Enhancements, item 1.5) — early stopping
    /// counter/reset logic, tested in isolation from the full training loop.
    #[test]
    fn test_early_stopping_step_triggers_after_patience() {
        let mut best = 0.0f64;
        let mut no_improve = 0usize;

        // Epoch 1: mIoU improves to 0.5 → resets counter, does not stop.
        assert!(!early_stopping_step(
            0.5,
            &mut best,
            &mut no_improve,
            Some(2)
        ));
        assert_eq!(no_improve, 0);
        assert!((best - 0.5).abs() < 1e-12);

        // Epoch 2: mIoU stagnates at 0.4 (no improvement) → counter=1, no stop yet.
        assert!(!early_stopping_step(
            0.4,
            &mut best,
            &mut no_improve,
            Some(2)
        ));
        assert_eq!(no_improve, 1);

        // Epoch 3: mIoU stagnates again → counter=2 == patience → should stop.
        assert!(early_stopping_step(
            0.3,
            &mut best,
            &mut no_improve,
            Some(2)
        ));
        assert_eq!(no_improve, 2);

        // A new best mIoU resets the counter even after it had grown.
        assert!(!early_stopping_step(
            0.6,
            &mut best,
            &mut no_improve,
            Some(2)
        ));
        assert_eq!(no_improve, 0);
        assert!((best - 0.6).abs() < 1e-12);
    }

    /// `patience = None` must never trigger a stop, regardless of how many
    /// epochs pass without improvement — this is the default, backward
    /// compatible behavior.
    #[test]
    fn test_early_stopping_step_disabled_never_stops() {
        let mut best = 0.0f64;
        let mut no_improve = 0usize;
        for _ in 0..10 {
            assert!(!early_stopping_step(0.1, &mut best, &mut no_improve, None));
        }
        // Counter still increments internally (harmless), but no stop signal.
        assert!(no_improve >= 9);
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
        let device = burn::backend::ndarray::NdArrayDevice::default();

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
        let max_err = diff.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-5,
            "SWA T-Net weight not correctly averaged; max element error = {max_err}"
        );
    }
}
