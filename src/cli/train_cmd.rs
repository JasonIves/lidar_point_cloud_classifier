//! `train` sub-command — runs the PointNet training loop.

#![allow(
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::collections::HashSet;
use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::training::{
    backend::{self, DevicePreference},
    dataset::LabeledBlockDataset,
    trainer::TrainConfig,
};

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let mut cfg = TrainConfig::default();
    let mut data_dirs: Vec<PathBuf> = Vec::new();
    let mut val_data_dirs: Vec<PathBuf> = Vec::new();
    let mut val_tile_blocks_path: Option<PathBuf> = None;
    let mut val_split_explicit = false;
    let mut val_tile_blocks_explicit = false;
    let mut device_pref = DevicePreference::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                data_dirs.push(PathBuf::from(next_value(args, &mut i, "--data-dir")?));
            }
            "--val-data-dir" => {
                val_data_dirs.push(PathBuf::from(next_value(args, &mut i, "--val-data-dir")?));
            }
            "--output-model" => {
                cfg.output_model = PathBuf::from(next_value(args, &mut i, "--output-model")?);
            }
            "--n-classes" => {
                cfg.n_classes =
                    parse_usize(next_value(args, &mut i, "--n-classes")?, "--n-classes")?;
            }
            "--epochs" => {
                cfg.epochs = parse_usize(next_value(args, &mut i, "--epochs")?, "--epochs")?;
            }
            "--batch-size" => {
                cfg.batch_size =
                    parse_usize(next_value(args, &mut i, "--batch-size")?, "--batch-size")?;
            }
            "--forward-batch-size" => {
                cfg.forward_batch_size = parse_usize(
                    next_value(args, &mut i, "--forward-batch-size")?,
                    "--forward-batch-size",
                )?;
            }
            "--learning-rate" => {
                cfg.learning_rate = parse_f64(
                    next_value(args, &mut i, "--learning-rate")?,
                    "--learning-rate",
                )?;
            }
            "--weight-decay" => {
                cfg.weight_decay = parse_f32(
                    next_value(args, &mut i, "--weight-decay")?,
                    "--weight-decay",
                )?;
            }
            "--val-split" => {
                cfg.val_split = parse_f64(next_value(args, &mut i, "--val-split")?, "--val-split")?;
                val_split_explicit = true;
            }
            "--val-tile-blocks" => {
                val_tile_blocks_path = Some(PathBuf::from(next_value(
                    args,
                    &mut i,
                    "--val-tile-blocks",
                )?));
                val_tile_blocks_explicit = true;
            }
            "--seed" => {
                cfg.seed = parse_u64(next_value(args, &mut i, "--seed")?, "--seed")?;
            }
            "--use-feature-tnet" => {
                cfg.use_feature_tnet = true;
            }
            "--no-class-weights" => {
                cfg.use_class_weights = false;
            }
            "--class-weight-beta" => {
                cfg.class_weight_beta = parse_f64(
                    next_value(args, &mut i, "--class-weight-beta")?,
                    "--class-weight-beta",
                )?;
            }
            "--checkpoint-dir" => {
                cfg.checkpoint_dir =
                    Some(PathBuf::from(next_value(args, &mut i, "--checkpoint-dir")?));
            }
            "--checkpoint-every" => {
                cfg.checkpoint_every = parse_usize(
                    next_value(args, &mut i, "--checkpoint-every")?,
                    "--checkpoint-every",
                )?;
            }
            "--keep-best-n" => {
                cfg.keep_best_n =
                    parse_usize(next_value(args, &mut i, "--keep-best-n")?, "--keep-best-n")?;
            }
            "--swa" => {
                cfg.swa = true;
            }
            "--metrics-out" => {
                cfg.metrics_out = PathBuf::from(next_value(args, &mut i, "--metrics-out")?);
            }
            "--threads" => {
                cfg.n_threads = Some(parse_usize(
                    next_value(args, &mut i, "--threads")?,
                    "--threads",
                )?);
            }
            "--device" => {
                device_pref = DevicePreference::parse(next_value(args, &mut i, "--device")?)?;
            }
            "--early-stopping-patience" => {
                cfg.early_stopping_patience = Some(parse_usize(
                    next_value(args, &mut i, "--early-stopping-patience")?,
                    "--early-stopping-patience",
                )?);
            }
            "--warmup-steps" => {
                cfg.warmup_steps = parse_usize(
                    next_value(args, &mut i, "--warmup-steps")?,
                    "--warmup-steps",
                )?;
            }
            "--grad-clip-norm" => {
                cfg.grad_clip_norm = Some(parse_f32(
                    next_value(args, &mut i, "--grad-clip-norm")?,
                    "--grad-clip-norm",
                )?);
            }
            "--cache-blocks-max-mb" => {
                cfg.cache_blocks_max_mb = Some(parse_usize(
                    next_value(args, &mut i, "--cache-blocks-max-mb")?,
                    "--cache-blocks-max-mb",
                )?);
            }
            "--halo-loss-weight" => {
                cfg.halo_loss_weight = parse_f32(
                    next_value(args, &mut i, "--halo-loss-weight")?,
                    "--halo-loss-weight",
                )?;
            }
            flag => {
                return Err(ClassifierError::Pipeline(format!(
                    "train: unknown flag '{flag}'"
                )));
            }
        }
        i += 1;
    }

    if data_dirs.is_empty() {
        return Err(ClassifierError::Pipeline(
            "at least one --data-dir is required".into(),
        ));
    }

    // Range validation.
    if cfg.n_classes < 2 {
        return Err(ClassifierError::Pipeline("--n-classes must be >= 2".into()));
    }
    if cfg.epochs == 0 {
        return Err(ClassifierError::Pipeline("--epochs must be >= 1".into()));
    }
    if cfg.batch_size == 0 {
        return Err(ClassifierError::Pipeline(
            "--batch-size must be >= 1".into(),
        ));
    }
    if cfg.forward_batch_size == 0 {
        return Err(ClassifierError::Pipeline(
            "--forward-batch-size must be >= 1".into(),
        ));
    }
    if cfg.learning_rate <= 0.0 || !cfg.learning_rate.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--learning-rate must be a positive finite number".into(),
        ));
    }
    if cfg.val_split <= 0.0 || cfg.val_split >= 1.0 {
        return Err(ClassifierError::Pipeline(
            "--val-split must be in the range (0.0, 1.0) exclusive".into(),
        ));
    }
    if cfg.keep_best_n == 0 {
        return Err(ClassifierError::Pipeline(
            "--keep-best-n must be >= 1".into(),
        ));
    }
    if cfg.checkpoint_every == 0 {
        return Err(ClassifierError::Pipeline(
            "--checkpoint-every must be >= 1".into(),
        ));
    }
    if let Some(t) = cfg.n_threads {
        if t == 0 {
            return Err(ClassifierError::Pipeline(
                "train: --threads must be >= 1".into(),
            ));
        }
    }
    // --class-weight-beta must be in [0.0, 1.0).
    // β = 1.0 is excluded because the effective-number formula is undefined there
    // (the limit is inverse-frequency, approximated at β = 0.9999).
    if cfg.class_weight_beta < 0.0 || cfg.class_weight_beta >= 1.0 {
        return Err(ClassifierError::Pipeline(
            "--class-weight-beta must be in the range [0.0, 1.0). \
             Use 0.0 for uniform weights, values near 1.0 for stronger minority-class emphasis."
                .into(),
        ));
    }
    // Stage 22 (Training Loop Enhancements): --grad-clip-norm, if provided,
    // must be a positive finite number (a zero or negative max-norm clip
    // threshold is meaningless).
    if let Some(g) = cfg.grad_clip_norm {
        if g <= 0.0 || !g.is_finite() {
            return Err(ClassifierError::Pipeline(
                "--grad-clip-norm must be a positive finite number".into(),
            ));
        }
    }
    // Stage 27 (Block Caching, audit finding 5.2): --cache-blocks-max-mb, if
    // provided, must be a nonzero budget (a 0 MB budget can never cache
    // anything and is almost certainly a user mistake).
    if let Some(mb) = cfg.cache_blocks_max_mb {
        if mb == 0 {
            return Err(ClassifierError::Pipeline(
                "--cache-blocks-max-mb must be >= 1".into(),
            ));
        }
    }
    // Stage 45: --halo-loss-weight must be finite and non-negative.
    if !cfg.halo_loss_weight.is_finite() || cfg.halo_loss_weight < 0.0 {
        return Err(ClassifierError::Pipeline(
            "--halo-loss-weight must be finite and >= 0.0".into(),
        ));
    }
    // Batch-size-1 BatchNorm regression advisory (see
    // docs/stages/stage-18-batchnorm-batched-forward.md).
    //
    // `--batch-size` caps how many blocks each `chunk.chunks(forward_batch_size)`
    // micro-batch can ever contain (trainer.rs: `for chunk in
    // shuffled.chunks(config.batch_size) { for micro in
    // chunk.chunks(config.forward_batch_size) { ... } }`), so the *effective*
    // forward-batch size — the effective BatchNorm batch size Stage 18
    // specifically introduced batching to fix — is `min(batch_size,
    // forward_batch_size)`, not `forward_batch_size` alone. Either flag set to
    // 1 silently reproduces the pre-Stage-18 degenerate regime (BatchNorm
    // normalizes each forward pass using only one block's own statistics),
    // which does not crash but is empirically documented to badly damage
    // generalization (val_loss ~10x train_loss, val_mIoU stuck below 0.10).
    // This is a warning, not a hard error, since a user may deliberately want
    // this behavior (e.g. reproducing the pre-fix regime for comparison).
    if effective_forward_batch_size(cfg.batch_size, cfg.forward_batch_size) == 1 {
        eprintln!(
            "[train] warning: effective forward-batch size is 1 \
             (min(--batch-size={}, --forward-batch-size={}) = 1). This reproduces the \
             pre-Stage-18 degenerate BatchNorm regime: each forward pass normalizes using only \
             one block's own statistics, while validation/deployment apply a single global \
             running average — blocks whose distribution differs from that average will be \
             systematically mis-normalized. Documented real-world symptom (see \
             docs/stages/stage-18-batchnorm-batched-forward.md): val_loss ~10x train_loss and \
             val_mIoU stuck below 0.10 despite normal-looking training loss, with no \
             converge-then-diverge curve. Consider raising --batch-size and/or \
             --forward-batch-size (e.g. >= 8) unless this is an intentional comparison run.",
            cfg.batch_size, cfg.forward_batch_size
        );
    }

    // Stage 32 (Dataset Split Materialization): if one or more
    // --val-data-dir directories were supplied, the split has already been
    // decided physically (via `wb_lidar_train split-dataset`) — every block
    // in --data-dir goes to train, every block in --val-data-dir goes to
    // val, with no macro-tile logic. --val-split/--val-tile-blocks are
    // ignored in this mode (warned, not errored, so a shared flag template
    // can be reused across both on-the-fly and pre-split invocations).
    let dataset = if val_data_dirs.is_empty() {
        // Load explicit val-tile-blocks override if provided.
        let val_tile_ids: Option<HashSet<u64>> = if let Some(ref p) = val_tile_blocks_path {
            let f = std::fs::File::open(p)?;
            let ids: Vec<u64> = serde_json::from_reader(f)
                .map_err(|e| ClassifierError::Pipeline(format!("--val-tile-blocks parse: {e}")))?;
            Some(ids.into_iter().collect())
        } else {
            None
        };

        // Load dataset — accepts one or more preprocessing directories.
        // Stage 27 (Block Caching, audit finding 5.2): .with_block_cache(None)
        // (the default, when --cache-blocks-max-mb is not passed) is a no-op —
        // load_block() behaves exactly as it did before Stage 27.
        let dataset =
            LabeledBlockDataset::load(&data_dirs, cfg.val_split, val_tile_ids.as_ref(), cfg.seed)?
                .with_block_cache(cfg.cache_blocks_max_mb)
                .with_halo_loss_weight(cfg.halo_loss_weight);

        cfg.val_tile_block_ids = val_tile_ids;
        dataset
    } else {
        if val_split_explicit {
            eprintln!(
                "[train] warning: --val-split is ignored because --val-data-dir was supplied"
            );
        }
        if val_tile_blocks_explicit {
            eprintln!(
                "[train] warning: --val-tile-blocks is ignored because --val-data-dir was supplied"
            );
        }
        LabeledBlockDataset::load_presplit(&data_dirs, &val_data_dirs)?
            .with_block_cache(cfg.cache_blocks_max_mb)
            .with_halo_loss_weight(cfg.halo_loss_weight)
    };

    // Default metrics output: <first_data_dir>/../metrics/metrics.csv
    if cfg.metrics_out.as_os_str() == "metrics.csv" {
        let anchor = &data_dirs[0];
        let metrics_dir = anchor
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(anchor.as_path())
            .join("metrics");
        std::fs::create_dir_all(&metrics_dir).ok();
        cfg.metrics_out = metrics_dir.join("metrics.csv");
    }

    // Dispatch to the selected backend (GPU or CPU) via runtime detection.
    backend::select_and_train(&dataset, &cfg, device_pref)?;

    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: wb_lidar_train train [options]\n\
         \n\
         Required:\n\
           --data-dir      <dir>    Directory from `preprocess-labeled`; repeat for multiple files\n\
           --output-model  <path>   Output .wbmodel file\n\
         \n\
         Optional:\n\
           --val-data-dir      <dir>    Pre-split validation directory (repeatable). When at\n\
                                        least one is supplied, ALL --data-dir directories are\n\
                                        used entirely for training (no on-the-fly split), and\n\
                                        ALL --val-data-dir directories are used entirely for\n\
                                        validation. --val-split/--val-tile-blocks are ignored\n\
                                        in this mode (with a warning if explicitly set).\n\
                                        See `wb_lidar_train split-dataset` to materialize a\n\
                                        pre-split train/val directory pair.\n\
           --n-classes         <u8>     Output classes (default: 8)\n\
           --epochs            <usize>  Training epochs (default: 50)\n\
           --batch-size        <usize>  Effective batch: blocks per optimizer step (default: 16)\n\
           --forward-batch-size <usize> Blocks per batched forward — effective BatchNorm\n\
                                        batch size; micro-batched then accumulated (default: 8)\n\
                                        Stage 28: keep forward_batch_size x target_points below\n\
                                        ~120,000 on 8GB-class GPUs to avoid VRAM oversubscription\n\
                                        (WDDM silently spills into slower shared system memory\n\
                                        instead of erroring, causing a severe slowdown)\n\
                                        NOTE: the true effective BatchNorm batch size is\n\
                                        min(--batch-size, --forward-batch-size), not either\n\
                                        flag alone. If that minimum is 1, a warning is printed\n\
                                        (see docs/stages/stage-18-batchnorm-batched-forward.md)\n\
                                        because it reproduces a known pre-fix regression that\n\
                                        harms validation generalization without crashing.\n\
           --learning-rate     <f64>    Initial AdamW LR (default: 1e-3)\n\
           --weight-decay      <f32>    AdamW weight decay (default: 1e-4)\n\
           --val-split         <f64>    Val macro-tile fraction (default: 0.20)\n\
           --val-tile-blocks   <path>   JSON file of explicit val block IDs\n\
           --seed              <u64>    Split/shuffle seed (default: 42)\n\
           --use-feature-tnet          Enable STN64d (default: off)\n\
           --no-class-weights          Disable class-weighted loss (default: on)\n\
           --class-weight-beta <f64>    β for effective-number weighting, range [0.0,1.0)\n\
                                        0.0=uniform, 0.999=default (strong), 0.9999≈inverse-freq\n\
           --checkpoint-dir    <path>   Save checkpoint .wbmodel files here\n\
           --checkpoint-every  <usize>  Checkpoint interval in epochs (default: 1)\n\
           --keep-best-n       <usize>  Max retained checkpoints (default: 5)\n\
           --swa                        Apply Stochastic Weight Averaging (default: off)\n\
           --metrics-out       <path>   Per-epoch metrics CSV\n\
           --threads           <usize>  Rayon thread pool size\n\
           --device            <auto|cpu|gpu>  Compute device (default: auto)\n\
           --early-stopping-patience <usize>  Stop after N epochs with no val_mIoU\n\
                                        improvement (default: disabled)\n\
           --warmup-steps      <usize>  Linear LR warmup steps before cosine\n\
                                        annealing begins (default: 0, disabled)\n\
           --grad-clip-norm    <f32>    Per-tensor L2-norm gradient clip threshold\n\
                                        (default: disabled)\n\
           --cache-blocks-max-mb <usize>  Enable in-memory block caching bounded to\n\
                                        this many megabytes (default: disabled)\n\
           --halo-loss-weight <f32>    Per-point loss weight for same-tile halo rows\n\
                                        (Stage 45; default: 1.0). Cross-tile halo rows\n\
                                        are always masked to 0 to prevent cross-split\n\
                                        label leakage. 0.0 masks all halo rows\n\
                                        (context-only mode)"
    );
}

/// If the flag at `args[*i]` requires a value, bounds-check and consume the
/// next token.  Returns a clear `ClassifierError::Pipeline` instead of
/// panicking (via unchecked indexing) if the flag is the last argument.
///
/// Stage 20 (Security Hardening) — mirrors the pattern already used in
/// `preprocess_cmd.rs`.
fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("flag '{flag}' requires a value")))
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f64 '{s}'")))
}
fn parse_f32(s: &str, flag: &str) -> Result<f32> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f32 '{s}'")))
}
fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid usize '{s}'")))
}
fn parse_u64(s: &str, flag: &str) -> Result<u64> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid u64 '{s}'")))
}

/// Compute the *effective* forward-batch size actually reaching a single
/// batched forward pass (and therefore `BatchNorm`), given the raw
/// `--batch-size`/`--forward-batch-size` CLI values.
///
/// Mirrors `trainer.rs`'s nested chunking exactly:
/// `shuffled.chunks(batch_size)` then `chunk.chunks(forward_batch_size)` —
/// since a `chunk` can never exceed `batch_size` elements, no micro-batch can
/// ever exceed `min(batch_size, forward_batch_size)` blocks, regardless of
/// which of the two flags is smaller. Both inputs are floored at 1 (matching
/// `trainer.rs`'s own `.max(1)` guards) so a caller can pass raw, unvalidated
/// CLI values without special-casing 0. In practice `run()`'s own validation
/// already rejects `--batch-size 0`/`--forward-batch-size 0` before this is
/// ever called, so the floor is defensive, not reachable via the CLI.
#[must_use]
fn effective_forward_batch_size(batch_size: usize, forward_batch_size: usize) -> usize {
    batch_size.max(1).min(forward_batch_size.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Stage 20 (Security Hardening) — a flag with no trailing value must
    // return a clear error instead of panicking via unchecked indexing.
    #[test]
    fn test_trailing_flag_without_value_errors_not_panics() {
        let args: Vec<String> = vec!["--data-dir".to_string()];
        let mut i = 0usize;
        let result = next_value(&args, &mut i, "--data-dir");
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_trailing_flag_returns_error() {
        // The full run() path with a dangling flag must error, not panic.
        let args: Vec<String> = vec!["--epochs".to_string()];
        let result = run(&args);
        assert!(result.is_err());
    }

    // Batch-size-1 BatchNorm regression advisory: effective_forward_batch_size()
    // must mirror trainer.rs's nested `shuffled.chunks(batch_size)` then
    // `chunk.chunks(forward_batch_size)` structure exactly, i.e. return
    // min(batch_size, forward_batch_size).
    #[test]
    fn test_effective_forward_batch_size_is_min_of_both() {
        assert_eq!(effective_forward_batch_size(16, 8), 8);
        assert_eq!(effective_forward_batch_size(4, 8), 4);
        assert_eq!(effective_forward_batch_size(1, 8), 1);
        assert_eq!(effective_forward_batch_size(8, 1), 1);
        assert_eq!(effective_forward_batch_size(8, 8), 8);
    }

    // Defensive floor: run()'s own validation already rejects
    // --batch-size 0 / --forward-batch-size 0 before this helper is ever
    // called from the CLI, so this only exercises the helper's own guard
    // for direct/unit-test callers passing raw, unvalidated values.
    #[test]
    fn test_effective_forward_batch_size_floors_zero_inputs_at_one() {
        assert_eq!(effective_forward_batch_size(0, 8), 1);
        assert_eq!(effective_forward_batch_size(8, 0), 1);
        assert_eq!(effective_forward_batch_size(0, 0), 1);
    }
}
