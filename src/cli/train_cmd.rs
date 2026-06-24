//! `train` sub-command — runs the PointNet training loop.

#![allow(clippy::missing_errors_doc, clippy::doc_markdown)]

use std::collections::HashSet;
use std::path::PathBuf;

use burn::backend::{Autodiff, NdArray};

use crate::error::{ClassifierError, Result};
use crate::training::{
    dataset::LabeledBlockDataset,
    trainer::{train, TrainConfig},
};

type TrainBackend = Autodiff<NdArray>;

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let mut cfg = TrainConfig::default();
    let mut data_dirs: Vec<PathBuf> = Vec::new();
    let mut val_tile_blocks_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir"          => { i += 1; data_dirs.push(PathBuf::from(&args[i])); }
            "--output-model"      => { i += 1; cfg.output_model = PathBuf::from(&args[i]); }
            "--n-classes"         => { i += 1; cfg.n_classes = parse_usize(&args[i], "--n-classes")?; }
            "--epochs"            => { i += 1; cfg.epochs = parse_usize(&args[i], "--epochs")?; }
            "--batch-size"        => { i += 1; cfg.batch_size = parse_usize(&args[i], "--batch-size")?; }
            "--learning-rate"     => { i += 1; cfg.learning_rate = parse_f64(&args[i], "--learning-rate")?; }
            "--weight-decay"      => { i += 1; cfg.weight_decay = parse_f32(&args[i], "--weight-decay")?; }
            "--val-split"         => { i += 1; cfg.val_split = parse_f64(&args[i], "--val-split")?; }
            "--val-tile-blocks"   => { i += 1; val_tile_blocks_path = Some(PathBuf::from(&args[i])); }
            "--seed"              => { i += 1; cfg.seed = parse_u64(&args[i], "--seed")?; }
            "--use-feature-tnet"  => { cfg.use_feature_tnet = true; }
            "--no-batch-norm"     => { cfg.use_batch_norm = false; }
            "--no-class-weights"  => { cfg.use_class_weights = false; }
            "--checkpoint-dir"    => { i += 1; cfg.checkpoint_dir = Some(PathBuf::from(&args[i])); }
            "--checkpoint-every"  => { i += 1; cfg.checkpoint_every = parse_usize(&args[i], "--checkpoint-every")?; }
            "--keep-best-n"       => { i += 1; cfg.keep_best_n = parse_usize(&args[i], "--keep-best-n")?; }
            "--swa"               => { cfg.swa = true; }
            "--metrics-out"       => { i += 1; cfg.metrics_out = PathBuf::from(&args[i]); }
            "--threads"           => { i += 1; cfg.n_threads = Some(parse_usize(&args[i], "--threads")?); }
            flag => {
                return Err(ClassifierError::Pipeline(format!(
                    "train: unknown flag '{flag}'"
                )));
            }
        }
        i += 1;
    }

    if data_dirs.is_empty() {
        return Err(ClassifierError::Pipeline("at least one --data-dir is required".into()));
    }

    // Range validation.
    if cfg.n_classes < 2 {
        return Err(ClassifierError::Pipeline("--n-classes must be >= 2".into()));
    }
    if cfg.epochs == 0 {
        return Err(ClassifierError::Pipeline("--epochs must be >= 1".into()));
    }
    if cfg.batch_size == 0 {
        return Err(ClassifierError::Pipeline("--batch-size must be >= 1".into()));
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
        return Err(ClassifierError::Pipeline("--keep-best-n must be >= 1".into()));
    }
    if cfg.checkpoint_every == 0 {
        return Err(ClassifierError::Pipeline("--checkpoint-every must be >= 1".into()));
    }
    if let Some(t) = cfg.n_threads {
        if t == 0 {
            return Err(ClassifierError::Pipeline("train: --threads must be >= 1".into()));
        }
    }

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
    let dataset = LabeledBlockDataset::load(
        &data_dirs,
        cfg.val_split,
        val_tile_ids.as_ref(),
        cfg.seed,
    )?;

    cfg.val_tile_block_ids = val_tile_ids;

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

    let device = burn::backend::ndarray::NdArrayDevice::default();

    train::<TrainBackend>(&dataset, &cfg, &device)?;

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
           --n-classes         <u8>     Output classes (default: 8)\n\
           --epochs            <usize>  Training epochs (default: 50)\n\
           --batch-size        <usize>  Gradient accumulation batch (default: 16)\n\
           --learning-rate     <f64>    Initial AdamW LR (default: 1e-3)\n\
           --weight-decay      <f32>    AdamW weight decay (default: 1e-4)\n\
           --val-split         <f64>    Val macro-tile fraction (default: 0.20)\n\
           --val-tile-blocks   <path>   JSON file of explicit val block IDs\n\
           --seed              <u64>    Split/shuffle seed (default: 42)\n\
           --use-feature-tnet          Enable STN64d (default: off)\n\
           --no-batch-norm             Disable BatchNorm (default: on)\n\
           --no-class-weights          Disable class-weighted loss (default: on)\n\
           --checkpoint-dir    <path>   Save checkpoint .wbmodel files here\n\
           --checkpoint-every  <usize>  Checkpoint interval in epochs (default: 1)\n\
           --keep-best-n       <usize>  Max retained checkpoints (default: 5)\n\
           --swa                        Apply Stochastic Weight Averaging (default: off)\n\
           --metrics-out       <path>   Per-epoch metrics CSV\n\
           --threads           <usize>  Rayon thread pool size"
    );
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f64 '{s}'")))
}
fn parse_f32(s: &str, flag: &str) -> Result<f32> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f32 '{s}'")))
}
fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid usize '{s}'")))
}
fn parse_u64(s: &str, flag: &str) -> Result<u64> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid u64 '{s}'")))
}
