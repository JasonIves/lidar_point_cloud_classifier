//! `evaluate` sub-command (Stage 39) — score a trained `.wbmodel` on a
//! labeled, held-out data directory (e.g. the `test/` split produced by
//! `wb_lidar_train split-dataset`) and emit segmentation metrics as two CSVs.
//!
//! Usage:
//! ```text
//! wb_lidar_train evaluate
//!     --model         <path.wbmodel>   trained model to evaluate (required)
//!     --data-dir      <dir>            labeled test dir (repeatable, required)
//!     --metrics-out   <path.csv>       per-class metrics CSV (required)
//!     --confusion-out <path.csv>       confusion-matrix CSV (required)
//!     [--n-classes    <n>]             optional cross-check against model/data
//!     [--threads      <n>]             Rayon thread-pool size (default: cores)
//! ```
//!
//! ## Index-space correctness (critical)
//!
//! `.lbl` files store **remapped model class indices** (`0..n-1`) — the same
//! space `MetricsAccumulator` operates in. `PointNetClassifier::classify()`
//! instead returns **ASPRS codes** via `label_map`. Evaluation therefore
//! compares the **argmax of `forward()` logits (a model index)** against the
//! `.lbl` model indices directly and never routes predictions through
//! `classify()` / `run_inference()`.
//!
//! See `docs/stages/stage-39-held-out-test-evaluation.md`.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::error::{ClassifierError, Result};
use crate::model::pointnet::PointNetClassifier;
use crate::model::weights::load_model;
use crate::training::dataset::LabeledBlockDataset;
use crate::training::metrics::{write_confusion_matrix_csv, EpochMetrics, MetricsAccumulator};

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `args` (everything after `evaluate`) and run the evaluation pipeline.
///
/// # Errors
/// Returns an error if argument parsing fails, the model or dataset cannot be
/// loaded, the model/data class counts disagree, inference fails, or a CSV
/// cannot be written.
pub fn run(args: &[String]) -> Result<()> {
    let cfg = parse_args(args)?;

    // ── Optionally configure the Rayon thread pool ─────────────────────────
    if let Some(threads) = cfg.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    // ── Load model ─────────────────────────────────────────────────────────
    eprintln!("[evaluate] loading model: {}", cfg.model.display());
    let model = load_model(&cfg.model)?;
    eprintln!(
        "[evaluate] model: {} encoder dims, {} classes, T-Nets: input={}, feature={}",
        model.config.encoder_dims.len(),
        model.config.n_classes,
        model.config.use_input_tnet,
        model.config.use_feature_tnet,
    );

    // ── Load the labeled test directory (all blocks, no split) ─────────────
    // `load` always builds an internal train/val split (and even prints a
    // "train blocks / val blocks" line) — that split is IRRELEVANT here.
    // Evaluation scores the ENTIRE dataset via `all_block_ids()`, so the
    // reported split can be safely ignored.
    eprintln!("[evaluate] loading {} data dir(s)…", cfg.data_dirs.len());
    let dataset = LabeledBlockDataset::load(&cfg.data_dirs, 0.0, None, 0)?;
    eprintln!(
        "[evaluate] note: the '[dataset] train/val blocks' line above is an \
         internal artifact of the loader; evaluation scores ALL {} block(s).",
        dataset.all_block_ids().len()
    );

    // ── Consistency checks ─────────────────────────────────────────────────
    let n_classes = reconcile_n_classes(&model, &dataset, cfg.n_classes)?;
    eprintln!("[evaluate] evaluating with {n_classes} classes");

    // ── Run evaluation ─────────────────────────────────────────────────────
    let (metrics, confusion) = run_evaluation(&model, &dataset, n_classes)?;

    // ── Write outputs ──────────────────────────────────────────────────────
    write_per_class_metrics_csv(&cfg.metrics_out, &metrics, &model.label_map)?;
    eprintln!(
        "[evaluate] wrote per-class metrics: {}",
        cfg.metrics_out.display()
    );
    write_confusion_matrix_csv(&cfg.confusion_out, &confusion).map_err(ClassifierError::Io)?;
    eprintln!(
        "[evaluate] wrote confusion matrix: {}",
        cfg.confusion_out.display()
    );

    print_summary(&metrics, &model.label_map);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Run the model over every block in `dataset` and return the aggregated
/// metrics plus the confusion matrix (rows = true class, cols = predicted).
///
/// Predictions are the **argmax of `forward()` logits** (model indices),
/// compared directly against the `.lbl` model indices.
///
/// # Errors
/// Returns an error if any block cannot be loaded or the forward pass fails.
fn run_evaluation(
    model: &PointNetClassifier,
    dataset: &LabeledBlockDataset,
    n_classes: usize,
) -> Result<(EpochMetrics, Vec<Vec<u64>>)> {
    // Score EVERY block exactly once, independent of the loader's internal
    // train/val split (which `load` always builds — and `spatial_split` forces
    // at least one held-out val macro-tile even at val_split = 0.0). Using
    // `all_block_ids()` makes the "evaluate the whole dataset" intent explicit
    // and robust rather than relying on reconstructing it from train_ids ∪
    // val_ids.
    let all_ids: Vec<u64> = dataset.all_block_ids();

    if all_ids.is_empty() {
        return Err(ClassifierError::Pipeline(
            "evaluate: no blocks found in the supplied --data-dir(s)".into(),
        ));
    }

    // ── Parallel phase — each worker owns its Result (no shared locks) ─────
    let per_block: Vec<Result<(Vec<u8>, Vec<u8>)>> = all_ids
        .par_iter()
        .map(|&gid| {
            let block = dataset.load_block(gid)?;
            let logits = model.forward(block.features)?;
            let n = logits.nrows();
            let mut preds = Vec::with_capacity(n);
            for i in 0..n {
                preds.push(argmax_row(&logits.row(i)));
            }
            Ok((preds, block.labels))
        })
        .collect();

    // ── Sequential drain — propagate first error, fold into accumulator ────
    let mut acc = MetricsAccumulator::new(n_classes);
    let mut n_blocks = 0usize;
    let mut n_points = 0u64;
    for item in per_block {
        let (preds, gts) = item?;
        n_points += u64::try_from(preds.len()).unwrap_or(u64::MAX);
        n_blocks += 1;
        acc.accumulate(&preds, &gts);
    }

    eprintln!("[evaluate] scored {n_blocks} block(s), {n_points} point(s)");

    let metrics = acc.compute(1, 0.0);
    let confusion = acc.confusion_matrix().clone();
    Ok((metrics, confusion))
}

/// Argmax of a logit row, returned as a **model class index** (`u8`).
///
/// Ties resolve to the **highest** index: `Iterator::max_by` keeps the last of
/// any equal maxima, which matches `PointNetClassifier::classify`. As a
/// consequence, an all-equal row (e.g. all-zero logits) yields `n_classes - 1`.
/// A class count larger than `u8::MAX` is not representable by the `.wbmodel`
/// header (it stores `n_classes` as a `u8`), so the saturating conversion here
/// is unreachable in practice.
fn argmax_row(row: &ArrayView1<f32>) -> u8 {
    let idx = row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx);
    u8::try_from(idx).unwrap_or(u8::MAX)
}

/// Reconcile the class count *and* label-map content across the model, the
/// dataset, and the optional `--n-classes` override; any disagreement is a
/// hard error.
///
/// # Stage 40 (item 2): count agreement alone is insufficient
///
/// Two label maps can declare the exact same class *count* while assigning
/// completely different ASPRS codes to the same model index (e.g. model
/// index 0 means "Ground" (ASPRS 2) in one label map, but "Building" (ASPRS
/// 6) in another). Prior to this check, `evaluate` only compared
/// `model.config.n_classes` against `dataset.n_classes()` — a count match
/// that silently permits this class-identity swap, which would corrupt
/// every emitted metric (each point's true/predicted class would be
/// compared under inconsistent semantics) without any error or warning.
///
/// This additionally inverts the dataset's ASPRS-code → model-index label
/// map (`LabeledBlockDataset::label_map`) into the same model-index →
/// ASPRS-code direction as `model.label_map`, and requires the two to agree
/// at every index.
fn reconcile_n_classes(
    model: &PointNetClassifier,
    dataset: &LabeledBlockDataset,
    requested: Option<usize>,
) -> Result<usize> {
    let model_n = model.config.n_classes;
    let data_n = dataset.n_classes();

    if model_n != data_n {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: model has {model_n} classes but the data directory declares \
             {data_n} classes. The model and the evaluation data must use the same \
             label map / class count."
        )));
    }

    if let Some(req) = requested {
        if req != model_n {
            return Err(ClassifierError::Pipeline(format!(
                "evaluate: --n-classes={req} disagrees with the model/data class \
                 count ({model_n}). Omit --n-classes to use the derived value."
            )));
        }
    }

    // ── Label-map content check (Stage 40, item 2) ─────────────────────────
    // Invert the dataset's ASPRS-code(string) -> model-index map into
    // model-index -> ASPRS-code, matching `model.label_map`'s direction.
    // Stage 41: this inversion is now a shared method on LabeledBlockDataset
    // (also used by `trainer::train()` to derive a freshly-saved model's
    // label_map correctly) rather than duplicated inline here.
    let derived = dataset.inverse_label_map()?;

    for (idx, expected_code) in derived.iter().enumerate() {
        let model_code = model.label_map.get(idx).copied();
        if model_code != Some(*expected_code) {
            return Err(ClassifierError::Pipeline(format!(
                "evaluate: label map mismatch at model class index {idx} — the \
                 model maps this index to ASPRS code {model_code:?}, but the \
                 evaluation data's label map maps it to ASPRS code {expected_code}. \
                 The model and the evaluation data must have been preprocessed \
                 with the exact same --label-map, not merely the same class count."
            )));
        }
    }

    Ok(model_n)
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV output
// ─────────────────────────────────────────────────────────────────────────────

/// Write one row per class: `class_idx,asprs_code,tp,fp,tn,fn,precision,recall,f1,iou`.
///
/// `label_map[class_idx]` supplies the human-readable ASPRS code; a missing
/// entry falls back to `1` (ASPRS Unassigned), matching the inference engine.
fn write_per_class_metrics_csv(
    path: &Path,
    metrics: &EpochMetrics,
    label_map: &[u8],
) -> Result<()> {
    let f = std::fs::File::create(path).map_err(ClassifierError::Io)?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "class_idx,asprs_code,tp,fp,tn,fn,precision,recall,f1,iou"
    )
    .map_err(ClassifierError::Io)?;
    for cm in &metrics.per_class {
        let asprs = label_map.get(cm.class_idx).copied().unwrap_or(1);
        writeln!(
            w,
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            cm.class_idx,
            asprs,
            cm.tp,
            cm.fp,
            cm.tn,
            cm.r#fn,
            cm.precision,
            cm.recall,
            cm.f1,
            cm.iou
        )
        .map_err(ClassifierError::Io)?;
    }
    w.flush().map_err(ClassifierError::Io)?;
    Ok(())
}

/// Print a concise, human-readable summary to stderr.
fn print_summary(metrics: &EpochMetrics, label_map: &[u8]) {
    eprintln!("[evaluate] ─── results ───────────────────────────────────────");
    eprintln!("[evaluate]   mean IoU        : {:.4}", metrics.miou);
    eprintln!(
        "[evaluate]   overall accuracy: {:.4}",
        metrics.overall_accuracy
    );
    eprintln!("[evaluate]   macro F1        : {:.4}", metrics.f1_macro);
    eprintln!("[evaluate]   per-class (idx/asprs: P / R / F1 / IoU):");
    for cm in &metrics.per_class {
        let asprs = label_map.get(cm.class_idx).copied().unwrap_or(1);
        eprintln!(
            "[evaluate]     {:>2}/{:<3}: {:.4} / {:.4} / {:.4} / {:.4}",
            cm.class_idx, asprs, cm.precision, cm.recall, cm.f1, cm.iou
        );
    }
    eprintln!("[evaluate] ────────────────────────────────────────────────────");
}

// ─────────────────────────────────────────────────────────────────────────────
// Config + argument parsing
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct EvaluateConfig {
    model: PathBuf,
    data_dirs: Vec<PathBuf>,
    metrics_out: PathBuf,
    confusion_out: PathBuf,
    n_classes: Option<usize>,
    threads: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<EvaluateConfig> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let mut model: Option<PathBuf> = None;
    let mut data_dirs: Vec<PathBuf> = Vec::new();
    let mut metrics_out: Option<PathBuf> = None;
    let mut confusion_out: Option<PathBuf> = None;
    let mut n_classes: Option<usize> = None;
    let mut threads: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(require_value(args, i, "--model")?));
            }
            "--data-dir" => {
                i += 1;
                data_dirs.push(PathBuf::from(require_value(args, i, "--data-dir")?));
            }
            "--metrics-out" => {
                i += 1;
                metrics_out = Some(PathBuf::from(require_value(args, i, "--metrics-out")?));
            }
            "--confusion-out" => {
                i += 1;
                confusion_out = Some(PathBuf::from(require_value(args, i, "--confusion-out")?));
            }
            "--n-classes" => {
                i += 1;
                let val = require_value(args, i, "--n-classes")?;
                n_classes = Some(parse_positive_usize(val, "--n-classes")?);
            }
            "--threads" => {
                i += 1;
                let val = require_value(args, i, "--threads")?;
                threads = Some(parse_positive_usize(val, "--threads")?);
            }
            unknown => {
                return Err(ClassifierError::Pipeline(format!(
                    "evaluate: unknown argument '{unknown}'"
                )));
            }
        }
        i += 1;
    }

    if data_dirs.is_empty() {
        return Err(ClassifierError::Pipeline(
            "evaluate: at least one --data-dir is required".into(),
        ));
    }

    Ok(EvaluateConfig {
        model: model
            .ok_or_else(|| ClassifierError::Pipeline("evaluate: --model is required".into()))?,
        data_dirs,
        metrics_out: metrics_out.ok_or_else(|| {
            ClassifierError::Pipeline("evaluate: --metrics-out is required".into())
        })?,
        confusion_out: confusion_out.ok_or_else(|| {
            ClassifierError::Pipeline("evaluate: --confusion-out is required".into())
        })?,
        n_classes,
        threads,
    })
}

fn require_value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("evaluate: {flag} requires a value")))
}

fn parse_positive_usize(val: &str, flag: &str) -> Result<usize> {
    let n = val.parse::<usize>().map_err(|_| {
        ClassifierError::Pipeline(format!(
            "evaluate: {flag} must be a positive integer, got '{val}'"
        ))
    })?;
    if n == 0 {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: {flag} must be >= 1"
        )));
    }
    Ok(n)
}

fn print_help() {
    eprintln!(
        "Usage: wb_lidar_train evaluate [options]\n\
         \n\
         Score a trained .wbmodel on a labeled, held-out data directory and\n\
         emit segmentation metrics as two CSV files.\n\
         \n\
         Options:\n\
           --model         <path>   Trained .wbmodel to evaluate (required)\n\
           --data-dir      <dir>    Labeled data dir with labeled_blocks.json +\n\
                                    .feat/.lbl blocks (repeatable, required)\n\
           --metrics-out   <path>   Per-class metrics CSV output (required)\n\
           --confusion-out <path>   Confusion-matrix CSV output (required)\n\
           --n-classes     <n>      Optional cross-check vs model/data (default: derived)\n\
           --threads       <n>      Rayon thread pool size (default: system cores)\n\
           --help, -h               Show this message\n\
         \n\
         Note: --data-dir must point at a `preprocess-labeled` / `split-dataset`\n\
           output directory (containing labeled_blocks.json). The model and the\n\
           data must share the same label map / class count."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::Linear;
    use crate::model::pointnet::PointNetConfig;
    use crate::preprocessing::labeled_pipeline::{
        LabeledBlockManifest, LabeledBlockMeta, SpatialTileGrid,
    };
    use crate::preprocessing::pipeline::BlockMeta;
    use crate::preprocessing::{FEAT_MAGIC, FEAT_VERSION, N_FEATURES};
    use ndarray::{Array1, Array2};
    use std::collections::HashMap;
    use std::path::Path;

    // ── argmax_row ─────────────────────────────────────────────────────────

    #[test]
    fn test_argmax_row_picks_max_and_breaks_ties_high() {
        let row = ndarray::arr1(&[0.1f32, 0.9, 0.3]);
        assert_eq!(argmax_row(&row.view()), 1u8);

        // Tie between idx 0 and 2 → highest index wins (`max_by` keeps last).
        let tie = ndarray::arr1(&[0.5f32, 0.2, 0.5]);
        assert_eq!(argmax_row(&tie.view()), 2u8);
    }

    // ── build a zero-weight model (all-zero logits → argmax = n_classes-1) ──

    fn make_zero_model(n_classes: usize) -> PointNetClassifier {
        let encoder_dims = vec![4usize, 8];
        let decoder_dims = vec![4usize];
        let config = PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: encoder_dims.clone(),
            decoder_dims: decoder_dims.clone(),
            n_classes,
            use_batch_norm: false,
            use_input_tnet: false,
            use_feature_tnet: false,
        };

        let mut encoder_layers = Vec::new();
        let mut prev = N_FEATURES;
        for &dim in &encoder_dims {
            encoder_layers.push((
                Linear::new(Array2::zeros((dim, prev)), Array1::zeros(dim)).unwrap(),
                None,
            ));
            prev = dim;
        }

        let concat_dim = config.concat_dim();
        let mut decoder_layers = Vec::new();
        let mut prev_d = concat_dim;
        for &dim in &decoder_dims {
            decoder_layers.push((
                Linear::new(Array2::zeros((dim, prev_d)), Array1::zeros(dim)).unwrap(),
                None,
            ));
            prev_d = dim;
        }

        let class_proj =
            Linear::new(Array2::zeros((n_classes, prev_d)), Array1::zeros(n_classes)).unwrap();

        // Model index → ASPRS code: class 0 → 2 (Ground), class 1 → 3 (Low veg).
        let label_map: Vec<u8> = (0..n_classes)
            .map(|i| u8::try_from(i + 2).unwrap_or(u8::MAX))
            .collect();

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

    // ── synthetic labeled dir on disk ───────────────────────────────────────

    fn write_feat(path: &Path, n_points: usize) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION);
        bytes.extend_from_slice(&(u32::try_from(n_points).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&(u32::try_from(N_FEATURES).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // block_id
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_x
        bytes.extend_from_slice(&0f64.to_le_bytes()); // origin_y
        for _ in 0..(n_points * N_FEATURES) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        std::fs::write(path, &bytes).expect("write feat");
    }

    fn dummy_manifest(label_map: HashMap<String, u8>) -> LabeledBlockManifest {
        let meta = BlockMeta {
            id: 0,
            file: "block_00000.feat".to_string(),
            origin_x: 0.0,
            origin_y: 0.0,
            raw_point_count: 6,
            sampled_point_count: 6,
            oversampled: false,
        };
        let lbm = LabeledBlockMeta {
            meta,
            lbl_file: "block_00000.lbl".to_string(),
            macro_tile_id: 0,
            class_distribution: HashMap::new(),
        };
        LabeledBlockManifest {
            source: "test.las".into(),
            block_size: 50.0,
            target_points: 6,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map,
            spatial_tile_grid: SpatialTileGrid {
                cols: 1,
                rows: 1,
                bbox_min_x: 0.0,
                bbox_min_y: 0.0,
                bbox_max_x: 50.0,
                bbox_max_y: 50.0,
            },
            blocks: vec![lbm],
        }
    }

    /// Build a one-block, 2-class labeled dir. Ground-truth labels are
    /// `[0,0,0,1,1,1]` (three class-0, three class-1). With a zero-weight
    /// model every logit row is all-equal, so argmax → the highest index
    /// (class `n_classes - 1`) and every prediction is class 1.
    fn build_test_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        write_feat(&dir.path().join("block_00000.feat"), 6);
        std::fs::write(dir.path().join("block_00000.lbl"), [0u8, 0, 0, 1, 1, 1])
            .expect("write lbl");

        let mut label_map = HashMap::new();
        label_map.insert("2".to_string(), 0u8);
        label_map.insert("3".to_string(), 1u8);
        let manifest = dummy_manifest(label_map);
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        dir
    }

    // ── end-to-end: deterministic confusion matrix ──────────────────────────

    #[test]
    fn test_run_evaluation_zero_model_all_highest_class() {
        let dir = build_test_dir();
        let model = make_zero_model(2);
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");

        let (metrics, confusion) = run_evaluation(&model, &dataset, 2).expect("evaluate");

        // All-zero logits → argmax = highest index (class 1). Everything
        // predicted class 1.
        // confusion[true][pred]: true0→pred1 = 3, true1→pred1 = 3.
        assert_eq!(confusion[0][0], 0);
        assert_eq!(confusion[0][1], 3);
        assert_eq!(confusion[1][0], 0);
        assert_eq!(confusion[1][1], 3);

        // Class 0: TP=0, FP=0, FN=3 (all class-0 points predicted 1).
        let c0 = &metrics.per_class[0];
        assert_eq!(c0.tp, 0);
        assert_eq!(c0.fp, 0);
        assert_eq!(c0.r#fn, 3);
        // Class 1: TP=3, FP=3 (the 3 class-0 points predicted 1), FN=0.
        let c1 = &metrics.per_class[1];
        assert_eq!(c1.tp, 3);
        assert_eq!(c1.fp, 3);
        assert_eq!(c1.r#fn, 0);

        // Overall accuracy = 3 correct / 6 = 0.5.
        assert!((metrics.overall_accuracy - 0.5).abs() < 1e-9);
    }

    // ── consistency checks ───────────────────────────────────────────────────

    #[test]
    fn test_reconcile_rejects_model_data_class_mismatch() {
        let dir = build_test_dir(); // data declares 2 classes
        let model = make_zero_model(3); // model has 3 classes
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");
        let result = reconcile_n_classes(&model, &dataset, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("classes"));
    }

    #[test]
    fn test_reconcile_rejects_bad_n_classes_override() {
        let dir = build_test_dir();
        let model = make_zero_model(2);
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");
        let result = reconcile_n_classes(&model, &dataset, Some(5));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--n-classes"));
    }

    #[test]
    fn test_reconcile_ok_when_all_agree() {
        let dir = build_test_dir();
        let model = make_zero_model(2);
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");
        assert_eq!(reconcile_n_classes(&model, &dataset, Some(2)).unwrap(), 2);
        assert_eq!(reconcile_n_classes(&model, &dataset, None).unwrap(), 2);
    }

    // Stage 40 (item 2): same class *count* but different label-map *content*
    // must now be rejected, not silently accepted.
    #[test]
    fn test_reconcile_rejects_label_map_content_mismatch() {
        let dir = build_test_dir(); // data: ASPRS "2"->0, "3"->1
        let mut model = make_zero_model(2); // model.label_map defaults to [2, 3]
                                            // Swap the model's label map so index 0 -> ASPRS 3, index 1 -> ASPRS 2
                                            // (same count, same *set* of codes, but different assignment).
        model.label_map = vec![3u8, 2u8];
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");
        let result = reconcile_n_classes(&model, &dataset, None);
        assert!(
            result.is_err(),
            "same class count but swapped label-map content must be rejected"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("label map mismatch"));
    }

    // ── CSV writer ───────────────────────────────────────────────────────────

    #[test]
    fn test_write_per_class_metrics_csv_shape() {
        let dir = build_test_dir();
        let model = make_zero_model(2);
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");
        let (metrics, _) = run_evaluation(&model, &dataset, 2).expect("evaluate");

        let out = tempfile::tempdir().expect("tempdir");
        let csv = out.path().join("metrics.csv");
        write_per_class_metrics_csv(&csv, &metrics, &model.label_map).expect("write csv");

        let text = std::fs::read_to_string(&csv).expect("read csv");
        let lines: Vec<&str> = text.lines().collect();
        // header + 2 class rows
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("class_idx,asprs_code,tp,fp,tn,fn"));
        // class 0 row → asprs code 2
        assert!(lines[1].starts_with("0,2,"));
        // class 1 row → asprs code 3
        assert!(lines[2].starts_with("1,3,"));
    }

    // ── arg parsing ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_args_requires_data_dir() {
        let args = vec![
            "--model".to_string(),
            "m.wbmodel".to_string(),
            "--metrics-out".to_string(),
            "m.csv".to_string(),
            "--confusion-out".to_string(),
            "c.csv".to_string(),
        ];
        let result = parse_args(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--data-dir is required"));
    }

    #[test]
    fn test_parse_args_rejects_unknown_flag() {
        let args = vec!["--bogus".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_parse_args_ok() {
        let args = vec![
            "--model".to_string(),
            "m.wbmodel".to_string(),
            "--data-dir".to_string(),
            "test".to_string(),
            "--metrics-out".to_string(),
            "m.csv".to_string(),
            "--confusion-out".to_string(),
            "c.csv".to_string(),
        ];
        let cfg = parse_args(&args).expect("parse");
        assert_eq!(cfg.data_dirs.len(), 1);
        assert!(cfg.n_classes.is_none());
    }
}
