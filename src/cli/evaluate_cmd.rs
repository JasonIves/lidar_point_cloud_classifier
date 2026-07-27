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

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::error::{ClassifierError, Result};
use crate::model::fusion::{default_proximity_sigma, fused_label, GridGeometry};
use crate::model::inference::{reconstruct_xy, BlockInferenceResult};
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

    // ── Stage 44: fused evaluation (cross-block prediction fusion) ─────────
    if cfg.fused_eval {
        return run_fused(&cfg, &model);
    }

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

// ─────────────────────────────────────────────────────────────────────────────
// Fused evaluation (Stage 44)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-directory statistics returned by [`run_evaluation_fused`].
#[derive(Debug, Default)]
struct FusedStats {
    n_blocks: usize,
    n_points: u64,
    band_points: u64,
    interior_points: u64,
    /// Radius actually used (explicit flag or derived `block_size / 4`).
    radius: f64,
}

/// Entry point for `evaluate --fused-eval` (Stage 44).
///
/// Replicates the deployed (write-time) decision rule — cross-block weighted
/// soft voting via [`fused_label`] — over every labeled point, and reports the
/// standard metric suite plus a **boundary-band vs. interior** split that
/// isolates the block-seam behaviour the fusion mechanism targets.
///
/// Each `--data-dir` is loaded and evaluated **independently** with its own
/// derived grid: blocks from different directories describe unrelated spatial
/// regions, so cross-directory voting would be physically meaningless (and
/// block IDs collide across directories by design).
///
/// # Errors
/// Propagates dataset/reconcile/fusion/CSV errors; rejects an invalid
/// `--fusion-temp`.
fn run_fused(cfg: &EvaluateConfig, model: &PointNetClassifier) -> Result<()> {
    let fusion_temp = cfg.fusion_temp.unwrap_or(1.0);
    if !fusion_temp.is_finite() || fusion_temp <= 0.0 {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: --fusion-temp must be finite and > 0.0, got {fusion_temp}"
        )));
    }

    let mut full_acc: Option<MetricsAccumulator> = None;
    let mut band_acc: Option<MetricsAccumulator> = None;
    let mut interior_acc: Option<MetricsAccumulator> = None;
    let mut total = FusedStats::default();

    for dir in &cfg.data_dirs {
        eprintln!("[evaluate] (fused) loading data dir: {}", dir.display());
        // Single-dir load ⇒ every GlobalBlockId has dir_idx 0, i.e. the
        // composite id equals the local block id — required for the grid
        // arithmetic in `fused_label`.
        let dataset = LabeledBlockDataset::load(std::slice::from_ref(dir), 0.0, None, 0)?;
        let n_classes = reconcile_n_classes(model, &dataset, cfg.n_classes)?;

        let full = full_acc.get_or_insert_with(|| MetricsAccumulator::new(n_classes));
        let band = band_acc.get_or_insert_with(|| MetricsAccumulator::new(n_classes));
        let interior = interior_acc.get_or_insert_with(|| MetricsAccumulator::new(n_classes));

        let stats = run_evaluation_fused(
            model,
            &dataset,
            cfg.fusion_radius,
            fusion_temp,
            full,
            band,
            interior,
        )?;
        total.n_blocks += stats.n_blocks;
        total.n_points += stats.n_points;
        total.band_points += stats.band_points;
        total.interior_points += stats.interior_points;
        total.radius = stats.radius;
    }

    // `parse_args` guarantees at least one --data-dir, so the accumulators
    // were necessarily initialised above.
    let (Some(full), Some(band), Some(interior)) = (full_acc, band_acc, interior_acc) else {
        return Err(ClassifierError::Pipeline(
            "evaluate: --fused-eval produced no accumulators (no data dirs?)".into(),
        ));
    };

    eprintln!(
        "[evaluate] (fused) scored {} block(s), {} point(s) — radius={}, temp={fusion_temp}",
        total.n_blocks, total.n_points, total.radius
    );

    // ── Standard outputs (full fused set) ──────────────────────────────────
    let metrics = full.compute(1, 0.0);
    let confusion = full.confusion_matrix().clone();
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

    // ── Boundary-band vs. interior analysis ────────────────────────────────
    eprintln!(
        "[evaluate] ─── fused boundary analysis (radius={}) ───",
        total.radius
    );
    print_band_subset("boundary band", &band, total.band_points, &model.label_map);
    print_band_subset(
        "interior     ",
        &interior,
        total.interior_points,
        &model.label_map,
    );
    eprintln!("[evaluate] ────────────────────────────────────────────────────");

    Ok(())
}

/// Print `mIoU` / overall accuracy / per-class `IoU` for one band subset, or
/// an explicit "n/a" line when the subset is empty (avoids degenerate
/// metrics).
fn print_band_subset(name: &str, acc: &MetricsAccumulator, n_points: u64, label_map: &[u8]) {
    if n_points == 0 {
        eprintln!("[evaluate]   {name}: n/a (0 points)");
        return;
    }
    let m = acc.compute(1, 0.0);
    eprintln!(
        "[evaluate]   {name}: mIoU {:.4}, accuracy {:.4} ({n_points} points)",
        m.miou, m.overall_accuracy
    );
    for cm in &m.per_class {
        let asprs = label_map.get(cm.class_idx).copied().unwrap_or(1);
        eprintln!(
            "[evaluate]     class {:>2}/{:<3}: IoU {:.4}",
            cm.class_idx, asprs, cm.iou
        );
    }
}

/// Per-block intermediate produced by the parallel forward pass of fused
/// evaluation (Stage 44): vote structure, absolute query coordinates,
/// boundary-band mask, and ground-truth labels.
type BlockVoteData = (
    u64,
    BlockInferenceResult,
    Vec<f64>,
    Vec<f64>,
    Vec<bool>,
    Vec<u8>,
);

/// Per-block query payload consumed by the fused-prediction phase.
type BlockQueryData = (Vec<f64>, Vec<f64>, Vec<bool>, Vec<u8>);

/// Derive the uniform grid geometry and validated fusion radius for one data
/// directory.
///
/// Grid geometry is derived from block origins (`LabeledBlockManifest`
/// carries no `grid_cols`): `x_min`/`y_min` are the minimum origins and cells
/// are `block_size`-aligned by partitioner construction.  The radius defaults
/// to `block_size / 4` when `--fusion-radius` is not given (the labeled
/// manifest does not carry `block_overlap`).
///
/// # Errors
/// Rejects mixed block sizes within the directory, non-positive block sizes,
/// empty `all_ids`, and an out-of-range or non-finite radius.
// Cell counts are small grid extents; the f64 → i64 cast cannot truncate
// meaningfully.
#[allow(clippy::cast_possible_truncation)]
fn derive_grid_and_radius(
    dataset: &LabeledBlockDataset,
    all_ids: &[u64],
    fusion_radius: Option<f64>,
) -> Result<(GridGeometry, f64)> {
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut east = f64::NEG_INFINITY;
    let mut north = f64::NEG_INFINITY;
    let mut block_size: Option<f64> = None;
    for &gid in all_ids {
        let sm = dataset.block_spatial_meta(gid)?;
        if let Some(bs) = block_size {
            if (bs - sm.block_size).abs() > 1e-9 {
                return Err(ClassifierError::Pipeline(format!(
                    "evaluate: mixed block sizes within one data dir ({bs} vs {}); \
                     --fused-eval requires a uniform grid",
                    sm.block_size
                )));
            }
        } else {
            block_size = Some(sm.block_size);
        }
        x_min = x_min.min(sm.origin_x);
        y_min = y_min.min(sm.origin_y);
        east = east.max(sm.origin_x);
        north = north.max(sm.origin_y);
    }
    let bs = block_size.ok_or_else(|| {
        ClassifierError::Pipeline("evaluate: no blocks to derive block_size from".into())
    })?;
    if bs <= 0.0 {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: non-positive block_size ({bs}) in data dir"
        )));
    }

    let radius = fusion_radius.unwrap_or(bs / 4.0);
    if !radius.is_finite() || radius < 0.0 || radius > bs / 2.0 {
        return Err(ClassifierError::Pipeline(format!(
            "evaluate: --fusion-radius must be finite and within [0, block_size/2 \
             (={})], got {radius}",
            bs / 2.0
        )));
    }

    let grid = GridGeometry {
        x_min,
        y_min,
        block_size: bs,
        grid_cols: ((east - x_min) / bs).round() as i64 + 1,
        grid_rows: ((north - y_min) / bs).round() as i64 + 1,
    };
    Ok((grid, radius))
}

/// Phase 1 (parallel): run the forward pass for every block and build the
/// whole-directory vote map plus per-block query data.  Each worker owns its
/// `Result` — no locks.
///
/// The boundary-band mask is computed from block-relative normalized coords:
/// distance to the canonical edge is `min(xn, 1−xn, yn, 1−yn) · block_size`.
fn build_vote_structures(
    model: &PointNetClassifier,
    dataset: &LabeledBlockDataset,
    all_ids: &[u64],
    block_size: f64,
    radius: f64,
    fusion_temp: f64,
) -> Result<(HashMap<u64, BlockInferenceResult>, Vec<BlockQueryData>)> {
    let phase1: Vec<Result<BlockVoteData>> = all_ids
        .par_iter()
        .map(|&gid| {
            let sm = dataset.block_spatial_meta(gid)?;
            let block = dataset.load_block(gid)?;
            let n = block.features.nrows();
            let band_mask: Vec<bool> = (0..n)
                .map(|i| {
                    let xn = f64::from(block.features[[i, 0]]);
                    let yn = f64::from(block.features[[i, 1]]);
                    xn.min(1.0 - xn).min(yn).min(1.0 - yn) * block_size < radius
                })
                .collect();
            let (xs, ys) = reconstruct_xy(&block.features, sm.origin_x, sm.origin_y, block_size);
            let logits = model.forward(block.features)?;
            let result = BlockInferenceResult::from_logits(&xs, &ys, &logits, fusion_temp)?;
            Ok((gid, result, xs, ys, band_mask, block.labels))
        })
        .collect();

    let mut map: HashMap<u64, BlockInferenceResult> = HashMap::with_capacity(all_ids.len());
    let mut query_data: Vec<BlockQueryData> = Vec::with_capacity(all_ids.len());
    for item in phase1 {
        let (gid, result, xs, ys, band_mask, labels) = item?;
        map.insert(gid, result);
        query_data.push((xs, ys, band_mask, labels));
    }
    Ok((map, query_data))
}

/// Phase 2 (parallel): compute the fused prediction for every labeled point.
///
/// `proximity_sigma` bounds the inverse-square proximity weight — without
/// it, every labeled point (which *is* one of its canonical block's own
/// samples, `d² = 0`) would let the canonical block dominate the blend by
/// orders of magnitude, and fused-eval could never differ from unfused eval
/// (the blind spot found in real-data validation).
///
/// A labeled point with no votes at all cannot occur in practice (its own
/// block is always in the map); any such point is skipped defensively rather
/// than panicking or injecting a spurious class.
fn fused_predictions(
    map: &HashMap<u64, BlockInferenceResult>,
    grid: &GridGeometry,
    radius: f64,
    proximity_sigma: f64,
    n_classes: usize,
    query_data: &[BlockQueryData],
) -> Vec<(Vec<u8>, Vec<u8>, Vec<bool>)> {
    query_data
        .par_iter()
        .map(|(xs, ys, band_mask, labels)| {
            let mut scratch = vec![0.0f64; n_classes];
            let mut preds = Vec::with_capacity(xs.len());
            let mut gts = Vec::with_capacity(xs.len());
            let mut bands = Vec::with_capacity(xs.len());
            for i in 0..xs.len() {
                if let Some(idx) = fused_label(
                    xs[i],
                    ys[i],
                    map,
                    grid,
                    radius,
                    proximity_sigma,
                    &mut scratch,
                ) {
                    preds.push(u8::try_from(idx).unwrap_or(u8::MAX));
                    gts.push(labels[i]);
                    bands.push(band_mask[i]);
                }
            }
            (preds, gts, bands)
        })
        .collect()
}

/// Fused evaluation over one single-directory dataset.
///
/// Predictions use the exact write-time fusion rule in model class-index
/// space (`.lbl` labels are already in that space, so no `label_map`
/// translation occurs anywhere in evaluation — the Stage 39 index-space
/// contract is preserved).
///
/// # Errors
/// Propagates block-loading/forward/fusion errors; rejects an out-of-range
/// or non-finite radius; rejects mixed block sizes within one directory.
fn run_evaluation_fused(
    model: &PointNetClassifier,
    dataset: &LabeledBlockDataset,
    fusion_radius: Option<f64>,
    fusion_temp: f64,
    full_acc: &mut MetricsAccumulator,
    band_acc: &mut MetricsAccumulator,
    interior_acc: &mut MetricsAccumulator,
) -> Result<FusedStats> {
    let all_ids = dataset.all_block_ids();
    if all_ids.is_empty() {
        return Err(ClassifierError::Pipeline(
            "evaluate: no blocks found in the supplied --data-dir".into(),
        ));
    }

    let (grid, radius) = derive_grid_and_radius(dataset, &all_ids, fusion_radius)?;
    let (map, query_data) = build_vote_structures(
        model,
        dataset,
        &all_ids,
        grid.block_size,
        radius,
        fusion_temp,
    )?;
    // Proximity bandwidth σ = characteristic inter-sample spacing (see
    // `fused_predictions` docs — the fused-eval blind-spot fix).
    let proximity_sigma = default_proximity_sigma(grid.block_size, dataset.target_points());
    let scored = fused_predictions(
        &map,
        &grid,
        radius,
        proximity_sigma,
        model.config.n_classes,
        &query_data,
    );

    // ── Sequential accumulation: full set + band/interior split ────────────
    let mut stats = FusedStats {
        radius,
        ..FusedStats::default()
    };
    for (preds, gts, bands) in scored {
        stats.n_blocks += 1;
        stats.n_points += u64::try_from(preds.len()).unwrap_or(u64::MAX);
        full_acc.accumulate(&preds, &gts);

        let mut band_preds = Vec::new();
        let mut band_gts = Vec::new();
        let mut int_preds = Vec::new();
        let mut int_gts = Vec::new();
        for ((&p, &g), &b) in preds.iter().zip(gts.iter()).zip(bands.iter()) {
            if b {
                band_preds.push(p);
                band_gts.push(g);
            } else {
                int_preds.push(p);
                int_gts.push(g);
            }
        }
        stats.band_points += u64::try_from(band_preds.len()).unwrap_or(u64::MAX);
        stats.interior_points += u64::try_from(int_preds.len()).unwrap_or(u64::MAX);
        band_acc.accumulate(&band_preds, &band_gts);
        interior_acc.accumulate(&int_preds, &int_gts);
    }

    Ok(stats)
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
    /// Stage 44: replicate the deployed cross-block fusion decision rule and
    /// report boundary-band vs. interior metrics.
    fused_eval: bool,
    fusion_radius: Option<f64>,
    fusion_temp: Option<f64>,
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
    let mut fused_eval = false;
    let mut fusion_radius: Option<f64> = None;
    let mut fusion_temp: Option<f64> = None;

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
            "--fused-eval" => {
                fused_eval = true;
            }
            "--fusion-radius" => {
                i += 1;
                let val = require_value(args, i, "--fusion-radius")?;
                fusion_radius = Some(val.parse::<f64>().map_err(|_| {
                    ClassifierError::Pipeline(format!(
                        "evaluate: --fusion-radius must be a number, got '{val}'"
                    ))
                })?);
            }
            "--fusion-temp" => {
                i += 1;
                let val = require_value(args, i, "--fusion-temp")?;
                fusion_temp = Some(val.parse::<f64>().map_err(|_| {
                    ClassifierError::Pipeline(format!(
                        "evaluate: --fusion-temp must be a number, got '{val}'"
                    ))
                })?);
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

    if !fused_eval && (fusion_radius.is_some() || fusion_temp.is_some()) {
        return Err(ClassifierError::Pipeline(
            "evaluate: --fusion-radius/--fusion-temp require --fused-eval \
             (otherwise they would be silently ignored)"
                .into(),
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
        fused_eval,
        fusion_radius,
        fusion_temp,
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
           --fused-eval           Replicate the deployed cross-block prediction-\n\
                                    fusion decision rule (Stage 44) instead of\n\
                                    per-block argmax, and additionally report\n\
                                    boundary-band vs. interior metrics\n\
           --fusion-radius <f>    Fusion voting reach in projection units\n\
                                    (default: block_size/4; max: block_size/2).\n\
                                    Requires --fused-eval\n\
           --fusion-temp <f>      Softmax temperature before voting (default: 1.0).\n\
                                    Requires --fused-eval\n\
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

// Test assertions compare against exact, exactly-representable constants
// (0.0, 0.5, 1.0, 12.5, …) produced by fixture construction and trivial
// metric arithmetic — strict float equality is intentional and safe here.
#[cfg(test)]
#[allow(clippy::float_cmp)]
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
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_halo (v2)
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
            n_halo: 0,
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
            halo_fraction: 0.0,
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
        assert!(!cfg.fused_eval);
        assert!(cfg.fusion_radius.is_none());
        assert!(cfg.fusion_temp.is_none());
    }

    // ── Stage 44: fused-eval flag validation ────────────────────────────────

    fn base_args() -> Vec<String> {
        vec![
            "--model".to_string(),
            "m.wbmodel".to_string(),
            "--data-dir".to_string(),
            "test".to_string(),
            "--metrics-out".to_string(),
            "m.csv".to_string(),
            "--confusion-out".to_string(),
            "c.csv".to_string(),
        ]
    }

    #[test]
    fn test_parse_args_fusion_flags_require_fused_eval() {
        // --fusion-radius without --fused-eval → hard error (not silently
        // ignored).
        let mut args = base_args();
        args.push("--fusion-radius".to_string());
        args.push("10.0".to_string());
        assert!(parse_args(&args).is_err());

        let mut args = base_args();
        args.push("--fusion-temp".to_string());
        args.push("0.8".to_string());
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_parse_args_fused_eval_with_fusion_flags() {
        let mut args = base_args();
        args.push("--fused-eval".to_string());
        args.push("--fusion-radius".to_string());
        args.push("12.5".to_string());
        args.push("--fusion-temp".to_string());
        args.push("0.8".to_string());
        let cfg = parse_args(&args).expect("parse");
        assert!(cfg.fused_eval);
        assert_eq!(cfg.fusion_radius, Some(12.5));
        assert_eq!(cfg.fusion_temp, Some(0.8));
    }

    // ── Stage 44: fused end-to-end with band split ──────────────────────────

    /// Write a `.feat` file whose rows sit at explicit normalized `(xn, yn)`
    /// positions (all other features zero).
    fn write_feat_at(
        path: &Path,
        block_id: u64,
        origin_x: f64,
        origin_y: f64,
        xy_norm: &[(f32, f32)],
    ) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION);
        bytes.extend_from_slice(&(u32::try_from(xy_norm.len()).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&(u32::try_from(N_FEATURES).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&block_id.to_le_bytes());
        bytes.extend_from_slice(&origin_x.to_le_bytes());
        bytes.extend_from_slice(&origin_y.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_halo (v2)
        for &(xn, yn) in xy_norm {
            for col in 0..N_FEATURES {
                let v = match col {
                    0 => xn,
                    1 => yn,
                    _ => 0.0,
                };
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, &bytes).expect("write feat");
    }

    /// Two-block labeled dir: block 0 at origin (0,0) with 3 points near the
    /// shared seam (`xn = 0.98` → 1 unit from the canonical edge, inside the
    /// default `block_size/4 = 12.5` fusion band); block 1 at origin (50,0)
    /// with 3 deep-interior points (`xn = 0.5` → 25 units from every edge).
    fn build_two_block_fused_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");

        write_feat_at(
            &dir.path().join("block_00000.feat"),
            0,
            0.0,
            0.0,
            &[(0.98, 0.5), (0.98, 0.5), (0.98, 0.5)],
        );
        std::fs::write(dir.path().join("block_00000.lbl"), [0u8, 0, 0]).expect("write lbl 0");

        write_feat_at(
            &dir.path().join("block_00001.feat"),
            1,
            50.0,
            0.0,
            &[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
        );
        std::fs::write(dir.path().join("block_00001.lbl"), [1u8, 1, 1]).expect("write lbl 1");

        let mut label_map = HashMap::new();
        label_map.insert("2".to_string(), 0u8);
        label_map.insert("3".to_string(), 1u8);

        let block_meta = |id: u64, origin_x: f64| LabeledBlockMeta {
            meta: BlockMeta {
                id,
                file: format!("block_{id:05}.feat"),
                origin_x,
                origin_y: 0.0,
                raw_point_count: 3,
                sampled_point_count: 3,
                oversampled: false,
                n_halo: 0,
            },
            lbl_file: format!("block_{id:05}.lbl"),
            macro_tile_id: 0,
            class_distribution: HashMap::new(),
        };
        let manifest = LabeledBlockManifest {
            source: "test.las".into(),
            block_size: 50.0,
            target_points: 3,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map,
            spatial_tile_grid: SpatialTileGrid {
                cols: 2,
                rows: 1,
                bbox_min_x: 0.0,
                bbox_min_y: 0.0,
                bbox_max_x: 100.0,
                bbox_max_y: 50.0,
            },
            halo_fraction: 0.0,
            blocks: vec![block_meta(0, 0.0), block_meta(1, 50.0)],
        };
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        dir
    }

    #[test]
    fn test_fused_eval_two_blocks_band_split() {
        let dir = build_two_block_fused_dir();
        let model = make_zero_model(2); // zero logits → uniform probs → argmax ties high (1)
        let dataset =
            LabeledBlockDataset::load(&[dir.path().to_path_buf()], 0.0, None, 0).expect("load");

        let mut full = MetricsAccumulator::new(2);
        let mut band = MetricsAccumulator::new(2);
        let mut interior = MetricsAccumulator::new(2);
        let stats = run_evaluation_fused(
            &model,
            &dataset,
            None,
            1.0,
            &mut full,
            &mut band,
            &mut interior,
        )
        .expect("fused evaluation must succeed");

        // Default radius = block_size/4 = 12.5.
        assert!((stats.radius - 12.5).abs() < 1e-9);
        assert_eq!(stats.n_blocks, 2);
        assert_eq!(stats.n_points, 6);
        // Block 0's seam-adjacent points form the band; block 1's are interior.
        assert_eq!(stats.band_points, 3);
        assert_eq!(stats.interior_points, 3);

        // Every prediction is class 1 (uniform probs tie-break high).
        // Full set: 3/6 correct → 0.5 accuracy.
        let full_m = full.compute(1, 0.0);
        assert!((full_m.overall_accuracy - 0.5).abs() < 1e-9);
        let confusion = full.confusion_matrix();
        assert_eq!(confusion[0][1], 3);
        assert_eq!(confusion[1][1], 3);

        // Band (block 0, gt all 0): all wrong → 0.0. Interior (block 1, gt
        // all 1): all correct → 1.0.
        let band_m = band.compute(1, 0.0);
        assert_eq!(band_m.overall_accuracy, 0.0);
        let int_m = interior.compute(1, 0.0);
        assert!((int_m.overall_accuracy - 1.0).abs() < 1e-9);
    }
}
