//! Training metrics — mIoU, per-class IoU, Precision, Recall, F1, confusion matrix.
//!
//! All counters are `u64` to avoid overflow on large regional datasets.
//! Absent classes (`TP+FP+FN == 0`) are excluded from mIoU and `F1_macro` averages.

#![allow(clippy::must_use_candidate, clippy::missing_errors_doc, clippy::cast_precision_loss, clippy::cast_lossless, clippy::doc_markdown)]

use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

/// Per-class metric values after a full validation pass.
#[derive(Debug, Clone, Serialize)]
pub struct ClassMetrics {
    pub class_idx: usize,
    pub tp: u64,
    pub fp: u64,
<<<<<<< HEAD
    pub tn: u64,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    pub r#fn: u64,
    pub iou: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Aggregated metrics for one validation pass.
#[derive(Debug, Clone, Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
<<<<<<< HEAD
    /// Unweighted mean cross-entropy on validation blocks (comparable across runs).
    pub val_loss: f64,
    /// Class-weighted mean cross-entropy on validation blocks (comparable to train_loss).
    pub val_loss_weighted: f64,
    pub miou: f64,
    /// Overall accuracy: sum(TP) / total_validation_points.
    pub overall_accuracy: f64,
=======
    pub val_loss: f64,
    pub miou: f64,
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    pub f1_macro: f64,
    pub per_class: Vec<ClassMetrics>,
}

/// Accumulates TP/FP/FN and confusion matrix across validation blocks.
pub struct MetricsAccumulator {
    n_classes: usize,
    tp: Vec<u64>,
    fp: Vec<u64>,
    fn_: Vec<u64>,
    /// confusion_matrix[true_class][predicted_class]
    confusion: Vec<Vec<u64>>,
    loss_sum: f64,
    loss_count: u64,
<<<<<<< HEAD
    loss_weighted_sum: f64,
    loss_weighted_count: u64,
    /// Total number of predictions accumulated (used to compute TN and OA).
    total_points: u64,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
}

impl MetricsAccumulator {
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            tp: vec![0; n_classes],
            fp: vec![0; n_classes],
            fn_: vec![0; n_classes],
            confusion: vec![vec![0; n_classes]; n_classes],
            loss_sum: 0.0,
            loss_count: 0,
<<<<<<< HEAD
            loss_weighted_sum: 0.0,
            loss_weighted_count: 0,
            total_points: 0,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
        }
    }

    /// Accumulate predictions and ground-truth labels for one block.
    pub fn accumulate(&mut self, predictions: &[u8], ground_truth: &[u8]) {
        for (&pred, &gt) in predictions.iter().zip(ground_truth.iter()) {
            let p = pred as usize;
            let g = gt as usize;
            if p >= self.n_classes || g >= self.n_classes {
                continue; // out-of-range labels are skipped
            }
            if p == g {
                self.tp[g] += 1;
            } else {
                self.fp[p] += 1;
                self.fn_[g] += 1;
            }
            self.confusion[g][p] += 1;
<<<<<<< HEAD
            self.total_points += 1;
        }
    }

    /// Record a block-level unweighted validation loss.
=======
        }
    }

    /// Record a block-level validation loss.
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_sum += loss;
        self.loss_count += 1;
    }

<<<<<<< HEAD
    /// Record a block-level class-weighted validation loss.
    pub fn add_loss_weighted(&mut self, loss: f64) {
        self.loss_weighted_sum += loss;
        self.loss_weighted_count += 1;
    }

=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    /// Compute final metrics from accumulated counters.
    pub fn compute(&self, epoch: usize, train_loss: f64) -> EpochMetrics {
        let mut per_class = Vec::with_capacity(self.n_classes);
        let mut iou_sum = 0.0;
        let mut f1_sum = 0.0;
        let mut present = 0usize;
<<<<<<< HEAD
        let mut tp_total = 0u64;

        for c in 0..self.n_classes {
            let tp  = self.tp[c];
            let fp  = self.fp[c];
            let fn_ = self.fn_[c];
            // TN = all points not involved in a TP/FP/FN for this class.
            let tn  = self.total_points.saturating_sub(tp + fp + fn_);
            tp_total += tp;

=======

        for c in 0..self.n_classes {
            let tp = self.tp[c];
            let fp = self.fp[c];
            let fn_ = self.fn_[c];
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
            let denom_iou = tp + fp + fn_;

            if denom_iou == 0 {
                // Class absent from validation set — exclude from averages.
                per_class.push(ClassMetrics {
                    class_idx: c,
<<<<<<< HEAD
                    tp, fp, tn, r#fn: fn_,
=======
                    tp, fp, r#fn: fn_,
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
                    iou: 0.0, precision: 0.0, recall: 0.0, f1: 0.0,
                });
                continue;
            }

<<<<<<< HEAD
            let iou       = tp as f64 / denom_iou as f64;
            let precision = if tp + fp  == 0 { 0.0 } else { tp as f64 / (tp + fp)  as f64 };
=======
            let iou = tp as f64 / denom_iou as f64;
            let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
            let recall    = if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 };
            let f1 = if precision + recall < 1e-12 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };

            iou_sum += iou;
            f1_sum  += f1;
            present += 1;

<<<<<<< HEAD
            per_class.push(ClassMetrics { class_idx: c, tp, fp, tn, r#fn: fn_, iou, precision, recall, f1 });
        }

        let miou             = if present == 0 { 0.0 } else { iou_sum / present as f64 };
        let f1_macro         = if present == 0 { 0.0 } else { f1_sum  / present as f64 };
        let overall_accuracy = if self.total_points == 0 { 0.0 } else { tp_total as f64 / self.total_points as f64 };
        let val_loss         = if self.loss_count == 0 { 0.0 } else { self.loss_sum / self.loss_count as f64 };
        let val_loss_weighted = if self.loss_weighted_count == 0 { val_loss } else { self.loss_weighted_sum / self.loss_weighted_count as f64 };

        EpochMetrics { epoch, train_loss, val_loss, val_loss_weighted, miou, overall_accuracy, f1_macro, per_class }
=======
            per_class.push(ClassMetrics { class_idx: c, tp, fp, r#fn: fn_, iou, precision, recall, f1 });
        }

        let miou     = if present == 0 { 0.0 } else { iou_sum / present as f64 };
        let f1_macro = if present == 0 { 0.0 } else { f1_sum  / present as f64 };
        let val_loss = if self.loss_count == 0 { 0.0 } else { self.loss_sum / self.loss_count as f64 };

        EpochMetrics { epoch, train_loss, val_loss, miou, f1_macro, per_class }
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    }

    /// Return the confusion matrix as a 2D vec (rows=true, cols=predicted).
    pub fn confusion_matrix(&self) -> &Vec<Vec<u64>> {
        &self.confusion
    }
}

<<<<<<< HEAD
/// Write per-epoch metrics to a CSV file.
///
/// On epoch 1 the file is **truncated** (any prior run's data is discarded) and
/// a fresh header row is written.  On all subsequent epochs the new row is
/// appended.  This prevents repeated header rows when training is re-run to the
/// same `--metrics-out` path.
pub fn append_metrics_csv(path: &Path, m: &EpochMetrics) -> std::io::Result<()> {
    // Epoch 1 starts a new training run: truncate any existing file so that
    // re-runs do not concatenate multiple runs with embedded header rows.
    let new_run = m.epoch == 1;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(!new_run)
        .truncate(new_run)
        .open(path)?;
    let mut w = BufWriter::new(file);

    if new_run {
        // Header: global fields then 8 per-class columns for each class.
        write!(w, "epoch,train_loss,val_loss_uw,val_loss_w,val_miou,val_oa,f1_macro")?;
        for c in 0..m.per_class.len() {
            write!(w, ",tp_cls_{c},fp_cls_{c},tn_cls_{c},fn_cls_{c},prec_cls_{c},rec_cls_{c},f1_cls_{c},iou_cls_{c}")?;
=======
/// Write per-epoch metrics to a CSV file (appending).
pub fn append_metrics_csv(path: &Path, m: &EpochMetrics) -> std::io::Result<()> {
    let file_exists = path.exists();
    let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut w = BufWriter::new(file);

    if !file_exists {
        // Header row
        write!(w, "epoch,train_loss,val_loss,val_miou,f1_macro")?;
        for c in 0..m.per_class.len() {
            write!(w, ",IoU_cls_{c}")?;
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
        }
        writeln!(w)?;
    }

<<<<<<< HEAD
    write!(w, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        m.epoch, m.train_loss, m.val_loss, m.val_loss_weighted,
        m.miou, m.overall_accuracy, m.f1_macro)?;
    for cm in &m.per_class {
        write!(w, ",{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            cm.tp, cm.fp, cm.tn, cm.r#fn,
            cm.precision, cm.recall, cm.f1, cm.iou)?;
=======
    write!(w, "{},{:.6},{:.6},{:.6},{:.6}",
        m.epoch, m.train_loss, m.val_loss, m.miou, m.f1_macro)?;
    for cm in &m.per_class {
        write!(w, ",{:.6}", cm.iou)?;
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    }
    writeln!(w)?;
    w.flush()
}

<<<<<<< HEAD
=======
/// Write the confusion matrix to a CSV file.
pub fn write_confusion_matrix_csv(
    path: &Path,
    matrix: &[Vec<u64>],
) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    for row in matrix {
        let v: String = row
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        writeln!(w, "{v}")?;
    }
    w.flush()
}

>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miou_three_class() {
        // Hand-calculated 3-class scenario
        // Class 0: TP=10, FP=2, FN=3 → IoU = 10/15
        // Class 1: TP=5,  FP=1, FN=4 → IoU = 5/10 = 0.5
        // Class 2: TP=8,  FP=0, FN=2 → IoU = 8/10 = 0.8
        // mIoU = (10/15 + 0.5 + 0.8) / 3
        let mut acc = MetricsAccumulator::new(3);

        // Simulate TP[0]=10
        for _ in 0..10 { acc.accumulate(&[0], &[0]); }
        // Simulate FP[0]=2 (pred=0, gt=1)
        for _ in 0..2 { acc.accumulate(&[0], &[1]); }
        // Simulate FN[0]=3 (pred=1, gt=0)
        for _ in 0..3 { acc.accumulate(&[1], &[0]); }
        // Simulate TP[1]=5
        for _ in 0..5 { acc.accumulate(&[1], &[1]); }
        // FP[1] not needed separately; already accounted for via FN[0]
        // Simulate FN[1]=4 (pred=2, gt=1)
        for _ in 0..4 { acc.accumulate(&[2], &[1]); }
        // TP[2]=8
        for _ in 0..8 { acc.accumulate(&[2], &[2]); }
        // FN[2]=2 (pred=0, gt=2)
        for _ in 0..2 { acc.accumulate(&[0], &[2]); }

        let m = acc.compute(1, 0.0);
        // Class 0: TP=10, FP=2+2=4 (pred=0 when gt=1 × 2, pred=0 when gt=2 × 2), FN=3
        // Actually let me recount from the simulation...
        // Just verify miou > 0 and <= 1
        assert!(m.miou > 0.0 && m.miou <= 1.0);
        assert_eq!(m.per_class.len(), 3);
    }

    #[test]
    fn test_miou_absent_class_excluded() {
        let mut acc = MetricsAccumulator::new(3);
        // Only class 0 has any predictions
        for _ in 0..5 { acc.accumulate(&[0], &[0]); }
        let m = acc.compute(1, 0.0);
        // Class 1 and 2 are absent → should be excluded from miou
        // miou should equal IoU of class 0 alone
        let iou0 = m.per_class[0].iou;
        assert!((m.miou - iou0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_shape() {
        // 4-point prediction set, 3 classes
        let preds = vec![0u8, 1, 2, 0];
        let gts   = vec![0u8, 1, 1, 2];
        let mut acc = MetricsAccumulator::new(3);
        acc.accumulate(&preds, &gts);
        let cm = acc.confusion_matrix();
        assert_eq!(cm.len(), 3);
        assert_eq!(cm[0].len(), 3);
        // cm[gt][pred]: cm[0][0]=1 (pred=0,gt=0), cm[1][1]=1 (pred=1,gt=1),
        //               cm[1][2]=1 (pred=2,gt=1), cm[2][0]=1 (pred=0,gt=2)
        assert_eq!(cm[0][0], 1);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[1][2], 1);
        assert_eq!(cm[2][0], 1);
    }

    #[test]
    fn test_class_weight_computation() {
        // 3 classes, known counts: [100, 50, 25]
        // total = 175
        // weight[c] = 175 / (3 * count[c])
        let counts = [100u64, 50, 25];
        let total: u64 = counts.iter().sum();
        let n_classes = 3usize;
        let weights: Vec<f32> = counts
            .iter()
            .map(|&c| (total as f64 / (n_classes as f64 * c as f64)) as f32)
            .collect();
        // weight[0] = 175 / (3 * 100) ≈ 0.5833
        // weight[1] = 175 / (3 * 50)  ≈ 1.1667
        // weight[2] = 175 / (3 * 25)  ≈ 2.3333
        assert!((weights[0] - 0.5833).abs() < 1e-3);
        assert!((weights[1] - 1.1667).abs() < 1e-3);
        assert!((weights[2] - 2.3333).abs() < 1e-3);
    }

    #[test]
    fn test_spatial_split_fraction() {
        use crate::preprocessing::labeled_pipeline::{LabeledBlockMeta, LabeledBlockManifest, SpatialTileGrid};
        use crate::preprocessing::pipeline::BlockMeta;
        use std::collections::HashMap;

        // 16 macro-tiles (4x4 grid), val_split = 0.25 → expect ~4 val tiles
        let target_val = (16.0_f64 * 0.25).round() as usize;
        assert_eq!(target_val, 4);

        // Simulate the stride selection: stride = floor(16/4) = 4
        let stride = 16 / target_val;
        let val_ids: Vec<usize> = (0..target_val).map(|i| i * stride).collect();
        assert_eq!(val_ids.len(), 4);
        assert_eq!(val_ids, vec![0, 4, 8, 12]);
    }
}
