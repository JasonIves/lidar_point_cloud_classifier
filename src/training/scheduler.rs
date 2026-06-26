//! Cosine annealing learning rate scheduler (no external deps).
//!
//! `CosineScheduler` computes:
//!
//! ```text
//! lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
//! ```
//!
//! where `t` is the current global step and `T` is the total number of steps.

#![allow(clippy::must_use_candidate, clippy::cast_precision_loss)]

use std::f64::consts::PI;

/// A stateless cosine annealing LR scheduler.
#[derive(Debug, Clone)]
pub struct CosineScheduler {
    lr_max: f64,
    lr_min: f64,
    total_steps: usize,
}

impl CosineScheduler {
    /// Create a scheduler.
    ///
    /// - `lr_max`       — peak learning rate (at step 0)
    /// - `lr_min`       — floor learning rate (default `1e-6`)
    /// - `total_steps`  — total gradient steps over the entire training run
    pub fn new(lr_max: f64, lr_min: f64, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            total_steps: total_steps.max(1),
        }
    }

    /// Return the learning rate for global step `t`.
    pub fn lr(&self, t: usize) -> f64 {
        let t_clamped = t.min(self.total_steps) as f64;
        let cos_val = (PI * t_clamped / self.total_steps as f64).cos();
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_schedule_values() {
        let sched = CosineScheduler::new(1e-3, 1e-6, 100);

        // At t=0: lr should equal lr_max
        let lr0 = sched.lr(0);
        assert!(
            (lr0 - 1e-3).abs() < 1e-10,
            "lr(0) should be lr_max, got {lr0}"
        );

        // At t=T: lr should be very close to lr_min (cos(π) = -1)
        let lr_t = sched.lr(100);
        assert!(
            (lr_t - 1e-6).abs() < 1e-10,
            "lr(T) should be lr_min, got {lr_t}"
        );

        // At t=T/2: lr should be (lr_max + lr_min) / 2 (cos(π/2) = 0)
        let lr_half = sched.lr(50);
        let expected = (1e-3 + 1e-6) / 2.0;
        assert!(
            (lr_half - expected).abs() < 1e-9,
            "lr(T/2) should be midpoint {expected}, got {lr_half}"
        );

        // Values should monotonically decrease
        let lr_25 = sched.lr(25);
        let lr_75 = sched.lr(75);
        assert!(lr0 > lr_25, "should decrease");
        assert!(lr_25 > lr_half, "should decrease");
        assert!(lr_half > lr_75, "should decrease");
        assert!(lr_75 > lr_t, "should decrease");
    }
}
