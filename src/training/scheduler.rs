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

/// A stateless cosine annealing LR scheduler with optional linear warmup.
#[derive(Debug, Clone)]
pub struct CosineScheduler {
    lr_max: f64,
    lr_min: f64,
    total_steps: usize,
    /// Stage 22 (Training Loop Enhancements): number of initial steps over
    /// which `lr(t)` ramps linearly from `0` to `lr_max`, before cosine
    /// annealing begins. `0` disables warmup entirely (the default via
    /// `new()`).
    warmup_steps: usize,
}

impl CosineScheduler {
    /// Create a scheduler with no warmup (`warmup_steps = 0`).
    ///
    /// - `lr_max`       — peak learning rate (at step 0)
    /// - `lr_min`       — floor learning rate (default `1e-6`)
    /// - `total_steps`  — total gradient steps over the entire training run
    pub fn new(lr_max: f64, lr_min: f64, total_steps: usize) -> Self {
        Self::with_warmup(lr_max, lr_min, total_steps, 0)
    }

    /// Create a scheduler with a linear LR warmup phase.
    ///
    /// - `lr_max`        — peak learning rate (reached at the end of warmup,
    ///   or at step 0 if `warmup_steps == 0`)
    /// - `lr_min`        — floor learning rate
    /// - `total_steps`   — total gradient steps over the entire training run
    ///   (includes the warmup steps)
    /// - `warmup_steps`  — number of initial steps over which `lr(t)` ramps
    ///   linearly from `0` to `lr_max`; `0` disables warmup
    pub fn with_warmup(lr_max: f64, lr_min: f64, total_steps: usize, warmup_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            total_steps: total_steps.max(1),
            warmup_steps,
        }
    }

    /// Return the learning rate for global step `t`.
    pub fn lr(&self, t: usize) -> f64 {
        if self.warmup_steps > 0 && t < self.warmup_steps {
            return self.lr_max * (t as f64 / self.warmup_steps as f64);
        }
        // Re-base the cosine curve so it spans the *post-warmup* remainder of
        // training: step `warmup_steps` maps to cosine-t=0 (lr_max), and step
        // `total_steps` still maps to cosine-t=post_warmup_total (lr_min).
        let post_warmup_t = t.saturating_sub(self.warmup_steps);
        let post_warmup_total = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let t_clamped = post_warmup_t.min(post_warmup_total) as f64;
        let cos_val = (PI * t_clamped / post_warmup_total as f64).cos();
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
        let expected = f64::midpoint(1e-3, 1e-6);
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

    /// Stage 22 (Training Loop Enhancements) — linear warmup ramp, followed by
    /// a cosine curve re-based over the post-warmup remainder of training.
    #[test]
    fn test_cosine_schedule_with_warmup() {
        let sched = CosineScheduler::with_warmup(1e-3, 1e-6, 110, 10);

        // During warmup (t < warmup_steps): lr ramps linearly from 0 to lr_max.
        let lr0 = sched.lr(0);
        assert!((lr0 - 0.0).abs() < 1e-12, "lr(0) should be 0.0, got {lr0}");

        let lr5 = sched.lr(5);
        let expected5 = 1e-3 * (5.0 / 10.0);
        assert!(
            (lr5 - expected5).abs() < 1e-10,
            "lr(5) should be half of lr_max during warmup, got {lr5}"
        );

        // At t == warmup_steps: lr should equal lr_max (start of cosine phase).
        let lr_at_warmup_end = sched.lr(10);
        assert!(
            (lr_at_warmup_end - 1e-3).abs() < 1e-9,
            "lr(warmup_steps) should be lr_max, got {lr_at_warmup_end}"
        );

        // At t == total_steps: lr should be lr_min (cos(π) = -1), same as
        // the no-warmup case, since the cosine curve is re-based to still
        // span the full post-warmup remainder.
        let lr_end = sched.lr(110);
        assert!(
            (lr_end - 1e-6).abs() < 1e-9,
            "lr(total_steps) should be lr_min, got {lr_end}"
        );

        // Midpoint of the post-warmup remainder (t = 10 + (110-10)/2 = 60)
        // should be the (lr_max + lr_min) / 2 midpoint, exactly as the
        // no-warmup cosine curve's T/2 point.
        let lr_mid = sched.lr(60);
        let expected_mid = f64::midpoint(1e-3, 1e-6);
        assert!(
            (lr_mid - expected_mid).abs() < 1e-9,
            "lr(post-warmup midpoint) should be {expected_mid}, got {lr_mid}"
        );

        // warmup_steps = 0 must behave identically to `new(...)` (regression
        // guard for the delegation in `new()`).
        let sched_no_warmup = CosineScheduler::with_warmup(1e-3, 1e-6, 100, 0);
        let sched_new = CosineScheduler::new(1e-3, 1e-6, 100);
        for t in [0, 25, 50, 75, 100] {
            assert!(
                (sched_no_warmup.lr(t) - sched_new.lr(t)).abs() < 1e-12,
                "with_warmup(..., 0) should match new(...) at t={t}"
            );
        }
    }
}
