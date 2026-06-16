//! Primitive neural-network layers used by `PointNet` and the T-Net sub-networks.
//!
//! All types are purely functional at inference time: weights are immutable after
//! construction and no gradient state is stored.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::error::{ClassifierError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Linear (fully-connected) layer
// ─────────────────────────────────────────────────────────────────────────────

/// A fully-connected linear layer: `output = input @ W^T + b`.
///
/// Weight matrix has shape `[dim_out, dim_in]` (row-major, matching `PyTorch`
/// convention so that weight tensors exported from training are layout-compatible).
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix `[dim_out, dim_in]`.
    pub weight: Array2<f32>,
    /// Bias vector `[dim_out]`.
    pub bias: Array1<f32>,
}

impl Linear {
    /// Create from raw weight and bias arrays.
    ///
    /// # Errors
    /// Returns an error if the bias length does not match `weight.nrows()`.
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Result<Self> {
        if weight.nrows() != bias.len() {
            return Err(ClassifierError::Pipeline(format!(
                "Linear: weight rows ({}) != bias len ({})",
                weight.nrows(),
                bias.len()
            )));
        }
        Ok(Self { weight, bias })
    }

    /// Apply the layer to an `[N, dim_in]` input, returning `[N, dim_out]`.
    ///
    /// # Errors
    /// Returns an error if `input.ncols() != self.weight.ncols()`.
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if input.ncols() != self.weight.ncols() {
            return Err(ClassifierError::Pipeline(format!(
                "Linear::forward: input cols ({}) != weight cols ({})",
                input.ncols(),
                self.weight.ncols()
            )));
        }
        // output[i, j] = sum_k input[i,k] * weight[j,k] + bias[j]
        let mut out = input.dot(&self.weight.t().to_owned());
        out += &self.bias;
        Ok(out)
    }

    /// Apply the layer to a 1-D vector `[dim_in]`, returning `[dim_out]`.
    ///
    /// # Errors
    /// Returns an error if `input.len() != self.weight.ncols()`.
    pub fn forward_1d(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        if input.len() != self.weight.ncols() {
            return Err(ClassifierError::Pipeline(format!(
                "Linear::forward_1d: input len ({}) != weight cols ({})",
                input.len(),
                self.weight.ncols()
            )));
        }
        Ok(self.weight.dot(input) + &self.bias)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchNorm1d (inference mode)
// ─────────────────────────────────────────────────────────────────────────────

/// Batch normalisation layer operating in **inference mode only**.
///
/// Uses stored running `mean` and `var` (from training), plus learnable affine
/// parameters `gamma` (scale) and `beta` (shift).
///
/// Formula per feature `j`:  `y_j = gamma_j * (x_j - mean_j) / sqrt(var_j + eps) + beta_j`
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub mean: Array1<f32>,
    pub var: Array1<f32>,
    pub eps: f32,
}

impl BatchNorm1d {
    /// Construct from pre-trained parameters (all length `features`).
    ///
    /// # Errors
    /// Returns an error if any parameter array has a different length than `gamma`.
    pub fn new(
        gamma: Array1<f32>,
        beta: Array1<f32>,
        mean: Array1<f32>,
        var: Array1<f32>,
    ) -> Result<Self> {
        let n = gamma.len();
        if beta.len() != n || mean.len() != n || var.len() != n {
            return Err(ClassifierError::Pipeline(format!(
                "BatchNorm1d: parameter length mismatch (gamma={}, beta={}, mean={}, var={})",
                n, beta.len(), mean.len(), var.len()
            )));
        }
        Ok(Self { gamma, beta, mean, var, eps: 1e-5 })
    }

    /// Normalise an `[N, features]` matrix in-place, returning the result.
    ///
    /// # Errors
    /// Returns an error if `input.ncols() != self.gamma.len()`.
    pub fn forward(&self, input: Array2<f32>) -> Result<Array2<f32>> {
        if input.ncols() != self.gamma.len() {
            return Err(ClassifierError::Pipeline(format!(
                "BatchNorm1d::forward: input cols ({}) != num features ({})",
                input.ncols(),
                self.gamma.len()
            )));
        }
        let inv_std: Array1<f32> =
            self.var.mapv(|v| 1.0_f32 / (v + self.eps).sqrt());
        // Broadcast: input[N, C] → subtract mean[C], multiply scale[C], add beta[C]
        let centered = input - &self.mean;
        let normed = centered * &inv_std;
        Ok(normed * &self.gamma + &self.beta)
    }

    /// Normalise a 1-D vector `[features]`.
    ///
    /// # Errors
    /// Returns an error if `input.len() != self.gamma.len()`.
    pub fn forward_1d(&self, input: Array1<f32>) -> Result<Array1<f32>> {
        if input.len() != self.gamma.len() {
            return Err(ClassifierError::Pipeline(format!(
                "BatchNorm1d::forward_1d: input len ({}) != num features ({})",
                input.len(), self.gamma.len()
            )));
        }
        let inv_std: Array1<f32> =
            self.var.mapv(|v| 1.0_f32 / (v + self.eps).sqrt());
        let normed = (input - &self.mean) * &inv_std;
        Ok(normed * &self.gamma + &self.beta)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Activations and pooling
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise `ReLU`: `max(0, x)`.
#[must_use]
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Element-wise `ReLU` on a 1-D vector.
#[must_use]
pub fn relu_1d(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Global max pooling over the N-point dimension.
///
/// Input `[N, C]` → output `[C]` (column-wise maximum).
#[must_use]
pub fn global_max_pool(features: &ArrayView2<f32>) -> Array1<f32> {
    features.fold_axis(Axis(0), f32::NEG_INFINITY, |&acc, &x| acc.max(x))
}

// ─────────────────────────────────────────────────────────────────────────────
// T-Net (Spatial Transformer Network)
// ─────────────────────────────────────────────────────────────────────────────

/// A single T-Net block (`STN3d` or `STN64d`) as described in Qi et al. 2017.
///
/// Architecture (fixed dims, not configurable):
/// ```text
/// Mini-encoder (shared MLP):  Linear(k→64) → BN → ReLU
///                             Linear(64→128) → BN → ReLU
///                             Linear(128→1024) → BN → ReLU
/// Global max pool:            [N, 1024] → [1024]
/// FC decoder:                 Linear(1024→512) → BN → ReLU
///                             Linear(512→256) → BN → ReLU
///                             Linear(256→k²)  (no BN, no ReLU)
/// Output: reshape [k²] → [k, k] + I_k  (identity-initialised)
/// ```
///
/// `k` is either 3 (`STN3d`) or 64 (`STN64d`).
#[derive(Debug, Clone)]
pub struct TNet {
    /// Input/output dimension k (3 or 64).
    pub k: usize,
    // Mini-encoder layers
    pub enc0: Linear,
    pub enc1: Linear,
    pub enc2: Linear,
    pub bn_enc0: Option<BatchNorm1d>,
    pub bn_enc1: Option<BatchNorm1d>,
    pub bn_enc2: Option<BatchNorm1d>,
    // FC decoder layers
    pub fc0: Linear,
    pub fc1: Linear,
    pub fc2: Linear,
    pub bn_fc0: Option<BatchNorm1d>,
    pub bn_fc1: Option<BatchNorm1d>,
}

impl TNet {
    /// Run the T-Net on `[N, k]` input features.
    ///
    /// Returns a `[k, k]` transformation matrix with identity added.
    ///
    /// # Errors
    /// Returns an error if `input.ncols() != self.k` or if the final FC
    /// output length does not equal `k²`.
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if input.ncols() != self.k {
            return Err(ClassifierError::Pipeline(format!(
                "TNet::forward: input cols ({}) != k ({})",
                input.ncols(), self.k
            )));
        }

        // ── Mini-encoder ──────────────────────────────────────────────────
        let h = self.enc0.forward(input)?;
        let h = apply_bn2d(h, self.bn_enc0.as_ref())?;
        let h = relu(&h);

        let h = self.enc1.forward(&h)?;
        let h = apply_bn2d(h, self.bn_enc1.as_ref())?;
        let h = relu(&h);

        let h = self.enc2.forward(&h)?;
        let h = apply_bn2d(h, self.bn_enc2.as_ref())?;
        let h = relu(&h);

        // ── Global max pool ────────────────────────────────────────────────
        let g = global_max_pool(&h.view()); // [1024]

        // ── FC decoder ────────────────────────────────────────────────────
        let g = self.fc0.forward_1d(&g)?;
        let g = apply_bn1d(g, self.bn_fc0.as_ref())?;
        let g = relu_1d(&g);

        let g = self.fc1.forward_1d(&g)?;
        let g = apply_bn1d(g, self.bn_fc1.as_ref())?;
        let g = relu_1d(&g);

        let g = self.fc2.forward_1d(&g)?;
        // No BN, no ReLU on final projection

        // ── Reshape + identity initialisation ─────────────────────────────
        let k = self.k;
        if g.len() != k * k {
            return Err(ClassifierError::Pipeline(format!(
                "TNet::forward: final FC output len ({}) != k² ({})",
                g.len(), k * k
            )));
        }
        let transform = Array2::from_shape_vec((k, k), g.to_vec())
            .map_err(|e| ClassifierError::Pipeline(e.to_string()))?;
        // Add identity matrix
        Ok(transform + Array2::<f32>::eye(k))
    }

    /// Apply the `[k, k]` transform matrix to an `[N, k]` feature slice.
    ///
    /// `output = input @ T^T`
    #[must_use]
    pub fn apply(features: &Array2<f32>, transform: &Array2<f32>) -> Array2<f32> {
        features.dot(&transform.t().to_owned())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply an optional `BatchNorm1d` to a 2-D array (no-op when `bn` is `None`).
pub(crate) fn apply_bn2d(
    x: Array2<f32>,
    bn: Option<&BatchNorm1d>,
) -> Result<Array2<f32>> {
    match bn {
        Some(b) => b.forward(x),
        None => Ok(x),
    }
}

/// Apply an optional `BatchNorm1d` to a 1-D vector (no-op when `bn` is `None`).
pub(crate) fn apply_bn1d(
    x: Array1<f32>,
    bn: Option<&BatchNorm1d>,
) -> Result<Array1<f32>> {
    match bn {
        Some(b) => b.forward_1d(x),
        None => Ok(x),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // DoD #4 — Linear forward shape and value correctness
    #[test]
    fn test_linear_forward_shape_and_values() -> Result<()> {
        // W: [2, 3]  b: [2]
        // output = X @ W^T + b   where X: [4, 3]
        let w = Array2::from_shape_vec((2, 3), vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let b = Array1::from_vec(vec![0.5f32, -0.5]);
        let linear = Linear::new(w, b)?;

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                0.0, 0.0, 0.0,
                -1.0, -2.0, -3.0,
            ],
        )
        .unwrap();

        let out = linear.forward(&x)?;
        assert_eq!(out.shape(), &[4, 2]);
        // Row 0: [1*1+2*0+3*0+0.5, 1*0+2*1+3*0-0.5] = [1.5, 1.5]
        assert!((out[[0, 0]] - 1.5).abs() < 1e-6, "out[0,0] = {}", out[[0,0]]);
        assert!((out[[0, 1]] - 1.5).abs() < 1e-6, "out[0,1] = {}", out[[0,1]]);
        // Row 2 (zero input): [0.5, -0.5]
        assert!((out[[2, 0]] - 0.5).abs() < 1e-6);
        assert!((out[[2, 1]] + 0.5).abs() < 1e-6);
        Ok(())
    }

    // DoD #4 — Linear dimension mismatch returns error, not panic
    #[test]
    fn test_linear_forward_dim_mismatch_is_error() {
        let w = Array2::from_shape_vec((2, 3), vec![0.0f32; 6]).unwrap();
        let b = Array1::zeros(2);
        let linear = Linear::new(w, b).unwrap();
        // Wrong number of columns
        let x = Array2::zeros((4, 5));
        assert!(linear.forward(&x).is_err());
    }

    // DoD #5 — BatchNorm1d inference mode
    #[test]
    fn test_batchnorm1d_inference_mode() -> Result<()> {
        // For gamma=1, beta=0, mean=0, var=1: output should equal input
        let gamma = Array1::from_vec(vec![1.0f32, 1.0]);
        let beta  = Array1::zeros(2);
        let mean  = Array1::zeros(2);
        let var   = Array1::from_vec(vec![1.0f32, 1.0]);
        let bn = BatchNorm1d::new(gamma, beta, mean, var)?;

        let x = Array2::from_shape_vec((3, 2), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = bn.forward(x.clone())?;
        // With mean=0, var=1: y = 1 * (x-0)/sqrt(1+1e-5) + 0 ≈ x
        for i in 0..3 {
            for j in 0..2 {
                assert!((out[[i, j]] - x[[i, j]]).abs() < 1e-4,
                    "bn out[{i},{j}] = {} expected ≈ {}", out[[i,j]], x[[i,j]]);
            }
        }

        // Non-trivial: gamma=2, beta=1, mean=1, var=4  → y = 2*(x-1)/sqrt(4+eps)+1
        let gamma2 = Array1::from_vec(vec![2.0f32, 2.0]);
        let beta2  = Array1::from_vec(vec![1.0f32, 1.0]);
        let mean2  = Array1::from_vec(vec![1.0f32, 1.0]);
        let var2   = Array1::from_vec(vec![4.0f32, 4.0]);
        let bn2 = BatchNorm1d::new(gamma2, beta2, mean2, var2)?;
        let x2 = array![[3.0f32, 5.0]];
        let out2 = bn2.forward(x2)?;
        // y[0,0] = 2*(3-1)/sqrt(4+1e-5)+1 ≈ 2*2/2+1 = 3.0
        // y[0,1] = 2*(5-1)/sqrt(4+1e-5)+1 ≈ 2*4/2+1 = 5.0
        assert!((out2[[0, 0]] - 3.0).abs() < 1e-3, "out2[0,0] = {}", out2[[0,0]]);
        assert!((out2[[0, 1]] - 5.0).abs() < 1e-3, "out2[0,1] = {}", out2[[0,1]]);
        Ok(())
    }

    // DoD #6 — ReLU
    #[test]
    fn test_relu_zeros_negatives() {
        let x = Array2::from_shape_vec(
            (2, 3),
            vec![-1.0f32, 0.0, 1.0, -100.0, 0.5, -0.001],
        )
        .unwrap();
        let out = relu(&x);
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 2]], 1.0);
        assert_eq!(out[[1, 0]], 0.0);
        assert_eq!(out[[1, 1]], 0.5);
        assert_eq!(out[[1, 2]], 0.0);
    }

    // DoD #7 — global_max_pool
    #[test]
    fn test_global_max_pool() {
        // 4 points, 3 features
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0f32, 5.0, -1.0,
                3.0,   2.0,  0.0,
                -2.0,  4.0,  7.0,
                0.0,   1.0,  3.0,
            ],
        )
        .unwrap();
        let pool = global_max_pool(&x.view());
        assert_eq!(pool.len(), 3);
        assert!((pool[0] - 3.0).abs() < 1e-6);  // max of col 0
        assert!((pool[1] - 5.0).abs() < 1e-6);  // max of col 1
        assert!((pool[2] - 7.0).abs() < 1e-6);  // max of col 2
    }

    // DoD #8 — STN3d produces [3,3] output and identity weights → identity
    #[test]
    fn test_stn3d_identity_weights_gives_identity_transform() -> Result<()> {
        // Build an STN3d with all-zero weights. Because the final output is
        // reshaped to [3,3] and the identity matrix is added, the transform
        // should be exactly I₃.
        let tnet = make_tnet_zeros(3)?;
        let pts = Array2::from_shape_vec(
            (5, 3),
            vec![1.0f32, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
                 1.0, 1.0, 0.0,  0.0, 1.0, 1.0],
        ).unwrap();
        let t = tnet.forward(&pts)?;
        assert_eq!(t.shape(), &[3, 3]);
        // T should equal I₃ (all linear layers output 0 → fc2 outputs 0 → reshape + I)
        let identity = Array2::<f32>::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((t[[i, j]] - identity[[i, j]]).abs() < 1e-5,
                    "T[{i},{j}] = {} expected {}", t[[i,j]], identity[[i,j]]);
            }
        }
        Ok(())
    }

    // DoD #10 — STN64d produces [64,64] output shape
    #[test]
    fn test_stn64d_output_shape() -> Result<()> {
        let tnet = make_tnet_zeros(64)?;
        let feats = Array2::<f32>::zeros((32, 64));
        let t = tnet.forward(&feats)?;
        assert_eq!(t.shape(), &[64, 64]);
        Ok(())
    }

    /// Construct a TNet with zero weights (and no BN) for a given k.
    fn make_tnet_zeros(k: usize) -> Result<TNet> {
        Ok(TNet {
            k,
            enc0: Linear::new(Array2::zeros((64, k)),    Array1::zeros(64))?,
            enc1: Linear::new(Array2::zeros((128, 64)),  Array1::zeros(128))?,
            enc2: Linear::new(Array2::zeros((1024, 128)),Array1::zeros(1024))?,
            bn_enc0: None,
            bn_enc1: None,
            bn_enc2: None,
            fc0: Linear::new(Array2::zeros((512, 1024)), Array1::zeros(512))?,
            fc1: Linear::new(Array2::zeros((256, 512)),  Array1::zeros(256))?,
            fc2: Linear::new(Array2::zeros((k*k, 256)),  Array1::zeros(k*k))?,
            bn_fc0: None,
            bn_fc1: None,
        })
    }
}
