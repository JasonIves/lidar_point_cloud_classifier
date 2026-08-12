# LiDAR Point Cloud Classifier

**A high-performance, pure-Rust PointNet-based classifier for airborne LiDAR point clouds, built for the Whitebox Next Gen geospatial ecosystem.**

[![Rust](https://img.shields.io/badge/rust-2021-edition?logo=rust)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Overview

The LiDAR Point Cloud Classifier ingests raw airborne LiDAR scans (`.las`, `.laz`, `.copc`), partitions them into spatial blocks, extracts a 17-dimensional feature vector per point (7 scalar + 10 eigenvalue-derived structural features), and runs a **PointNet** deep neural network to produce a fully classified point cloud — labeling each return as Ground, Low/Medium/High Vegetation, Building, Water, or other ASPRS-standard classes.

The project ships two CLI binaries:

| Binary | Purpose | Build |
|--------|---------|-------|
| `wb_lidar_classify` | Preprocess raw LiDAR + run inference with a pre-trained model | `cargo build --release` |
| `wb_lidar_train` | Preprocess labeled data, split datasets, train models, evaluate | `cargo build --release --features training` |

It is designed as a native plugin for the **Whitebox Next Gen** ecosystem and follows Whitebox's core philosophy: **fast, lightweight, pure Rust, platform-agnostic, with graceful GPU fallback.**

---

## Key Features

- **Streaming Block-Based Pipeline** — Processes LiDAR files of any size by dividing them into fixed-size spatial blocks. Memory usage scales with block size and thread count, not file size.
- **17-D Feature Vectors** — Each point is represented by 7 scalar features (normalized coordinates, intensity, height above ground) and 10 eigenvalue-derived structural features (linearity, planarity, sphericity, omnivariance, eigentropy, slope, residual).
- **PointNet Architecture** — A canonical PointNet segmentation network (Qi et al., 2017) with 5 encoder layers (17→64→64→64→128→1024), global max-pooling, local+global feature concatenation (1088-dim), and a 2-layer decoder (512→256→*n_classes*).
- **GPU-Accelerated Training** — Leverages the `burn` deep-learning framework with automatic GPU detection and graceful CPU fallback. No GPU? Training runs on CPU without error.
- **Prediction Fusion** — Cross-block soft-voting at inference time mitigates "patchwork quilt" seam artifacts between adjacent blocks. No retraining required.
- **Halo-Augmented Blocks** — Extends each block's input tensor across its boundaries with overlap-margin points, giving the model genuine cross-boundary context. Fixed-N budget (no extra VRAM).
- **Spatial Dataset Splitting** — Materialize train/val/test splits with class-stratified, spatially-aware macro-tile partitioning to prevent data leakage from adjacent blocks and mitigate spatial autocorrelation bias.
- **Class-Weighted Loss** — Tunable effective-number class weighting for imbalanced point-cloud classification tasks.
- **Automatic DTM Generation** — When no external DTM is provided, the pipeline auto-generates a bare-earth digital terrain model from the input cloud for accurate height-above-ground computation.
- **Absolute Feature Normalization** — Height and elevation features are normalized against fixed, whole-file references (not per-block statistics), so identical physical heights map to identical feature values regardless of block neighbours.

---

## Quick Start

### Prerequisites

- **Rust toolchain** (edition 2021) — install via [rustup.rs](https://rustup.rs/)
- **Git** (for fetching Whitebox Next Gen dependencies)

### Build

```bash
# Inference only (no GPU, no training)
cargo build --release

# Full build with training support
cargo build --release --features training
```

### Basic Usage

**Preprocess a raw LiDAR file into feature blocks:**

```bash
wb_lidar_classify preprocess \
    --input "area51.las" \
    --output "blocks/area51" \
    --block-size 20.0 \
    --target-points 1024
```

**Classify using a pre-trained model:**

```bash
wb_lidar_classify classify \
    --input "area51.las" \
    --model "urban_model.wbmodel" \
    --blocks "blocks/area51/blocks.json" \
    --output "classified/area51.las"
```

**Train a new model from labeled data:**

```bash
wb_lidar_train preprocess-labeled \
    --input "training_data.las" \
    --output "blocks/training"

wb_lidar_train split-dataset \
    --input "blocks/training" \
    --output "split/merged" \
    --val-split 0.20 --test-split 0.10

wb_lidar_train train \
    --data-dir "split/merged/train" \
    --val-data-dir "split/merged/val" \
    --output-model "models/my_model.wbmodel" \
    --epochs 100
```

**Evaluate a trained model on held-out data:**

```bash
wb_lidar_train evaluate \
    --model "models/my_model.wbmodel" \
    --data-dir "split/merged/test" \
    --metrics-out "eval/per_class.csv" \
    --confusion-out "eval/confusion.csv"
```

---

## Pre-Trained Models & Helper Scripts

### Pre-Trained Models (`models/`)

The `models/` directory is a **versioned, git-tracked library of user-approved pre-trained PointNet weights** (`.wbmodel` files). It is treated as a resource library:

- Only **final, user-approved** models are placed here.
- Approval is signified by **manually copying** the model into `models/` and committing + pushing it — the file's presence in the repository *is* the approval marker.
- There is **no automatic discovery or download**: the CLI never looks in `models/` on its own. You always name the model explicitly with `--model`:

```bash
wb_lidar_classify classify \
    --input area51.las \
    --model models/urban_model.wbmodel \
    --blocks blocks/area51/blocks.json \
    --output classified/area51.las
```

Each approved model is catalogued in [`models/README.md`](models/README.md) (provenance, class count / label map, feature contract, checksum, training summary, approval date).

### Helper Scripts (`scripts/`)

The `scripts/` directory contains **minimal, intentionally "dumb" passthrough wrappers** for the two binaries (`wb_lidar_classify`, `wb_lidar_train`). They perform **no** directory discovery, model lookup, downloads, or file writes — they only forward your command-line arguments verbatim to the binary. The binary must be on your `PATH` (install once with `cargo install --path .`). Full-logic workflow scripts (batch pipelines) live under [`scripts/workflows/`](scripts/workflows/README.md) and call these passthrough wrappers. See [`scripts/README.md`](scripts/README.md).

---

## Architecture

### Pipeline

```
Raw LiDAR (.las/.laz/.copc)
        │
        ▼
┌─────────────────────────────┐
│  Preprocessing              │
│  • Outlier removal (opt.)   │
│  • Eigenvalue pre-pass      │
│  • Auto-DTM generation      │
│  • Block partitioning       │
│  • Feature extraction       │
│  • Halo augmentation (opt.) │
└─────────────┬───────────────┘
              │  .feat blocks
              ▼
┌─────────────────────────────┐
│  Inference (PointNet)       │
│  • Per-block forward pass   │
│  • Prediction fusion (opt.) │
└─────────────┬───────────────┘
              │
              ▼
    Classified LAS/LAZ output
```

### PointNet Model

```
Input: N × 17 features
    │
    ├── T-Net (STN-3d) — 3×3 spatial transform
    │
    ├── Encoder: 17→64→64→64→128→1024
    │   └── Layer 0 output saved as local_feat (N × 64)
    │
    ├── Optional T-Net (STN-64d) — 64×64 feature transform
    │
    ├── Global Max Pool → 1024-dim descriptor
    │
    ├── Concat: local_feat (64) + global (1024) = 1088
    │
    └── Decoder: 1088→512→256→n_classes
```

### File Formats

| Format | Description |
|--------|-------------|
| `.feat` | Per-point feature tensor (17 × f32 per point); v2 format with halo support |
| `.lbl` | Ground-truth class labels (u8 per point), paired with `.feat` |
| `.wbmodel` | Serialized PointNet model weights (JSON header + binary weights) |
| `blocks.json` | Manifest of `.feat` files from unlabeled preprocessing |
| `labeled_blocks.json` | Manifest for labeled preprocessing (includes class distributions, label map) |

---

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:

| Document | Description |
|----------|-------------|
| [`docs/user/user_guide.md`](docs/user/user_guide.md) | Complete user guide with CLI reference, architecture details, and troubleshooting |
| [`docs/stages/`](docs/stages/) | Stage-by-stage specification files detailing every feature and design decision |
| [`PROJECT_SPEC.md`](../PROJECT_SPEC.md) | High-level project specification and architectural pillars |

---

## Project Structure

```
lidar_point_cloud_classifier/
├── src/
│   ├── main.rs                  # wb_lidar_classify entry point
│   ├── bin/wb_lidar_train.rs    # wb_lidar_train entry point
│   ├── cli/                     # CLI argument parsing and sub-commands
│   ├── preprocessing/           # Block partitioning, feature extraction, normalization
│   ├── model/                   # PointNet architecture, inference, weights, fusion
│   ├── training/                # Burn-based training loop, dataset loading, metrics
│   ├── output/                  # LAS/LAZ writer, format guard
│   └── error.rs                 # Error types
├── models/                      # Versioned library of user-approved pre-trained .wbmodel weights
├── scripts/                     # Minimal passthrough CLI wrappers (bash + PowerShell)
├── tests/                       # Integration tests
├── docs/                        # Specifications and user documentation
│   ├── user/user_guide.md       # User manual
│   └── stages/                  # Stage-by-stage design specs
└── Cargo.toml                   # Dependencies and build configuration
```

---

## Dependencies

The project depends on the **Whitebox Next Gen** crates (`wblidar`, `wbraster`, `wbcore`, `wbtools_oss`) via pinned git dependencies — not from crates.io. These provide LAS/LAZ/COPC streaming I/O, eigenvalue feature extraction, DTM generation, and outlier removal.

Key Rust crates:
- **`ndarray`** — N-dimensional array operations for the PointNet forward pass
- **`rayon`** — Data-parallel concurrency for embarrassingly parallel block processing
- **`burn`** (optional) — GPU-accelerated deep learning framework for training
- **`kdtree`** — Spatial indexing for neighbourhood queries
- **`nalgebra`** — Linear algebra for eigenvalue decomposition

---

## Known Limitations

- **LAZ output is disabled by default** — The upstream `wblidar` LASzip encoder produces streams that reference decoders reject mid-chunk. Classified output defaults to uncompressed `.las`. Use `--allow-laz` at your own risk, or compress the result with a reference `laszip` implementation. See [`docs/LAZ_CODEC_DEFECT_REPORT.md`](docs/LAZ_CODEC_DEFECT_REPORT.md) for details.
- **VLR passthrough** — Original file VLRs (including CRS/GeoTIFF) are not currently preserved in classified output. This is a known gap being tracked for a future fix.

---

## License

This project is part of the Whitebox Next Gen ecosystem. See the repository root for license information.

---

## References

- Qi, C. R., et al. (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR.
- Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* Ecography.
- Ploton, P., et al. (2020). *Spatial validation reveals poor predictive performance of large-scale ecological mapping models.* Nature Communications.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press.
- Kernighan, B. W., & Lin, S. (1970). *An efficient heuristic procedure for partitioning graphs.* The Bell System Technical Journal.