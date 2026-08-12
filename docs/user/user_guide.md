# LiDAR Point Cloud Classifier — User Guide

**Version:** 0.1.0  
**Project:** `lidar_point_cloud_classifier`  
**Author:** Whitebox Next Gen Ecosystem

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Building](#2-installation--building)
3. [Conceptual Overview: The Block-Based Pipeline](#3-conceptual-overview-the-block-based-pipeline)
4. [Command-Line Interface: `wb_lidar_classify`](#4-command-line-interface-wb_lidar_classify)
   - [4.1 `preprocess` — Spatial Preprocessing](#41-preprocess--spatial-preprocessing)
   - [4.2 `classify` — Inference](#42-classify--inference)
5. [Command-Line Interface: `wb_lidar_train`](#5-command-line-interface-wb_lidar_train)
   - [5.1 `preprocess-labeled` — Labeled Preprocessing](#51-preprocess-labeled--labeled-preprocessing)
   - [5.2 `split-dataset` — Dataset Split Materialization](#52-split-dataset--dataset-split-materialization)
   - [5.3 `train` — PointNet Training](#53-train--pointnet-training)
   - [5.4 `evaluate` — Held-Out Test Evaluation](#54-evaluate--held-out-test-evaluation)
6. [GPU Acceleration & Device Selection](#6-gpu-acceleration--device-selection)
7. [Model Architecture: PointNet](#7-model-architecture-pointnet)
8. [The `.feat` Binary Format](#8-the-feat-binary-format)
9. [Output & Evaluation](#9-output--evaluation)
   - [Output Format Behavior](#output-format-behavior)
   - [Why LAZ Output Is Disabled](#why-laz-output-is-disabled)
10. [Advanced Topics](#10-advanced-topics)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)
12. [Appendix](#12-appendix)

---

## 1. Introduction

The **LiDAR Point Cloud Classifier** is a Rust-based toolchain for classifying airborne LiDAR point clouds using a **PointNet** deep-learning architecture. It is designed as a native plugin for the **Whitebox Next Gen** geospatial ecosystem and leverages Whitebox's LAS/LAZ/COPC streaming I/O, eigenvalue feature extraction, and outlier removal tools.

### Who Is This Tool For?

- **GIS / Remote Sensing Analysts** who need to classify raw LiDAR point clouds by land-cover or object type (ground, vegetation, buildings, water, etc.).
- **Machine Learning Practitioners** who want to train custom PointNet classifiers on their own labeled LiDAR datasets.
- **Researchers** working with 3D point cloud segmentation who need a high-performance, GPU-accelerated training pipeline.

### Two-Binary Architecture

The project ships two CLI binaries:

| Binary | Description | Build Requirement |
|---|---|---|
| `wb_lidar_classify` | Unlabeled preprocessing + inference. Works out of the box with a pre-trained model. | `cargo build --release` |
| `wb_lidar_train` | Labeled preprocessing, dataset splitting, and model training. | `cargo build --release --features training` |

### What is PointNet?

PointNet (Qi et al., 2017) is a deep neural network that consumes raw point clouds directly without voxelization or rasterization. It uses shared multi-layer perceptrons (MLPs) and a symmetric max-pooling function to achieve permutation invariance — the network produces the same classification regardless of the order in which points are fed to it. The LiDAR Point Cloud Classifier implements a PointNet variant optimized for airborne LiDAR data.

### Key Capabilities

- **Streaming Preprocessing:** Process LAS/LAZ/COPC files of any size with a block-based streaming pipeline that keeps memory usage bounded.
- **17-D Feature Vectors:** Each point is represented by 7 scalar features (coordinates, intensity, height) and 10 eigenvalue-derived structural features (linearity, planarity, sphericity, etc.).
- **GPU-Accelerated Training:** Leverages the `burn` deep-learning framework with automatic GPU detection and graceful CPU fallback.
- **Class-Weighted Loss:** Tunable effective-number class weighting to handle imbalanced point-cloud classification tasks.
- **Spatial Dataset Splitting:** Materialize train/val/test splits with class-stratified, spatially-aware macro-tile partitioning.
- **Block Overlap:** Eliminate edge artifacts at block seams during feature extraction.

---

## 2. Installation & Building

### Prerequisites

- **Rust toolchain** (edition 2021). Install via [rustup.rs](https://rustup.rs/).
- **Git** (for fetching Whitebox Next Gen dependencies).

The project has no system-level dependencies — all LiDAR I/O and computation is handled by Rust crates.

### Building from Source

Clone the repository:

```bash
git clone https://github.com/JasonIves/lidar_point_cloud_classifier.git
cd lidar_point_cloud_classifier
```

**Inference-only build** (no GPU support, no training commands):

```bash
cargo build --release
```

This produces `target/release/wb_lidar_classify.exe` (Windows) or `target/release/wb_lidar_classify` (Linux/macOS). The `preprocess` and `classify` sub-commands are available.

**Full build with training support:**

```bash
cargo build --release --features training
```

This produces both binaries:
- `target/release/wb_lidar_classify.exe`
- `target/release/wb_lidar_train.exe`

The `preprocess-labeled`, `split-dataset`, and `train` sub-commands become available in `wb_lidar_train`.
**Optional: put the binaries on your `PATH`** so the helper wrappers in [`scripts/`](../../scripts/README.md) can find them:

```bash
cargo install --path .
```

This installs `wb_lidar_classify` (inference) and, with the `training` feature, `wb_lidar_train` into your Cargo bin directory. The `scripts/` wrappers are intentionally minimal passthroughs — they forward your arguments to the binary and perform no automatic model lookup or file management. You always supply model and data paths explicitly on the command line. Full-logic workflow scripts (batch pipelines) live under [`scripts/workflows/`](../../scripts/workflows/README.md) and call these passthrough wrappers.

### Feature Flags

| Flag | Effect |
|---|---|
| `training` | Enables the GPU-accelerated training framework (`burn` + `wgpu`), the `wb_lidar_train` binary, and all training sub-commands. |
| `gpu` | Alias for `training`. Kept for backward compatibility. |

### Note on Dependencies

The Whitebox Next Gen crates (`wblidar`, `wbraster`, `wbcore`, `wbtools_oss`) are pinned via **git dependency** in `Cargo.toml`, not from crates.io. They are automatically fetched and compiled when you run `cargo build`. This requires a working internet connection for the first build.

---

## 3. Conceptual Overview: The Block-Based Pipeline

Processing an entire LiDAR point cloud as a single monolithic tensor is impractical — files can easily contain tens of millions of points, far exceeding GPU memory limits. The LiDAR Point Cloud Classifier solves this with a **block-based architecture**.

### The Full Lifecycle

```
┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│   Raw LiDAR      │    │  .feat Blocks +    │    │  Classified        │
│   (.las/.laz)    │───▶│  blocks.json       │───▶│  LAS/LAZ           │
│                  │    │  (Preprocessing)   │    │  (Inference)       │
└──────────────────┘    └───────────────────┘    └────────────────────┘
         │                       │                        │
         │ (labeled only)        │                        │
         ▼                       ▼                        ▼
┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│   Labeled LiDAR  │───▶│  .feat + .lbl     │───▶│  Trained           │
│   (.las/.laz)    │    │  Blocks           │    │  .wbmodel          │
│                  │    │  (Training Prep)  │    │  (Training)        │
└──────────────────┘    └───────────────────┘    └────────────────────┘
                                   │
                                   ▼
                          ┌────────────────────┐
                          │  train/ val/ test/  │
                          │  (Dataset Split)   │
                          └────────────────────┘
```

### What Is a "Block"?

A **block** is a fixed-size 2D spatial tile of the LiDAR point cloud. The input file's bounding box is divided into a regular grid (default: 50 × 50 projection units per cell). Each cell becomes a block containing a fixed number of points (default: 1,024), sampled and normalized from the raw points within that cell's footprint.

Blocks serve three purposes:
1. **Memory efficiency:** Only one block's worth of points is in memory at a time during inference.
2. **Fixed tensor size:** The PointNet model expects a fixed-size input tensor (`target_points × 17` features). Blocks with fewer points are zero-padded or jitter-oversampled.
3. **Parallelism:** Blocks are independent and can be processed in parallel across CPU threads or GPU batches.

### File Formats at a Glance

| Format | Description |
|---|---|
| `.feat` | Binary file containing a single block's per-point feature tensor (17 f32 values per point). |
| `.lbl` | Binary file containing ground-truth class labels (one `u8` per point) for a single block. |
| `.wbmodel` | Serialized PointNet model weights (JSON header + binary weight data). |
| `blocks.json` | JSON manifest listing all `.feat` files produced by an unlabeled preprocess run. |
| `labeled_blocks.json` | JSON manifest for labeled preprocessing (includes class distributions, tile grid, label map). |

---

## 4. Command-Line Interface: `wb_lidar_classify`

### Usage

```
wb_lidar_classify <sub-command> [options]
```

### Sub-Commands

| Sub-Command | Description |
|---|---|
| `preprocess` | Stream a LAS/LAZ/COPC file and produce `.feat` block files + `blocks.json` manifest |
| `classify` | Run inference on `.feat` files using a pre-trained model and write classified LAS/LAZ |
| `help` | Show usage information |

---

### 4.1 `preprocess` — Spatial Preprocessing

Transforms a raw LiDAR point cloud into block-based feature files ready for inference.

#### Required Arguments

| Argument | Description |
|---|---|
| `--input <path>` | Path to the input LAS, LAZ, or COPC file |
| `--output <dir>` | Output directory where `.feat` block files and `blocks.json` will be written |

#### Optional Arguments — Block Tiling

| Argument | Default | Description |
|---|---|---|
| `--block-size <f64>` | `50.0` | Edge length of each 2D block cell, in projection units (e.g., metres). |
| `--target-points <uint>` | `1024` | Number of points per block after density-gated sampling and padding. |
| `--min-density <f64>` | `1.0` | Minimum point density (points/m²) required to retain a block. Blocks below this threshold are discarded. |
| `--block-overlap <f64>` | `0.0` | Radius of the border strip collected from neighbouring blocks, in projection units. On its own it only feeds halo sampling and the `classify --fusion-radius` default (no change to `.feat` contents); pair with `--halo-fraction` to write strip points into `.feat` as halo rows. Recommended: `block-size / 4`. Must be < `block-size`. |

#### Optional Arguments — Feature Extraction

| Argument | Default | Description |
|---|---|---|
| `--search-radius <f64>` | `1.0` | Base neighbourhood radius (projection units) for the whole-file eigenvalue feature pre-pass. |
| `--min-neighbors <uint>` | `8` | Minimum number of neighbours for adaptive radius expansion. |
| `--eigen-memory-budget-mb <uint>` | `2048` | Memory budget (in MB) for the whole-file eigenvalue feature pre-pass. When the estimated point-record buffer exceeds this budget, the pre-pass is split into spatial strips. |
| `--z-norm-block-relative` | *(off)* | Opt into the legacy per-block z-normalisation instead of the default whole-file absolute elevation range. Retained for reproducibility and A/B comparison only. |

#### Optional Arguments — Outlier Removal

All outlier arguments are disabled by default (`--outlier-removal` must be explicitly passed to enable).

| Argument | Default | Description |
|---|---|---|
| `--outlier-removal` | *(flag)* | Enable a whole-file outlier removal pre-pass before block partitioning. |
| `--outlier-radius <f64>` | `2.0` | Neighbourhood radius for outlier elevation residual calculation. |
| `--outlier-elev-diff <f64>` | `50.0` | Elevation residual threshold. Points whose Z deviates from the neighbourhood mean/median by more than this value are removed. |
| `--outlier-use-median` | *(flag)* | Use neighbourhood median instead of mean for baseline Z. |

#### Optional Arguments — Jitter Oversampling

| Argument | Default | Description |
|---|---|---|
| `--oversample-jitter <f64>` | `0.0` | Standard deviation (projection units) of per-axis Gaussian jitter applied to padding-only points when a block has fewer than `--target-points` raw points. Offsets are clipped to ±3σ. When 0.0 (default), exact-duplicate padding is used. |

#### Optional Arguments — Height Above Ground (HAG)

| Argument | Default | Description |
|---|---|---|
| `--hag-model <path>` | *(auto-DTM)* | Path to an external DTM raster file. When provided, it takes priority over auto-DTM. When not provided, a bare-earth DTM is auto-generated from the input cloud. |
| `--hag-max <f64>` | `50.0` | Fixed absolute reference height (projection z-units) for normalising raw HAG into `[0,1]`. Raw HAG is clamped to `raw / hag_max`. Must be positive and finite. |
| `--hag-norm-percentile` | *(off)* | Opt into the legacy per-block-99th-percentile HAG normalisation instead of the fixed-absolute-reference default. Ignores `--hag-max`. Retained for reproducibility and A/B comparison only. |
| `--no-auto-dtm` | *(off)* | Disable auto-DTM generation. When set and no `--hag-model` is provided, falls back to the per-block Z-min proxy for ground elevation. |
| `--dtm-resolution <f64>` | `1.0` | Cell size (projection units) for the auto-generated ground DTM raster. Controls the spatial grain of the bare-earth surface. See [Auto-DTM](#auto-dtm) for tuning guidance. |
| `--keep-auto-dtm` | *(off)* | Retain the intermediate `_auto_ground.las` and `_auto_dtm.tif` files after the run instead of deleting them. Useful for inspection and diagnostics. |

#### Optional Arguments — Diagnostics & Performance

| Argument | Default | Description |
|---|---|---|
| `--threads <uint>` | *(system cores)* | Size of the Rayon thread pool for parallel block processing. |
| `--debug-csv` | *(flag)* | Also emit per-block CSV files alongside `.feat` files for debugging. |

#### Outputs

- **`*.feat` files:** One per retained block, named `block_{id:05}.feat`.
- **`blocks.json`:** A JSON manifest containing block metadata (origin, point count, file names, CRS EPSG code, etc.).

#### Example

```bash
wb_lidar_classify preprocess \
    --input "C:/data/lidar/area51.las" \
    --output "C:/data/blocks/area51" \
    --block-size 50.0 \
    --target-points 1024 \
    --block-overlap 25.0 \
    --threads 8
```

---

### 4.2 `classify` — Inference

Runs a pre-trained PointNet model on preprocessed block files and writes a classified LAS/LAZ output.

#### Required Arguments

| Argument | Description |
|---|---|
| `--input <path>` | Path to the original LAS, LAZ, or COPC source file (used for the output file header and point geometry). |
| `--model <path>` | Path to a pre-trained `.wbmodel` weights file. |
| `--blocks <path>` | Path to the `blocks.json` manifest produced by a `preprocess` run on the same input file. |
| `--output <path>` | Path for the classified output file. Use a `.las` extension — a `.laz` or `.copc` extension is **redirected to `.las`** with a warning. See [Why LAZ output is disabled](#why-laz-output-is-disabled). |

#### Using a Pre-Trained Model

Pre-trained weights are distributed as a **versioned, git-tracked resource library** in the `models/` directory of this repository. Only **user-approved** final models live there — approval is signified by manually dropping the `.wbmodel` into `models/` and committing it. The CLI performs **no** automatic model discovery or download; you always pass the model's full path to `--model`:

```bash
wb_lidar_classify classify \
    --input area51.las \
    --model models/urban_model.wbmodel \
    --blocks blocks/area51/blocks.json \
    --output classified/area51.las
```

Each approved model is catalogued in [`models/README.md`](../../models/README.md) (class count / label map, feature contract, provenance, checksum, training summary, approval date). Ensure the model's `n_classes` and expected input features match your data — the model is validated against the manifest at load time.

> [!NOTE]
> `--model` accepts any file path; `models/` is simply the curated, version-controlled home for the approved set.

#### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--threads <uint>` | *(system cores)* | Size of the Rayon thread pool for parallel block inference. |
| `--allow-laz` | *(off)* | Write `.laz` as requested instead of redirecting to `.las`. **Not recommended** — the resulting file is very likely unreadable by other LiDAR software. See [Why LAZ output is disabled](#why-laz-output-is-disabled). |
| `--fusion-radius <f64>` | `block_overlap` from `blocks.json`, else `0` (off) | Cross-block prediction-fusion voting reach in projection units. Adjacent blocks within this distance of a point also vote (confidence-weighted), smoothing block-seam misclassifications ("patchwork quilt" edges). `0` disables fusion. Maximum: `block-size / 2`. See [Prediction Fusion](#prediction-fusion-stage-44). |
| `--fusion-temp <f64>` | `1.0` | Softmax temperature applied per block before voting (`>1` softens each block's class distribution, `<1` sharpens it). |

#### Workflow

1. The model weights are loaded from the `.wbmodel` file.
2. The `blocks.json` manifest is loaded, and the directory containing it is scanned for corresponding `.feat` files.
3. Each block is processed through the PointNet forward pass to produce per-point class predictions.
4. Every original point is labeled by fusing the softmax probability vectors of the block(s) whose footprint covers it (weighted soft voting — with `--fusion-radius 0` this is exactly the legacy nearest-block behaviour).
5. The classified points are written back to the original LAS/LAZ file, preserving all original fields except the classification byte, which is overwritten with the predicted class.

> [!NOTE]
> The `--blocks` argument must point to the `blocks.json` produced by a `preprocess` run on the **same** `--input` file. The `.feat` block files must exist in the same directory as `blocks.json`.

> [!WARNING]
> Compressed **LAZ output is disabled by default**. If you pass `--output …laz`, the tool writes `…las` instead and prints a warning explaining why. Compressed input (`--input …laz`) is unaffected — reading LAZ works correctly. See [Why LAZ output is disabled](#why-laz-output-is-disabled).

#### Example

```bash
wb_lidar_classify classify \
    --input "C:/data/lidar/area51.las" \
    --model "C:/data/models/urban_model.wbmodel" \
    --blocks "C:/data/blocks/area51/blocks.json" \
    --output "C:/data/classified/area51_classified.las" \
    --threads 8
```

---

## 5. Command-Line Interface: `wb_lidar_train`

> [!IMPORTANT]
> The `wb_lidar_train` binary is only available when compiled with the `training` feature:
> ```bash
> cargo build --release --features training
> ```

### Usage

```
wb_lidar_train <sub-command> [options]
```

### Sub-Commands

| Sub-Command | Description |
|---|---|
| `preprocess-labeled` | Preprocess labeled LiDAR ⇒ `.feat` + `.lbl` block pairs |
| `split-dataset` | Materialize a physical train/val/test directory split |
| `train` | Train a PointNet model and produce a `.wbmodel` file |
| `evaluate` | Score a trained model against a held-out labeled dataset |
| `help` | Show usage information |

---

### 5.1 `preprocess-labeled` — Labeled Preprocessing

Similar to `wb_lidar_classify preprocess`, but the input LAS/LAZ file is expected to contain ground-truth classification labels (e.g., from a manually classified reference dataset). The pipeline produces paired `.feat` (features) and `.lbl` (labels) files for each block.

#### Required Arguments

| Argument | Description |
|---|---|
| `--input <path>` | Path to the input LAS, LAZ, or COPC file with ground-truth classification. |
| `--output <dir>` | Output directory where `.feat`, `.lbl`, and `labeled_blocks.json` will be written. |

#### Optional Arguments

All optional arguments from `preprocess` are supported (`--block-size`, `--target-points`, `--min-density`, `--search-radius`, `--min-neighbors`, `--hag-model`, `--threads`, `--debug-csv`, `--outlier-removal`, `--outlier-radius`, `--outlier-elev-diff`, `--outlier-use-median`, `--block-overlap`, `--oversample-jitter`, `--eigen-memory-budget-mb`), plus:

| Argument | Default | Description |
|---|---|---|
| `--label-map <path>` | *(default ASPRS map)* | Path to a JSON file mapping ASPRS classification codes (integers as strings) to zero-based class indices. |
| `--tile-grid <uint>` | `4` | N×N grid of macro-tiles for spatial dataset splitting. The bounding box is divided into an N×N grid, and each block is assigned to a macro-tile. This enables spatially-aware train/val/test splits that prevent data leakage from adjacent blocks. |

#### The Label Map

The label map defines how ASPRS standard classification codes (e.g., 2 = Ground, 6 = Building, 9 = Water) are mapped to the zero-based class indices used by the PointNet model.

The spillover mechanism works as follows:

1. The pipeline looks up **ASPRS code 1** (Unassigned) in the label map. The index it maps to becomes the **unassigned index**.
2. Any point whose ASPRS code is **not found** in the map is assigned to this unassigned index.
3. If ASPRS code 1 is not present in a custom map, the hardcoded fallback unassigned index is **7**.

**Default label map:**

| ASPRS Code | Class Name | Index |
|---|---|---|
| 2 | Ground | 0 |
| 3 | Low Vegetation | 1 |
| 4 | Medium Vegetation | 2 |
| 5 | High Vegetation | 3 |
| 6 | Building | 4 |
| 9 | Water | 5 |
| 7 | Low Point (noise) | 6 |
| 1 | Unassigned | 7 |
| All others | *(fall through)* | 7 |

> [!IMPORTANT]
> The spillover class is **index 7** (the last index), not index 0. The fallback is determined by the mapping of ASPRS code 1 (Unassigned). If you build a custom map and want unrecognised codes to fall into a specific index, assign ASPRS 1 to that index.

**Custom label map JSON format:**

```json
{
  "1": 0,
  "2": 1,
  "3": 2,
  "4": 3,
  "5": 4,
  "6": 5,
  "9": 6,
  "17": 7
}
```

This maps:
- ASPRS 1 (Unassigned) → index 0 — this becomes the spillover class
- ASPRS 2 (Ground) → index 1
- ASPRS 3 (Low Vegetation) → index 2
- ASPRS 4 (Medium Vegetation) → index 3
- ASPRS 5 (High Vegetation) → index 4
- ASPRS 6 (Building) → index 5
- ASPRS 9 (Water) → index 6
- ASPRS 17 (Bridge Deck) → index 7

**Unrecognised codes** such as ASPRS 42 or 44 will fall into **index 0** (because ASPRS 1 maps to 0). You would set `--n-classes 8` to accommodate indices 0–7.

> [!WARNING]
> Custom label maps must use ASPRS classification codes as **string keys** in the JSON file (e.g., `"1"`, `"2"`, `"9"`), not numeric keys. This is because JSON object keys are always strings.

#### Outputs

- **`*.feat` files:** Feature tensors (one per block).
- **`*.lbl` files:** Label tensors — one `u8` per point, aligned with the feature tensor rows.
- **`labeled_blocks.json`:** Extended manifest including class distributions, the label map, and spatial tile grid metadata.

#### Example

```bash
wb_lidar_train preprocess-labeled \
    --input "C:/data/training/block42.las" \
    --output "C:/data/training/blocks/block42" \
    --block-size 50.0 \
    --target-points 1024 \
    --label-map "C:/data/training/my_label_map.json" \
    --tile-grid 4
```

---

### 5.2 `split-dataset` — Dataset Split Materialization

Physically splits one or more `preprocess-labeled` output directories into `train/`, `val/`, and optionally `test/` subdirectories. The split is spatially aware: blocks are assigned to macro-tiles, and entire macro-tiles are allocated to each split. This prevents data leakage where nearly-identical adjacent blocks could appear in both training and validation sets — a standard best practice for geospatial machine learning (Roberts et al., 2017; Ploton et al., 2020). When class stratification is enabled (default), macro-tiles are assigned via a greedy, cost-minimizing bin-packing heuristic that balances per-class proportions across splits while respecting the requested size fractions (Boyd & Vandenberghe, 2004).

#### Required Arguments

| Argument | Description |
|---|---|
| `--input <dir>` | Directory produced by `preprocess-labeled` (must contain `labeled_blocks.json`). **Repeatable** — pass once per source directory to merge them into a single global split. |
| `--input-list <file>` | Text file containing one input directory path per line. Lines starting with `#` and blank lines are ignored. **Repeatable** — multiple files are concatenated in order. May be combined with `--input`. |
| `--output <dir>` | Output directory; `train/`, `val/`, and optionally `test/` subdirectories are created inside it. |

#### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--val-split <f64>` | `0.20` | Fraction of macro-tiles allocated to validation (0.0–1.0). |
| `--test-split <f64>` | `0.0` | Fraction of macro-tiles allocated to test (disabled when 0.0). |
| `--seed <u64>` | `42` | Seed for deterministic assignment and tie-breaking. |
| `--no-stratify-classes` | *(off)* | Disable class-stratified assignment. When enabled (default), macro-tile assignment attempts to balance class distributions across splits. |
| `--move` | *(off)* | Move files instead of copying. Faster for same-volume operations but destructive — the source files are removed. |

#### Multi-Input Merging

Multiple `--input` directories can be merged into a single split. This is useful when your training data spans multiple LAS/LAZ files that were preprocessed independently. The tool:

1. Loads all `labeled_blocks.json` manifests.
2. Validates that their preprocessing parameters are compatible (same `block_size`, `target_points`, etc.).
3. Merges them into a single global macro-tile pool.
4. Assigns macro-tiles to train/val/test with optional stratification.
5. Materializes the split, **renumbering blocks sequentially** to avoid filename collisions.

#### Input List Files

When merging hundreds or thousands of input directories, the OS command-line length limit (approximately 32,767 characters on Windows) may be exceeded. Use `--input-list` to specify one or more text files containing input directory paths:

```
# inputs.txt — comment lines are ignored
C:/data/training/blocks/area1
C:/data/training/blocks/area2
C:/data/training/blocks/area3
```

Combined with `--input`:

```bash
wb_lidar_train split-dataset \
    --input-list "C:/data/inputs_a.txt" \
    --input-list "C:/data/inputs_b.txt" \
    --input "C:/data/training/blocks/special_area" \
    --output "C:/data/split/merged"
```

Entries from `--input-list` files are placed first (in file order), followed by explicit `--input` entries (in flag order).

#### Output Structure

```
output_dir/
  train/
    block_00000.feat
    block_00000.lbl
    block_00001.feat
    block_00001.lbl
    ...
    labeled_blocks.json
  val/
    block_00000.feat
    block_00000.lbl
    ...
    labeled_blocks.json
  test/                          (only if --test-split > 0.0)
    block_00000.feat
    block_00000.lbl
    ...
    labeled_blocks.json
```

Each subset directory contains its own `labeled_blocks.json` manifest scoped to only that subset's blocks.

#### Example

```bash
wb_lidar_train split-dataset \
    --input "C:/data/training/blocks/area1" \
    --input "C:/data/training/blocks/area2" \
    --output "C:/data/split/merged" \
    --val-split 0.20 \
    --test-split 0.10 \
    --seed 42
```

---

### 5.3 `train` — PointNet Training

Runs the PointNet training loop on labeled block data, producing a trained `.wbmodel` file.

#### Required Arguments

| Argument | Description |
|---|---|
| `--data-dir <dir>` | One or more directories from `preprocess-labeled` or the `train/` subdirectory from `split-dataset`. **Repeatable.** |
| `--output-model <path>` | Path for the output `.wbmodel` file. |

#### Optional Arguments — Training Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--n-classes <uint>` | `8` | Number of output classes. Must be ≥ 2 and match the label map's index range. |
| `--epochs <uint>` | `50` | Number of training epochs. |
| `--batch-size <uint>` | `16` | Effective batch size: number of blocks per optimizer step. |
| `--forward-batch-size <uint>` | `8` | Blocks per batched forward pass (micro-batched then accumulated). This is the effective BatchNorm batch size. |
| `--learning-rate <f64>` | `0.001` | Initial AdamW learning rate. |
| `--weight-decay <f32>` | `0.0001` | AdamW weight decay. |
| `--warmup-steps <uint>` | `0` | Number of linear LR warmup steps before cosine annealing starts. 0 = disabled. |
| `--grad-clip-norm <f32>` | *(disabled)* | Per-tensor L2-norm gradient clipping threshold. Helpful for stabilizing training on noisy data. |

#### Optional Arguments — Model Architecture

| Argument | Default | Description |
|---|---|---|
| `--use-feature-tnet` | *(off)* | Enable the STN-64d feature transform network (T-Net after the encoder). Adds parameters and regularization; can improve accuracy on rotationally-variant features. |
| `--no-class-weights` | *(off)* | Disable class-weighted loss. By default, class weights based on the effective number of samples are used. |
| `--class-weight-beta <f64>` | `0.999` | Beta parameter for effective-number class weighting. Range: `[0.0, 1.0)`. `0.0` = uniform weights. Values near `1.0` provide stronger minority-class emphasis. `0.9999` approximates inverse-frequency weighting. |

#### Optional Arguments — Validation

| Argument | Default | Description |
|---|---|---|
| `--val-split <f64>` | `0.20` | Fraction of macro-tiles held out for on-the-fly validation. Only used when `--val-data-dir` is not supplied. |
| `--val-data-dir <dir>` | *(none)* | Pre-split validation directory (repeatable). When supplied, all `--data-dir` directories are used entirely for training, and `--val-data-dir` directories entirely for validation. `--val-split`/`--val-tile-blocks` are ignored (with a warning). |
| `--val-tile-blocks <path>` | *(none)* | Path to a JSON file containing explicit validation block IDs (array of `u64` values). Overrides the automatic macro-tile split. |

#### Optional Arguments — Checkpointing & Model Selection

| Argument | Default | Description |
|---|---|---|
| `--checkpoint-dir <dir>` | *(none)* | Directory to save checkpoint `.wbmodel` files periodically during training. |
| `--checkpoint-every <uint>` | `1` | Save a checkpoint every N epochs. |
| `--keep-best-n <uint>` | `5` | Maximum number of best checkpoints to retain (by validation mIoU). Older checkpoints beyond this limit are pruned. |
| `--swa` | *(off)* | Enable **Stochastic Weight Averaging** (SWA) for improved generalization. SWA averages model weights from the last portion of training. |

#### Optional Arguments — Metrics & Logging

| Argument | Default | Description |
|---|---|---|
| `--metrics-out <path>` | `metrics.csv` in a `metrics/` sibling directory | Path for the per-epoch metrics CSV file. |
| `--seed <uint>` | `42` | Random seed for dataset splitting, shuffling, and initialization. |

#### Optional Arguments — Performance

| Argument | Default | Description |
|---|---|---|
| `--threads <uint>` | *(system cores)* | Rayon thread pool size for data loading and preprocessing. |
| `--device <auto\|cpu\|gpu>` | `auto` | Compute device selection. See [Section 6](#6-gpu-acceleration--device-selection). |
| `--cache-blocks-max-mb <uint>` | *(disabled)* | Enable in-memory block caching, bounded to the specified number of megabytes. Reduces I/O when the dataset fits in RAM. |

#### Optional Arguments — Early Stopping

| Argument | Default | Description |
|---|---|---|
| `--early-stopping-patience <uint>` | *(disabled)* | Stop training after N epochs with no improvement in validation mIoU. |

#### Validation Strategies

The trainer supports two mutually exclusive validation strategies:

**1. On-the-fly macro-tile split (default)**

When only `--data-dir` is provided (no `--val-data-dir`), the trainer loads the data and performs a macro-tile-based split using `--val-split` and optionally `--val-tile-blocks`. This is convenient for quick experiments but the split is re-determined each run.

**2. Pre-split data**

When `--val-data-dir` is provided, the trainer assumes the directories have already been physically split by `wb_lidar_train split-dataset`. All blocks in `--data-dir` go to training, all blocks in `--val-data-dir` go to validation. This guarantees reproducible, reviewable splits and is recommended for production workflows.

#### Metrics Output

Metrics are written to the CSV file specified by `--metrics-out`. Each row contains:

| Column | Description |
|---|---|
| `epoch` | Epoch number (1-indexed) |
| `train_loss` | Cross-entropy loss on the training set |
| `val_loss` | Cross-entropy loss on the validation set |
| `val_mIoU` | Mean Intersection-over-Union on the validation set |
| `val_F1` | Macro-averaged F1 score on the validation set |
| `train_accuracy` | Per-point accuracy on the training set |
| `val_accuracy` | Per-point accuracy on the validation set |
| `class_0_iou` … `class_N_iou` | Per-class IoU scores |
| `confusion_matrix_flat` | Flattened confusion matrix (row-major, one long string) |
| `learning_rate` | Current learning rate |
| `mean_epoch_duration_s` | Mean time per epoch (seconds) |

#### Example

```bash
wb_lidar_train train \
    --data-dir "C:/data/split/merged/train" \
    --val-data-dir "C:/data/split/merged/val" \
    --output-model "C:/data/models/urban_model.wbmodel" \
    --n-classes 8 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --use-feature-tnet \
    --class-weight-beta 0.999 \
    --checkpoint-dir "C:/data/models/checkpoints" \
    --checkpoint-every 10 \
    --swa \
    --device auto
```

### 5.4 `evaluate` — Held-Out Test Evaluation

Measures a trained model's classification performance on a labeled, held-out dataset (e.g., the `test/` split produced by `split-dataset`, or any `preprocess-labeled` output directory the model never saw during training).

Evaluation runs through the **pure-Rust inference engine** (`PointNetClassifier`), the same engine used by `wb_lidar_classify classify`. This ensures the metrics reflect the actually-deployed model's performance.

> [!NOTE]
> The `evaluate` sub-command is CPU-only, pure-Rust, and requires no GPU. Unlike `train`, it does not use the `burn` framework.

#### Required Arguments

| Argument | Description |
|---|---|
| `--model <path>` | Path to a trained `.wbmodel` file to evaluate. |
| `--data-dir <dir>` | One or more labeled data directories from `preprocess-labeled` or the `test/` subdirectory from `split-dataset`. **Repeatable.** |
| `--metrics-out <path>` | Path for the per-class metrics CSV file. |
| `--confusion-out <path>` | Path for the confusion matrix CSV file. |

#### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--n-classes <uint>` | *(auto)* | Optional cross-check against the model and data. If supplied, must agree with both `model.config.n_classes` and `dataset.n_classes()`. |
| `--threads <uint>` | *(system cores)* | Size of the Rayon thread pool for parallel block evaluation. |
| `--fused-eval` | *(off)* | Replicate the deployed cross-block prediction-fusion decision rule (Stage 44) instead of per-block argmax, and additionally print **boundary-band vs. interior** metrics (per-class IoU and accuracy split by distance to the block edge). Use this to quantify block-seam error and the effect of fusion. |
| `--fusion-radius <f64>` | `block-size / 4` | Fusion voting reach in projection units for `--fused-eval`. Maximum: `block-size / 2`. Requires `--fused-eval`. |
| `--fusion-temp <f64>` | `1.0` | Softmax temperature before voting. Requires `--fused-eval`. |

> [!NOTE]
> **`--fused-eval` requires a Stage-47+ manifest.** Fused evaluation needs the
> grid-geometry fields (`grid_cols` / `grid_rows` / `grid_x_min` / `grid_y_min`)
> that `preprocess-labeled` persists from Stage 47 onward. A `labeled_blocks.json`
> created before Stage 47 has no such fields and is **rejected**, as is any
> `split-dataset` output that merged blocks from **multiple `--input` sources**
> (those have no single coherent grid). In both cases evaluation exits with
> `labeled_blocks.json is missing grid_cols/grid_rows — required for --fused-eval`.
> Fix: re-run `preprocess-labeled` to regenerate the manifest; for a merged
> multi-input split, re-run `split-dataset` with a single `--input` (or evaluate
> without `--fused-eval`).

#### Output CSV: `--metrics-out`

One row per class, plus a header:

```
class_idx,asprs_code,tp,fp,tn,fn,precision,recall,f1,iou
```

`asprs_code` is the ASPRS classification code from the model's label map for human readability. Aggregate scores (mean IoU, overall accuracy, macro-F1) are printed to stderr.

#### Output CSV: `--confusion-out`

A confusion matrix written by the existing `write_confusion_matrix_csv` utility: rows = true class (model index), columns = predicted class (model index). Cell values are point counts.

#### Workflow

1. The `.wbmodel` file is loaded via `model::weights::load_model`.
2. The labeled data directory is loaded via `LabeledBlockDataset::load` with `val_split=0.0` so **every** block is evaluated.
3. Model class count (`model.config.n_classes`) is validated against the dataset's class count derived from the manifest label map. A mismatch produces a hard error.
4. Each block is processed in parallel (Rayon): features → `model.forward()` → argmax → class predictions.
5. Predictions and ground-truth labels (from `.lbl` files) are accumulated into a `MetricsAccumulator`.
6. Two CSV files are written; a summary is printed to stderr.

#### Example

```bash
wb_lidar_train evaluate \
    --model "C:/data/models/urban_model.wbmodel" \
    --data-dir "C:/data/split/merged/test" \
    --metrics-out "C:/data/eval/per_class_metrics.csv" \
    --confusion-out "C:/data/eval/confusion_matrix.csv" \
    --threads 8
```

---

## 6. GPU Acceleration & Device Selection

### How GPU Detection Works

When compiled with the `training` feature, the training binary includes `wgpu` for GPU compute via `burn`'s `Wgpu` backend. At runtime, the tool:

1. **Enumerates adapters** using `wgpu::Instance::enumerate_adapters()`.
2. If one or more GPU adapters are found, the `Wgpu` backend is initialised.
3. If no GPU is found, the `NdArray` (CPU) backend is used.

### Device Preference (`--device`)

| Value | Behavior |
|---|---|
| `auto` (default) | Try GPU first; fall back to CPU if GPU is unavailable or fails to initialise. |
| `cpu` | Force CPU (NdArray backend). Skips GPU detection entirely. |
| `gpu` | Require GPU. Errors immediately if no GPU is found or if the binary was compiled without the `training` feature. |

### Building Without GPU Support

If you build `cargo build --release` (without `--features training`):

- `wb_lidar_classify` works normally.
- `wb_lidar_train` is **not compiled**.

### VRAM Considerations

> [!NOTE]
> This section applies to Windows users in particular, where the WDDM driver model silently spills oversubscribed GPU memory into shared system memory rather than raising an error. This causes a severe training slowdown without an obvious crash.

The `--forward-batch-size` parameter controls how many blocks are processed in each batched forward pass. The total points per batch is approximately:

```
forward_batch_size × max_block_size (--target-points, default 1024)
```

**Empirical guidance (8 GB-class GPU, e.g., RTX 2070 SUPER):**

- **81,920 points/batch** (`16 × 5120`): Safe — 7.6 GB VRAM, sustained 55–70% GPU utilization.
- **163,840 points/batch** (`32 × 5120`): Oversubscribed — visible slowdown.

The tool logs an informational warning when your configuration exceeds a conservative threshold (120,000 points per batch). This is purely advisory — training proceeds unmodified.

To reduce VRAM usage:
- Lower `--forward-batch-size`.
- Lower `--target-points` (requires re-running preprocessing).
- Use `--device cpu`.

### GPU Adapter Information

When training on GPU, the tool logs the name, backend, and device type of the first enumerated adapter:

```
[device] GPU adapter: NVIDIA GeForce RTX 2070 SUPER (Vulkan, DiscreteGpu)
```

On multi-adapter systems (e.g., laptops with integrated + discrete GPUs), the logged adapter may not be the one `WgpuDevice::default()` actually binds. The tool notes this in the log.

---

## 7. Model Architecture: PointNet

The LiDAR Point Cloud Classifier implements a PointNet variant specifically designed for airborne LiDAR point cloud classification. The architecture is identical between the training model (`BurnPointNet`) and the inference model (`PointNetClassifier`) — weights trained by one can be directly loaded by the other.

### Architecture Diagram

```
Input Tensor (N × 17)
        │
    ┌───┴───┐
    │ T-Net │  ← input transform (STN-3d, always enabled)
    └───┬───┘
        │
    ┌───────┐
    │ Encoder│  ← layer 0: 64  (saved as local_feat, N × 64)
    └───┬───┘
        │
    ┌───┴───┐
    │ T-Net │  ← feature transform (STN-64d, optional)
    └───┬───┘   (--use-feature-tnet)
        │
    ┌───────┐
    │ Encoder│  ← layers 1-4: 64 → 64 → 128 → 1024
    └───┬───┘
        │
    ┌───────┐
    │Max Pool│  ← symmetric aggregation over point dimension → N × 1024 (broadcast)
    └───┬───┘
        │
    ┌─────────┐
    │  Concat  │  ← local_feat (N × 64) + global (N × 1024) = N × 1088
    └─────┬───┘
          │
    ┌─────────┐
    │ Decoder  │  ← 2 FC layers: 1088 → 512 → 256, then 256 → n_classes (proj)
    └─────┬───┘
          │
     Class Scores (N × n_classes)
```

### Components

#### Input Transform T-Net (STN-3d)

A small sub-network that predicts a 3×3 affine transformation matrix applied to the input point coordinates. This makes the network invariant to rotations, scaling, and shearing of the input point cloud.

- **Always enabled** — there is no CLI flag to disable the input T-Net.
- Predicts a 3×3 matrix, initialized to the identity.

#### Encoder

Five shared MLP layers (applied independently to each point), matching the
canonical PointNet (Qi et al. 2017) segmentation architecture:

1. Layer 0: `n_features_in (17) → 64` — this output is saved as `local_feat`
   and later concatenated with the global descriptor.
2. Layers 1-4: `64 → 64 → 64 → 128 → 1024` — the "deep" branch, ending in the
   1024-dimensional global feature vector.

Each layer uses 1D convolution (conv1d with kernel size 1) to apply the same
MLP to every point independently.

#### Feature Transform T-Net (STN-64d)

An optional sub-network that predicts a 64×64 affine transformation matrix applied to the 64-dimensional feature vectors after the first encoder block.

- **Disabled** by default. Enable with `--use-feature-tnet` during training.
- Adds significant model capacity but also regularization.
- Can improve accuracy on datasets with complex spatial relationships.

#### Max Pooling

A symmetric max-pooling operation across the point dimension. This is the key innovation of PointNet: it aggregates per-point features into a single 1024-dimensional global feature vector, and because max is symmetric, the result is invariant to point order. The global vector is broadcast back to `N × 1024` for the segmentation concat.

#### Segmentation Concat

The 64-dimensional `local_feat` (saved from encoder layer 0, optionally
T-Net-transformed) is concatenated with the broadcast 1024-dimensional global
descriptor, producing an `N × 1088` segmentation context vector. This
local+global concatenation is what gives PointNet its per-point reasoning
ability — the decoder below never sees the deep-encoder output directly.

#### Decoder Head

Two fully connected layers with dropout and batch normalization, followed by
a final class-projection layer:

- 1088 → 512 (ReLU + BN + Dropout 0.3)
- 512 → 256 (ReLU + BN + Dropout 0.3)
- 256 → n_classes (Linear projection to class scores; no BN/ReLU)

The output is a per-point class score tensor of shape `N × n_classes`.

### Model Configuration (`.wbmodel`)

The `.wbmodel` file stores the complete model architecture and weights:

| Field | Type | Description |
|---|---|---|
| `encoder_dims` | `Vec<usize>` | Layer widths of the encoder (canonical default: `[64, 64, 64, 128, 1024]`). |
| `decoder_dims` | `Vec<usize>` | Layer widths of the decoder, before the final class-projection layer (canonical default: `[512, 256]`). Decoder input dim = `encoder_dims[0] + encoder_dims.last()` = `64 + 1024 = 1088`. |
| `n_classes` | `usize` | Number of output classes. |
| `use_input_tnet` | `bool` | Whether the input transform T-Net is enabled. |
| `use_feature_tnet` | `bool` | Whether the feature transform T-Net is enabled. |
| `weights` | *(binary)* | Serialized weight tensors for all layers. |

> [!WARNING]
> The `.wbmodel` file is architecture-specific. A model trained with `--use-feature-tnet` cannot be loaded by an inference pipeline compiled without feature T-Net support, and vice versa.

---

## 8. The `.feat` Binary Format

The `.feat` file is the fundamental block data format used throughout the pipeline. Both preprocessing (unlabeled and labeled) produce `.feat` files, and both inference and training consume them.

### File Layout

The `.feat` format has two versions. **v2** (current, introduced in Stage 45) adds a 4-byte `n_halo` field to the header; **v1** (legacy) is identical except it lacks this field and has `version = 1`. All readers accept both versions (v1 ⇒ `n_halo = 0`).

**v2 header (41 bytes):**

```
┌──────────────────────────────────────────────────────┐
│ Magic Bytes: 'W' 'B' 'F' 'T' (4 bytes, ASCII)      │
├──────────────────────────────────────────────────────┤
│ Version: u8 (currently 2)                           │
├──────────────────────────────────────────────────────┤
│ Number of Points: u32 (little-endian)               │
├──────────────────────────────────────────────────────┤
│ Number of Features: u32 (little-endian, currently 17)│
├──────────────────────────────────────────────────────┤
│ Block ID: u64 (little-endian)                       │
├──────────────────────────────────────────────────────┤
│ Origin X: f64 (little-endian)                       │
├──────────────────────────────────────────────────────┤
│ Origin Y: f64 (little-endian)                       │
├──────────────────────────────────────────────────────┤
│ Halo Rows: u32 (little-endian)                      │
│   → Rows [0 .. n_points − n_halo) are core samples  │
│   → Rows [n_points − n_halo .. n_points) are halo   │
├──────────────────────────────────────────────────────┤
│ Feature Data: f32[points][features] (row-major)     │
│   → Each point has 17 consecutive f32 values        │
│   → Total bytes = points × features × 4            │
└──────────────────────────────────────────────────────┘
```

When `n_halo = 0` (v1 files or v2 files with halo disabled), the payload contains only core samples and the layout is identical to the pre-Stage-45 format.

### Feature Layout (17 Features Per Point)

#### Scalar Features (7 values, indices 0–6)

| Index | Feature | Description |
|---|---|---|
| 0 | x | Normalized X coordinate (relative to block origin) |
| 1 | y | Normalized Y coordinate (relative to block origin) |
| 2 | z | Normalized Z coordinate (relative to block min Z) |
| 3 | intensity | Raw intensity value (normalized to [0, 1]) |
| 4 | intensity_norm | Intensity normalized by block mean and standard deviation |
| 5 | height_above_ground | Z minus terrain elevation (from DTM or block-min-Z proxy) |
| 6 | height_norm | Height above ground normalized by block statistics |

#### Eigenvalue-Derived Features (10 values, indices 7–16)

| Index | Feature | Description |
|---|---|---|
| 7 | lambda_1 | First eigenvalue (largest) |
| 8 | lambda_2 | Second eigenvalue |
| 9 | lambda_3 | Third eigenvalue (smallest) |
| 10 | linearity | (λ₁ − λ₂) / λ₁ |
| 11 | planarity | (λ₂ − λ₃) / λ₁ |
| 12 | sphericity | λ₃ / λ₁ |
| 13 | omnivariance | (λ₁ × λ₂ × λ₃)^(1/3) |
| 14 | eigentropy | -Σ(λᵢ/Σλ × log(λᵢ/Σλ)) |
| 15 | slope | Z-gradient magnitude |
| 16 | residual | Residual from plane fitting |

These eigenvalue features are produced by a single whole-file pre-pass using `wbtools_oss::LidarEigenvalueFeaturesTool`. Prior to Stage 30, the pipeline computed only 5 eigenvalue features locally with multi-scale radii. This was a **breaking change** — all `.feat` files generated before Stage 30 are incompatible and must be regenerated.

### Security Note

The pipeline validates all `.feat` filenames from manifests against path-traversal attacks. Filenames containing `..`, `/`, or `\` are rejected with an error. This guards against maliciously crafted or hand-edited manifest files.

### The `.lbl` File Format

A companion to `.feat` for labeled data. The `.lbl` file is a simple flat array of `u8` values, one per point:

```
┌──────────────────────────────────────────┐
│ Label Data: u8[points] (packed)          │
│   → Total bytes = points × 1            │
└──────────────────────────────────────────┘
```

The label at index `i` corresponds to the point at row `i` in the `.feat` file.

---

## 9. Output & Evaluation

### Classified LAS

The `classify` sub-command produces a LAS file that preserves the original file's:

- Point coordinates (X, Y, Z) — unchanged.
- Intensity, return number, scan angle, etc. — unchanged.
- **Classification byte** — overwritten with the predicted class from the PointNet model.

The output file is always written as a **LAS 1.4** container, regardless of the input's container (LAS, LAZ, or COPC) or its source LAS version — see [Output Format Behavior](#output-format-behavior) below.

> [!NOTE]
> Custom VLRs (variable-length records) from the input file are **not currently copied** to the output. The CRS is carried through the header, but any additional VLRs present in the source are lost. This is a known deficiency, tracked separately from the LAZ issue described below.

### Output Format Behavior

The `classify` sub-command always writes a **LAS 1.4** file, regardless of the input's container (LAS, LAZ, or COPC) or its source LAS version. The output version is set by the writer itself, so older inputs (e.g. LAS 1.2 / 1.3) are automatically promoted to 1.4.

With that fixed version, the writer **mirrors the rest of the record layout from the source header** rather than forcing a canonical profile:

- **Point-data record format (PDRF)** — copied from the input (e.g. a PDRF 3 source stays PDRF 3, just re-versioned to 1.4). LAS 1.4 permits the legacy formats 0–5 as well as 6–8, so the file is valid, but it will not be a "native" 1.4 PDRF 6/7/8 file unless the input was.
- **Scale / offset / extra bytes** — copied from the input.
- **CRS** — re-emitted from the source as a projection VLR.
- **Other VLRs** — not copied (see the note above).

> [!NOTE]
> "Regardless of input" describes the **container and LAS version** only. The exact record layout depends on the source (PDRF is mirrored), so outputs from heterogeneous inputs are not guaranteed byte-identical in point format.

### Why LAZ Output Is Disabled

**Short version:** the LAZ compressor in the Whitebox Next Gen LiDAR library produces files that other LiDAR software cannot read, so this tool writes uncompressed `.las` instead and tells you when it does.

#### What you will see

If you pass a `.laz` (or `.copc`) path to `--output`, the run proceeds but prints a banner like this *before* doing any work:

```text
================================ WARNING ================================
Compressed LAZ output is DISABLED because the LAZ encoder in this build
of Whitebox Next Gen (wblidar) produces files that reference LASzip
decoders (LAStools, laszip, CloudCompare, PDAL) reject mid-stream.
Affected files appear to load but render as a sparse scatter of points;
all coordinates after the failure point are garbage.

  requested: C:\data\classified\area51_classified.laz
  writing:   C:\data\classified\area51_classified.las
...
```

The warning appears before the model loads, so you find out immediately rather than after a long inference run.

#### What the underlying defect looks like

A classified tile written as `.laz` opens in CloudCompare with an error like:

```text
laszip error: reading point 1596 of 4080355 total points
```

CloudCompare then loads the file anyway, but only the points decoded before the failure have valid coordinates. The result renders as a sparse scatter of points floating in space — a "3D night sky" — rather than a terrain surface. The same points written as uncompressed `.las` from the same run open and render correctly.

The defect is in the **compressor**, not in the classification. Your classifications are correct; only the compressed container is unreadable.

#### Reading LAZ is unaffected

This limitation applies **only to writing**. `--input` accepts `.las`, `.laz`, and `.copc` normally — the decompressor is not implicated.

| Operation | LAZ status |
|---|---|
| `preprocess --input …laz` | ✅ Works |
| `classify --input …laz` | ✅ Works |
| `classify --output …laz` | ❌ Redirected to `.las` |

#### If you need a `.laz` deliverable

Write `.las`, then compress it with a reference LASzip implementation. This produces a fully standard, interoperable file:

```bash
# Using LAStools / laszip (https://laszip.org)
laszip -i "C:/data/classified/area51_classified.las"

# Or using PDAL
pdal translate area51_classified.las area51_classified.laz
```

Because LASzip compression is lossless, the round-trip does not alter your classifications.

#### The `--allow-laz` escape hatch

`--allow-laz` restores the old behaviour and writes `.laz` as requested, with a sterner warning. It exists so the defect stays reproducible for whoever investigates it upstream.

> [!WARNING]
> Do not use `--allow-laz` for analysis or delivery. The output is very likely corrupt in a way that no error message will report at write time. `.copc` output is rejected outright even with the flag, because the underlying library has no COPC writer.

#### Status

This is an **upstream defect** in `whitebox_next_gen`, which this project is not permitted to modify. The guard is therefore a mitigation, not a fix, and is intended to be removed once the compressor is repaired.

- Full technical analysis, shareable with the Whitebox Next Gen maintainer: [`LAZ_CODEC_DEFECT_REPORT.md`](../LAZ_CODEC_DEFECT_REPORT.md)
- Design specification for the guard: [`stage-46-laz-output-integrity-guard.md`](../stages/stage-46-laz-output-integrity-guard.md)

### Training Metrics CSV

When training, metrics are written to the file specified by `--metrics-out` (default: `metrics/metrics.csv` relative to the first `--data-dir`). The CSV contains one row per epoch with columns:

- `epoch` — Epoch number (1-indexed).
- `train_loss` — Training cross-entropy loss.
- `val_loss` — Validation cross-entropy loss.
- `val_mIoU` — Validation mean Intersection-over-Union.
- `val_F1` — Validation macro-averaged F1 score.
- `train_accuracy` — Point-wise training accuracy.
- `val_accuracy` — Point-wise validation accuracy.
- `class_0_iou` … `class_N_iou` — Per-class Intersection-over-Union.
- `confusion_matrix_flat` — Row-major confusion matrix as a comma-separated string.
- `learning_rate` — Current learning rate after scheduling.
- `mean_epoch_duration_s` — Mean wall-clock time per epoch.

### Model Checkpoints

When `--checkpoint-dir` is provided, the trainer saves a checkpoint `.wbmodel` file at the interval specified by `--checkpoint-every`. Checkpoints are named:

```
{output_model_stem}_epoch_{epoch:04}_miou_{val_mIoU:.4}.wbmodel
```

The `--keep-best-n` parameter limits how many of the best checkpoints (by validation mIoU) are retained. When this limit is exceeded, the lowest-performing checkpoint is deleted.

### SWA (Stochastic Weight Averaging)

When `--swa` is enabled, the trainer maintains a running average of model weights during the last portion of training. The SWA model often generalizes better than the final model and is saved as a separate `.wbmodel` file with `_swa` appended to the filename.

---

## 10. Advanced Topics

### Block Overlap

When `--block-overlap` is set to a positive value, each block collects a **border strip** — points from adjacent blocks that fall within the overlap radius — written to internal spill files during preprocessing. Two features consume this strip:

- **Halo augmentation (Stage 45):** with `--halo-fraction > 0`, strip points are sampled into the `.feat` payload as halo rows (see next section).
- **Prediction fusion default (Stage 44):** `classify --fusion-radius` defaults to the manifest's `block_overlap` value when no flag is given.

Key facts about block overlap:
- On its own (halo off), the strip does **not** change `.feat` contents — since Stage 30/42, eigenvalue features come from a whole-file pre-pass and the strip is otherwise unused.
- Recommended: `block-size / 4` when pairing with halo. Must be `≥ 0.0` and strictly `< block-size`.

### Prediction Fusion (Stage 44)

Because each block is classified independently, objects that straddle a block boundary (e.g., a building half in one block and half in another) can receive different classifications on either side of the seam — the "patchwork quilt" artifact. **Prediction fusion** smooths these seams by letting adjacent blocks *vote* on points near their borders:

1. During inference, every block retains a full softmax **probability vector** per sampled point (not just the winning class).
2. When the output file is written, each original point is labeled by the weighted mixture of the softmax vectors of **all blocks whose footprint covers it**:
   - **Centrality weight** — a vote from a block is strongest when the point lies deep inside that block and ramps linearly to zero at the fusion radius beyond its edge (continuous blending, no new seams).
   - **Proximity weight** — inverse-square falloff in the distance to the block's nearest sampled point, bounded by a bandwidth σ = `block-size / √target-points` (the characteristic sample spacing), so a sparse block's far-away sample cannot outvote the local one — and a point sitting exactly on a block's own sample cannot dominate the blend.
3. The final class is the argmax of the fused mixture, mapped through the model's label map.

Points deep inside a block are unaffected (single-block decision, identical to legacy behaviour); only points within `--fusion-radius` of a block edge are blended. Fusion requires no retraining and no preprocessing changes.

| Flag | Command | Default | Description |
|---|---|---|---|
| `--fusion-radius <f64>` | `classify` | `block_overlap` from `blocks.json`, else `0` | Voting reach. `0` disables fusion (legacy behaviour). Max: `block-size / 2`. |
| `--fusion-temp <f64>` | `classify`, `evaluate` | `1.0` | Softmax temperature before voting. |
| `--fused-eval` | `evaluate` | *(off)* | Score the test set with the deployed fusion rule and print boundary-band vs. interior metrics. |

> [!NOTE]
> **Is fusion actually active?** `classify` prints exactly one status line on
> every run so you can verify the fusion state at a glance:
>
> - `[classify] prediction fusion: ON (radius=..., temperature=..., source=--fusion-radius flag)`
> - `[classify] prediction fusion: ON (radius=..., temperature=..., source=manifest block_overlap default)`
> - `[classify] prediction fusion: OFF`
>
> The `source=` field tells you *where the radius came from*. Because an omitted
> `--fusion-radius` **defaults to the manifest's `block_overlap` value** when
> that value is `> 0.0`, fusion can be ON even when you never passed the flag —
> it silently falls back to the `block_overlap` you set at
> `preprocess --block-overlap` time. To force fusion off on such a dataset,
> pass `--fusion-radius 0` explicitly and confirm the `OFF` line in the log.

**Measuring the effect:** run `wb_lidar_train evaluate --fused-eval` on a held-out split. The boundary-band vs. interior per-class IoU breakdown shows exactly how much error concentrates at block seams — and how much fusion recovers. Example:

```bash
wb_lidar_train evaluate \
    --model "C:/data/models/urban_model.wbmodel" \
    --data-dir "C:/data/split/merged/test" \
    --metrics-out "C:/data/eval/fused_metrics.csv" \
    --confusion-out "C:/data/eval/fused_confusion.csv" \
    --fused-eval --fusion-radius 12.5
```

### Halo-Augmented Blocks (Stage 45)

Fusion (Stage 44) reconciles *predictions* across block seams, but each block's own PointNet forward pass still sees only its own territory. **Halo augmentation** attacks the root cause: each block's input tensor is extended across its boundaries with **halo rows** — points sampled from the overlap margin — so the model's global max-pool aggregates the full local structure (e.g., the whole roof, not the half that fits in one block), and border-zone points receive genuine predictions in multiple blocks.

Key facts:

- **Fixed-N budget split:** the per-block tensor stays exactly `target-points × 17`. Halo rows are a reallocation *inside* N (`N = core + halo`), so VRAM, batch sizing, and `.feat` sizes are unchanged. Recommended: `--halo-fraction 0.25` with `--block-overlap = block-size / 4`.
- **`.feat` format v2:** header gains one `n_halo` field (41 bytes vs. 37). Payload layout is `[core rows | halo rows]`; v1 files remain readable.
- **Retraining required:** halo rows extend the normalized x/y coordinates slightly outside `[0,1]` — a genuine input-distribution shift, so models trained on pre-Stage-45 data should be retrained (same precedent as earlier feature changes).
- **Split-aware training (mandatory):** halo rows sampled from *neighbouring* blocks' territory carry real labels. If a halo row lies across a train/val/test macro-tile boundary, its loss weight is automatically **0** (context-only), so held-out labels can never leak into training. Same-tile halo rows train at `--halo-loss-weight` (default 1.0).
- **Fusion composes:** `classify` defaults `--fusion-radius` to the manifest's `block_overlap` (the halo reach) automatically — preprocessing with overlap + halo gives you both mechanisms with no extra flags.

| Flag | Command | Default | Description |
|---|---|---|---|
| `--halo-fraction <f64>` | `preprocess`, `preprocess-labeled` | `0.0` (off) | Fraction of each block's rows reserved for halo samples (0–0.5). Requires `--block-overlap > 0`. Recommended: `0.25`. |
| `--halo-loss-weight <f32>` | `train` | `1.0` | Loss weight for same-tile halo rows. `0.0` masks all halo rows from the loss (context-only). |

### Jitter-Based Oversampling

When a block has fewer points than `--target-points`, the pipeline pads the block with copies of existing points. By default, these are exact duplicates. When `--oversample-jitter` is set to a positive value, the duplicate points' (x, y, z) coordinates are perturbed by Gaussian noise with the specified standard deviation:

```
x_jittered = x + N(0, σ)  (clipped to ±3σ)
y_jittered = y + N(0, σ)  (clipped to ±3σ)
z_jittered = z + N(0, σ)  (clipped to ±3σ)
```

This produces distinct eigenvalue features for the padded points, which can improve model generalization on sparse blocks. The jitter is applied in projection units (typically metres).

### Dataset Merging

Multiple LiDAR files covering different areas can be combined into a single training dataset:

1. Run `preprocess-labeled` on each file independently.
2. Run `split-dataset` with multiple `--input` directories (or an `--input-list` file).
3. The tool merges all manifests, validates compatibility, performs a single global split, and renumbers blocks to prevent filename collisions.

### Class-Stratified Splitting

When `--no-stratify-classes` is **not** set (the default), the `split-dataset` command uses class-stratified assignment: macro-tiles are allocated to train/val/test in a way that balances the class distributions across all splits. This is particularly important for datasets with rare classes (e.g., water, bridge decks) that might otherwise be entirely absent from the validation set.

### Auto-DTM (Automatic Ground DTM Generation)

When no `--hag-model` is supplied, the preprocessing pipeline automatically generates a bare-earth DTM from the input point cloud. This replaces the historical block-minimum-Z proxy for ground elevation, which suffered from slope bias, block-boundary discontinuities, and empty-ground-block artefacts.

The auto-DTM pipeline uses two Whitebox tools in sequence:

1. **`improved_ground_point_filter`** — a multi-stage ground classifier (percentile filter → TIN grid → pit fill → off-terrain removal → reference-surface filter) that extracts bare-earth returns, explicitly removing buildings and large flat roofs.
2. **`lidar_tin_gridding`** — Delaunay triangulation-based linear interpolation of the filtered ground points into a continuous raster surface. Produces nodata cells over large areas without ground returns (safe worst-case: falls back to the block-min-Z proxy per `NodataPolicy::Strict`).

**Priority order:** `--hag-model` (external DTM) → auto-DTM → block-min-Z proxy (legacy fallback). If an external DTM is supplied, auto-DTM is skipped entirely.

#### Choosing `--dtm-resolution`

`--dtm-resolution` sets the cell size of the auto-generated DTM raster (in projection units — e.g., metres for a metric CRS). It controls the spatial grain of the bare-earth surface against which every point's HAG is measured.

| Scenario | Suggested Resolution | Rationale |
|---|---|---|
| Typical airborne (5–20 pts/m²) | `1.0` (default) | Matches ground spacing; balanced |
| High-density / drone (>50 pts/m²) | `0.3–0.5` | Data supports finer relief |
| Sparse / older ALS (1–3 pts/m²) | `2.0–3.0` | Avoids nodata holes |
| Flat terrain (floodplain, playa) | `2.0–5.0` | Little relief to resolve; faster |
| Steep / dissected terrain | `1.0` (don't go coarser) | Coarse cells smooth away needed slope |

**Diagnosing a bad choice:** Run with `--keep-auto-dtm` and inspect `_auto_dtm.tif`. Speckled/nodata holes → too fine → increase resolution. Flat terraced surface → too coarse → decrease resolution.

#### Intermediate Files

Auto-generated intermediates are deleted after the run unless `--keep-auto-dtm` is passed:
- `_auto_ground.las` — ground-only point cloud from the filter stage.
- `_auto_dtm.tif` — bare-earth DTM raster.

#### Breaking Change

Auto-DTM being the default changes the HAG column values compared to the legacy block-min-Z proxy. Any model trained against pre-Stage-38 features must be retrained with features produced using auto-DTM.

### Block Caching

During training, each epoch reloads blocks from disk. When `--cache-blocks-max-mb` is set, the trainer maintains an in-memory LRU cache of recently loaded blocks. This can dramatically reduce I/O when:

- The dataset fits within the cache budget.
- The dataset is large enough that loading dominates training time.

The cache is bounded by the specified megabyte budget. When the budget is exhausted, the least-recently-used block is evicted.

---

## 11. Troubleshooting & FAQ

### Common Errors

| Error Message | Likely Cause | Solution |
|---|---|---|
| `classify: --input is required` | Missing required argument | Check that all required arguments are provided. |
| `Model/config mismatch` | `.wbmodel` file was trained with different architecture (e.g., T-Net enabled/disabled mismatch) | Retrain the model with matching configuration, or use the correct `.wbmodel` file. |
| `manifest file name '...' is not a valid bare file name` | Hand-edited or corrupted manifest contains a path traversal sequence | Regenerate the manifest with the preprocessing pipeline. |
| `flag '--input' requires a value` | Flag was specified without a value | Provide a value after the flag. |
| `--block-overlap must be less than --block-size` | Overlap value is too large | Reduce `--block-overlap` to strictly less than `--block-size`. |
| `--device gpu was requested but no GPU adapter was found` | No compatible GPU or the `training` feature was not compiled | Use `--device auto` for CPU fallback, or rebuild with `--features training`. |
| `labeled_blocks.json is missing grid_cols/grid_rows — required for --fused-eval` | Manifest predates Stage 47, or is a `split-dataset` output that merged blocks from multiple `--input` sources (no single coherent grid) | Re-run `preprocess-labeled` to regenerate the manifest; for a merged multi-input split, re-run `split-dataset` with a single `--input`, or evaluate without `--fused-eval`. |
| `evaluate --fused-eval` reports all-zero metrics | Stale (pre-Stage-47) labeled manifest without persisted grid geometry, so block votes never resolve | Regenerate the labeled manifest (`preprocess-labeled`), then re-run the evaluation. |

### GPU Training Crashes / Fallback Behavior

When using `--device auto`, if GPU training panics (e.g., due to a driver error, VRAM exhaustion, or an unsupported GPU feature), the trainer catches the panic and automatically falls back to CPU training. A message is logged:

```
[device] GPU training failed: ...
[device] Falling back to CPU NdArray backend (--device auto).
```

If `--device gpu` was explicitly set, a panic results in an error rather than a silent fallback.

### Out-of-Memory Issues

**During preprocessing:** Reduce `--eigen-memory-budget-mb` to force the eigenvalue pre-pass to use smaller spatial strips, consuming less peak memory.

**During training on GPU:** Reduce `--forward-batch-size` or `--target-points`. See [Section 6](#vram-considerations) for guidance.

**During training on CPU:** Reduce `--batch-size` or `--cache-blocks-max-mb`.

### Performance Tuning Tips

- **Thread count:** Set `--threads` to match your CPU's physical core count (not logical threads). Oversubscribing can reduce performance.
- **Block caching:** If the training dataset fits in system RAM, enable `--cache-blocks-max-mb` with a generous budget (e.g., 80% of available RAM) to eliminate disk I/O.
- **Forward batch size:** On GPU, a larger `--forward-batch-size` improves GPU utilization but increases VRAM usage. Monitor GPU utilization with Task Manager or `nvidia-smi`.
- **Preprocessing I/O:** Preprocessing is primarily I/O-bound. An SSD provides significant speedup over an HDD.

### File Format Compatibility

> [!WARNING]
> The `.feat` file format changed in Stage 30. Previously, files contained 12 features per point (7 scalar + 5 eigenvalue features). The current format contains 17 features per point (7 scalar + 10 eigenvalue features). Any `.feat` files or `.wbmodel` files generated before Stage 30 **must be regenerated**.

> [!WARNING]
> **Stage 37 (HAG normalisation) and Stage 38 (auto-DTM) both changed HAG column semantics.** The HAG feature (index 5) and normalised HAG (index 6) values differ from earlier runs. Any model trained against pre-Stage-37 or pre-Stage-38 features must be retrained with features produced under the current default settings.

> [!WARNING]
> **LAZ output is disabled (Stage 46).** `classify --output …laz` writes `…las` instead, because the underlying Whitebox Next Gen LAZ compressor produces files that reference LASzip decoders (LAStools, CloudCompare, PDAL) cannot read. LAZ **input** is unaffected. See [Why LAZ Output Is Disabled](#why-laz-output-is-disabled).

### "My classified LAZ file renders as a sparse scatter of points"

If you have an older `.laz` output from this tool that opens with a `laszip error: reading point N of M` message and renders as scattered points floating in empty space, that file is corrupt — see [Why LAZ Output Is Disabled](#why-laz-output-is-disabled) for the cause. Re-run the classification with a `.las` output path; the guard now prevents this by default.

---

## 12. Appendix

### A. Quick Reference: All CLI Flags

#### `wb_lidar_classify preprocess`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--input` | Path | — | ✓ |
| `--output` | Path | — | ✓ |
| `--block-size` | f64 | 50.0 | |
| `--target-points` | usize | 1024 | |
| `--min-density` | f64 | 1.0 | |
| `--search-radius` | f64 | 1.0 | |
| `--min-neighbors` | usize | 8 | |
| `--hag-model` | Path | None | |
| `--hag-max` | f64 | 50.0 | |
| `--hag-norm-percentile` | bool flag | false | |
| `--no-auto-dtm` | bool flag | false | |
| `--dtm-resolution` | f64 | 1.0 | |
| `--keep-auto-dtm` | bool flag | false | |
| `--threads` | usize | System cores | |
| `--debug-csv` | bool flag | false | |
| `--block-overlap` | f64 | 0.0 | |
| `--halo-fraction` | f64 | 0.0 | |
| `--outlier-removal` | bool flag | false | |
| `--outlier-radius` | f64 | 2.0 | |
| `--outlier-elev-diff` | f64 | 50.0 | |
| `--outlier-use-median` | bool flag | false | |
| `--oversample-jitter` | f64 | 0.0 | |
| `--eigen-memory-budget-mb` | usize | 2048 | |
| `--z-norm-block-relative` | bool flag | false | |

#### `wb_lidar_classify classify`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--input` | Path | — | ✓ |
| `--model` | Path | — | ✓ |
| `--blocks` | Path | — | ✓ |
| `--output` | Path | — | ✓ |
| `--threads` | usize | System cores | |
| `--allow-laz` | bool flag | false | |
| `--fusion-radius` | f64 | `block_overlap`, else `0` | |
| `--fusion-temp` | f64 | 1.0 | |

#### `wb_lidar_train preprocess-labeled`

All `preprocess` flags plus:

| Flag | Type | Default | Required |
|---|---|---|---|
| `--label-map` | Path | *(default ASPRS map)* | |
| `--tile-grid` | usize | 4 | |

#### `wb_lidar_train split-dataset`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--input` | Path (repeatable) | — | *(at least one input)* |
| `--input-list` | Path (repeatable) | — | |
| `--output` | Path | — | ✓ |
| `--val-split` | f64 | 0.20 | |
| `--test-split` | f64 | 0.0 | |
| `--seed` | u64 | 42 | |
| `--no-stratify-classes` | bool flag | false | |
| `--move` | bool flag | false | |

#### `wb_lidar_train train`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--data-dir` | Path (repeatable) | — | ✓ |
| `--output-model` | Path | — | ✓ |
| `--val-data-dir` | Path (repeatable) | None | |
| `--n-classes` | usize | 8 | |
| `--epochs` | usize | 50 | |
| `--batch-size` | usize | 16 | |
| `--forward-batch-size` | usize | 8 | |
| `--learning-rate` | f64 | 0.001 | |
| `--weight-decay` | f32 | 0.0001 | |
| `--val-split` | f64 | 0.20 | |
| `--val-tile-blocks` | Path | None | |
| `--seed` | u64 | 42 | |
| `--use-feature-tnet` | bool flag | false | |
| `--no-class-weights` | bool flag | false | |
| `--class-weight-beta` | f64 | 0.999 | |
| `--checkpoint-dir` | Path | None | |
| `--checkpoint-every` | usize | 1 | |
| `--keep-best-n` | usize | 5 | |
| `--swa` | bool flag | false | |
| `--metrics-out` | Path | `metrics/metrics.csv` | |
| `--threads` | usize | System cores | |
| `--device` | string | `auto` | |
| `--early-stopping-patience` | usize | None | |
| `--warmup-steps` | usize | 0 | |
| `--grad-clip-norm` | f32 | None | |
| `--cache-blocks-max-mb` | usize | None | |
| `--halo-loss-weight` | f32 | 1.0 | |

#### `wb_lidar_train evaluate`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--model` | Path | — | ✓ |
| `--data-dir` | Path (repeatable) | — | ✓ |
| `--metrics-out` | Path | — | ✓ |
| `--confusion-out` | Path | — | ✓ |
| `--n-classes` | usize | *(auto)* | |
| `--threads` | usize | System cores | |
| `--fused-eval` | bool flag | false | |
| `--fusion-radius` | f64 | `block-size / 4` | |
| `--fusion-temp` | f64 | 1.0 | |

### B. ASPRS LAS Classification Standard (Common Codes)

| Code | Meaning |
|---|---|
| 0 | Never Classified |
| 1 | Unassigned |
| 2 | Ground |
| 3 | Low Vegetation |
| 4 | Medium Vegetation |
| 5 | High Vegetation |
| 6 | Building |
| 7 | Low Point (noise) |
| 8 | Model Key Point |
| 9 | Water |
| 10 | Rail |
| 11 | Road Surface |
| 12 | Wire Guard |
| 13 | Wire Conductor |
| 14 | Transmission Tower |
| 15 | Wire Structure Connector |
| 16 | Bridge Deck |
| 17 | High Noise |

### C. Glossary

| Term | Definition |
|---|---|
| **ASPRS** | American Society for Photogrammetry and Remote Sensing; defines the standard LAS classification codes. |
| **Block** | A fixed-size 2D spatial tile of the LiDAR point cloud, containing a fixed number of points. |
| **COPC** | Cloud Optimized Point Cloud; a LAS-like format with spatial indexing for efficient cloud access. |
| **DTM** | Digital Terrain Model; a raster representing bare-earth elevation. |
| **Eigenvalue Features** | Structural features derived from the eigenvalue decomposition of the 3D covariance matrix of a point's neighbourhood (linearity, planarity, sphericity, etc.). |
| **HAG / Height Above Ground** | The height of a point relative to the underlying terrain surface. |
| **LAS / LAZ** | Standard file formats for storing LiDAR point cloud data (LAZ is the compressed version). |
| **Macro-Tile** | A coarse N×N spatial grid used for dataset splitting. |
| **mIoU** | Mean Intersection-over-Union; a common metric for semantic segmentation tasks. |
| **PointNet** | A deep neural network architecture that consumes raw point clouds directly. |
| **T-Net (Transform Net)** | A sub-network that predicts an affine transformation matrix for spatial or feature alignment. |
| **WDDM** | Windows Display Driver Model; the graphics driver framework on Windows, which silently spills oversubscribed GPU memory to system RAM. |
| **Whitebox Next Gen** | A geospatial analysis platform and Rust crate ecosystem; the parent project of this LiDAR classifier. |
| **`.wbmodel`** | Serialized PointNet model weights file format. |

### D. File Format Summary

| Format | Extension | Producer | Consumer | Content |
|---|---|---|---|---|
| Feature block | `.feat` | `preprocess` / `preprocess-labeled` | `classify` / `train` | Per-point feature tensor (17 × f32 per point); v2 format (41-byte header with `n_halo` field) since Stage 45, v1 backward-compatible |
| Label block | `.lbl` | `preprocess-labeled` | `train` | Per-point class labels (u8 per point) |
| Block manifest | `blocks.json` | `preprocess` | `classify` | Block metadata (origin, point count, CRS) |
| Labeled manifest | `labeled_blocks.json` | `preprocess-labeled` / `split-dataset` | `train` | Block metadata + class distributions + label map |
| Model weights | `.wbmodel` | `train` | `classify` / `train` (resume) | Network config + serialized weight tensors |
| Training metrics | `metrics.csv` | `train` | *(analysis)* | Per-epoch loss, IoU, accuracy, confusion matrix |
| Evaluation per-class metrics | *(user-specified)* | `evaluate` | *(analysis)* | Per-class TP/FP/TN/FN, precision, recall, F1, IoU |
| Evaluation confusion matrix | *(user-specified)* | `evaluate` | *(analysis)* | Row-major confusion matrix (true × predicted class) |
| Validation tile blocks | *(JSON array)* | *(manual or script)* | `train` | Explicit list of `u64` block IDs for validation |

---

> **Document Version:** 1.6  
> **Last Updated:** 2026-08-08  
> **Corresponding Code Revision:** Stages 47 & 48 (fused-eval grid-geometry fix; classify fusion-status logging clarity)
