# Pre-Trained Model Library

This directory is a **versioned, git-tracked library of user-approved pre-trained
PointNet weights** (`.wbmodel` files) for the LiDAR Point Cloud Classifier. It is
intended to be treated like a resource library: the models committed here are the
curated, trusted set that the project ships.

## How models get here (approval workflow)

- Only **final, user-approved** models are placed in this directory.
- Approval is signified by **manually copying** the approved `.wbmodel` into
  `models/` and committing + pushing it. A model's presence in the repository
  *is* the approval marker — nothing more is required.
- In-development, scratch, or intermediate training checkpoints should **not** be
  committed here. Keep those out of version control (e.g. in a local `data/` or a
  scratch directory) until they are promoted to an approved final model.

## How models are used

There is **no automatic model discovery or download**. The CLI never looks into
`models/` on its own — you always name the model explicitly via `--model`:

```bash
wb_lidar_classify classify \
    --input area51.las \
    --model models/urban_model.wbmodel \
    --blocks blocks/area51/blocks.json \
    --output classified/area51.las
```

The `--model` argument accepts any file path; `models/` is simply the
version-controlled home for the approved set.

## Model catalog

Each approved model should be added to the table below. Before committing a model,
record its SHA-256 checksum and the training/label information needed for a user to
invoke it correctly.

| File | Classes (n) / label map | Feature contract | Provenance / training summary | SHA-256 | Approved | Notes |
|------|-------------------------|------------------|-------------------------------|---------|----------|-------|
| `5c3ffffe-8947-43c4-9080-7a9148a66806.wbmodel` | 8 — index→ASPRS: 0→1 (Unassigned), 1→2 (Ground), 2→3 (Low Vegetation), 3→4 (Medium Vegetation), 4→5 (High Vegetation), 5→6 (Building), 6→9 (Water), 7→17 (Bridge deck) | 17-D (7 scalar + 10 eigen) | Trained 2026-07-31 on 12 manually sampled IGN HD tiles (block size 15 m, target points 16, halo fraction 0.48); see [training script](../scripts/workflows/5c3ffffe-8947-43c4-9080-7a9148a66806_train.ps1). Header: WBML v1, encoder (64,64,64,128,1024), decoder (512,256), batch norm + input T-Net + feature T-Net | `DF73AE3FAC91523D15EBBF646F8F906AE1C01CEADB97F47CBB184A80AC71C3DF` | 2026-08-12 | Pair with the [classify script](../scripts/workflows/5c3ffffe-8947-43c4-9080-7a9148a66806_classify.ps1) for inference |

> [!IMPORTANT]
> A user must know a model's `n_classes` and expected input features before running
> inference. Fill in the row above (or the deployment notes) whenever a model is
> added, and keep it in sync with the model's actual `.wbmodel` header.