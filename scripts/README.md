# Helper Scripts

This directory contains two tiers of helper scripts for the LiDAR Point Cloud
Classifier:

- **Passthrough wrappers** (the `.sh` / `.ps1` files at this level) — minimal,
  intentionally "dumb" wrappers that forward your arguments verbatim to a binary.
- **Workflow scripts** (in `workflows/`) — full-logic orchestration scripts that
  call the tools (see [Workflow scripts (`workflows/`)](#workflow-scripts-workflows)).

## What these scripts do (and do not do)

- **They only forward arguments.** Each wrapper passes your command-line arguments
  verbatim to its binary and propagates the exit code.
- **No automation.** They perform no directory discovery, no model lookup, no
  downloads, no environment probing, and no file reads or writes of their own.
- **You supply everything.** Model paths, input/output paths, and all other options
  are passed through to the CLI exactly as you write them.

## Prerequisite

The target binary must be on your `PATH`. Install it once:

```bash
cargo install --path .
```

This installs `wb_lidar_classify` (inference) and, with the `training` feature,
`wb_lidar_train`. Alternatively, call the wrapper with the binary's directory
prepended to `PATH`, or edit the wrapper to name an absolute binary path.

## The wrappers

| Script (bash) | Script (PowerShell) | Binary it calls |
|---|---|---|
| `wb_lidar_classify.sh` | `wb_lidar_classify.ps1` | `wb_lidar_classify` |
| `wb_lidar_train.sh` | `wb_lidar_train.ps1` | `wb_lidar_train` |

Make the bash scripts executable once if needed:

```bash
chmod +x scripts/*.sh
```

## Examples

```bash
# Classify (Unix / Git Bash)
./scripts/wb_lidar_classify.sh classify \
    --input area51.las \
    --model models/urban_model.wbmodel \
    --blocks blocks/area51/blocks.json \
    --output classified/area51.las
```

```powershell
# Classify (PowerShell on Windows)
.\scripts\wb_lidar_classify.ps1 classify`
    --input area51.las`
    --model models\urban_model.wbmodel`
    --blocks blocks\area51\blocks.json`
    --output classified\area51.las
```

The `wb_lidar_train.*` wrappers work the same way for `preprocess-labeled`,
`split-dataset`, `train`, `evaluate`, and `fix-label-map`.

## Workflow scripts (`workflows/`)

The [`workflows/`](workflows/README.md) subdirectory contains **full-logic
orchestration scripts** — the "smart" layer of this directory. These are
batch pipelines (e.g. classify a set of tiles, or drive a labeled
preprocess → split → train → evaluate flow) that loop over inputs, validate
parameters, and build argument lists.

The integration rule for workflow scripts:

- Workflows call the **passthrough wrapper** as their single execution entry
  point (not the binary directly, not a reimplementation of tool logic).
- From a PowerShell workflow in `workflows/`, reach the wrapper via
  `$PSScriptRoot`:

  ```powershell
  $classify = Join-Path $PSScriptRoot '..\wb_lidar_classify.ps1'
  & $classify classify --input $tile --model $model --blocks $blocks --output $out
  ```

- Workflows may be "smart" (validation, looping, directory handling) but must
  still pass every model and data path to the CLI explicitly.
- The Rust CLI remains the only component that reads/writes files and does the
  actual work; the layers stay clean: workflow (decisions) → wrapper
  (invocation) → binary (work).

## Notes

- The bash wrappers run on Linux, macOS, and Windows (under Git Bash / WSL). The
  PowerShell wrappers run natively on Windows, keeping the tooling platform-agnostic.
- Because these wrappers are trivial passthroughs, they are purely optional — the
  binaries can always be invoked directly.