# Workflow Scripts

This directory holds **full-logic PowerShell (and optional bash) workflow scripts**
for the LiDAR Point Cloud Classifier — the "smart" layer that sits on top of the
minimal passthrough wrappers in the parent `scripts/` directory.

## What belongs here

- Multi-step pipelines (e.g. classify a batch of tiles, or a labeled
  `preprocess-labeled` → `split-dataset` → `train` → `evaluate` run).
- Scripts that loop over inputs, validate parameters, build argument lists, or
  otherwise orchestrate several invocations of the CLI.

## Integration rule

Workflows call the **passthrough wrapper** in the parent directory as their single
execution entry point, rather than invoking the binary directly or reimplementing
any tool logic. From a workflow in `scripts/workflows/`, reach it via `$PSScriptRoot`:

```powershell
# The passthrough wrapper handles the single binary invocation:
$classify = Join-Path $PSScriptRoot '..\wb_lidar_classify.ps1'
& $classify classify --input $tile --model $model --blocks $blocks --output $out
```

Keep the layers separate:

- **Workflow** — owns orchestration and decisions (loops, validation, arg lists).
- **Passthrough wrapper** — owns the single binary invocation.
- **Rust CLI** — the only thing that reads/writes files and does the actual work.

## Notes

- Workflows **may** be "smart" (that is their purpose) — unlike the passthrough
  wrappers, they are allowed to do parameter validation, directory handling, and
  looping. They must still pass every model and data path to the CLI explicitly.
- The passthrough wrapper assumes the binary is on `PATH`
  (`cargo install --path .`); workflows inherit that requirement.
- Scripts added here are version-controlled and should be platform-appropriate:
  PowerShell for Windows; where a workflow must run cross-platform, add a matching
  `.sh` twin for Linux/macOS.
