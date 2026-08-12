#!/usr/bin/env bash
# Minimal passthrough wrapper for the `wb_lidar_train` binary.
#
# Intentionally "dumb": performs no directory discovery, model lookup, downloads,
# or file I/O of its own. It forwards your arguments verbatim to the binary and
# exits with the binary's status.
#
# The binary must be on your PATH and built with the `training` feature
# (install once with: cargo install --path . --features training)
#
# Example:
#   ./scripts/wb_lidar_train.sh train \
#       --data-dir split/merged/train \
#       --val-data-dir split/merged/val \
#       --output-model models/my_model.wbmodel \
#       --epochs 100
set -euo pipefail
exec wb_lidar_train "$@"