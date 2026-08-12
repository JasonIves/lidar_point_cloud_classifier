# Minimal passthrough wrapper for the `wb_lidar_classify` binary.
#
# Intentionally "dumb": performs no directory discovery, model lookup, downloads,
# or file I/O of its own. It forwards your arguments verbatim to the binary and
# propagates the binary's exit code.
#
# The binary must be on your PATH (install once with: cargo install --path .)
#
# Example:
#   .\scripts\wb_lidar_classify.ps1 classify `
#       --input area51.las `
#       --model models\urban_model.wbmodel `
#       --blocks blocks\area51\blocks.json `
#       --output classified\area51.las
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Forward
)
& wb_lidar_classify @Forward
exit $LASTEXITCODE