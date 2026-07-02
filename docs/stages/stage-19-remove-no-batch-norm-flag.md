# Stage 19 — Remove the Misleading `--no-batch-norm` CLI Flag

## Status: CLOSED — `--no-batch-norm` removed; build/tests/clippy/fmt clean

## The Goal

Remove the `--no-batch-norm` training flag. In its current form it is a
**foot-gun**: it does not disable BatchNorm during training, yet it silently
strips BatchNorm parameters from the serialized `.wbmodel`, producing a
train-with-BN / deploy-without-BN mismatch. Since BatchNorm is load-bearing for
this architecture (Stages 16–18 exist specifically to make its train/eval
statistics behave), there is no legitimate use for turning it off. Per the
AGENTS.md lean / minimal-surface philosophy, the correct action is to delete the
flag rather than invest in gating BatchNorm throughout the burn forward for a
capability nobody needs.

## Background — why the flag is broken

- `BurnPointNet::forward` / `forward_batched` **always** apply BatchNorm
  (`apply_bn2d` / `apply_bn3d`); neither checks `use_batch_norm`. There is no
  gating field on `BurnPointNet`.
- The flag is only consulted by the weight bridge (`save_model_from_burn` →
  `extract_pair` / `extract_tnet*`), which **drops** the BN parameters from the
  `.wbmodel` when `use_batch_norm == false`.
- Deployed ndarray inference (`model/pointnet.rs`) then runs without BatchNorm,
  so the deployed model no longer matches what was trained. The result is a
  silent accuracy regression with no error surfaced to the user.

## Scope & Decisions

- **Remove** the `--no-batch-norm` argument branch and its help-text line from
  `cli/train_cmd.rs`.
- **Keep** `TrainConfig.use_batch_norm` and `PointNetConfig.use_batch_norm`
  fields, pinned to their default `true`. Rationale: they are threaded through
  the bridge, `weights.rs` (de)serialization, and numerous tests; ripping them
  out is a large, risky change for no functional benefit. With the CLI flag gone
  they can only ever be `true` from the training path.
- **Preserve the `.wbmodel` binary format unchanged.** The `use_batch_norm` byte
  stays in the header (always written as `1` from the training path). The
  **reader keeps honoring a `0` byte** so any pre-existing `--no-batch-norm`
  models still load correctly — backward compatibility is maintained.
- No changes to the burn forward pass, the bridge, or `weights.rs` logic.

## Inputs & Outputs

- **Inputs:** CLI surface loses one flag; `--no-batch-norm` becomes an unknown
  flag (rejected with the standard `train: unknown flag '...'` error).
- **Outputs:** `.wbmodel` and `metrics.csv` formats unchanged. Models trained
  after this stage always carry BatchNorm. Old models (with or without BN) still
  load.

## Steps & Specifications

1. Delete the `"--no-batch-norm" => { cfg.use_batch_norm = false; }` match arm in
   `cli/train_cmd.rs`.
2. Delete the `--no-batch-norm` line from `print_usage()`.
3. Leave `TrainConfig::default().use_batch_norm = true` and all bridge /
   `weights.rs` handling intact (reader still accepts the `0` byte).
4. Verify build, tests, clippy, fmt.

## Definition of Done

- [x] `--no-batch-norm` removed from the argument parser and help text.
- [x] `use_batch_norm` remains `true` on every training-produced model
      (`TrainConfig::default().use_batch_norm = true`; no code path sets it false).
- [x] `.wbmodel` format byte-compatible; reader (`weights.rs`) still reads the
      `use_batch_norm` byte and honors a `0` value, so legacy BN-stripped models
      still load.
- [x] Passing `--no-batch-norm` now yields the standard unknown-flag error
      (`train: unknown flag '--no-batch-norm'`).
- [x] `cargo build/test/clippy/fmt` clean; all existing tests pass.
- [x] This spec synchronized with the final implementation.

## Results

Implemented as specified. Changes were confined to `cli/train_cmd.rs`:

- Removed the `"--no-batch-norm" => { cfg.use_batch_norm = false; }` match arm.
- Removed the `--no-batch-norm` line from `print_usage()`.

No changes to the burn forward, the weight bridge, `weights.rs`, or the
`.wbmodel` format. `TrainConfig.use_batch_norm` / `PointNetConfig.use_batch_norm`
remain as `true`-pinned fields (still consumed by the bridge and serialization),
and the reader continues to accept the `0` byte for backward compatibility with
any previously-produced BN-stripped models.

Verification: `cargo test --features training` → 66 passed / 0 failed;
`cargo clippy --features training` clean (only the pre-existing
`whitebox_next_gen/wbraster` `CmpNe` warning, outside this crate);
`cargo fmt --check` clean. Passing `--no-batch-norm` is now rejected as an
unknown flag, as intended.
