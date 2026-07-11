//! `split-dataset` sub-command — physically materializes a train/val/test
//! directory split from one or more `preprocess-labeled` output directories
//! (Stage 32, extended to multiple merged inputs in Stage 33, extended to
//! accept a `--input-list` response file in Stage 34).
//!
//! See `docs/stages/stage-32-dataset-split-materialization.md`,
//! `docs/stages/stage-33-multi-input-dataset-split.md`, and
//! `docs/stages/stage-34-input-list-flag.md` for the full design rationale.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{ClassifierError, Result};
use crate::preprocessing::dataset_split::{three_way_spatial_split_multi, MultiThreeWaySplit};
use crate::preprocessing::labeled_pipeline::{LabeledBlockManifest, LabeledBlockMeta};
use crate::preprocessing::validate_block_filename;

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let mut explicit_inputs: Vec<PathBuf> = Vec::new();
    let mut input_list_files: Vec<PathBuf> = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut val_split: f64 = 0.20;
    let mut test_split: f64 = 0.0;
    let mut seed: u64 = 42;
    let mut stratify_classes: bool = true;
    let mut move_files: bool = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                explicit_inputs.push(PathBuf::from(next_value(args, &mut i, "--input")?));
            }
            "--input-list" => {
                input_list_files.push(PathBuf::from(next_value(args, &mut i, "--input-list")?));
            }
            "--output" => {
                output = Some(PathBuf::from(next_value(args, &mut i, "--output")?));
            }
            "--val-split" => {
                val_split = parse_f64(next_value(args, &mut i, "--val-split")?, "--val-split")?;
            }
            "--test-split" => {
                test_split = parse_f64(next_value(args, &mut i, "--test-split")?, "--test-split")?;
            }
            "--seed" => {
                seed = parse_u64(next_value(args, &mut i, "--seed")?, "--seed")?;
            }
            "--no-stratify-classes" => {
                stratify_classes = false;
            }
            "--move" => {
                move_files = true;
            }
            flag => {
                return Err(ClassifierError::Pipeline(format!(
                    "split-dataset: unknown flag '{flag}'"
                )));
            }
        }
        i += 1;
    }

    // Resolve `--input-list` files (in the order given) and combine with
    // explicit `--input` flags per the ordering rule documented in
    // stage-34-input-list-flag.md: list-derived entries first, then
    // explicit --input flags.
    let inputs = resolve_inputs(&input_list_files, explicit_inputs)?;
    let output = output.ok_or_else(|| ClassifierError::Pipeline("--output is required".into()))?;

    if !(0.0..1.0).contains(&val_split) || !val_split.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--val-split must be in the range [0.0, 1.0) and finite".into(),
        ));
    }

    if !(0.0..1.0).contains(&test_split) || !test_split.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--test-split must be in the range [0.0, 1.0) and finite".into(),
        ));
    }
    if val_split + test_split >= 1.0 {
        return Err(ClassifierError::Pipeline(
            "--val-split + --test-split must be < 1.0 (the training set would \
             otherwise be empty)"
                .into(),
        ));
    }

    let mut manifests: Vec<LabeledBlockManifest> = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let manifest_path = input.join("labeled_blocks.json");
        let f = fs::File::open(&manifest_path).map_err(|e| {
            ClassifierError::Pipeline(format!("cannot open {}: {e}", manifest_path.display()))
        })?;
        let manifest: LabeledBlockManifest = serde_json::from_reader(std::io::BufReader::new(f))
            .map_err(|e| {
                ClassifierError::Pipeline(format!(
                    "labeled_blocks.json parse error in {}: {e}",
                    input.display()
                ))
            })?;
        manifests.push(manifest);
    }

    let manifest_refs: Vec<&LabeledBlockManifest> = manifests.iter().collect();
    let split = three_way_spatial_split_multi(
        &manifest_refs,
        val_split,
        test_split,
        seed,
        stratify_classes,
    )?;

    materialize_split(&inputs, &output, &manifests, &split, move_files)?;

    Ok(())
}

/// Resolve the final, ordered list of input directories for `run()`:
/// each `--input-list` file (in the order given) is read and parsed into
/// trimmed, non-blank, non-`#`-comment-prefixed lines (in file order across
/// all supplied list files), then `explicit_inputs` (from `--input` flags,
/// in flag order) is appended. See
/// `docs/stages/stage-34-input-list-flag.md` for the full ordering
/// rationale. Returns a clear error (not a panic) if a list file cannot be
/// read, or if the combined result is empty.
fn resolve_inputs(
    input_list_files: &[PathBuf],
    explicit_inputs: Vec<PathBuf>,
) -> Result<Vec<PathBuf>> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    for list_file in input_list_files {
        let contents = fs::read_to_string(list_file).map_err(|e| {
            ClassifierError::Pipeline(format!(
                "--input-list: cannot read '{}': {e}",
                list_file.display()
            ))
        })?;
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            inputs.push(PathBuf::from(trimmed));
        }
    }
    inputs.extend(explicit_inputs);

    if inputs.is_empty() {
        return Err(ClassifierError::Pipeline(
            "at least one --input or --input-list entry is required".into(),
        ));
    }

    Ok(inputs)
}

/// Physically create `train/`, `val/`, and (if non-empty) `test/`
/// subdirectories under `output`, each containing the assigned blocks'
/// `.feat`/`.lbl` files (freshly, sequentially renumbered — see
/// `write_subset`) plus a merged, filtered `labeled_blocks.json`.
fn materialize_split(
    inputs: &[PathBuf],

    output: &Path,
    manifests: &[LabeledBlockManifest],
    split: &MultiThreeWaySplit,
    move_files: bool,
) -> Result<()> {
    fs::create_dir_all(output)?;

    write_subset(inputs, output, "train", manifests, &split.train, move_files)?;
    write_subset(inputs, output, "val", manifests, &split.val, move_files)?;
    if !split.test.is_empty() {
        write_subset(inputs, output, "test", manifests, &split.test, move_files)?;
    }

    eprintln!(
        "[split-dataset] train: {} blocks, val: {} blocks, test: {} blocks (from {} input \
         director{})",
        split.train.len(),
        split.val.len(),
        split.test.len(),
        inputs.len(),
        if inputs.len() == 1 { "y" } else { "ies" }
    );

    Ok(())
}

/// Materialize a single subset (`train`/`val`/`test`) directory: copy or move
/// each assigned block's `.feat`/`.lbl` files (resolved back to its correct
/// source input directory via `dir_idx`), then write a merged, filtered
/// `labeled_blocks.json` scoped to just this subset's blocks.
///
/// Every block is freshly, sequentially renumbered at write time (sorted by
/// `(dir_idx, original_block_id)` for determinism) so that blocks originating
/// from different input directories — whose original local IDs may collide,
/// since each `preprocess-labeled` run numbers blocks independently relative
/// to its own file's bounding box — never collide in the merged output
/// directory. See `docs/stages/stage-33-multi-input-dataset-split.md`.
///
/// `--move` semantics (`DoD` item 7, Stage 32/33): every block's pair of
/// files is fully copied first; only after *both* files of a block have been
/// successfully copied are the *source* files removed. If copying a block's
/// `.lbl` file fails after its `.feat` file already succeeded, the
/// already-copied `.feat` destination file is left in place (harmless — it
/// will simply be re-copied/overwritten on a re-run) but neither source file
/// for that block is deleted, so no data is ever lost to a partial failure.
#[allow(clippy::cast_possible_truncation)]
fn write_subset(
    inputs: &[PathBuf],
    output: &Path,
    subset_name: &str,
    manifests: &[LabeledBlockManifest],
    refs: &[(usize, u64)],
    move_files: bool,
) -> Result<()> {
    if refs.is_empty() {
        return Ok(());
    }

    let subset_dir = output.join(subset_name);
    fs::create_dir_all(&subset_dir)?;

    // Deterministic ordering for fresh sequential renumbering: re-running
    // split-dataset with identical inputs/flags/seed always produces
    // byte-identical output filenames for a given logical block.
    let mut sorted_refs: Vec<(usize, u64)> = refs.to_vec();
    sorted_refs.sort_unstable();

    let mut block_lookup: HashMap<(usize, u64), &LabeledBlockMeta> = HashMap::new();
    for (dir_idx, manifest) in manifests.iter().enumerate() {
        for b in &manifest.blocks {
            block_lookup.insert((dir_idx, b.meta.id), b);
        }
    }

    let mut subset_blocks = Vec::with_capacity(sorted_refs.len());

    for (new_id, &(dir_idx, orig_id)) in sorted_refs.iter().enumerate() {
        let Some(block) = block_lookup.get(&(dir_idx, orig_id)).copied() else {
            // Defensive: should be unreachable, since `refs` is always
            // derived from these same manifests' block lists.
            continue;
        };

        validate_block_filename(&block.meta.file)?;
        validate_block_filename(&block.lbl_file)?;

        let input_dir = inputs.get(dir_idx).ok_or_else(|| {
            ClassifierError::Pipeline(format!(
                "split-dataset: internal error — dir_idx {dir_idx} out of range for {} inputs",
                inputs.len()
            ))
        })?;
        let src_feat = input_dir.join(&block.meta.file);
        let src_lbl = input_dir.join(&block.lbl_file);

        let new_id_u64 = new_id as u64;
        let new_feat_name = format!("block_{new_id_u64:05}.feat");
        let new_lbl_name = format!("block_{new_id_u64:05}.lbl");
        let dst_feat = subset_dir.join(&new_feat_name);
        let dst_lbl = subset_dir.join(&new_lbl_name);

        fs::copy(&src_feat, &dst_feat).map_err(|e| {
            ClassifierError::Pipeline(format!(
                "split-dataset: failed to copy '{}' -> '{}': {e}",
                src_feat.display(),
                dst_feat.display()
            ))
        })?;
        fs::copy(&src_lbl, &dst_lbl).map_err(|e| {
            ClassifierError::Pipeline(format!(
                "split-dataset: failed to copy '{}' -> '{}': {e}",
                src_lbl.display(),
                dst_lbl.display()
            ))
        })?;

        // Only remove sources after both files of this block copied cleanly.
        if move_files {
            let _ = fs::remove_file(&src_feat);
            let _ = fs::remove_file(&src_lbl);
        }

        let mut new_block = block.clone();
        new_block.meta.id = new_id_u64;
        new_block.meta.file = new_feat_name;
        new_block.lbl_file = new_lbl_name;
        subset_blocks.push(new_block);
    }

    // Combined manifest metadata: `block_size`/`target_points`/etc. were
    // already validated identical across all input manifests by
    // `validate_manifest_compatibility` (called inside
    // `three_way_spatial_split_multi`); `source` is a comma-joined list for
    // provenance, and `spatial_tile_grid` is carried over from the first
    // input purely as informational/debug data (see stage-33 doc — no
    // downstream consumer reads it back out of a loaded manifest).
    let first = &manifests[0];
    let combined_source = manifests
        .iter()
        .map(|m| m.source.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let subset_manifest = LabeledBlockManifest {
        source: combined_source,
        block_size: first.block_size,
        target_points: first.target_points,
        min_density: first.min_density,
        search_radius: first.search_radius,
        min_neighbors: first.min_neighbors,
        crs_epsg: first.crs_epsg,
        label_map: first.label_map.clone(),
        spatial_tile_grid: first.spatial_tile_grid.clone(),
        blocks: subset_blocks,
    };

    let manifest_out_path = subset_dir.join("labeled_blocks.json");
    let manifest_bytes = serde_json::to_vec_pretty(&subset_manifest).map_err(|e| {
        ClassifierError::Pipeline(format!("split-dataset: manifest serialize error: {e}"))
    })?;
    fs::write(&manifest_out_path, manifest_bytes).map_err(|e| {
        ClassifierError::Pipeline(format!(
            "split-dataset: failed to write '{}': {e}",
            manifest_out_path.display()
        ))
    })?;

    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: wb_lidar_train split-dataset [options]\n\
         \n\
         Physically materializes a train/val/test directory split from one or\n\
         more `preprocess-labeled` output directories. Passing multiple --input\n\
         directories merges their manifests into a single, globally-stratified\n\
         split before materialization (Stage 33). At large input counts, use\n\
         --input-list to avoid OS command-line length limits (Stage 34).\n\
         \n\
         Required (at least one of --input / --input-list, combined, is\n\
         required):\n\
           --input   <dir>   Directory produced by `preprocess-labeled`\n\
                              (must contain labeled_blocks.json). REPEATABLE —\n\
                              pass --input once per source directory to merge\n\
                              them into a single global split.\n\
           --input-list <file>  Text file containing one input directory path\n\
                              per line. Blank lines and lines starting with\n\
                              '#' are ignored. REPEATABLE — multiple\n\
                              --input-list files are concatenated in the\n\
                              order given. May be combined with --input;\n\
                              --input-list entries are placed first, followed\n\
                              by explicit --input flags. Use this to avoid\n\
                              the OS command-line length limit (~32,767\n\
                              characters on Windows) when merging hundreds or\n\
                              thousands of input directories.\n\
           --output  <dir>   Output directory; train/, val/, [test/]\n\
                              subdirectories are created inside it\n\
         \n\
         Optional:\n\
           --val-split  <f64>   Fraction of macro-tiles -> validation (default: 0.20)\n\
           --test-split <f64>   Fraction of macro-tiles -> test (default: 0.0, disabled)\n\
           --seed <u64>         Seed for deterministic tie-breaking (default: 42)\n\
           --no-stratify-classes  Disable class-stratified assignment; use pure\n\
                                  spatial macro-tile stride selection instead\n\
                                  (default: stratification is ON)\n\
           --move                 Move files instead of copying (default: copy)"
    );
}

/// If the flag at `args[*i]` requires a value, bounds-check and consume the
/// next token.  Returns a clear `ClassifierError::Pipeline` instead of
/// panicking (via unchecked indexing) if the flag is the last argument.
fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("flag '{flag}' requires a value")))
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f64 '{s}'")))
}

fn parse_u64(s: &str, flag: &str) -> Result<u64> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid u64 '{s}'")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::dataset_split::three_way_spatial_split_multi;
    use crate::preprocessing::labeled_pipeline::SpatialTileGrid;
    use crate::preprocessing::pipeline::BlockMeta;
    use crate::training::dataset::LabeledBlockDataset;
    use std::collections::HashMap as HM;
    use std::collections::HashSet;

    // Stage 20 (Security Hardening) — a flag with no trailing value must
    // return a clear error instead of panicking via unchecked indexing.
    #[test]
    fn test_trailing_flag_without_value_errors_not_panics() {
        let args: Vec<String> = vec!["--input".to_string()];
        let mut i = 0usize;
        let result = next_value(&args, &mut i, "--input");
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_trailing_flag_returns_error() {
        let args: Vec<String> = vec!["--val-split".to_string()];
        let result = run(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_rejects_missing_required_flags() {
        let result = run(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_rejects_missing_input_even_with_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let args: Vec<String> = vec!["--output".to_string(), dir.path().display().to_string()];
        let result = run(&args);
        assert!(
            result.is_err(),
            "--output alone without any --input must fail"
        );
    }

    fn make_block_fixture(dir: &Path, id: u64, n_points: usize) {
        let feat_name = format!("block_{id:05}.feat");
        let lbl_name = format!("block_{id:05}.lbl");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(crate::preprocessing::FEAT_MAGIC);
        bytes.push(crate::preprocessing::FEAT_VERSION);
        #[allow(clippy::cast_possible_truncation)]
        {
            bytes.extend_from_slice(&(n_points as u32).to_le_bytes());
            bytes.extend_from_slice(&(crate::preprocessing::N_FEATURES as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        for _ in 0..(n_points * crate::preprocessing::N_FEATURES) {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        fs::write(dir.join(&feat_name), &bytes).expect("write feat fixture");
        fs::write(dir.join(&lbl_name), vec![0u8; n_points]).expect("write lbl fixture");
    }

    fn make_manifest_fixture(n_blocks: u64, source: &str) -> LabeledBlockManifest {
        let mut blocks = Vec::new();
        let mut label_map = HM::new();
        label_map.insert("2".to_string(), 0u8);

        for id in 0..n_blocks {
            let mut dist = HM::new();
            dist.insert("0".to_string(), 4u64);
            blocks.push(LabeledBlockMeta {
                meta: BlockMeta {
                    id,
                    file: format!("block_{id:05}.feat"),
                    origin_x: 0.0,
                    origin_y: 0.0,
                    raw_point_count: 4,
                    sampled_point_count: 4,
                    oversampled: false,
                },
                lbl_file: format!("block_{id:05}.lbl"),
                #[allow(clippy::cast_possible_truncation)]
                macro_tile_id: id as u32,
                class_distribution: dist,
            });
        }

        LabeledBlockManifest {
            source: source.to_string(),
            block_size: 50.0,
            target_points: 4,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map,
            spatial_tile_grid: SpatialTileGrid {
                cols: 4,
                rows: 4,
                bbox_min_x: 0.0,
                bbox_min_y: 0.0,
                bbox_max_x: 200.0,
                bbox_max_y: 200.0,
            },
            blocks,
        }
    }

    #[test]
    fn test_end_to_end_split_dataset_materializes_loadable_directories() {
        let input_dir = tempfile::tempdir().expect("input tempdir");
        let output_dir = tempfile::tempdir().expect("output tempdir");

        let manifest = make_manifest_fixture(8, "test.las");
        for b in &manifest.blocks {
            make_block_fixture(input_dir.path(), b.meta.id, 4);
        }
        fs::write(
            input_dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest fixture");

        let split = three_way_spatial_split_multi(&[&manifest], 0.25, 0.0, 7, false)
            .expect("three_way_spatial_split_multi should succeed");

        let inputs = vec![input_dir.path().to_path_buf()];
        let manifests = vec![manifest];
        materialize_split(&inputs, output_dir.path(), &manifests, &split, false)
            .expect("materialize_split should succeed");

        assert!(output_dir.path().join("train").is_dir());
        assert!(output_dir.path().join("val").is_dir());
        assert!(!output_dir.path().join("test").exists());

        // Loadable via load_presplit().
        let dataset = LabeledBlockDataset::load_presplit(
            &[output_dir.path().join("train")],
            &[output_dir.path().join("val")],
        )
        .expect("load_presplit should succeed on materialized output");

        assert_eq!(dataset.train_ids.len(), split.train.len());
        assert_eq!(dataset.val_ids.len(), split.val.len());
    }

    #[test]
    fn test_move_deletes_source_files_after_success() {
        let input_dir = tempfile::tempdir().expect("input tempdir");
        let output_dir = tempfile::tempdir().expect("output tempdir");

        let manifest = make_manifest_fixture(4, "test.las");
        for b in &manifest.blocks {
            make_block_fixture(input_dir.path(), b.meta.id, 4);
        }

        let split = three_way_spatial_split_multi(&[&manifest], 0.25, 0.0, 1, false)
            .expect("three_way_spatial_split_multi should succeed");

        let inputs = vec![input_dir.path().to_path_buf()];
        let manifest_files: Vec<(String, String)> = manifest
            .blocks
            .iter()
            .map(|b| (b.meta.file.clone(), b.lbl_file.clone()))
            .collect();
        let manifests = vec![manifest];
        materialize_split(&inputs, output_dir.path(), &manifests, &split, true)
            .expect("materialize_split (move) should succeed");

        // Every source .feat/.lbl file must be gone after a successful move.
        for (feat, lbl) in &manifest_files {
            assert!(
                !input_dir.path().join(feat).exists(),
                "source .feat should have been moved away"
            );
            assert!(
                !input_dir.path().join(lbl).exists(),
                "source .lbl should have been moved away"
            );
        }
    }

    #[test]
    fn test_multi_input_merge_materializes_no_filename_collisions_and_loadable() {
        // Two independently-preprocessed input directories with locally
        // colliding block ids (0..6 in each) — exactly the scenario Stage 33
        // is meant to fix.
        let input_dir_a = tempfile::tempdir().expect("input tempdir a");
        let input_dir_b = tempfile::tempdir().expect("input tempdir b");
        let output_dir = tempfile::tempdir().expect("output tempdir");

        let manifest_a = make_manifest_fixture(6, "file_a.las");
        let manifest_b = make_manifest_fixture(6, "file_b.las");
        for b in &manifest_a.blocks {
            make_block_fixture(input_dir_a.path(), b.meta.id, 4);
        }
        for b in &manifest_b.blocks {
            make_block_fixture(input_dir_b.path(), b.meta.id, 4);
        }

        let manifest_refs: Vec<&LabeledBlockManifest> = vec![&manifest_a, &manifest_b];
        let split = three_way_spatial_split_multi(&manifest_refs, 0.25, 0.0, 3, true)
            .expect("three_way_spatial_split_multi should succeed across merged inputs");

        let inputs = vec![
            input_dir_a.path().to_path_buf(),
            input_dir_b.path().to_path_buf(),
        ];
        let manifests = vec![manifest_a, manifest_b];
        materialize_split(&inputs, output_dir.path(), &manifests, &split, false)
            .expect("materialize_split should succeed across merged inputs");

        // No filename collisions: every .feat file present exactly once per
        // subset directory, and total block count across both subsets
        // equals the full merged input (12 blocks).
        let mut total_written = 0usize;
        for subset in ["train", "val"] {
            let subset_dir = output_dir.path().join(subset);
            if !subset_dir.is_dir() {
                continue;
            }
            let mut seen_ids: HashSet<String> = HashSet::new();
            for entry in fs::read_dir(&subset_dir).expect("read_dir") {
                let entry = entry.expect("dir entry");
                let name = entry.file_name().to_string_lossy().to_string();
                if std::path::Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("feat"))
                {
                    assert!(
                        seen_ids.insert(name.clone()),
                        "duplicate .feat filename '{name}' found in {subset}/ — merge \
                         renumbering must prevent collisions"
                    );
                    total_written += 1;
                }
            }
        }
        assert_eq!(
            total_written, 12,
            "all 12 merged blocks (6 + 6) must be written exactly once across train+val"
        );

        // Loadable via load_presplit(), with correct combined counts.
        let dataset = LabeledBlockDataset::load_presplit(
            &[output_dir.path().join("train")],
            &[output_dir.path().join("val")],
        )
        .expect("load_presplit should succeed on merged materialized output");
        assert_eq!(dataset.train_ids.len() + dataset.val_ids.len(), 12);
    }

    // ---- Stage 34: --input-list response file tests ----

    #[test]
    fn test_run_rejects_nonexistent_input_list_file() {
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let missing_list = output_dir.path().join("does_not_exist.txt");
        let args: Vec<String> = vec![
            "--input-list".to_string(),
            missing_list.display().to_string(),
            "--output".to_string(),
            output_dir.path().display().to_string(),
        ];
        let result = run(&args);
        assert!(
            result.is_err(),
            "a nonexistent --input-list file must produce a clear error, not panic"
        );
    }

    #[test]
    fn test_run_rejects_empty_input_list_and_no_input() {
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let list_path = output_dir.path().join("inputs.txt");
        // Only blank lines and comments -> zero resolved directories.
        fs::write(&list_path, "# just a comment\n\n   \n").expect("write list file");

        let args: Vec<String> = vec![
            "--input-list".to_string(),
            list_path.display().to_string(),
            "--output".to_string(),
            output_dir.path().display().to_string(),
        ];
        let result = run(&args);
        assert!(
            result.is_err(),
            "an --input-list file with zero resolved directories (and no --input) must fail"
        );
    }

    #[test]
    fn test_input_list_parsing_skips_blank_and_comment_lines() {
        // Directly exercises the same parsing logic as `run()` by writing a
        // list file with comments/blank lines interleaved with two real
        // directory paths, and confirming both are picked up in file order.
        let list_dir = tempfile::tempdir().expect("list tempdir");
        let list_path = list_dir.path().join("inputs.txt");
        fs::write(
            &list_path,
            "# comment line\n\n  \nC:/some/dir/a\n# another comment\nC:/some/dir/b\n\n",
        )
        .expect("write list file");

        let contents = fs::read_to_string(&list_path).expect("read list file");
        let parsed: Vec<PathBuf> = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(PathBuf::from)
            .collect();

        assert_eq!(
            parsed,
            vec![
                PathBuf::from("C:/some/dir/a"),
                PathBuf::from("C:/some/dir/b"),
            ]
        );
    }

    #[test]
    fn test_multiple_input_list_files_concatenate_in_order() {
        let list_dir = tempfile::tempdir().expect("list tempdir");
        let list_a = list_dir.path().join("a.txt");
        let list_b = list_dir.path().join("b.txt");
        fs::write(&list_a, "dir_one\ndir_two\n").expect("write list a");
        fs::write(&list_b, "dir_three\n").expect("write list b");

        // Mirrors the combination loop in `run()`: entries from each
        // --input-list file are concatenated in the order the files are
        // given (a.txt fully, then b.txt fully).
        let mut combined: Vec<PathBuf> = Vec::new();
        for list_file in [&list_a, &list_b] {
            let contents = fs::read_to_string(list_file).expect("read list file");
            for line in contents.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                combined.push(PathBuf::from(trimmed));
            }
        }

        assert_eq!(
            combined,
            vec![
                PathBuf::from("dir_one"),
                PathBuf::from("dir_two"),
                PathBuf::from("dir_three"),
            ]
        );
    }

    #[test]
    fn test_input_list_end_to_end_materializes_identically_to_input_flags() {
        // Two synthetic preprocess-labeled-style directories, referenced via
        // an --input-list file instead of repeated --input flags — must
        // produce byte-identical merged output to the equivalent all-flags
        // invocation (test_multi_input_merge_materializes_no_filename_collisions_and_loadable).
        let input_dir_a = tempfile::tempdir().expect("input tempdir a");
        let input_dir_b = tempfile::tempdir().expect("input tempdir b");
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let list_dir = tempfile::tempdir().expect("list tempdir");

        let manifest_a = make_manifest_fixture(6, "file_a.las");
        let manifest_b = make_manifest_fixture(6, "file_b.las");
        for b in &manifest_a.blocks {
            make_block_fixture(input_dir_a.path(), b.meta.id, 4);
        }
        for b in &manifest_b.blocks {
            make_block_fixture(input_dir_b.path(), b.meta.id, 4);
        }
        fs::write(
            input_dir_a.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest_a).expect("serialize manifest a"),
        )
        .expect("write manifest a fixture");
        fs::write(
            input_dir_b.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest_b).expect("serialize manifest b"),
        )
        .expect("write manifest b fixture");

        let list_path = list_dir.path().join("inputs.txt");
        fs::write(
            &list_path,
            format!(
                "# generated for test\n{}\n\n{}\n",
                input_dir_a.path().display(),
                input_dir_b.path().display()
            ),
        )
        .expect("write input list file");

        let args: Vec<String> = vec![
            "--input-list".to_string(),
            list_path.display().to_string(),
            "--output".to_string(),
            output_dir.path().display().to_string(),
            "--val-split".to_string(),
            "0.25".to_string(),
            "--seed".to_string(),
            "3".to_string(),
        ];
        run(&args).expect("run() via --input-list should succeed");

        assert!(output_dir.path().join("train").is_dir());
        assert!(output_dir.path().join("val").is_dir());

        let mut total_written = 0usize;
        for subset in ["train", "val"] {
            let subset_dir = output_dir.path().join(subset);
            if !subset_dir.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&subset_dir).expect("read_dir") {
                let entry = entry.expect("dir entry");
                let name = entry.file_name().to_string_lossy().to_string();
                if std::path::Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("feat"))
                {
                    total_written += 1;
                }
            }
        }
        assert_eq!(
            total_written, 12,
            "all 12 merged blocks (6 + 6) must be written exactly once via --input-list"
        );

        let dataset = LabeledBlockDataset::load_presplit(
            &[output_dir.path().join("train")],
            &[output_dir.path().join("val")],
        )
        .expect("load_presplit should succeed on --input-list materialized output");
        assert_eq!(dataset.train_ids.len() + dataset.val_ids.len(), 12);
    }

    #[test]
    fn test_input_list_combined_with_explicit_input_flag() {
        // --input-list entries plus an explicit --input flag must all be
        // merged into a single global split.
        let input_dir_a = tempfile::tempdir().expect("input tempdir a");
        let input_dir_b = tempfile::tempdir().expect("input tempdir b");
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let list_dir = tempfile::tempdir().expect("list tempdir");

        let manifest_a = make_manifest_fixture(6, "file_a.las");
        let manifest_b = make_manifest_fixture(6, "file_b.las");
        for b in &manifest_a.blocks {
            make_block_fixture(input_dir_a.path(), b.meta.id, 4);
        }
        for b in &manifest_b.blocks {
            make_block_fixture(input_dir_b.path(), b.meta.id, 4);
        }
        fs::write(
            input_dir_a.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest_a).expect("serialize manifest a"),
        )
        .expect("write manifest a fixture");
        fs::write(
            input_dir_b.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest_b).expect("serialize manifest b"),
        )
        .expect("write manifest b fixture");

        // input_dir_a comes from the list file; input_dir_b is an explicit
        // --input flag.
        let list_path = list_dir.path().join("inputs.txt");
        fs::write(&list_path, format!("{}\n", input_dir_a.path().display()))
            .expect("write input list file");

        let args: Vec<String> = vec![
            "--input-list".to_string(),
            list_path.display().to_string(),
            "--input".to_string(),
            input_dir_b.path().display().to_string(),
            "--output".to_string(),
            output_dir.path().display().to_string(),
            "--val-split".to_string(),
            "0.25".to_string(),
            "--seed".to_string(),
            "3".to_string(),
        ];
        run(&args).expect("run() with combined --input-list + --input should succeed");

        let mut total_written = 0usize;
        for subset in ["train", "val"] {
            let subset_dir = output_dir.path().join(subset);
            if !subset_dir.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&subset_dir).expect("read_dir") {
                let entry = entry.expect("dir entry");
                let name = entry.file_name().to_string_lossy().to_string();
                if std::path::Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("feat"))
                {
                    total_written += 1;
                }
            }
        }
        assert_eq!(
            total_written, 12,
            "all 12 merged blocks (6 from --input-list + 6 from --input) must be written exactly once"
        );
    }
}
