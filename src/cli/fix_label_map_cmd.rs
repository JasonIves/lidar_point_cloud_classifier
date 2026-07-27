//! `fix-label-map` sub-command (Stage 41 follow-up) — patch an existing
//! `.wbmodel`'s `label_map` field in place, without retraining.
//!
//! ## Why this exists
//!
//! Every `.wbmodel` trained before Stage 41
//! (`docs/stages/stage-41-model-label-map-identity-bug-fix.md`) has an
//! incorrect (hardcoded-identity) `label_map`. Since `label_map` is a small,
//! self-contained `Vec<u8>` field in the `.wbmodel` file — entirely separate
//! from the trained weight tensors — it can be corrected without retraining:
//! reload the model, re-derive the correct mapping from the *same data
//! directory(ies) originally used to train it*, overwrite just that field,
//! and re-save. `model::weights::save_model`/`load_model` round-trip every
//! weight tensor bit-identically (`test_wbmodel_round_trip`), so no weights
//! are touched.
//!
//! This is a minimal, standalone utility deliberately kept out of the
//! primary `train`/`evaluate` workflows and documentation — see Stage 41's
//! doc for the full incident writeup; this sub-command is referenced there
//! only as a footnote.
//!
//! Usage:
//! ```text
//! wb_lidar_train fix-label-map
//!     --model    <path.wbmodel>   model to patch (required)
//!     --data-dir <dir>            the ORIGINAL training data dir(s), same
//!                                  label map as used to train this model
//!                                  (repeatable, required)
//!     [--output  <path>]          write to a new file instead of overwriting
//!                                  --model in place (default: overwrite)
//! ```

use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::model::weights::{load_model, save_model};
use crate::training::dataset::LabeledBlockDataset;

/// Parse `args` (everything after `fix-label-map`) and patch the model.
///
/// # Errors
/// Returns an error if argument parsing fails, the model or dataset cannot
/// be loaded, the model/data class counts disagree, or the patched model
/// cannot be saved.
pub fn run(args: &[String]) -> Result<()> {
    let cfg = parse_args(args)?;

    eprintln!("[fix-label-map] loading model: {}", cfg.model.display());
    let mut model = load_model(&cfg.model)?;

    eprintln!(
        "[fix-label-map] loading {} data dir(s)…",
        cfg.data_dirs.len()
    );
    let dataset = LabeledBlockDataset::load(&cfg.data_dirs, 0.0, None, 0)?;

    let model_n = model.config.n_classes;
    let data_n = dataset.n_classes();
    if model_n != data_n {
        return Err(ClassifierError::Pipeline(format!(
            "fix-label-map: model has {model_n} classes but the data directory \
             declares {data_n} classes — this is not the data the model was \
             trained on. Pass the ORIGINAL --data-dir(s) used to train this model."
        )));
    }

    let derived = dataset.inverse_label_map()?;

    if model.label_map == derived {
        eprintln!(
            "[fix-label-map] model.label_map already matches the dataset's mapping \
             {derived:?} — no changes needed."
        );
        if cfg.output.is_none() {
            return Ok(());
        }
    } else {
        eprintln!(
            "[fix-label-map] patching label_map: {:?} -> {derived:?}",
            model.label_map
        );
        model.label_map = derived;
    }

    let out_path = cfg.output.unwrap_or(cfg.model);
    save_model(&out_path, &model)?;
    eprintln!(
        "[fix-label-map] wrote patched model: {}",
        out_path.display()
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Config + argument parsing
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct FixLabelMapConfig {
    model: PathBuf,
    data_dirs: Vec<PathBuf>,
    output: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<FixLabelMapConfig> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let mut model: Option<PathBuf> = None;
    let mut data_dirs: Vec<PathBuf> = Vec::new();
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(require_value(args, i, "--model")?));
            }
            "--data-dir" => {
                i += 1;
                data_dirs.push(PathBuf::from(require_value(args, i, "--data-dir")?));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(require_value(args, i, "--output")?));
            }
            unknown => {
                return Err(ClassifierError::Pipeline(format!(
                    "fix-label-map: unknown argument '{unknown}'"
                )));
            }
        }
        i += 1;
    }

    if data_dirs.is_empty() {
        return Err(ClassifierError::Pipeline(
            "fix-label-map: at least one --data-dir is required".into(),
        ));
    }

    Ok(FixLabelMapConfig {
        model: model.ok_or_else(|| {
            ClassifierError::Pipeline("fix-label-map: --model is required".into())
        })?,
        data_dirs,
        output,
    })
}

fn require_value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("fix-label-map: {flag} requires a value")))
}

fn print_help() {
    eprintln!(
        "Usage: wb_lidar_train fix-label-map [options]\n\
         \n\
         Patch an existing .wbmodel's label_map field in place, without\n\
         retraining (Stage 41 follow-up). Only the label_map bytes are\n\
         changed; all weight tensors are preserved bit-identically.\n\
         \n\
         Options:\n\
           --model    <path>   Trained .wbmodel to patch (required)\n\
           --data-dir <dir>    The ORIGINAL training data dir(s) — same label\n\
                               map used to train this model (repeatable,\n\
                               required)\n\
           --output   <path>   Write to a new file instead of overwriting\n\
                               --model in place (default: overwrite)\n\
           --help, -h          Show this message"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::Linear;
    use crate::model::pointnet::{PointNetClassifier, PointNetConfig};
    use crate::preprocessing::labeled_pipeline::{
        LabeledBlockManifest, LabeledBlockMeta, SpatialTileGrid,
    };
    use crate::preprocessing::pipeline::BlockMeta;
    use crate::preprocessing::{FEAT_MAGIC, FEAT_VERSION, N_FEATURES};
    use ndarray::{Array1, Array2};
    use std::collections::HashMap;

    fn make_identity_model(n_classes: usize) -> PointNetClassifier {
        let encoder_dims = vec![4usize, 8];
        let decoder_dims = vec![4usize];
        let config = PointNetConfig {
            n_features_in: N_FEATURES,
            encoder_dims: encoder_dims.clone(),
            decoder_dims: decoder_dims.clone(),
            n_classes,
            use_batch_norm: false,
            use_input_tnet: false,
            use_feature_tnet: false,
        };

        let mut encoder_layers = Vec::new();
        let mut prev = N_FEATURES;
        for &dim in &encoder_dims {
            encoder_layers.push((
                Linear::new(Array2::zeros((dim, prev)), Array1::zeros(dim)).unwrap(),
                None,
            ));
            prev = dim;
        }

        let concat_dim = config.concat_dim();
        let mut decoder_layers = Vec::new();
        let mut prev_d = concat_dim;
        for &dim in &decoder_dims {
            decoder_layers.push((
                Linear::new(Array2::zeros((dim, prev_d)), Array1::zeros(dim)).unwrap(),
                None,
            ));
            prev_d = dim;
        }

        let class_proj =
            Linear::new(Array2::zeros((n_classes, prev_d)), Array1::zeros(n_classes)).unwrap();

        PointNetClassifier {
            config,
            input_tnet: None,
            feature_tnet: None,
            encoder_layers,
            decoder_layers,
            class_proj,
            // The pre-Stage-41 bug: identity label_map.
            label_map: (0..n_classes).map(|i| u8::try_from(i).unwrap()).collect(),
        }
    }

    fn write_feat(path: &std::path::Path, n_points: usize) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FEAT_MAGIC);
        bytes.push(FEAT_VERSION);
        bytes.extend_from_slice(&(u32::try_from(n_points).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&(u32::try_from(N_FEATURES).unwrap()).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        bytes.extend_from_slice(&0f64.to_le_bytes());
        for _ in 0..(n_points * N_FEATURES) {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        std::fs::write(path, &bytes).expect("write feat");
    }

    /// Build a one-block, 2-class labeled dir with a deliberately non-identity
    /// label map (ASPRS "2"->0, "3"->1 — happens to be identity-shifted-by-2
    /// here to mirror the codebase's real default map's first two entries).
    fn build_test_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        write_feat(&dir.path().join("block_00000.feat"), 4);
        std::fs::write(dir.path().join("block_00000.lbl"), [0u8, 0, 1, 1]).expect("write lbl");

        let mut label_map = HashMap::new();
        label_map.insert("2".to_string(), 0u8);
        label_map.insert("3".to_string(), 1u8);

        let meta = BlockMeta {
            id: 0,
            file: "block_00000.feat".to_string(),
            origin_x: 0.0,
            origin_y: 0.0,
            raw_point_count: 4,
            sampled_point_count: 4,
            oversampled: false,
            n_halo: 0,
        };
        let lbm = LabeledBlockMeta {
            meta,
            lbl_file: "block_00000.lbl".to_string(),
            macro_tile_id: 0,
            class_distribution: HashMap::new(),
        };
        let manifest = LabeledBlockManifest {
            source: "test.las".into(),
            block_size: 50.0,
            target_points: 4,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 8,
            crs_epsg: None,
            label_map,
            spatial_tile_grid: SpatialTileGrid {
                cols: 1,
                rows: 1,
                bbox_min_x: 0.0,
                bbox_min_y: 0.0,
                bbox_max_x: 50.0,
                bbox_max_y: 50.0,
            },
            halo_fraction: 0.0,
            blocks: vec![lbm],
        };
        std::fs::write(
            dir.path().join("labeled_blocks.json"),
            serde_json::to_vec(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        dir
    }

    #[test]
    fn test_run_patches_identity_label_map_in_place() {
        let data_dir = build_test_dir();
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let model_path = model_dir.path().join("model.wbmodel");

        let model = make_identity_model(2);
        assert_eq!(model.label_map, vec![0u8, 1u8]); // pre-fix bug reproduced
        save_model(&model_path, &model).expect("save fixture model");

        let args = vec![
            "--model".to_string(),
            model_path.to_string_lossy().to_string(),
            "--data-dir".to_string(),
            data_dir.path().to_string_lossy().to_string(),
        ];
        run(&args).expect("fix-label-map run");

        let patched = load_model(&model_path).expect("reload patched model");
        assert_eq!(
            patched.label_map,
            vec![2u8, 3u8],
            "label_map must now be the dataset-derived mapping, not identity"
        );
    }

    #[test]
    fn test_run_is_a_noop_when_already_correct() {
        let data_dir = build_test_dir();
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let model_path = model_dir.path().join("model.wbmodel");

        let mut model = make_identity_model(2);
        model.label_map = vec![2u8, 3u8]; // already correct
        save_model(&model_path, &model).expect("save fixture model");

        let args = vec![
            "--model".to_string(),
            model_path.to_string_lossy().to_string(),
            "--data-dir".to_string(),
            data_dir.path().to_string_lossy().to_string(),
        ];
        run(&args).expect("fix-label-map run");

        let patched = load_model(&model_path).expect("reload patched model");
        assert_eq!(patched.label_map, vec![2u8, 3u8]);
    }

    #[test]
    fn test_run_rejects_class_count_mismatch() {
        let data_dir = build_test_dir(); // 2 classes
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let model_path = model_dir.path().join("model.wbmodel");

        let model = make_identity_model(3); // model has 3 classes
        save_model(&model_path, &model).expect("save fixture model");

        let args = vec![
            "--model".to_string(),
            model_path.to_string_lossy().to_string(),
            "--data-dir".to_string(),
            data_dir.path().to_string_lossy().to_string(),
        ];
        let result = run(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("classes"));
    }

    #[test]
    fn test_run_writes_to_output_path_when_given() {
        let data_dir = build_test_dir();
        let model_dir = tempfile::tempdir().expect("model tempdir");
        let model_path = model_dir.path().join("model.wbmodel");
        let output_path = model_dir.path().join("patched.wbmodel");

        let model = make_identity_model(2);
        save_model(&model_path, &model).expect("save fixture model");

        let args = vec![
            "--model".to_string(),
            model_path.to_string_lossy().to_string(),
            "--data-dir".to_string(),
            data_dir.path().to_string_lossy().to_string(),
            "--output".to_string(),
            output_path.to_string_lossy().to_string(),
        ];
        run(&args).expect("fix-label-map run");

        // Original untouched, still identity.
        let original = load_model(&model_path).expect("reload original model");
        assert_eq!(original.label_map, vec![0u8, 1u8]);

        // New file has the patched mapping.
        let patched = load_model(&output_path).expect("reload output model");
        assert_eq!(patched.label_map, vec![2u8, 3u8]);
    }

    #[test]
    fn test_parse_args_requires_data_dir() {
        let args = vec!["--model".to_string(), "m.wbmodel".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--data-dir is required"));
    }

    #[test]
    fn test_parse_args_requires_model() {
        let args = vec!["--data-dir".to_string(), "d".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--model is required"));
    }
}
