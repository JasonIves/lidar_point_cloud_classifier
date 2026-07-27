//! `classify` sub-command — argument parsing and pipeline orchestration.
//!
//! Usage:
//! ```text
//! wb_lidar_classify classify
//!     --input   <path>   LAS, LAZ, or COPC source file
//!     --model   <path>   Pre-trained .wbmodel weights file
//!     --blocks  <path>   blocks.json manifest from the preprocess run
//!     --output  <path>   Classified output file (.las or .laz)
//!     [--threads <n>]    Rayon thread pool size (default: system cores)
//!     [--fusion-radius <f64>]  Cross-block fusion voting reach (default:
//!                              manifest block_overlap, else 0 = off)
//!     [--fusion-temp <f64>]    Softmax temperature before voting (default: 1)
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{ClassifierError, Result};
use crate::model::fusion::FusionConfig;
use crate::model::inference::run_inference;
use crate::model::weights::load_model;
use crate::output::las_writer::write_classified;
use crate::preprocessing::BlockManifest;

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `args` (everything after `classify`) and run the classify pipeline.
///
/// # Errors
/// Returns an error if argument parsing fails, the model or manifest cannot
/// be loaded, inference fails, or the output file cannot be written.
pub fn run(args: &[String]) -> Result<()> {
    let cfg = parse_args(args)?;

    // ── Optionally configure Rayon thread pool ─────────────────────────────
    if let Some(threads) = cfg.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    // ── Load model ─────────────────────────────────────────────────────────
    eprintln!("[classify] loading model: {}", cfg.model.display());
    let model = Arc::new(load_model(&cfg.model)?);
    eprintln!(
        "[classify] model: {} encoder dims, {} classes, T-Nets: input={}, feature={}",
        model.config.encoder_dims.len(),
        model.config.n_classes,
        model.config.use_input_tnet,
        model.config.use_feature_tnet,
    );

    // ── Load block manifest ────────────────────────────────────────────────
    eprintln!("[classify] loading manifest: {}", cfg.blocks.display());
    let manifest_json = std::fs::read_to_string(&cfg.blocks)?;
    let manifest: BlockManifest = serde_json::from_str(&manifest_json)?;
    eprintln!("[classify] blocks in manifest: {}", manifest.blocks.len());

    // ── Feat directory: same directory as blocks.json ─────────────────────
    // Path::new("blocks.json").parent() returns Some(""), not None, so we
    // must also guard against the empty-string case.
    let feat_dir = cfg
        .blocks
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();

    // ── Resolve fusion behaviour (Stage 44) ────────────────────────────────
    let fusion_radius = resolve_fusion_radius(cfg.fusion_radius, &manifest)?;
    let fusion_temp = validate_fusion_temp(cfg.fusion_temp)?;
    if fusion_radius > 0.0 {
        eprintln!(
            "[classify] prediction fusion enabled: radius={fusion_radius}, \
             temperature={fusion_temp}"
        );
    }

    // ── Run per-block inference ────────────────────────────────────────────
    eprintln!(
        "[classify] running inference on {} blocks…",
        manifest.blocks.len()
    );
    let inference_map = run_inference(&manifest, &model, &feat_dir, fusion_temp)?;
    eprintln!(
        "[classify] inference complete ({} blocks processed)",
        inference_map.len()
    );

    // ── Write classified output ────────────────────────────────────────────
    eprintln!("[classify] writing output: {}", cfg.output.display());
    write_classified(
        &cfg.input,
        &cfg.output,
        &inference_map,
        &manifest,
        &model.label_map,
        &FusionConfig {
            radius: fusion_radius,
        },
    )?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Fusion parameter resolution (Stage 44)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the fusion voting radius: the explicit CLI value when given, else
/// the manifest's `block_overlap` when positive (forward-compatible with
/// Stage 45 halo-augmented manifests, where the overlap radius *is* the halo
/// reach), else `0.0` (fusion off, legacy behaviour).
///
/// # Errors
/// Returns an error for negative/non-finite values, or a radius exceeding
/// `block_size / 2` (the candidacy bound of 4 blocks per query).
fn resolve_fusion_radius(explicit: Option<f64>, manifest: &BlockManifest) -> Result<f64> {
    let radius = explicit.unwrap_or(if manifest.block_overlap > 0.0 {
        manifest.block_overlap
    } else {
        0.0
    });
    if !radius.is_finite() || radius < 0.0 {
        return Err(ClassifierError::Pipeline(format!(
            "classify: --fusion-radius must be finite and >= 0.0, got {radius}"
        )));
    }
    let max_radius = manifest.block_size / 2.0;
    if radius > max_radius {
        return Err(ClassifierError::Pipeline(format!(
            "classify: --fusion-radius ({radius}) exceeds block_size/2 ({max_radius}); \
             the voting reach may be at most half the block size"
        )));
    }
    Ok(radius)
}

/// Resolve the softmax temperature applied per block before voting
/// (default `1.0`).
///
/// # Errors
/// Returns an error for non-finite or non-positive values.
fn validate_fusion_temp(explicit: Option<f64>) -> Result<f64> {
    let temp = explicit.unwrap_or(1.0);
    if !temp.is_finite() || temp <= 0.0 {
        return Err(ClassifierError::Pipeline(format!(
            "classify: --fusion-temp must be finite and > 0.0, got {temp}"
        )));
    }
    Ok(temp)
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

struct ClassifyConfig {
    input: PathBuf,
    model: PathBuf,
    blocks: PathBuf,
    output: PathBuf,
    threads: Option<usize>,
    fusion_radius: Option<f64>,
    fusion_temp: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<ClassifyConfig> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let mut input: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut blocks: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut fusion_radius: Option<f64> = None;
    let mut fusion_temp: Option<f64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(require_value(args, i, "--input")?));
            }
            "--model" => {
                i += 1;
                model = Some(PathBuf::from(require_value(args, i, "--model")?));
            }
            "--blocks" => {
                i += 1;
                blocks = Some(PathBuf::from(require_value(args, i, "--blocks")?));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(require_value(args, i, "--output")?));
            }
            "--threads" => {
                i += 1;
                let val = require_value(args, i, "--threads")?;
                threads = Some(val.parse::<usize>().map_err(|_| {
                    ClassifierError::Pipeline(format!(
                        "--threads must be a positive integer, got '{val}'"
                    ))
                })?);
            }
            "--fusion-radius" => {
                i += 1;
                let val = require_value(args, i, "--fusion-radius")?;
                fusion_radius = Some(val.parse::<f64>().map_err(|_| {
                    ClassifierError::Pipeline(format!(
                        "--fusion-radius must be a number, got '{val}'"
                    ))
                })?);
            }
            "--fusion-temp" => {
                i += 1;
                let val = require_value(args, i, "--fusion-temp")?;
                fusion_temp = Some(val.parse::<f64>().map_err(|_| {
                    ClassifierError::Pipeline(format!(
                        "--fusion-temp must be a number, got '{val}'"
                    ))
                })?);
            }
            unknown => {
                return Err(ClassifierError::Pipeline(format!(
                    "classify: unknown argument '{unknown}'"
                )));
            }
        }
        i += 1;
    }

    let cfg = ClassifyConfig {
        input: input
            .ok_or_else(|| ClassifierError::Pipeline("classify: --input is required".into()))?,
        model: model
            .ok_or_else(|| ClassifierError::Pipeline("classify: --model is required".into()))?,
        blocks: blocks
            .ok_or_else(|| ClassifierError::Pipeline("classify: --blocks is required".into()))?,
        output: output
            .ok_or_else(|| ClassifierError::Pipeline("classify: --output is required".into()))?,
        threads,
        fusion_radius,
        fusion_temp,
    };

    if let Some(t) = cfg.threads {
        if t == 0 {
            return Err(ClassifierError::Pipeline(
                "classify: --threads must be >= 1".to_string(),
            ));
        }
    }

    Ok(cfg)
}

fn require_value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("classify: {flag} requires a value")))
}

fn print_help() {
    eprintln!(
        "Usage: wb_lidar_classify classify [options]\n\
         \n\
         Options:\n\
           --input   <path>      LAS, LAZ, or COPC source file (required)\n\
           --model   <path>      Pre-trained .wbmodel weights file (required)\n\
           --blocks  <path>      blocks.json manifest from preprocess run (required)\n\
           --output  <path>      Classified output file (.las or .laz) (required)\n\
           --threads <n>         Rayon thread pool size (default: system cores)\n\
           --fusion-radius <f>   Cross-block prediction-fusion voting reach, in\n\
                                 projection units. Blocks within this distance of a\n\
                                 point also vote (confidence-weighted), smoothing\n\
                                 block-seam misclassifications. 0 disables fusion.\n\
                                 Default: the manifest's block_overlap when > 0,\n\
                                 else 0. Max: block_size/2.\n\
           --fusion-temp <f>     Softmax temperature applied per block before\n\
                                 voting (>1 softens, <1 sharpens; default: 1.0)\n\
           --help, -h            Show this message\n\
         \n\
         Note: --blocks must point to the blocks.json produced by running\n\
           `wb_lidar_classify preprocess` on the same --input file.\n\
         The .feat block files must exist in the same directory as blocks.json."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

// Test assertions compare against exact, exactly-representable constants
// (0.0, 1.0, 12.5, 25.0, …) produced by trivial parameter resolution —
// strict float equality is intentional and safe here.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::preprocessing::BlockManifest;

    fn manifest_with(block_size: f64, block_overlap: f64) -> BlockManifest {
        BlockManifest {
            source: "t.las".into(),
            block_size,
            target_points: 4,
            min_density: 1.0,
            search_radius: 1.0,
            min_neighbors: 1,
            crs_epsg: None,
            grid_cols: 1,
            grid_rows: 1,
            grid_x_min: 0.0,
            grid_y_min: 0.0,
            outlier_removal: false,
            outlier_radius: 2.0,
            outlier_elev_diff: 50.0,
            outlier_use_median: false,
            block_overlap,
            oversample_jitter: 0.0,
            z_norm_use_block_relative: false,
            halo_fraction: 0.0,
            blocks: vec![],
        }
    }

    fn base_args() -> Vec<String> {
        vec![
            "--input".into(),
            "in.las".into(),
            "--model".into(),
            "m.wbmodel".into(),
            "--blocks".into(),
            "blocks.json".into(),
            "--output".into(),
            "out.las".into(),
        ]
    }

    // ── DoD #8 — fusion radius resolution & validation ──────────────────────

    #[test]
    fn test_fusion_radius_defaults_to_zero_without_overlap() {
        let m = manifest_with(50.0, 0.0);
        assert_eq!(resolve_fusion_radius(None, &m).unwrap(), 0.0);
    }

    #[test]
    fn test_fusion_radius_defaults_to_manifest_block_overlap() {
        let m = manifest_with(50.0, 12.5);
        assert_eq!(resolve_fusion_radius(None, &m).unwrap(), 12.5);
    }

    #[test]
    fn test_fusion_radius_explicit_overrides_manifest() {
        let m = manifest_with(50.0, 12.5);
        assert_eq!(resolve_fusion_radius(Some(5.0), &m).unwrap(), 5.0);
        // Explicit 0 disables fusion even when the manifest has overlap.
        assert_eq!(resolve_fusion_radius(Some(0.0), &m).unwrap(), 0.0);
    }

    #[test]
    fn test_fusion_radius_rejects_negative_nan_and_too_large() {
        let m = manifest_with(50.0, 0.0);
        assert!(resolve_fusion_radius(Some(-1.0), &m).is_err());
        assert!(resolve_fusion_radius(Some(f64::NAN), &m).is_err());
        assert!(resolve_fusion_radius(Some(f64::INFINITY), &m).is_err());
        // block_size/2 = 25 is allowed; 25.001 is not.
        assert!(resolve_fusion_radius(Some(25.0), &m).is_ok());
        assert!(resolve_fusion_radius(Some(25.001), &m).is_err());
    }

    #[test]
    fn test_fusion_temp_validation() {
        assert_eq!(validate_fusion_temp(None).unwrap(), 1.0);
        assert_eq!(validate_fusion_temp(Some(2.5)).unwrap(), 2.5);
        assert!(validate_fusion_temp(Some(0.0)).is_err());
        assert!(validate_fusion_temp(Some(-0.5)).is_err());
        assert!(validate_fusion_temp(Some(f64::NAN)).is_err());
    }

    // ── argument parsing ────────────────────────────────────────────────────

    #[test]
    fn test_parse_args_ok_with_fusion_flags() {
        let mut args = base_args();
        args.push("--fusion-radius".into());
        args.push("10.5".into());
        args.push("--fusion-temp".into());
        args.push("0.8".into());
        let cfg = parse_args(&args).expect("parse must succeed");
        assert_eq!(cfg.fusion_radius, Some(10.5));
        assert_eq!(cfg.fusion_temp, Some(0.8));
    }

    #[test]
    fn test_parse_args_fusion_flags_default_none() {
        let cfg = parse_args(&base_args()).expect("parse must succeed");
        assert_eq!(cfg.fusion_radius, None);
        assert_eq!(cfg.fusion_temp, None);
    }

    #[test]
    fn test_parse_args_rejects_bad_fusion_values_and_unknown_flags() {
        let mut args = base_args();
        args.push("--fusion-radius".into());
        args.push("abc".into());
        assert!(parse_args(&args).is_err());

        let mut args = base_args();
        args.push("--bogus".into());
        assert!(parse_args(&args).is_err());
    }
}
