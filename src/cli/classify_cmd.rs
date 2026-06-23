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
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{ClassifierError, Result};
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

    // ── Run per-block inference ────────────────────────────────────────────
    eprintln!("[classify] running inference on {} blocks…", manifest.blocks.len());
    let inference_map = run_inference(&manifest, &model, &feat_dir)?;
    eprintln!("[classify] inference complete ({} blocks processed)", inference_map.len());

    // ── Write classified output ────────────────────────────────────────────
    eprintln!("[classify] writing output: {}", cfg.output.display());
    write_classified(&cfg.input, &cfg.output, &inference_map, &manifest)?;

    Ok(())
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<ClassifyConfig> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(0);
    }

    let mut input:   Option<PathBuf> = None;
    let mut model:   Option<PathBuf> = None;
    let mut blocks:  Option<PathBuf> = None;
    let mut output:  Option<PathBuf> = None;
    let mut threads: Option<usize>   = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"   => { i += 1; input   = Some(PathBuf::from(require_value(args, i, "--input")?)); }
            "--model"   => { i += 1; model   = Some(PathBuf::from(require_value(args, i, "--model")?)); }
            "--blocks"  => { i += 1; blocks  = Some(PathBuf::from(require_value(args, i, "--blocks")?)); }
            "--output"  => { i += 1; output  = Some(PathBuf::from(require_value(args, i, "--output")?)); }
            "--threads" => {
                i += 1;
                let val = require_value(args, i, "--threads")?;
                threads = Some(val.parse::<usize>().map_err(|_| {
                    ClassifierError::Pipeline(format!("--threads must be a positive integer, got '{val}'"))
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
        input:   input.ok_or_else(|| ClassifierError::Pipeline("classify: --input is required".into()))?,
        model:   model.ok_or_else(|| ClassifierError::Pipeline("classify: --model is required".into()))?,
        blocks:  blocks.ok_or_else(|| ClassifierError::Pipeline("classify: --blocks is required".into()))?,
        output:  output.ok_or_else(|| ClassifierError::Pipeline("classify: --output is required".into()))?,
        threads,
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
    args.get(i).map(String::as_str).ok_or_else(|| {
        ClassifierError::Pipeline(format!("classify: {flag} requires a value"))
    })
}

fn print_help() {
    eprintln!(
        "Usage: wb_lidar_classify classify [options]\n\
         \n\
         Options:\n\
           --input   <path>   LAS, LAZ, or COPC source file (required)\n\
           --model   <path>   Pre-trained .wbmodel weights file (required)\n\
           --blocks  <path>   blocks.json manifest from preprocess run (required)\n\
           --output  <path>   Classified output file (.las or .laz) (required)\n\
           --threads <n>      Rayon thread pool size (default: system cores)\n\
           --help, -h         Show this message\n\
         \n\
         Note: --blocks must point to the blocks.json produced by running\n\
           `wb_lidar_classify preprocess` on the same --input file.\n\
         The .feat block files must exist in the same directory as blocks.json."
    );
}
