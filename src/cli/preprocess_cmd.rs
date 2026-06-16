//! `preprocess` sub-command — CLI argument parsing and pipeline invocation.
//!
//! Usage:
//! ```text
//! wb_lidar_classify preprocess \
//!     --input    <path>      \
//!     --output   <dir>       \
//!     [--block-size     <f64>]   # default 50.0
//!     [--target-points  <usize>] # default 1024
//!     [--min-density    <f64>]   # default 1.0 pts/m²
//!     [--search-radius  <f64>]   # default 1.0
//!     [--min-neighbors  <usize>] # default 8
//!     [--hag-model      <path>]
//!     [--threads        <usize>]
//!     [--debug-csv]
//! ```

use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::{PreprocessConfig, PreprocessingPipeline};

/// Parse `args` and run the preprocessing pipeline.
///
/// # Errors
/// Propagates any error from argument parsing or the preprocessing pipeline.
pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    let config = parse_args(args)?;
    let manifest = PreprocessingPipeline::run(&config)?;

    eprintln!(
        "[preprocess] done — {} blocks written to {}",
        manifest.blocks.len(),
        config.output_dir.display()
    );
    Ok(())
}

// ── Argument parsing ──────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<PreprocessConfig> {
    let mut cfg = PreprocessConfig::default();
    let mut i = 0_usize;

    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                cfg.input = PathBuf::from(next_value(args, &mut i, "--input")?);
            }
            "--output" => {
                cfg.output_dir = PathBuf::from(next_value(args, &mut i, "--output")?);
            }
            "--block-size" => {
                cfg.block_size = parse_f64(next_value(args, &mut i, "--block-size")?, "--block-size")?;
            }
            "--target-points" => {
                cfg.target_points = parse_usize(next_value(args, &mut i, "--target-points")?, "--target-points")?;
            }
            "--min-density" => {
                cfg.min_density = parse_f64(next_value(args, &mut i, "--min-density")?, "--min-density")?;
            }
            "--search-radius" => {
                cfg.search_radius = parse_f64(next_value(args, &mut i, "--search-radius")?, "--search-radius")?;
            }
            "--min-neighbors" => {
                cfg.min_neighbors = parse_usize(next_value(args, &mut i, "--min-neighbors")?, "--min-neighbors")?;
            }
            "--hag-model" => {
                cfg.hag_model = Some(PathBuf::from(next_value(args, &mut i, "--hag-model")?));
            }
            "--threads" => {
                cfg.threads = Some(parse_usize(next_value(args, &mut i, "--threads")?, "--threads")?);
            }
            "--debug-csv" => {
                cfg.debug_csv = true;
            }
            unknown => {
                return Err(ClassifierError::Pipeline(format!(
                    "unknown argument: '{unknown}'"
                )));
            }
        }
        i += 1;
    }

    // Validate required arguments.
    if cfg.input.as_os_str().is_empty() {
        return Err(ClassifierError::Pipeline(
            "--input is required".to_string(),
        ));
    }
    if cfg.output_dir.as_os_str().is_empty() {
        return Err(ClassifierError::Pipeline(
            "--output is required".to_string(),
        ));
    }

    Ok(cfg)
}

fn next_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| ClassifierError::Pipeline(format!("flag '{flag}' requires a value")))
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse::<f64>().map_err(|_| {
        ClassifierError::Pipeline(format!("'{flag}' expects a decimal number, got '{s}'"))
    })
}

fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse::<usize>().map_err(|_| {
        ClassifierError::Pipeline(format!("'{flag}' expects a positive integer, got '{s}'"))
    })
}

fn print_help() {
    eprintln!(
        "wb_lidar_classify preprocess — Spatial preprocessing pipeline\n\
         \n\
         REQUIRED:\n\
           --input  <path>      LAS, LAZ, or COPC input file\n\
           --output <dir>       Output directory for .feat blocks and blocks.json\n\
         \n\
         OPTIONAL:\n\
           --block-size    <f64>   2-D cell edge length in projection units (default: 50.0)\n\
           --target-points <uint>  Points per block after sampling (default: 1024)\n\
           --min-density   <f64>   Minimum pts/m² to retain a block (default: 1.0)\n\
           --search-radius <f64>   Base neighbourhood radius for eigenvalue features (default: 1.0)\n\
           --min-neighbors <uint>  Minimum neighbours; radius expands adaptively (default: 8)\n\
           --hag-model     <path>  DTM raster for Height Above Ground (default: block-min-z proxy)\n\
           --threads       <uint>  Rayon thread pool size (default: system cores)\n\
           --debug-csv             Also emit per-block .csv files alongside .feat files\n"
    );
}
