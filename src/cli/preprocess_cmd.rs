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
//!     [--block-overlap  <f64>]   # default 0.0 (disabled)
//! ```

use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::{HagNormalization, PreprocessConfig, PreprocessingPipeline};

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

#[allow(clippy::too_many_lines)]
fn parse_args(args: &[String]) -> Result<PreprocessConfig> {
    let mut cfg = PreprocessConfig::default();
    let mut i = 0_usize;

    // HAG normalisation (Stage 37) is resolved after the parse loop so the
    // two flags interact deterministically regardless of order:
    // `--hag-norm-percentile` selects the legacy mode; otherwise `--hag-max`
    // (or its default) selects the fixed-absolute-metre mode.
    let mut hag_max = crate::preprocessing::DEFAULT_HAG_MAX_METERS;
    let mut hag_use_percentile = false;

    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                cfg.input = PathBuf::from(next_value(args, &mut i, "--input")?);
            }
            "--output" => {
                cfg.output_dir = PathBuf::from(next_value(args, &mut i, "--output")?);
            }
            "--block-size" => {
                cfg.block_size =
                    parse_f64(next_value(args, &mut i, "--block-size")?, "--block-size")?;
            }
            "--target-points" => {
                cfg.target_points = parse_usize(
                    next_value(args, &mut i, "--target-points")?,
                    "--target-points",
                )?;
            }
            "--min-density" => {
                cfg.min_density =
                    parse_f64(next_value(args, &mut i, "--min-density")?, "--min-density")?;
            }
            "--search-radius" => {
                cfg.search_radius = parse_f64(
                    next_value(args, &mut i, "--search-radius")?,
                    "--search-radius",
                )?;
            }
            "--min-neighbors" => {
                cfg.min_neighbors = parse_usize(
                    next_value(args, &mut i, "--min-neighbors")?,
                    "--min-neighbors",
                )?;
            }
            "--hag-model" => {
                cfg.hag_model = Some(PathBuf::from(next_value(args, &mut i, "--hag-model")?));
            }
            "--no-auto-dtm" => {
                // Invert: presence (or explicit `true`) disables auto-DTM.
                cfg.auto_dtm = !parse_optional_bool(args, &mut i, true);
            }
            "--dtm-resolution" => {
                cfg.auto_dtm_resolution = parse_f64(
                    next_value(args, &mut i, "--dtm-resolution")?,
                    "--dtm-resolution",
                )?;
            }
            "--keep-auto-dtm" => {
                cfg.keep_auto_dtm = parse_optional_bool(args, &mut i, true);
            }
            "--hag-max" => {
                hag_max = parse_f64(next_value(args, &mut i, "--hag-max")?, "--hag-max")?;
            }
            "--hag-norm-percentile" => {
                hag_use_percentile = parse_optional_bool(args, &mut i, true);
            }
            "--threads" => {
                cfg.threads = Some(parse_usize(
                    next_value(args, &mut i, "--threads")?,
                    "--threads",
                )?);
            }
            "--debug-csv" => {
                cfg.debug_csv = parse_optional_bool(args, &mut i, true);
            }
            "--outlier-removal" => {
                cfg.outlier_removal = parse_optional_bool(args, &mut i, true);
            }
            "--outlier-radius" => {
                cfg.outlier_radius = parse_f64(
                    next_value(args, &mut i, "--outlier-radius")?,
                    "--outlier-radius",
                )?;
            }
            "--outlier-elev-diff" => {
                cfg.outlier_elev_diff = parse_f64(
                    next_value(args, &mut i, "--outlier-elev-diff")?,
                    "--outlier-elev-diff",
                )?;
            }
            "--outlier-use-median" => {
                cfg.outlier_use_median = parse_optional_bool(args, &mut i, true);
            }
            "--block-overlap" => {
                cfg.block_overlap = parse_f64(
                    next_value(args, &mut i, "--block-overlap")?,
                    "--block-overlap",
                )?;
            }
            "--oversample-jitter" => {
                cfg.oversample_jitter = parse_f64(
                    next_value(args, &mut i, "--oversample-jitter")?,
                    "--oversample-jitter",
                )?;
            }
            "--eigen-memory-budget-mb" => {
                let mb = parse_usize(
                    next_value(args, &mut i, "--eigen-memory-budget-mb")?,
                    "--eigen-memory-budget-mb",
                )?;
                cfg.eigen_memory_budget_bytes = mb.saturating_mul(1024 * 1024);
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
        return Err(ClassifierError::Pipeline("--input is required".to_string()));
    }
    if cfg.output_dir.as_os_str().is_empty() {
        return Err(ClassifierError::Pipeline(
            "--output is required".to_string(),
        ));
    }

    // Range validation: catch pathological values before they cause silent
    // misbehaviour or confusing panics deep in the pipeline.
    if cfg.block_size <= 0.0 || !cfg.block_size.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--block-size must be a positive finite number".to_string(),
        ));
    }
    if cfg.target_points == 0 {
        return Err(ClassifierError::Pipeline(
            "--target-points must be >= 1".to_string(),
        ));
    }
    if cfg.min_density < 0.0 || !cfg.min_density.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--min-density must be >= 0.0 and finite".to_string(),
        ));
    }
    if cfg.search_radius <= 0.0 || !cfg.search_radius.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--search-radius must be a positive finite number".to_string(),
        ));
    }
    if cfg.min_neighbors == 0 {
        return Err(ClassifierError::Pipeline(
            "--min-neighbors must be >= 1".to_string(),
        ));
    }
    if cfg.outlier_radius <= 0.0 || !cfg.outlier_radius.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--outlier-radius must be a positive finite number".to_string(),
        ));
    }
    if cfg.outlier_elev_diff < 0.0 || !cfg.outlier_elev_diff.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--outlier-elev-diff must be a non-negative finite number".to_string(),
        ));
    }
    if cfg.block_overlap < 0.0 || !cfg.block_overlap.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--block-overlap must be >= 0.0 and finite".to_string(),
        ));
    }
    if cfg.block_overlap >= cfg.block_size {
        return Err(ClassifierError::Pipeline(
            "--block-overlap must be less than --block-size".to_string(),
        ));
    }
    if cfg.oversample_jitter < 0.0 || !cfg.oversample_jitter.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--oversample-jitter must be >= 0.0 and finite".to_string(),
        ));
    }
    if cfg.eigen_memory_budget_bytes == 0 {
        return Err(ClassifierError::Pipeline(
            "--eigen-memory-budget-mb must be >= 1".to_string(),
        ));
    }
    // Stage 38: auto-DTM resolution must be positive & finite, even when
    // auto-DTM is disabled or an external --hag-model overrides it, so that
    // obviously bad input is rejected early.
    if cfg.auto_dtm_resolution <= 0.0 || !cfg.auto_dtm_resolution.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--dtm-resolution must be a positive finite number".to_string(),
        ));
    }

    // ── HAG normalisation resolution + validation (Stage 37) ──────────────

    // `--hag-norm-percentile` selects the legacy per-block mode and takes
    // precedence over `--hag-max`; otherwise the fixed-absolute-metre mode
    // is used (default 50.0 m). `--hag-max` must be positive and finite even
    // when percentile mode is selected, to reject obviously bad input early.
    if hag_max <= 0.0 || !hag_max.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--hag-max must be a positive finite number".to_string(),
        ));
    }
    cfg.hag_normalization = if hag_use_percentile {
        HagNormalization::BlockPercentile99
    } else {
        HagNormalization::FixedMeters(hag_max)
    };

    Ok(cfg)
}

/// If the next token after a boolean flag is literally `"true"` or `"false"`,
/// consume it and return the parsed value.  Otherwise leave `i` unchanged and
/// return `default_val`.  This lets callers use either style:
///   `--outlier-removal`        (flag-only, sets true)
///   `--outlier-removal true`   (explicit value)
///   `--outlier-removal false`  (explicit disable)
fn parse_optional_bool(args: &[String], i: &mut usize, default_val: bool) -> bool {
    if let Some(next) = args.get(*i + 1) {
        match next.as_str() {
            "true" | "1" | "yes" => {
                *i += 1;
                return true;
            }
            "false" | "0" | "no" => {
                *i += 1;
                return false;
            }
            _ => {}
        }
    }
    default_val
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
           --search-radius <f64>   Base neighbourhood radius for eigenvalue features (default: 1.0)
           --min-neighbors <uint>  Minimum neighbours; radius expands adaptively (default: 8)\n\
           --hag-model     <path>  DTM raster for Height Above Ground (default: block-min-z proxy)\n\
           --hag-max       <f64>   Fixed absolute reference height (projection units) used to\n\
                                   normalise HAG into [0,1]; preserves absolute vertical scale\n\
                                   so identical physical heights map to identical features.\n\
                                   (default: 50.0 — Stage 37)\n\
           --hag-norm-percentile   Opt into the legacy per-block 99th-percentile HAG\n\
                                   normalisation (ignores --hag-max). Neighbour-dependent;\n\
                                   retained for reproducibility/comparison only.\n\
           --threads       <uint>  Rayon thread pool size (default: system cores)\n\
           --debug-csv             Also emit per-block .csv files alongside .feat files\n\
         \n\
         AUTOMATIC GROUND DTM (Stage 38 — enabled by default):\n\
           A bare-earth DTM is auto-generated from the input and used for HAG.\n\
           An explicit --hag-model always takes priority over auto-generation.\n\
           --dtm-resolution <f64>  Grid cell size (projection units) for the auto-generated\n\
                                   bare-earth DTM (default: 1.0)\n\
           --no-auto-dtm           Disable auto-DTM; fall back to the block-min-z HAG proxy\n\
                                   (unless --hag-model is given)\n\
           --keep-auto-dtm         Keep the intermediate _auto_dtm.tif / _auto_ground.las\n\
                                   files instead of deleting them after the run\n\
         \n\
         BLOCK OVERLAP (disabled by default):\n\
           --block-overlap   <f64>     Border-point context radius in projection units (default: 0.0)\n\
                                       Recommended: block-size / 2.  Must be < block-size.\n\
                                       Border points augment the k-d tree for edge-accurate features\n\
                                       but are never written to .feat output files.\n\
         \n\
         OUTLIER REMOVAL (disabled by default):\n\
           --outlier-removal           Enable lidar_remove_outliers pre-pass (whole-file)\n\
           --outlier-radius  <f64>     Neighbourhood radius for residual calculation (default: 2.0)\n\
           --outlier-elev-diff <f64>   Residual threshold; points exceeding this are removed (default: 50.0)\n\
           --outlier-use-median        Use neighbourhood median instead of mean\n\
         \n\
         JITTER-BASED OVERSAMPLING (disabled by default):\n\
           --oversample-jitter <f64>   Std-dev (projection units) of per-axis Gaussian jitter\n\
                                       applied to padding-only points when a block is\n\
                                       oversampled. Offsets are clipped to ±3σ. (default: 0.0)\n\
         \n\
         EIGENVALUE-FEATURE PRE-PASS MEMORY BUDGET:\n\
           --eigen-memory-budget-mb <uint>  Memory budget (MB) gating whether the whole-file\n\
                                       eigenvalue-feature pre-pass runs in one pass or is\n\
                                       split into memory-gated spatial strips (default: 2048)\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Vec<String> {
        vec![
            "--input".to_string(),
            "in.las".to_string(),
            "--output".to_string(),
            "out_dir".to_string(),
        ]
    }

    // Stage 38: auto-DTM is enabled by default with a 1.0 resolution.
    // Exact comparison against the `Default`-derived literal 1.0 constant;
    // no floating-point arithmetic occurs, so precision is not a concern.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_auto_dtm_defaults() {
        let cfg = parse_args(&base_args()).unwrap();
        assert!(cfg.auto_dtm, "auto-DTM must default on");
        assert_eq!(cfg.auto_dtm_resolution, 1.0);
        assert!(!cfg.keep_auto_dtm);
    }

    // Stage 38: `--no-auto-dtm` clears the flag.
    #[test]
    fn test_no_auto_dtm_disables() {
        let mut args = base_args();
        args.push("--no-auto-dtm".to_string());
        let cfg = parse_args(&args).unwrap();
        assert!(!cfg.auto_dtm, "--no-auto-dtm must disable auto-DTM");
    }

    // Stage 38: `--dtm-resolution` is parsed into `auto_dtm_resolution`.
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_dtm_resolution_parsed() {
        let mut args = base_args();
        args.push("--dtm-resolution".to_string());
        args.push("0.5".to_string());
        let cfg = parse_args(&args).unwrap();
        assert_eq!(cfg.auto_dtm_resolution, 0.5);
    }

    // Stage 38: `--keep-auto-dtm` retains intermediates.
    #[test]
    fn test_keep_auto_dtm_flag() {
        let mut args = base_args();
        args.push("--keep-auto-dtm".to_string());
        let cfg = parse_args(&args).unwrap();
        assert!(cfg.keep_auto_dtm);
    }

    // Stage 38: a non-positive `--dtm-resolution` is rejected.
    #[test]
    fn test_bad_dtm_resolution_rejected() {
        let mut args = base_args();
        args.push("--dtm-resolution".to_string());
        args.push("0".to_string());
        assert!(parse_args(&args).is_err());

        let mut neg = base_args();
        neg.push("--dtm-resolution".to_string());
        neg.push("-2.0".to_string());
        assert!(parse_args(&neg).is_err());
    }
}
