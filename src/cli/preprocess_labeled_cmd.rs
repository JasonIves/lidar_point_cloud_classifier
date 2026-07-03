//! `preprocess-labeled` sub-command — runs the labeled preprocessing pipeline.

#![allow(clippy::missing_errors_doc, clippy::too_many_lines)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::{run_labeled_pipeline, LabeledPreprocessConfig};
use crate::preprocessing::PreprocessConfig;

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut block_size: f64 = 50.0;
    let mut target_points: usize = 1024;
    let mut min_density: f64 = 1.0;
    let mut search_radius: f64 = 1.0;
    let mut search_radii: Vec<f64> = Vec::new();
    let mut min_neighbors: usize = 8;
    let mut hag_model: Option<PathBuf> = None;
    let mut label_map_path: Option<PathBuf> = None;
    let mut tile_grid: usize = 4;
    let mut threads: Option<usize> = None;
    let mut debug_csv: bool = false;
    let mut outlier_removal: bool = false;
    let mut outlier_radius: f64 = 2.0;
    let mut outlier_elev_diff: f64 = 50.0;
    let mut outlier_use_median: bool = false;
    let mut block_overlap: f64 = 0.0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                input = Some(PathBuf::from(next_value(args, &mut i, "--input")?));
            }
            "--output" => {
                output = Some(PathBuf::from(next_value(args, &mut i, "--output")?));
            }
            "--block-size" => {
                block_size = parse_f64(next_value(args, &mut i, "--block-size")?, "--block-size")?;
            }
            "--target-points" => {
                target_points = parse_usize(
                    next_value(args, &mut i, "--target-points")?,
                    "--target-points",
                )?;
            }
            "--min-density" => {
                min_density =
                    parse_f64(next_value(args, &mut i, "--min-density")?, "--min-density")?;
            }
            "--search-radius" => {
                search_radius = parse_f64(
                    next_value(args, &mut i, "--search-radius")?,
                    "--search-radius",
                )?;
            }
            "--search-radii" => {
                search_radii = parse_radii(
                    next_value(args, &mut i, "--search-radii")?,
                    "--search-radii",
                )?;
            }
            "--min-neighbors" => {
                min_neighbors = parse_usize(
                    next_value(args, &mut i, "--min-neighbors")?,
                    "--min-neighbors",
                )?;
            }
            "--hag-model" => {
                hag_model = Some(PathBuf::from(next_value(args, &mut i, "--hag-model")?));
            }
            "--label-map" => {
                label_map_path = Some(PathBuf::from(next_value(args, &mut i, "--label-map")?));
            }
            "--tile-grid" => {
                tile_grid = parse_usize(next_value(args, &mut i, "--tile-grid")?, "--tile-grid")?;
            }
            "--threads" => {
                threads = Some(parse_usize(
                    next_value(args, &mut i, "--threads")?,
                    "--threads",
                )?);
            }
            "--debug-csv" => {
                debug_csv = parse_optional_bool(args, &mut i, true);
            }
            "--outlier-removal" => {
                outlier_removal = parse_optional_bool(args, &mut i, true);
            }
            "--outlier-radius" => {
                outlier_radius = parse_f64(
                    next_value(args, &mut i, "--outlier-radius")?,
                    "--outlier-radius",
                )?;
            }
            "--outlier-elev-diff" => {
                outlier_elev_diff = parse_f64(
                    next_value(args, &mut i, "--outlier-elev-diff")?,
                    "--outlier-elev-diff",
                )?;
            }
            "--outlier-use-median" => {
                outlier_use_median = parse_optional_bool(args, &mut i, true);
            }
            "--block-overlap" => {
                block_overlap = parse_f64(
                    next_value(args, &mut i, "--block-overlap")?,
                    "--block-overlap",
                )?;
            }
            flag => {
                return Err(ClassifierError::Pipeline(format!(
                    "preprocess-labeled: unknown flag '{flag}'"
                )));
            }
        }
        i += 1;
    }

    let input = input.ok_or_else(|| ClassifierError::Pipeline("--input is required".into()))?;
    let output = output.ok_or_else(|| ClassifierError::Pipeline("--output is required".into()))?;

    // Range validation.
    if block_size <= 0.0 || !block_size.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--block-size must be a positive finite number".to_string(),
        ));
    }
    if target_points == 0 {
        return Err(ClassifierError::Pipeline(
            "--target-points must be >= 1".to_string(),
        ));
    }
    if min_density < 0.0 || !min_density.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--min-density must be >= 0.0 and finite".to_string(),
        ));
    }
    if search_radius <= 0.0 || !search_radius.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--search-radius must be a positive finite number".to_string(),
        ));
    }
    if min_neighbors == 0 {
        return Err(ClassifierError::Pipeline(
            "--min-neighbors must be >= 1".to_string(),
        ));
    }
    if tile_grid == 0 {
        return Err(ClassifierError::Pipeline(
            "--tile-grid must be >= 1".to_string(),
        ));
    }
    if let Some(t) = threads {
        if t == 0 {
            return Err(ClassifierError::Pipeline(
                "preprocess-labeled: --threads must be >= 1".to_string(),
            ));
        }
    }
    if outlier_radius <= 0.0 || !outlier_radius.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--outlier-radius must be a positive finite number".to_string(),
        ));
    }
    if outlier_elev_diff < 0.0 || !outlier_elev_diff.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--outlier-elev-diff must be a non-negative finite number".to_string(),
        ));
    }
    if block_overlap < 0.0 || !block_overlap.is_finite() {
        return Err(ClassifierError::Pipeline(
            "--block-overlap must be >= 0.0 and finite".to_string(),
        ));
    }
    if block_overlap >= block_size {
        return Err(ClassifierError::Pipeline(
            "--block-overlap must be less than --block-size".to_string(),
        ));
    }

    // Load label map from JSON file, or use default.
    let label_map: HashMap<u8, u8> = if let Some(ref p) = label_map_path {
        let f = std::fs::File::open(p)?;
        let raw: HashMap<String, u8> = serde_json::from_reader(f)
            .map_err(|e| ClassifierError::Pipeline(format!("label-map JSON parse: {e}")))?;
        raw.into_iter()
            .filter_map(|(k, v)| k.parse::<u8>().ok().map(|kk| (kk, v)))
            .collect()
    } else {
        LabeledPreprocessConfig::default_label_map()
    };

    let preprocess = PreprocessConfig {
        input,
        output_dir: output,
        block_size,
        target_points,
        min_density,
        search_radius,
        search_radii,
        min_neighbors,
        hag_model,
        threads,
        debug_csv,
        outlier_removal,
        outlier_radius,
        outlier_elev_diff,
        outlier_use_median,
        block_overlap,
    };

    let config = LabeledPreprocessConfig {
        preprocess,
        label_map,
        tile_grid,
    };
    run_labeled_pipeline(&config)?;

    Ok(())
}

fn print_usage() {
    eprintln!(
        "Usage: wb_lidar_train preprocess-labeled [options]\n\
         \n\
         Required:\n\
           --input   <path>   LAS/LAZ/COPC with ground-truth classification field\n\
           --output  <dir>    Output directory for .feat, .lbl, labeled_blocks.json\n\
         \n\
         Optional:\n\
           --block-size    <f64>    2D cell size in projection units (default: 50.0)\n\
           --target-points <usize>  Points per block after sampling (default: 1024)\n\
           --min-density   <f64>    Minimum pts/m² (default: 1.0)\n\
           --search-radius <f64>    Base eigenvalue query radius (default: 1.0)\n\
           --search-radii  <f64,..> Comma-separated radii for multi-scale features\n\
                                    (overrides --search-radius when provided)\n\
           --min-neighbors <usize>  Min neighbours for adaptive radius (default: 8)\n\
           --hag-model     <path>   DTM raster for Height Above Ground\n\
           --label-map     <path>   JSON ASPRS→class-index remapping\n\
           --tile-grid     <usize>  NxN macro-tile grid for spatial split (default: 4)\n\
           --threads       <usize>  Rayon thread pool size\n\
           --debug-csv              Also emit per-block .csv files\n\
         \n\
         Block overlap (disabled by default):\n\
           --block-overlap   <f64>     Border-point context radius in projection units (default: 0.0)\n\
                                       Recommended: block-size / 2.  Must be < block-size.\n\
         \n\
         Outlier removal (disabled by default):\n\
           --outlier-removal           Enable outlier removal pre-pass (whole-file)\n\
           --outlier-radius  <f64>     Neighbourhood radius for residual calc (default: 2.0)\n\
           --outlier-elev-diff <f64>   Residual threshold; exceeding removes point (default: 50.0)\n\
           --outlier-use-median        Use neighbourhood median instead of mean"
    );
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

/// If the flag at `args[*i]` requires a value, bounds-check and consume the
/// next token.  Returns a clear `ClassifierError::Pipeline` instead of
/// panicking (via unchecked indexing) if the flag is the last argument.
///
/// Stage 20 (Security Hardening) — mirrors the pattern already used in
/// `preprocess_cmd.rs`.
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

fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse()
        .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid usize '{s}'")))
}

fn parse_radii(s: &str, flag: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|v| {
            let v = v.trim();
            v.parse::<f64>()
                .map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid radius '{v}'")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // The full run() path with a dangling flag must error, not panic.
        let args: Vec<String> = vec!["--block-size".to_string()];
        let result = run(&args);
        assert!(result.is_err());
    }
}
