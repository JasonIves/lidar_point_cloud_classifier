//! `preprocess-labeled` sub-command — runs the labeled preprocessing pipeline.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{ClassifierError, Result};
use crate::preprocessing::labeled_pipeline::{LabeledPreprocessConfig, run_labeled_pipeline};
use crate::preprocessing::PreprocessConfig;

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let mut input:         Option<PathBuf>   = None;
    let mut output:        Option<PathBuf>   = None;
    let mut block_size:    f64               = 50.0;
    let mut target_points: usize             = 1024;
    let mut min_density:   f64               = 1.0;
    let mut search_radius: f64               = 1.0;
    let mut min_neighbors: usize             = 8;
    let mut hag_model:     Option<PathBuf>   = None;
    let mut label_map_path: Option<PathBuf>  = None;
    let mut tile_grid:     usize             = 4;
    let mut threads:       Option<usize>     = None;
    let mut debug_csv:     bool              = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"          => { i += 1; input         = Some(PathBuf::from(&args[i])); }
            "--output"         => { i += 1; output        = Some(PathBuf::from(&args[i])); }
            "--block-size"     => { i += 1; block_size    = parse_f64(&args[i], "--block-size")?; }
            "--target-points"  => { i += 1; target_points = parse_usize(&args[i], "--target-points")?; }
            "--min-density"    => { i += 1; min_density   = parse_f64(&args[i], "--min-density")?; }
            "--search-radius"  => { i += 1; search_radius = parse_f64(&args[i], "--search-radius")?; }
            "--min-neighbors"  => { i += 1; min_neighbors = parse_usize(&args[i], "--min-neighbors")?; }
            "--hag-model"      => { i += 1; hag_model     = Some(PathBuf::from(&args[i])); }
            "--label-map"      => { i += 1; label_map_path = Some(PathBuf::from(&args[i])); }
            "--tile-grid"      => { i += 1; tile_grid     = parse_usize(&args[i], "--tile-grid")?; }
            "--threads"        => { i += 1; threads       = Some(parse_usize(&args[i], "--threads")?); }
            "--debug-csv"      => { debug_csv = true; }
            flag => {
                return Err(ClassifierError::Pipeline(format!(
                    "preprocess-labeled: unknown flag '{flag}'"
                )));
            }
        }
        i += 1;
    }

    let input  = input.ok_or_else(|| ClassifierError::Pipeline("--input is required".into()))?;
    let output = output.ok_or_else(|| ClassifierError::Pipeline("--output is required".into()))?;

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
        min_neighbors,
        hag_model,
        threads,
        debug_csv,
    };

    let config = LabeledPreprocessConfig { preprocess, label_map, tile_grid };
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
           --min-neighbors <usize>  Min neighbours for adaptive radius (default: 8)\n\
           --hag-model     <path>   DTM raster for Height Above Ground\n\
           --label-map     <path>   JSON ASPRS→class-index remapping\n\
           --tile-grid     <usize>  NxN macro-tile grid for spatial split (default: 4)\n\
           --threads       <usize>  Rayon thread pool size\n\
           --debug-csv              Also emit per-block .csv files"
    );
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f64 '{s}'")))
}

fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid usize '{s}'")))
}
