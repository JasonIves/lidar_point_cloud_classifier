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
<<<<<<< HEAD
    let mut search_radii:  Vec<f64>          = Vec::new();
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    let mut min_neighbors: usize             = 8;
    let mut hag_model:     Option<PathBuf>   = None;
    let mut label_map_path: Option<PathBuf>  = None;
    let mut tile_grid:     usize             = 4;
    let mut threads:       Option<usize>     = None;
    let mut debug_csv:     bool              = false;
<<<<<<< HEAD
    let mut outlier_removal:   bool  = false;
    let mut outlier_radius:    f64   = 2.0;
    let mut outlier_elev_diff: f64   = 50.0;
    let mut outlier_use_median: bool = false;
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input"          => { i += 1; input         = Some(PathBuf::from(&args[i])); }
            "--output"         => { i += 1; output        = Some(PathBuf::from(&args[i])); }
            "--block-size"     => { i += 1; block_size    = parse_f64(&args[i], "--block-size")?; }
            "--target-points"  => { i += 1; target_points = parse_usize(&args[i], "--target-points")?; }
            "--min-density"    => { i += 1; min_density   = parse_f64(&args[i], "--min-density")?; }
            "--search-radius"  => { i += 1; search_radius = parse_f64(&args[i], "--search-radius")?; }
<<<<<<< HEAD
            "--search-radii"   => {
                i += 1;
                search_radii = parse_radii(&args[i], "--search-radii")?;
            }
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
            "--min-neighbors"  => { i += 1; min_neighbors = parse_usize(&args[i], "--min-neighbors")?; }
            "--hag-model"      => { i += 1; hag_model     = Some(PathBuf::from(&args[i])); }
            "--label-map"      => { i += 1; label_map_path = Some(PathBuf::from(&args[i])); }
            "--tile-grid"      => { i += 1; tile_grid     = parse_usize(&args[i], "--tile-grid")?; }
            "--threads"        => { i += 1; threads       = Some(parse_usize(&args[i], "--threads")?); }
            "--debug-csv"      => { debug_csv = true; }
<<<<<<< HEAD
            "--outlier-removal"    => { outlier_removal = true; }
            "--outlier-radius"     => { i += 1; outlier_radius    = parse_f64(&args[i], "--outlier-radius")?; }
            "--outlier-elev-diff"  => { i += 1; outlier_elev_diff = parse_f64(&args[i], "--outlier-elev-diff")?; }
            "--outlier-use-median" => { outlier_use_median = true; }
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
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

<<<<<<< HEAD
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

=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
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
<<<<<<< HEAD
        search_radii,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
        min_neighbors,
        hag_model,
        threads,
        debug_csv,
<<<<<<< HEAD
        outlier_removal,
        outlier_radius,
        outlier_elev_diff,
        outlier_use_median,
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
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
<<<<<<< HEAD
           --search-radius <f64>    Base eigenvalue query radius (default: 1.0)
           --search-radii  <f64,..> Comma-separated radii for multi-scale features
                                    (overrides --search-radius when provided)\n\
=======
           --search-radius <f64>    Base eigenvalue query radius (default: 1.0)\n\
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
           --min-neighbors <usize>  Min neighbours for adaptive radius (default: 8)\n\
           --hag-model     <path>   DTM raster for Height Above Ground\n\
           --label-map     <path>   JSON ASPRS→class-index remapping\n\
           --tile-grid     <usize>  NxN macro-tile grid for spatial split (default: 4)\n\
           --threads       <usize>  Rayon thread pool size\n\
<<<<<<< HEAD
           --debug-csv              Also emit per-block .csv files

         Outlier removal (disabled by default):
           --outlier-removal           Enable outlier removal pre-pass (whole-file)
           --outlier-radius  <f64>     Neighbourhood radius for residual calc (default: 2.0)
           --outlier-elev-diff <f64>   Residual threshold; exceeding removes point (default: 50.0)
           --outlier-use-median        Use neighbourhood median instead of mean"
=======
           --debug-csv              Also emit per-block .csv files"
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
    );
}

fn parse_f64(s: &str, flag: &str) -> Result<f64> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid f64 '{s}'")))
}

fn parse_usize(s: &str, flag: &str) -> Result<usize> {
    s.parse().map_err(|_| ClassifierError::Pipeline(format!("{flag}: invalid usize '{s}'")))
}
<<<<<<< HEAD

fn parse_radii(s: &str, flag: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|v| {
            let v = v.trim();
            v.parse::<f64>().map_err(|_| ClassifierError::Pipeline(
                format!("{flag}: invalid radius '{v}'")
            ))
        })
        .collect()
}
=======
>>>>>>> cf241b7a93ef85c278c70d77292d38d1c3a9def4
