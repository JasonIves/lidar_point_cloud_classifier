//! Entry point for `wb_lidar_train` — the training binary.
//!
//! Requires `--features training` to compile.

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> lidar_point_cloud_classifier::Result<()> {
    use lidar_point_cloud_classifier::cli::{
        evaluate_cmd, fix_label_map_cmd, preprocess_labeled_cmd, split_dataset_cmd, train_cmd,
    };

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "preprocess-labeled" => preprocess_labeled_cmd::run(&args[2..]),
        "split-dataset" => split_dataset_cmd::run(&args[2..]),
        "train" => train_cmd::run(&args[2..]),
        "evaluate" => evaluate_cmd::run(&args[2..]),
        // Stage 41 follow-up: minimal, standalone utility — deliberately
        // omitted from print_usage()'s sub-command list to avoid adding
        // permanent overhead to the primary documented toolset. See
        // docs/stages/stage-41-model-label-map-identity-bug-fix.md.
        "fix-label-map" => fix_label_map_cmd::run(&args[2..]),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        unknown => {
            eprintln!("Unknown sub-command: '{unknown}'");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: wb_lidar_train <sub-command> [options]\n\
         \n\
         Sub-commands:\n\
         \n\
           preprocess-labeled   Preprocess labeled LiDAR → .feat + .lbl blocks\n\
           split-dataset        Materialize a physical train/val/test directory split\n\
           train                Train a PointNet model → .wbmodel\n\
           evaluate             Score a trained model on a labeled held-out dir → metrics CSVs\n\
           help                 Show this message\n\
         \n\
         Run `wb_lidar_train <sub-command> --help` for sub-command options."
    );
}
