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
    use lidar_point_cloud_classifier::cli::{preprocess_labeled_cmd, train_cmd};

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "preprocess-labeled" => preprocess_labeled_cmd::run(&args[2..]),
        "train" => train_cmd::run(&args[2..]),
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
           train                Train a PointNet model → .wbmodel\n\
           help                 Show this message\n\
         \n\
         Run `wb_lidar_train <sub-command> --help` for sub-command options."
    );
}
