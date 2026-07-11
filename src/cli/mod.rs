//! CLI module — sub-command dispatch.

pub mod classify_cmd;
pub mod preprocess_cmd;

#[cfg(feature = "training")]
pub mod preprocess_labeled_cmd;
#[cfg(feature = "training")]
pub mod split_dataset_cmd;
#[cfg(feature = "training")]
pub mod train_cmd;

use crate::error::Result;

/// Top-level CLI entry point called from `main`.
///
/// Parses `std::env::args()` and dispatches to the appropriate sub-command.
///
/// # Errors
/// Propagates any error returned by the dispatched sub-command.
pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "preprocess" => preprocess_cmd::run(&args[2..]),
        "classify" => classify_cmd::run(&args[2..]),
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
        "Usage: wb_lidar_classify <sub-command> [options]\n\
         \n\
         Sub-commands:\n\
         \n\
           preprocess   Stream a LAS/LAZ/COPC file and produce .feat block files\n\
           classify     Run inference on .feat files and write classified LAS/LAZ\n\
           help         Show this message\n\
         \n\
         Run `wb_lidar_classify preprocess --help` for preprocessing options."
    );
}
