//! Entry point — delegates entirely to the CLI module.

fn main() {
    if let Err(e) = lidar_point_cloud_classifier::cli::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
