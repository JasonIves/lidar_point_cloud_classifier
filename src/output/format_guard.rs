//! Stage 46 — LAZ output integrity guard.
//!
//! The in-house `LASzip` encoder in this build of `wblidar` emits compressed
//! streams that reference `LASzip` decoders (`LAStools`, `laszip`,
//! `CloudCompare`, PDAL) reject mid-chunk.  An affected file appears to open but
//! only the points decoded before the failure carry valid coordinates, so the
//! cloud renders as a sparse scatter ("night sky").  Uncompressed LAS output
//! from the same run is verified correct.
//!
//! `whitebox_next_gen` is off-limits to this repository (AGENTS.md — "Greenfield
//! Only"), so the codec cannot be repaired here.  Instead this module makes the
//! broken path unreachable by default: a `.laz` (or `.copc`) `--output`
//! extension is redirected to a sibling `.las` path with a loud warning, unless
//! the caller explicitly opts back in with `--allow-laz`.
//!
//! See `docs/stages/stage-46-laz-output-integrity-guard.md` for the
//! specification and `docs/LAZ_CODEC_DEFECT_REPORT.md` for the upstream
//! analysis.
//!
//! This module is deliberately pure: it performs no filesystem access and never
//! panics, so every branch of the redirect rule is unit-testable.

use std::path::{Path, PathBuf};

use crate::error::{ClassifierError, Result};

/// Extension written when a compressed output request is redirected.
const UNCOMPRESSED_EXT: &str = "las";

/// Output extensions that route through the defective `LASzip` encoder.
///
/// COPC is included because a COPC payload *is* `LASzip`-chunked, so it shares
/// the same codec and the same defect.
const COMPRESSED_EXTS: [&str; 2] = ["laz", "copc"];

/// Outcome of applying the Stage 46 guard to a requested output path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOutput {
    /// The path that will actually be written.
    pub path: PathBuf,
    /// Whether the requested path's extension was overridden.
    pub redirected: bool,
}

/// True when `path`'s final extension routes through the defective LAZ codec.
///
/// Comparison is case-insensitive to stay consistent with
/// `wblidar::LidarFormat::detect`, which also lowercases before matching — a
/// `--output OUT.LAZ` on Windows must not slip past the guard.
fn is_compressed_output(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .is_some_and(|ext| COMPRESSED_EXTS.contains(&ext.as_str()))
}

/// Emit the guard warning banner to stderr.
///
/// `redirected_to` is `Some(actual_path)` when the request was overridden, or
/// `None` when `--allow-laz` forced the compressed write through.
fn warn_compressed_output(requested: &Path, redirected_to: Option<&Path>) {
    let actual = redirected_to.unwrap_or(requested);
    eprintln!("================================ WARNING ================================");
    eprintln!("Compressed LAZ output is DISABLED because the LAZ encoder in this build");
    eprintln!("of Whitebox Next Gen (wblidar) produces files that reference LASzip");
    eprintln!("decoders (LAStools, laszip, CloudCompare, PDAL) reject mid-stream.");
    eprintln!("Affected files appear to load but render as a sparse scatter of points;");
    eprintln!("all coordinates after the failure point are garbage.");
    eprintln!();
    eprintln!("  requested: {}", requested.display());
    eprintln!("  writing:   {}", actual.display());
    eprintln!();
    if redirected_to.is_some() {
        eprintln!("Your classified output is being written as uncompressed LAS instead,");
        eprintln!("which is fully valid. To obtain a .laz, compress the result with a");
        eprintln!("reference implementation, e.g.:");
        eprintln!();
        eprintln!("  laszip -i \"{}\"", actual.display());
        eprintln!();
        eprintln!("See docs/LAZ_CODEC_DEFECT_REPORT.md. Override with --allow-laz (NOT");
        eprintln!("recommended: the output will very likely be unreadable).");
    } else {
        eprintln!("--allow-laz was supplied: writing LAZ anyway. THE OUTPUT IS LIKELY");
        eprintln!("CORRUPT AND UNREADABLE BY OTHER TOOLS. Do not use it for analysis");
        eprintln!("or delivery. See docs/LAZ_CODEC_DEFECT_REPORT.md.");
    }
    eprintln!("=========================================================================");
}

/// Apply the Stage 46 guard to a requested `--output` path.
///
/// - `.las` (or any non-compressed extension) passes through untouched and
///   silently; format validation itself remains the writer's responsibility.
/// - `.laz` / `.copc` is redirected to a sibling `.las` path, with a warning.
/// - With `allow_laz`, a `.laz` request is honoured (with a sterner warning);
///   a `.copc` request is rejected outright, because `wblidar` exposes no COPC
///   writer and so there is nothing for the override to enable.
///
/// Emitting the warning is a deliberate side effect: the guard is a
/// user-visible safety behaviour, and keeping the message adjacent to the
/// decision prevents the two from drifting apart.
///
/// # Errors
/// Returns [`ClassifierError::Pipeline`] when `allow_laz` is set on a `.copc`
/// output, so the failure surfaces during argument resolution rather than after
/// a long inference run.
pub fn resolve_output_path(requested: &Path, allow_laz: bool) -> Result<ResolvedOutput> {
    if !is_compressed_output(requested) {
        return Ok(ResolvedOutput {
            path: requested.to_path_buf(),
            redirected: false,
        });
    }

    let is_copc = requested
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("copc"));

    if allow_laz {
        if is_copc {
            return Err(ClassifierError::Pipeline(format!(
                "classify: --allow-laz cannot enable COPC output for '{}': wblidar provides \
                 no COPC writer. Use a .las output path instead.",
                requested.display()
            )));
        }
        warn_compressed_output(requested, None);
        return Ok(ResolvedOutput {
            path: requested.to_path_buf(),
            redirected: false,
        });
    }

    let redirected = requested.with_extension(UNCOMPRESSED_EXT);
    warn_compressed_output(requested, Some(&redirected));
    Ok(ResolvedOutput {
        path: redirected,
        redirected: true,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DoD: .las passes through untouched ──────────────────────────────────

    #[test]
    fn test_las_output_is_untouched() {
        let out = resolve_output_path(Path::new("out.las"), false).expect("las must be accepted");
        assert_eq!(out.path, PathBuf::from("out.las"));
        assert!(!out.redirected, "a .las request must not be redirected");
    }

    #[test]
    fn test_las_output_untouched_with_allow_laz() {
        // The flag must not perturb an already-safe request.
        let out = resolve_output_path(Path::new("out.las"), true).expect("las must be accepted");
        assert_eq!(out.path, PathBuf::from("out.las"));
        assert!(!out.redirected);
    }

    // ── DoD: .laz is redirected to .las ─────────────────────────────────────

    #[test]
    fn test_laz_output_is_redirected_to_las() {
        let out = resolve_output_path(Path::new("out.laz"), false).expect("laz must be redirected");
        assert_eq!(out.path, PathBuf::from("out.las"));
        assert!(out.redirected, "a .laz request must report redirection");
    }

    #[test]
    fn test_laz_redirect_preserves_full_path_and_stem() {
        let out = resolve_output_path(Path::new("C:/data/classified/area51_out.laz"), false)
            .expect("laz must be redirected");
        assert_eq!(
            out.path,
            PathBuf::from("C:/data/classified/area51_out.las"),
            "directory and file stem must survive the redirect"
        );
    }

    /// `with_extension` replaces only the final extension — a dotted stem such
    /// as `area51.classified.laz` must not lose its `.classified` segment.
    #[test]
    fn test_laz_redirect_only_replaces_final_extension() {
        let out = resolve_output_path(Path::new("area51.classified.laz"), false)
            .expect("laz must be redirected");
        assert_eq!(out.path, PathBuf::from("area51.classified.las"));
    }

    // ── DoD: case-insensitivity ─────────────────────────────────────────────

    #[test]
    fn test_uppercase_laz_is_caught() {
        let out = resolve_output_path(Path::new("OUT.LAZ"), false).expect("laz must be redirected");
        assert_eq!(out.path, PathBuf::from("OUT.las"));
        assert!(out.redirected, "extension match must be case-insensitive");
    }

    #[test]
    fn test_mixed_case_las_is_untouched() {
        let out = resolve_output_path(Path::new("OUT.LAS"), false).expect("las must be accepted");
        assert_eq!(out.path, PathBuf::from("OUT.LAS"));
        assert!(!out.redirected);
    }

    // ── DoD: --allow-laz honours the request ────────────────────────────────

    #[test]
    fn test_allow_laz_honours_laz_request() {
        let out =
            resolve_output_path(Path::new("out.laz"), true).expect("override must be allowed");
        assert_eq!(out.path, PathBuf::from("out.laz"));
        assert!(
            !out.redirected,
            "an honoured request is not a redirected one"
        );
    }

    // ── DoD: COPC handling ──────────────────────────────────────────────────

    #[test]
    fn test_copc_is_redirected_to_las_without_flag() {
        let out =
            resolve_output_path(Path::new("out.copc"), false).expect("copc must be redirected");
        assert_eq!(out.path, PathBuf::from("out.las"));
        assert!(out.redirected);
    }

    #[test]
    fn test_copc_with_allow_laz_is_rejected() {
        let err = resolve_output_path(Path::new("out.copc"), true);
        assert!(
            err.is_err(),
            "--allow-laz must not pretend COPC writing is possible"
        );
    }

    // ── Non-LiDAR / extensionless paths are left to the writer ──────────────

    /// The guard is not a format validator: an unsupported extension passes
    /// through so `LidarFormat::detect` in `write_classified` produces the
    /// existing, well-tested error message rather than a duplicate one here.
    #[test]
    fn test_unrelated_extension_passes_through_unchanged() {
        let out = resolve_output_path(Path::new("out.txt"), false).expect("guard must not judge");
        assert_eq!(out.path, PathBuf::from("out.txt"));
        assert!(!out.redirected);
    }

    #[test]
    fn test_extensionless_path_passes_through_unchanged() {
        let out = resolve_output_path(Path::new("out"), false).expect("guard must not judge");
        assert_eq!(out.path, PathBuf::from("out"));
        assert!(!out.redirected);
    }
}
