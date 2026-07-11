//! Per-block 3-D k-d tree with adaptive radius neighbourhood search.
//!
//! The tree is built once from the full (unsampled) block point set so that
//! eigenvalue neighbourhoods use the original spatial distribution.  Sampled
//! point indices are queried against this tree during feature extraction.

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

use crate::preprocessing::lite_point::LitePoint;

/// Wrapper around `kdtree::KdTree<f64, usize, [f64; 3]>` that stores the
/// 3-D coordinates and exposes an adaptive-radius search.
pub struct BlockSpatialIndex {
    tree: KdTree<f64, usize, [f64; 3]>,
}

impl BlockSpatialIndex {
    /// Build a 3-D k-d tree from a slice of `LitePoint`s.
    ///
    /// Each point's index into `pts` is stored as the tree payload so callers
    /// can look up the original record after a neighbourhood query.
    #[must_use]
    pub fn build(pts: &[LitePoint]) -> Self {
        // bucket_size = 32 is a good default for medium-density LiDAR clouds.
        let mut tree: KdTree<f64, usize, [f64; 3]> = KdTree::with_capacity(3, pts.len());
        for (i, pt) in pts.iter().enumerate() {
            // `add` only errors when the point dimension doesn't match the tree
            // dimension, which can never happen here (fixed [f64; 3] key type).
            // However, NaN coordinates could theoretically cause a rejection, so
            // we log a warning instead of silently discarding — this surfaces
            // bad input during integration testing without panicking.
            if let Err(e) = tree.add([pt.x, pt.y, pt.z], i) {
                eprintln!(
                    "[warn] spatial_index: skipped point {i} \
                     (x={:.3}, y={:.3}, z={:.3}): {e}",
                    pt.x, pt.y, pt.z
                );
            }
        }
        Self { tree }
    }

    /// Return the indices of all points within `radius` of `center`.
    ///
    /// The kdtree crate uses *squared* Euclidean distance internally, so we
    /// square `radius` before querying.
    #[must_use]
    pub fn radius_search(&self, center: [f64; 3], radius: f64) -> Vec<usize> {
        let r2 = radius * radius;
        self.tree
            .within(&center, r2, &squared_euclidean)
            .unwrap_or_default()
            .into_iter()
            .map(|(_, &idx)| idx)
            .collect()
    }

    /// Adaptive-radius neighbourhood search.
    ///
    /// Starting at `base_radius`, the search radius is expanded in steps of
    /// `base_radius × 0.5` until at least `min_neighbors` points are found
    #[must_use]
    /// or the hard cap `base_radius × 4.0` is reached.
    ///
    /// Returns the indices of all neighbours found at the first radius that
    /// satisfies the `min_neighbors` requirement (or the cap-radius set if
    /// the requirement is never met).
    pub fn adaptive_radius_search(
        &self,
        center: [f64; 3],
        base_radius: f64,
        min_neighbors: usize,
    ) -> Vec<usize> {
        let max_radius = base_radius * 4.0;
        let step = base_radius * 0.5;
        let mut radius = base_radius;

        loop {
            let result = self.radius_search(center, radius);
            if result.len() >= min_neighbors || radius >= max_radius {
                return result;
            }
            radius = (radius + step).min(max_radius);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pts_from_coords(coords: &[(f64, f64, f64)]) -> Vec<LitePoint> {
        coords
            .iter()
            .map(|&(x, y, z)| LitePoint {
                x,
                y,
                z,
                ..LitePoint::default()
            })
            .collect()
    }

    /// Brute-force reference search for test verification.
    fn brute_radius(pts: &[LitePoint], center: [f64; 3], r: f64) -> Vec<usize> {
        pts.iter()
            .enumerate()
            .filter(|(_, p)| {
                let dx = p.x - center[0];
                let dy = p.y - center[1];
                let dz = p.z - center[2];
                dx * dx + dy * dy + dz * dz <= r * r
            })
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn test_radius_search_matches_brute_force() {
        let coords = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (5.0, 5.0, 5.0), // far point — should never be returned
        ];
        let pts = pts_from_coords(&coords);
        let idx = BlockSpatialIndex::build(&pts);

        let center = [0.0, 0.0, 0.0];
        let r = 1.5;

        let mut got = idx.radius_search(center, r);
        got.sort_unstable();

        let mut expected = brute_radius(&pts, center, r);
        expected.sort_unstable();

        assert_eq!(got, expected);
    }

    #[test]
    fn test_adaptive_radius_expands_when_needed() {
        // Place 10 points just beyond base_radius = 1.0, within 1.5.
        let far: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (1.1 + f64::from(i) * 0.01, 0.0, 0.0))
            .collect();
        let pts = pts_from_coords(&far);
        let idx = BlockSpatialIndex::build(&pts);

        // With base_radius = 1.0, min_neighbors = 5: no points found at 1.0,
        // first expansion step (1.5) should capture all 10.
        let result = idx.adaptive_radius_search([0.0, 0.0, 0.0], 1.0, 5);
        assert!(
            result.len() >= 5,
            "expected at least 5 neighbours after expansion, got {}",
            result.len()
        );
    }

    #[test]
    fn test_adaptive_radius_caps_at_4x() {
        // Single isolated point, far away — should return just that point or
        // the empty set when it's beyond 4× base_radius.
        let pts = pts_from_coords(&[(100.0, 0.0, 0.0)]);
        let idx = BlockSpatialIndex::build(&pts);

        // 4× base_radius (4.0) is nowhere near 100.0, so expect empty.
        let result = idx.adaptive_radius_search([0.0, 0.0, 0.0], 1.0, 5);
        assert_eq!(result.len(), 0);
    }
}
