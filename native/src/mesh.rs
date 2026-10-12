//! Deterministic seam welding and a compact surface-neighborhood graph.
use glam::Vec3;
use std::collections::BTreeMap;

pub struct Surface {
    pub points: Vec<Vec3>,
    pub triangles: Vec<[usize; 3]>,
    pub original_to_welded: Vec<usize>,
    pub neighbors: Vec<Vec<(usize, f32)>>,
}

impl Surface {
    pub fn new(points: &[Vec3], triangles: &[[usize; 3]]) -> Self {
        let min = points
            .iter()
            .fold(Vec3::splat(f32::INFINITY), |a, &p| a.min(p));
        let max = points
            .iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), |a, &p| a.max(p));
        let scale = (max - min).max_element().max(1e-6);
        let tolerance = scale * 1e-6;
        let mut map = BTreeMap::new();
        let mut welded = vec![];
        let mut ids = vec![];
        for &p in points {
            let key = p.to_array().map(|x| (x / tolerance).round() as i64);
            let id = *map.entry(key).or_insert_with(|| {
                let id = welded.len();
                welded.push(p);
                id
            });
            ids.push(id);
        }
        let triangles: Vec<_> = triangles
            .iter()
            .map(|t| t.map(|i| ids[i]))
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
            .collect();
        let mut edges = BTreeMap::new();
        for t in &triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                edges
                    .entry(key)
                    .or_insert_with(|| welded[a].distance(welded[b]));
            }
        }
        let mut neighbors = vec![vec![]; welded.len()];
        for ((a, b), distance) in edges {
            neighbors[a].push((b, distance));
            neighbors[b].push((a, distance));
        }
        for row in &mut neighbors {
            row.sort_unstable_by_key(|&(i, _)| i);
        }
        Self {
            points: welded,
            triangles,
            original_to_welded: ids,
            neighbors,
        }
    }
}
