//! Surface-contour ownership distinguishes a continuous skirt from nearby feet.
//! A mask is a garment attachment policy, not a fabric or collision simulation.

use crate::{anatomy::Skeleton, mesh::Surface};
use glam::Vec3;
use std::collections::BTreeMap;

struct Cut {
    point: Vec3,
    a: usize,
    b: usize,
    fraction: f32,
}

#[derive(Default)]
struct Contour {
    low: f32,
    high: f32,
    perimeter: f32,
    segments: Vec<(usize, usize, usize)>,
}

struct Evidence {
    vertex_yes: Vec<f32>,
    vertex_no: Vec<f32>,
    face_yes: Vec<f32>,
    face_no: Vec<f32>,
}

fn memberships(yes: &[f32], no: &[f32]) -> Vec<f32> {
    yes.iter()
        .zip(no)
        .map(|(&yes, &no)| {
            let total = yes + no;
            if total > 1e-12 {
                (yes / total).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn find(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}

fn add_votes(
    surface: &Surface,
    up: Vec3,
    lateral: Vec3,
    plane: f32,
    center: f32,
    hip_width: f32,
    evidence: &mut Evidence,
) {
    let mut cuts: Vec<Cut> = Vec::new();
    let mut lookup = BTreeMap::new();
    let mut parents = Vec::new();
    let mut segments = Vec::new();
    for (face, tri) in surface.triangles.iter().enumerate() {
        let mut crossings = [0_usize; 3];
        let mut count = 0;
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let p = surface.points[a];
            let q = surface.points[b];
            let py = p.dot(up);
            let qy = q.dot(up);
            if (py > plane) == (qy > plane) {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let id = *lookup.entry(key).or_insert_with(|| {
                let fraction = (plane - py) / (qy - py);
                let id = cuts.len();
                cuts.push(Cut {
                    point: p.lerp(q, fraction),
                    a,
                    b,
                    fraction,
                });
                parents.push(id);
                id
            });
            crossings[count] = id;
            count += 1;
        }
        if count == 2 && crossings[0] != crossings[1] {
            let a = find(&mut parents, crossings[0]);
            let b = find(&mut parents, crossings[1]);
            if a != b {
                parents[a.max(b)] = a.min(b);
            }
            segments.push((crossings[0], crossings[1], face));
        }
    }
    let mut contours: BTreeMap<usize, Contour> = BTreeMap::new();
    for (a, b, face) in segments {
        let root = find(&mut parents, a);
        let contour = contours.entry(root).or_insert_with(|| Contour {
            low: f32::INFINITY,
            high: f32::NEG_INFINITY,
            ..Default::default()
        });
        let p = cuts[a].point;
        let q = cuts[b].point;
        contour.low = contour.low.min(p.dot(lateral)).min(q.dot(lateral));
        contour.high = contour.high.max(p.dot(lateral)).max(q.dot(lateral));
        contour.perimeter += p.distance(q);
        contour.segments.push((a, b, face));
    }
    for contour in contours.into_values() {
        // A garment surrounds both thighs. A foot/hand contour remains on one side.
        let garment = contour.low < center - 0.45 * hip_width
            && contour.high > center + 0.45 * hip_width
            && contour.high - contour.low > 1.4 * hip_width
            && contour.perimeter > 2.0 * hip_width;
        let (votes, face_votes) = if garment {
            (&mut evidence.vertex_yes, &mut evidence.face_yes)
        } else {
            (&mut evidence.vertex_no, &mut evidence.face_no)
        };
        for (a, b, face) in contour.segments {
            let mass = 0.5 * cuts[a].point.distance(cuts[b].point);
            face_votes[face] += 2.0 * mass;
            for id in [a, b] {
                let cut = &cuts[id];
                // Barycentric support assigns evidence to the true edge endpoints;
                // a long skirt triangle does not become a short foot-distance capsule.
                votes[cut.a] += mass * (1.0 - cut.fraction);
                votes[cut.b] += mass * cut.fraction;
            }
        }
    }
}

/// Return welded-vertex garment ownership in [0,1], zero for uncovered anatomy.
/// Feed this mask to pelvis attachment instead of testing initial foot influence.
/// Ordinary upper-body falloff and the weight solver still control the waist seam.
fn ownership(surface: &Surface, skeleton: &Skeleton) -> Evidence {
    let n = surface.points.len();
    let mut evidence = Evidence {
        vertex_yes: vec![0.0; n],
        vertex_no: vec![0.0; n],
        face_yes: vec![0.0; surface.triangles.len()],
        face_no: vec![0.0; surface.triangles.len()],
    };
    if !skeleton.covers_legs || n == 0 {
        return evidence;
    }
    let joint = |name: &str| skeleton.names.iter().position(|n| n == name);
    let (Some(hips), Some(left), Some(right)) =
        (joint("Hips"), joint("LeftLeg"), joint("RightLeg"))
    else {
        return evidence;
    };
    let up = skeleton.up;
    let lateral = up.cross(skeleton.forward).normalize_or_zero();
    let center = skeleton.joints[hips].dot(lateral);
    let hip_width = (skeleton.joints[left] - skeleton.joints[right])
        .dot(lateral)
        .abs()
        .max(0.06 * skeleton.height);
    let floor = surface
        .points
        .iter()
        .map(|p| p.dot(up))
        .fold(f32::INFINITY, f32::min);
    let top = skeleton.joints[hips].dot(up) + 0.025 * skeleton.height;
    let bottom = floor + 0.012 * skeleton.height;
    for k in 0..128 {
        let plane = bottom + (top - bottom) * (k as f32 + 0.413) / 128.0;
        add_votes(
            surface,
            up,
            lateral,
            plane,
            center,
            hip_width,
            &mut evidence,
        );
    }
    evidence
}

pub fn lower_garment_mask(surface: &Surface, skeleton: &Skeleton) -> Vec<f32> {
    let evidence = ownership(surface, skeleton);
    memberships(&evidence.vertex_yes, &evidence.vertex_no)
}

/// Per-source-triangle clothing roles for a topology seam, preserving vertex positions.
/// Contour evidence classifies faces directly; vertex votes only resolve unsampled caps.
/// A seam separates garment and articulated weight domains, not physical collision.
pub fn lower_garment_faces(
    points: &[Vec3],
    triangles: &[[usize; 3]],
    skeleton: &Skeleton,
) -> Vec<bool> {
    let surface = Surface::new(points, triangles);
    let evidence = ownership(&surface, skeleton);
    let face_mask = memberships(&evidence.face_yes, &evidence.face_no);
    let vertex_mask = memberships(&evidence.vertex_yes, &evidence.vertex_no);
    let mut welded_face = 0;
    triangles
        .iter()
        .map(|tri| {
            let ids = tri.map(|i| surface.original_to_welded[i]);
            if ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0] {
                return false;
            }
            let index = welded_face;
            welded_face += 1;
            let sampled = evidence.face_yes[index] + evidence.face_no[index] > 1e-12;
            if sampled {
                face_mask[index] > 0.5
            } else {
                ids.iter().map(|i| vertex_mask[*i]).sum::<f32>() / 3.0 > 0.5
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_tube(
        points: &mut Vec<Vec3>,
        triangles: &mut Vec<[usize; 3]>,
        center: Vec3,
        radii: Vec3,
        low: f32,
        high: f32,
    ) {
        let first = points.len();
        for y in [low, high] {
            for i in 0..16 {
                let a = std::f32::consts::TAU * i as f32 / 16.0;
                points.push(center + Vec3::new(radii.x * a.cos(), y, radii.z * a.sin()));
            }
        }
        for i in 0..16 {
            let next = (i + 1) % 16;
            triangles.push([first + i, first + next, first + i + 16]);
            triangles.push([first + next, first + next + 16, first + i + 16]);
        }
    }

    #[test]
    fn broad_skirt_is_preserved_while_nearby_shoes_and_hands_remain_articulated() {
        let mut points = Vec::new();
        let mut triangles = Vec::new();
        ring_tube(
            &mut points,
            &mut triangles,
            Vec3::ZERO,
            Vec3::new(0.14, 0.0, 0.07),
            0.06,
            0.54,
        );
        let garment_end = points.len();
        for side in [1.0, -1.0] {
            ring_tube(
                &mut points,
                &mut triangles,
                Vec3::new(side * 0.11, 0.0, 0.0),
                Vec3::new(0.035, 0.0, 0.055),
                0.0,
                0.07,
            );
            ring_tube(
                &mut points,
                &mut triangles,
                Vec3::new(side * 0.19, 0.0, 0.0),
                Vec3::new(0.025, 0.0, 0.025),
                0.40,
                0.55,
            );
        }
        let surface = Surface::new(&points, &triangles);
        let mut joints = vec![Vec3::ZERO; 27];
        joints[0] = Vec3::new(0.0, 0.54, 0.0);
        joints[12] = Vec3::new(0.055, 0.5, 0.0);
        joints[22] = Vec3::new(-0.055, 0.5, 0.0);
        let skeleton = Skeleton {
            names: crate::anatomy::NAMES.iter().map(|s| (*s).into()).collect(),
            parents: crate::anatomy::PARENTS.to_vec(),
            joints,
            forward: Vec3::Z,
            up: Vec3::Y,
            height: 1.0,
            notes: Vec::new(),
            covers_legs: true,
        };
        let mask = lower_garment_mask(&surface, &skeleton);
        for (i, &welded) in surface.original_to_welded.iter().enumerate() {
            if i < garment_end {
                assert!(mask[welded] > 0.999);
            } else {
                assert!(mask[welded] < 0.001);
            }
        }
    }
}
