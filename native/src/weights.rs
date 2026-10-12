//! Surface distances and screened diffusion avoid Euclidean limb leakage and hard label seams.
use crate::{anatomy::Skeleton, mesh::Surface};
use glam::Vec3;
use serde_json::{Value, json};
use std::{cmp::Ordering, collections::BinaryHeap};

pub struct SkinWeights {
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub report: Value,
}

#[derive(Copy, Clone)]
struct Visit {
    distance: f32,
    index: usize,
}
impl PartialEq for Visit {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.index == other.index
    }
}
impl Eq for Visit {}
impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + t * ab)
}
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn endpoint(s: &Skeleton, i: usize) -> Vec3 {
    let first = s
        .parents
        .iter()
        .enumerate()
        .find(|(_, p)| **p == i as i32)
        .map(|(j, _)| j);
    match first {
        Some(j) => s.joints[j],
        None => s.joints[i] + s.up * s.height * 0.015,
    }
}

pub fn solve(surface: &Surface, skeleton: &Skeleton) -> Result<SkinWeights, String> {
    let n = surface.points.len();
    let b = skeleton.joints.len();
    let h = skeleton.height;
    if b > u16::MAX as usize {
        return Err("Too many joints".into());
    }
    let names = &skeleton.names;
    let joint = |name: &str| {
        names
            .iter()
            .position(|n| n == name)
            .expect("Canonical anatomy names")
    };
    let hips = joint("Hips");
    let neck = joint("Neck1");
    let head = joint("Head");
    let hip_y = skeleton.joints[hips].dot(skeleton.up);
    let covered = skeleton.covers_legs;
    let garment_mask = crate::cloth::lower_garment_mask(surface, skeleton);
    let arm_support = crate::capsules::arm_support(surface, skeleton);
    let mut distances = vec![f32::INFINITY; n * b];
    for bone in 0..b {
        let a = skeleton.joints[bone];
        let end = endpoint(skeleton, bone);
        let mut heap = BinaryHeap::new();
        let direction = (end - a).normalize_or_zero();
        let radial_a = direction
            .cross(if direction.x.abs() < 0.8 {
                Vec3::X
            } else {
                Vec3::Y
            })
            .normalize();
        let radial_b = direction.cross(radial_a);
        for fraction in [0.15, 0.4, 0.65, 0.85] {
            let target = a.lerp(end, fraction);
            let mut sectors = [(usize::MAX, f32::INFINITY); 8];
            for (i, &point) in surface.points.iter().enumerate() {
                let delta = point - target;
                if delta.dot(direction).abs() > 0.055 * h {
                    continue;
                }
                let angle = delta.dot(radial_b).atan2(delta.dot(radial_a));
                let sector = (((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 8.0)
                    as usize)
                    .min(7);
                let distance = delta.length();
                if distance < sectors[sector].1 {
                    sectors[sector] = (i, distance);
                }
            }
            let nearest = sectors
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::INFINITY, f32::min);
            for (seed, distance) in sectors {
                if distance > 1.8 * nearest + 0.01 * h {
                    continue;
                }
                if seed != usize::MAX && distance < distances[seed * b + bone] {
                    distances[seed * b + bone] = distance;
                    heap.push(Visit {
                        distance,
                        index: seed,
                    });
                }
            }
        }
        while let Some(Visit { distance, index }) = heap.pop() {
            if distance > distances[index * b + bone] {
                continue;
            }
            for &(neighbor, length) in &surface.neighbors[index] {
                let next = distance + length;
                if next < distances[neighbor * b + bone] {
                    distances[neighbor * b + bone] = next;
                    heap.push(Visit {
                        distance: next,
                        index: neighbor,
                    });
                }
            }
        }
        // Disconnected accessories receive geometry-based support instead of zero weights.
        for (i, &point) in surface.points.iter().enumerate() {
            if !distances[i * b + bone].is_finite() {
                distances[i * b + bone] = segment_distance(point, a, end) + 0.03 * h;
            }
        }
    }
    let mut initial = vec![0.0; n * b];
    for (i, &p) in surface.points.iter().enumerate() {
        let min = distances[i * b..(i + 1) * b]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let mut total = 0.0;
        for bone in 0..b {
            let d = distances[i * b + bone] - min;
            let value = (-d / (0.026 * h)).exp();
            initial[i * b + bone] = value;
            total += value;
        }
        for v in &mut initial[i * b..(i + 1) * b] {
            *v /= total;
        }
        // Compare body volumes, not just bone axes, before smoothing the attachment field.
        for bone in 0..b {
            if names[bone].contains("Arm") || names[bone].contains("Hand") {
                initial[i * b + bone] *= arm_support[i];
            }
        }
        total = initial[i * b..(i + 1) * b].iter().sum();
        for v in &mut initial[i * b..(i + 1) * b] {
            *v /= total.max(1e-20);
        }
        let y = p.dot(skeleton.up);
        // Facial detail pieces and hair stay coherent with the cranium.
        if y > skeleton.joints[head].dot(skeleton.up) - 0.02 * h {
            let t = smoothstep(
                skeleton.joints[neck].dot(skeleton.up),
                skeleton.joints[head].dot(skeleton.up) + 0.04 * h,
                y,
            );
            for v in &mut initial[i * b..(i + 1) * b] {
                *v *= 1.0 - t;
            }
            initial[i * b + head] += t;
        }
        if covered && y < hip_y + 0.04 * h {
            let t = (1.0 - smoothstep(hip_y - 0.08 * h, hip_y + 0.04 * h, y)) * garment_mask[i];
            // Pelvis attachment preserves a continuous robe rather than splitting it between legs.
            for v in &mut initial[i * b..(i + 1) * b] {
                *v *= 1.0 - t;
            }
            initial[i * b + hips] += t;
        }
    }
    let (weights, diffusion_report) =
        crate::diffusion::solve(&surface.neighbors, &initial, b, h, 64.0)?;
    let mut welded_joints = vec![[0u16; 4]; n];
    let mut welded_weights = vec![[0.0f32; 4]; n];
    let mut error = 0.0f32;
    for i in 0..n {
        let mut order: Vec<_> = (0..b).collect();
        order.sort_unstable_by(|&a, &c| {
            weights[i * b + c]
                .total_cmp(&weights[i * b + a])
                .then_with(|| a.cmp(&c))
        });
        let sum: f32 = order[..4].iter().map(|&j| weights[i * b + j]).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(format!("Invalid skin weights at vertex {i}"));
        }
        for k in 0..4 {
            welded_joints[i][k] = order[k] as u16;
            welded_weights[i][k] = weights[i * b + order[k]] / sum;
        }
        error = error.max((welded_weights[i].iter().sum::<f32>() - 1.0).abs());
    }
    let joints = surface
        .original_to_welded
        .iter()
        .map(|&i| welded_joints[i])
        .collect();
    let weights = surface
        .original_to_welded
        .iter()
        .map(|&i| welded_weights[i])
        .collect();
    Ok(SkinWeights {
        joints,
        weights,
        report: json!({"method":"surface-geodesic-screened-diffusion","welded_vertices":n,"original_vertices":surface.original_to_welded.len(),"diffusion":diffusion_report,"continuous_lower_garment":covered,"max_weight_sum_error":error,"cloth_policy":if covered{"pelvis-attached continuous cloth; no cloth collision simulation"}else{"articulated limbs"}}),
    })
}
