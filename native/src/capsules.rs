//! Radius-aware support prevents a nearby diagonal arm axis from capturing the waist.
//! This is an initial prior; surface diffusion must run after applying it.

use crate::{
    anatomy::{self, Skeleton},
    mesh::Surface,
};
use glam::Vec3;

#[derive(Clone, Copy)]
struct ArmCapsule {
    a: Vec3,
    b: Vec3,
    radius: f32,
}

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn segment_distance(point: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let t = ((point - a).dot(ab) / ab.length_squared().max(1e-20)).clamp(0.0, 1.0);
    (point.distance(a + t * ab), t)
}

fn interpolate(profile: &[[f32; 5]], y: f32) -> [f32; 5] {
    let upper = profile.partition_point(|slice| slice[0] < y);
    if upper == 0 {
        return profile[0];
    }
    if upper == profile.len() {
        return profile[upper - 1];
    }
    let a = profile[upper - 1];
    let b = profile[upper];
    let t = ((y - a[0]) / (b[0] - a[0]).max(1e-20)).clamp(0.0, 1.0);
    std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
}

fn torso_distance(point: Vec3, slice: [f32; 5]) -> f32 {
    let rx = slice[3].max(1e-20);
    let rz = slice[4].max(1e-20);
    let x = (point.x - slice[1]) / rx;
    let z = (point.z - slice[2]) / rz;
    // An approximate signed distance is sufficient for a smoothly varying prior.
    ((x * x + z * z).sqrt() - 1.0) * rx.min(rz)
}

fn median(values: &mut [f32]) -> f32 {
    let middle = values.len() / 2;
    values.select_nth_unstable_by(middle, f32::total_cmp);
    values[middle]
}

pub fn arm_support(surface: &Surface, skeleton: &Skeleton) -> Vec<f32> {
    let h = skeleton.height;
    let find = |name: &str| {
        skeleton
            .names
            .iter()
            .position(|candidate| candidate == name)
    };
    let (Some(hips), Some(left_arm), Some(right_arm)) =
        (find("Hips"), find("LeftArm"), find("RightArm"))
    else {
        return vec![1.0; surface.points.len()];
    };
    let up = skeleton.up;
    let forward = skeleton.forward;
    let lateral = up.cross(forward).normalize();
    let project = |point: Vec3| Vec3::new(point.dot(lateral), point.dot(up), point.dot(forward));
    let hips_y = skeleton.joints[hips].dot(up);
    let shoulder_y = skeleton.joints[left_arm]
        .dot(up)
        .min(skeleton.joints[right_arm].dot(up));
    let min_y = hips_y - 0.04 * h;
    let max_y = shoulder_y - 0.04 * h;
    if max_y <= min_y + 0.04 * h {
        return vec![1.0; surface.points.len()];
    }
    let profile = anatomy::torso_profile(
        &surface.points,
        &surface.triangles,
        up,
        forward,
        min_y,
        max_y,
        33,
    );
    if profile.len() < 5 {
        return vec![1.0; surface.points.len()];
    }
    let projected: Vec<_> = surface.points.iter().copied().map(project).collect();
    let torso: Vec<_> = projected
        .iter()
        .map(|&point| torso_distance(point, interpolate(&profile, point.y)))
        .collect();
    let mut capsules = Vec::with_capacity(6);
    for side in ["Left", "Right"] {
        for (bone, child) in [("Arm", "ForeArm"), ("ForeArm", "Hand"), ("Hand", "HandEnd")] {
            let (Some(a), Some(b)) = (
                find(&format!("{side}{bone}")),
                find(&format!("{side}{child}")),
            ) else {
                continue;
            };
            let (a, b) = (skeleton.joints[a], skeleton.joints[b]);
            let mut radii = Vec::new();
            for (i, &point) in surface.points.iter().enumerate() {
                let (distance, t) = segment_distance(point, a, b);
                if t > 0.15 && t < 0.85 && distance < 0.09 * h && torso[i] > 0.01 * h {
                    radii.push(distance);
                }
            }
            let radius = if radii.len() >= 10 {
                median(&mut radii)
            } else {
                0.03 * h
            };
            capsules.push(ArmCapsule {
                a,
                b,
                radius: radius.clamp(0.012 * h, 0.065 * h),
            });
        }
    }
    if capsules.is_empty() {
        return vec![1.0; surface.points.len()];
    }
    surface
        .points
        .iter()
        .enumerate()
        .map(|(i, &point)| {
            let y = projected[i].y;
            if y <= min_y || y >= max_y {
                return 1.0;
            }
            let arm_distance = capsules
                .iter()
                .map(|capsule| segment_distance(point, capsule.a, capsule.b).0 - capsule.radius)
                .fold(f32::INFINITY, f32::min);
            let support = smoothstep(-0.01 * h, 0.04 * h, torso[i] - arm_distance);
            // The bounded waist/rib prior fades before shoulder cuts merge with deltoids.
            let body_presence = smoothstep(min_y, hips_y + 0.02 * h, y)
                * (1.0 - smoothstep(shoulder_y - 0.12 * h, max_y, y));
            (1.0 - body_presence * (1.0 - support)).clamp(0.0, 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> (Surface, Skeleton) {
        let mut points = Vec::new();
        let mut triangles = Vec::new();
        for row in 0..17 {
            let y = 0.46 + row as f32 * 0.02;
            for sector in 0..32 {
                let theta = sector as f32 * std::f32::consts::TAU / 32.0;
                points.push(Vec3::new(0.14 * theta.cos(), y, 0.10 * theta.sin()));
                if row > 0 {
                    let a = (row - 1) * 32 + sector;
                    let b = (row - 1) * 32 + (sector + 1) % 32;
                    let c = row * 32 + sector;
                    let d = row * 32 + (sector + 1) % 32;
                    triangles.extend([[a, b, c], [b, d, c]]);
                }
            }
        }
        let mut joints = vec![Vec3::ZERO; anatomy::NAMES.len()];
        joints[0] = Vec3::new(0.0, 0.53, 0.0);
        joints[1] = Vec3::new(0.0, 0.57, 0.0);
        joints[2] = Vec3::new(0.0, 0.64, 0.0);
        joints[3] = Vec3::new(0.0, 0.73, 0.0);
        joints[4] = Vec3::new(0.0, 0.84, 0.0);
        for (base, sign) in [(7, 1.0), (17, -1.0)] {
            joints[base] = Vec3::new(sign * 0.025, 0.77, 0.0);
            joints[base + 1] = Vec3::new(sign * 0.15, 0.78, 0.0);
            joints[base + 2] = Vec3::new(sign * 0.30, 0.63, 0.0);
            joints[base + 3] = Vec3::new(sign * 0.40, 0.50, 0.0);
            joints[base + 4] = Vec3::new(sign * 0.43, 0.46, 0.0);
        }
        // Probe vertices are independent accessories; a genuine arm stays supported.
        points.extend([
            Vec3::new(0.30, 0.63, 0.02),
            Vec3::new(-0.30, 0.63, 0.02),
            Vec3::new(0.1, 0.9, 0.0),
        ]);
        let surface = Surface::new(&points, &triangles);
        let skeleton = Skeleton {
            names: anatomy::NAMES.iter().map(|name| name.to_string()).collect(),
            parents: anatomy::PARENTS.to_vec(),
            joints,
            forward: Vec3::Z,
            up: Vec3::Y,
            height: 1.0,
            notes: vec![],
            covers_legs: false,
        };
        (surface, skeleton)
    }

    #[test]
    fn elliptic_torso_surface_has_zero_signed_distance() {
        let slice = [0.6, 0.0, 0.0, 0.14, 0.10];
        assert!(torso_distance(Vec3::new(0.14, 0.6, 0.0), slice).abs() < 1e-7);
        assert!(torso_distance(Vec3::new(0.0, 0.6, 0.0), slice) < 0.0);
    }

    #[test]
    fn waist_is_protected_but_arm_and_head_are_unrestricted() {
        let (surface, skeleton) = body();
        let support = arm_support(&surface, &skeleton);
        let waist = surface
            .points
            .iter()
            .position(|p| p.distance(Vec3::new(0.14, 0.60, 0.0)) < 1e-5)
            .unwrap();
        assert!(support[waist] < 0.1, "waist support {}", support[waist]);
        assert_eq!(support[support.len() - 2], 1.0);
        assert_eq!(support[support.len() - 1], 1.0);
        assert!(
            support
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
    }

    #[test]
    fn empty_geometry_does_not_suppress_arms() {
        let (mut surface, skeleton) = body();
        surface.triangles.clear();
        assert!(
            arm_support(&surface, &skeleton)
                .iter()
                .all(|&value| value == 1.0)
        );
    }
}
