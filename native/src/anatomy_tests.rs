use super::*;
use glam::Quat;

#[derive(Default)]
struct Fixture {
    points: Vec<Vec3>,
    triangles: Vec<[usize; 3]>,
}

impl Fixture {
    fn ellipsoid(&mut self, center: Vec3, radii: Vec3) {
        let top = self.points.len();
        self.points.push(center + Vec3::Y * radii.y);
        let rings = 8;
        let count = 16;
        for ring in 1..rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            for i in 0..count {
                let theta = std::f32::consts::TAU * i as f32 / count as f32;
                self.points.push(
                    center
                        + radii
                            * Vec3::new(
                                phi.sin() * theta.cos(),
                                phi.cos(),
                                phi.sin() * theta.sin(),
                            ),
                );
            }
        }
        let bottom = self.points.len();
        self.points.push(center - Vec3::Y * radii.y);
        for i in 0..count {
            let next = (i + 1) % count;
            self.triangles.push([top, top + 1 + i, top + 1 + next]);
            self.triangles
                .push([bottom, bottom - count + next, bottom - count + i]);
        }
        for ring in 0..rings - 2 {
            for i in 0..count {
                let next = (i + 1) % count;
                let a = top + 1 + ring * count + i;
                let b = top + 1 + ring * count + next;
                self.triangles.push([a, a + count, b]);
                self.triangles.push([b, a + count, b + count]);
            }
        }
    }

    fn tube(&mut self, start: Vec3, end: Vec3, radius: f32) {
        let first = self.points.len();
        let axis = (end - start).normalize();
        let a = orthogonal(axis);
        let b = axis.cross(a);
        let count = 12;
        let rings = 8;
        for ring in 0..rings {
            let center = start.lerp(end, ring as f32 / (rings - 1) as f32);
            for i in 0..count {
                let theta = std::f32::consts::TAU * i as f32 / count as f32;
                self.points
                    .push(center + radius * (a * theta.cos() + b * theta.sin()));
            }
        }
        for ring in 0..rings - 1 {
            for i in 0..count {
                let next = (i + 1) % count;
                let a = first + ring * count + i;
                let b = first + ring * count + next;
                self.triangles.push([a, a + count, b]);
                self.triangles.push([b, a + count, b + count]);
            }
        }
    }

    fn humanoid() -> Self {
        let mut f = Self::default();
        f.ellipsoid(Vec3::new(0.0, 0.65, 0.0), Vec3::new(0.15, 0.20, 0.095));
        f.ellipsoid(Vec3::new(0.0, 0.925, 0.0), Vec3::new(0.085, 0.075, 0.075));
        f.tube(Vec3::new(0.0, 0.80, 0.0), Vec3::new(0.0, 0.88, 0.0), 0.035);
        for side in [1.0, -1.0] {
            f.tube(
                Vec3::new(side * 0.075, 0.51, 0.0),
                Vec3::new(side * 0.105, 0.065, 0.0),
                0.042,
            );
            f.ellipsoid(
                Vec3::new(side * 0.105, 0.032, 0.055),
                Vec3::new(0.048, 0.032, 0.11),
            );
            f.tube(
                Vec3::new(side * 0.125, 0.78, 0.0),
                Vec3::new(side * 0.425, 0.46, 0.0),
                0.03,
            );
            f.ellipsoid(Vec3::new(side * 0.44, 0.445, 0.0), Vec3::splat(0.032));
        }
        f
    }
}

#[test]
fn plane_cut_interpolates_real_surface_instead_of_using_slab_endpoints() {
    let p = [
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(2.0, 2.0, 2.0),
    ];
    let cut = section(&p, &[[0, 1, 2]], 1.0);
    assert_eq!(cut.len(), 1);
    assert!(cut[0].center.abs_diff_eq(Vec3::new(0.0, 1.0, 0.5), 1e-6));
    assert!(cut[0].lo.abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1e-6));
    assert!(cut[0].hi.abs_diff_eq(Vec3::new(0.0, 1.0, 1.0), 1e-6));
}

#[test]
fn separated_surfaces_do_not_join_because_they_are_close() {
    let p = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        Vec3::new(1.0, 2.0, 1.0),
        Vec3::new(0.0001, 0.0, 0.0),
        Vec3::new(1.0001, 2.0, 0.0),
        Vec3::new(1.0001, 2.0, 1.0),
    ];
    assert_eq!(section(&p, &[[0, 1, 2], [3, 4, 5]], 1.0).len(), 2);
}

#[test]
fn rejects_nonfinite_geometry_invalid_indices_and_parallel_axes() {
    let mut p = vec![Vec3::ZERO; 12];
    p[0].x = f32::NAN;
    assert!(
        fit(&p, &[[0, 1, 2]], Some(Vec3::Y), None)
            .unwrap_err()
            .contains("finite")
    );
    p[0] = Vec3::ZERO;
    assert!(
        fit(&p, &[[0, 1, 12]], Some(Vec3::Y), None)
            .unwrap_err()
            .contains("index")
    );
    let f = Fixture::humanoid();
    assert!(
        fit(&f.points, &f.triangles, Some(Vec3::Y), Some(Vec3::Y))
            .unwrap_err()
            .contains("perpendicular")
    );
}

#[test]
fn generated_anatomy_has_separate_moving_limb_landmarks() {
    let f = Fixture::humanoid();
    let s = fit(&f.points, &f.triangles, Some(Vec3::Y), Some(Vec3::Z)).unwrap();
    assert!(!s.covers_legs);
    assert_eq!(s.joints.len(), 27);
    assert!(s.joints[4].y > s.joints[3].y && s.joints[3].y > s.joints[0].y);
    for (start, sign) in [(7, 1.0), (17, -1.0)] {
        let shoulder = s.joints[start + 1];
        let elbow = s.joints[start + 2];
        let wrist = s.joints[start + 3];
        let tip = s.joints[start + 4];
        assert!(sign * wrist.x > sign * shoulder.x + 0.15);
        assert!(elbow.distance(wrist) > 0.04);
        assert!(sign * tip.x >= sign * wrist.x - 0.025);
    }
    for start in [12, 22] {
        assert!(s.joints[start].y > s.joints[start + 1].y);
        assert!(s.joints[start + 1].y > s.joints[start + 2].y);
        assert!(s.joints[start + 4].z > s.joints[start + 2].z);
    }
}

#[test]
fn fit_commutes_with_scale_translation_and_rigid_orientation() {
    let f = Fixture::humanoid();
    let baseline = fit(&f.points, &f.triangles, Some(Vec3::Y), Some(Vec3::Z)).unwrap();
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.63, -0.41, 0.28);
    let translation = Vec3::new(2.7, -1.3, 0.8);
    let scale = 3.2;
    let p: Vec<Vec3> = f
        .points
        .iter()
        .map(|p| rotation * (*p * scale) + translation)
        .collect();
    let transformed = fit(
        &p,
        &f.triangles,
        Some(rotation * Vec3::Y),
        Some(rotation * Vec3::Z),
    )
    .unwrap();
    for (a, b) in baseline.joints.iter().zip(transformed.joints) {
        let recovered = rotation.conjugate() * (b - translation) / scale;
        assert!(a.distance(recovered) / baseline.height < 8e-5);
    }
}
