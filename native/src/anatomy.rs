//! Deterministic skeleton fitting from exact surface cross-sections.
//! Hidden anatomy uses explicit priors; input geometry is never modified.

use glam::Vec3;
use std::collections::BTreeMap;

pub const NAMES: &[&str] = &[
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "HeadEnd",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
];
pub const PARENTS: [i32; 27] = [
    -1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 10, 0, 12, 13, 14, 15, 3, 17, 18, 19, 20, 0, 22, 23, 24, 25,
];

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub joints: Vec<Vec3>,
    pub forward: Vec3,
    pub up: Vec3,
    pub height: f32,
    pub notes: Vec<String>,
    pub covers_legs: bool,
}

#[derive(Clone, Debug)]
struct Section {
    center: Vec3,
    lo: Vec3,
    hi: Vec3,
    perimeter: f32,
}

impl Section {
    fn width(&self) -> f32 {
        self.hi.x - self.lo.x
    }
    fn area_proxy(&self) -> f32 {
        self.width().max(0.002) * (self.hi.z - self.lo.z).max(0.002)
    }
}

struct Frame {
    origin: Vec3,
    left: Vec3,
    up: Vec3,
    forward: Vec3,
    height: f32,
}

impl Frame {
    fn project(&self, p: Vec3) -> Vec3 {
        let d = (p - self.origin) / self.height;
        Vec3::new(d.dot(self.left), d.dot(self.up), d.dot(self.forward))
    }
    fn world(&self, p: Vec3) -> Vec3 {
        self.origin + self.height * (self.left * p.x + self.up * p.y + self.forward * p.z)
    }
}

fn bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    points.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    )
}

fn range(points: &[Vec3], axis: Vec3) -> (f32, f32) {
    points
        .iter()
        .map(|p| p.dot(axis))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), t| {
            (lo.min(t), hi.max(t))
        })
}

fn median(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn orthogonal(up: Vec3) -> Vec3 {
    let seed = if up.x.abs() < 0.8 { Vec3::X } else { Vec3::Z };
    (seed - up * seed.dot(up)).normalize()
}

fn lateral_axis(points: &[Vec3], up: Vec3) -> Vec3 {
    let a = orthogonal(up);
    let b = up.cross(a).normalize();
    let mean = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let (mut aa, mut ab, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for p in points {
        let d = *p - mean;
        let x = d.dot(a) as f64;
        let z = d.dot(b) as f64;
        aa += x * x;
        ab += x * z;
        bb += z * z;
    }
    let angle = 0.5 * (2.0 * ab).atan2(aa - bb);
    let mut left = a * angle.cos() as f32 + b * angle.sin() as f32;
    // Eigenvector sign is arbitrary; stabilize it before choosing facing.
    let dominant = if left.x.abs() >= left.z.abs() {
        left.x
    } else {
        left.z
    };
    if dominant < 0.0 {
        left = -left;
    }
    left.normalize()
}

fn make_frame(points: &[Vec3], up: Vec3, forward: Option<Vec3>) -> Result<Frame, String> {
    let up = up.normalize_or_zero();
    if up.length_squared() < 0.5 {
        return Err("Up axis must be a nonzero finite vector".into());
    }
    let mut left = lateral_axis(points, up);
    let mut facing = left.cross(up).normalize();
    if let Some(f) = forward {
        let tangent = f - up * f.dot(up);
        if !tangent.is_finite() || tangent.length_squared() < 1e-8 {
            return Err("Forward axis must be finite and perpendicular to the up axis".into());
        }
        facing = tangent.normalize();
        left = up.cross(facing).normalize();
    }
    let (bottom, top) = range(points, up);
    let height = top - bottom;
    if height <= 1e-8 {
        return Err("Character height is zero".into());
    }
    let (lx, hx) = range(points, left);
    let (lz, hz) = range(points, facing);
    Ok(Frame {
        origin: left * (0.5 * (lx + hx)) + up * bottom + facing * (0.5 * (lz + hz)),
        left,
        up,
        forward: facing,
        height,
    })
}

fn orientation_score(points: &[Vec3], up: Vec3) -> f32 {
    let Ok(frame) = make_frame(points, up, None) else {
        return f32::NEG_INFINITY;
    };
    let p: Vec<Vec3> = points.iter().map(|p| frame.project(*p)).collect();
    let head: Vec<Vec3> = p.iter().copied().filter(|p| p.y > 0.82).collect();
    let torso: Vec<Vec3> = p
        .iter()
        .copied()
        .filter(|p| p.y > 0.43 && p.y < 0.75)
        .collect();
    let feet: Vec<Vec3> = p.iter().copied().filter(|p| p.y < 0.07).collect();
    if head.len() < 3 || torso.len() < 3 || feet.len() < 3 {
        return f32::NEG_INFINITY;
    }
    let (hl, hh) = bounds(&head);
    let (tl, th) = bounds(&torso);
    let (fl, fh) = bounds(&feet);
    let hw = (hh.x - hl.x).max(0.005);
    let tw = (th.x - tl.x).max(0.005);
    let fw = (fh.x - fl.x).max(0.005);
    let head_center = (0.5 * (hl.x + hh.x)).abs();
    let depth = th.z - tl.z;
    // Head narrower than torso, two feet across the bottom, little depth.
    1.8 * (tw / hw).min(4.0) + 1.0 * (fw / hw).min(3.0)
        - 4.0 * head_center
        - 2.0 * depth
        - 6.0 * (hw - 0.32).max(0.0)
        - 15.0 * (tw - 0.80).max(0.0)
        - 10.0 * (depth - 0.45).max(0.0)
}

fn choose_up(points: &[Vec3], notes: &mut Vec<String>) -> Vec3 {
    let candidates = [Vec3::Y, -Vec3::Y, Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z];
    let mut best = (f32::NEG_INFINITY, Vec3::Y);
    let mut second = f32::NEG_INFINITY;
    for axis in candidates {
        let score = orientation_score(points, axis);
        if score > best.0 {
            second = best.0;
            best = (score, axis);
        } else {
            second = second.max(score);
        }
    }
    notes.push(format!(
        "Automatic up selected from six coordinate directions; score margin {:.3}",
        best.0 - second
    ));
    if best.0 - second < 0.25 {
        notes.push("Up direction is ambiguous; verify the skeleton overlay or provide --up".into());
    }
    best.1
}

fn facing_sign(points: &[Vec3]) -> f32 {
    let head: Vec<Vec3> = points
        .iter()
        .copied()
        .filter(|p| p.y > 0.84 && p.y < 0.95)
        .collect();
    if !head.is_empty() {
        let (lo, hi) = bounds(&head);
        let tip = 0.012;
        let front: Vec<Vec3> = head.iter().copied().filter(|p| p.z > hi.z - tip).collect();
        let back: Vec<Vec3> = head.iter().copied().filter(|p| p.z < lo.z + tip).collect();
        if !front.is_empty() && !back.is_empty() {
            let (fl, fh) = bounds(&front);
            let (bl, bh) = bounds(&back);
            let fw = fh.x - fl.x;
            let bw = bh.x - bl.x;
            if fw > 1.25 * bw {
                return -1.0;
            }
            if bw > 1.25 * fw {
                return 1.0;
            }
        }
    }
    // Tactical vests can protrude farther than toes: compare to lower shins,
    // never to the chest or belly, when the head silhouette is ambiguous.
    let shins = median(
        points
            .iter()
            .filter(|p| p.y > 0.09 && p.y < 0.17)
            .map(|p| p.z)
            .collect(),
    );
    let feet: Vec<Vec3> = points.iter().copied().filter(|p| p.y < 0.045).collect();
    if feet.is_empty() {
        return 1.0;
    }
    let (lo, hi) = bounds(&feet);
    if hi.z - shins >= shins - lo.z {
        1.0
    } else {
        -1.0
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Intersect real triangles with a plane, not a slab of original vertices.
fn section(points: &[Vec3], triangles: &[[usize; 3]], y: f32) -> Vec<Section> {
    let mut edge_index = BTreeMap::new();
    let mut intersections = Vec::new();
    let mut parents = Vec::new();
    let mut segments = Vec::new();
    for t in triangles {
        let mut ids = [0_usize; 3];
        let mut n = 0;
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let pa = points[a];
            let pb = points[b];
            if (pa.y > y) == (pb.y > y) {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let id = *edge_index.entry(key).or_insert_with(|| {
                let id = intersections.len();
                intersections.push(pa.lerp(pb, (y - pa.y) / (pb.y - pa.y)));
                parents.push(id);
                id
            });
            ids[n] = id;
            n += 1;
        }
        if n == 2 && ids[0] != ids[1] {
            let a = find(&mut parents, ids[0]);
            let b = find(&mut parents, ids[1]);
            if a != b {
                parents[a.max(b)] = a.min(b);
            }
            segments.push((ids[0], ids[1]));
        }
    }
    let mut result: BTreeMap<usize, Section> = BTreeMap::new();
    for (a, b) in segments {
        let root = find(&mut parents, a);
        let (p, q) = (intersections[a], intersections[b]);
        let length = p.distance(q);
        let s = result.entry(root).or_insert(Section {
            center: Vec3::ZERO,
            lo: Vec3::splat(f32::INFINITY),
            hi: Vec3::splat(f32::NEG_INFINITY),
            perimeter: 0.0,
        });
        s.center += 0.5 * (p + q) * length;
        s.lo = s.lo.min(p).min(q);
        s.hi = s.hi.max(p).max(q);
        s.perimeter += length;
    }
    let mut result: Vec<Section> = result
        .into_values()
        .filter(|s| s.perimeter > 0.004)
        .collect();
    for s in &mut result {
        s.center /= s.perimeter;
    }
    result.sort_by(|a, b| {
        b.area_proxy()
            .total_cmp(&a.area_proxy())
            .then(a.center.x.total_cmp(&b.center.x))
            .then(a.center.z.total_cmp(&b.center.z))
    });
    result
}

fn torso(sections: &[Section]) -> Option<&Section> {
    sections
        .iter()
        .filter(|s| s.center.x.abs() < 0.12)
        .max_by(|a, b| {
            (a.area_proxy() / (1.0 + 8.0 * a.center.x.abs()))
                .total_cmp(&(b.area_proxy() / (1.0 + 8.0 * b.center.x.abs())))
        })
}

/// Central surface sections as [world up, lateral center, forward center, lateral radius, forward radius].
pub(crate) fn torso_profile(
    points: &[Vec3],
    triangles: &[[usize; 3]],
    up: Vec3,
    forward: Vec3,
    min_y: f32,
    max_y: f32,
    samples: usize,
) -> Vec<[f32; 5]> {
    if points.is_empty() || samples == 0 || max_y <= min_y {
        return Vec::new();
    }
    let Ok(frame) = make_frame(points, up, Some(forward)) else {
        return Vec::new();
    };
    let local: Vec<Vec3> = points.iter().map(|p| frame.project(*p)).collect();
    let origin_y = frame.origin.dot(up);
    let origin_x = frame.origin.dot(frame.left);
    let origin_z = frame.origin.dot(frame.forward);
    (0..samples)
        .filter_map(|i| {
            let y = min_y + (max_y - min_y) * (i as f32 + 0.413) / samples as f32;
            let contours = section(&local, triangles, (y - origin_y) / frame.height);
            let core = torso(&contours)?;
            Some([
                y,
                origin_x + core.center.x * frame.height,
                origin_z + core.center.z * frame.height,
                0.5 * core.width() * frame.height,
                0.5 * (core.hi.z - core.lo.z) * frame.height,
            ])
        })
        .collect()
}

fn leg_pair(sections: &[Section]) -> Option<[&Section; 2]> {
    let mut pair = [None, None];
    for (i, sign) in [1.0, -1.0].iter().enumerate() {
        pair[i] = sections
            .iter()
            .filter(|s| {
                sign * s.center.x > 0.022
                    && s.center.x.abs() < 0.25
                    && s.width() < 0.22
                    && s.perimeter > 0.03
            })
            .max_by(|a, b| a.area_proxy().total_cmp(&b.area_proxy()));
    }
    match pair {
        [Some(a), Some(b)] => Some([a, b]),
        _ => None,
    }
}

fn outer_arm(sections: &[Section], sign: f32, previous: Option<Vec3>) -> Option<&Section> {
    let core = torso(sections)?;
    let boundary = if sign > 0.0 { core.hi.x } else { -core.lo.x };
    sections
        .iter()
        .filter(|s| {
            sign * s.center.x > (0.90 * boundary).max(0.075)
                && s.width() < 0.18
                && !std::ptr::eq(*s, core)
        })
        .min_by(|a, b| {
            let cost = |s: &Section| {
                previous.map_or(-s.area_proxy(), |p| {
                    s.center.distance_squared(p) - 0.05 * s.area_proxy()
                })
            };
            cost(a).total_cmp(&cost(b))
        })
        .filter(|s| previous.is_none_or(|p| s.center.distance(p) < 0.095))
}

fn polyline_point(points: &[Vec3], fraction: f32) -> Vec3 {
    let length: f32 = points.windows(2).map(|p| p[0].distance(p[1])).sum();
    if length < 1e-8 {
        return points[0];
    }
    let mut target = fraction * length;
    for p in points.windows(2) {
        let l = p[0].distance(p[1]);
        if target <= l {
            return p[0].lerp(p[1], target / l.max(1e-8));
        }
        target -= l;
    }
    *points.last().unwrap()
}

fn arm_tracks(slices: &[(f32, Vec<Section>)]) -> Option<[Vec<Vec3>; 2]> {
    let mut tracks = [Vec::new(), Vec::new()];
    let mut best = [Vec::new(), Vec::new()];
    let mut started = false;
    for (y, sections) in slices
        .iter()
        .rev()
        .filter(|(y, _)| *y >= 0.32 && *y <= 0.86)
    {
        let arms = [
            outer_arm(sections, 1.0, tracks[0].last().copied()),
            outer_arm(sections, -1.0, tracks[1].last().copied()),
        ];
        if let [Some(a), Some(b)] = arms {
            tracks[0].push(a.center);
            tracks[1].push(b.center);
            started = true;
        } else if started {
            // A single missing section can occur at a detached cuff or wrist.
            let last_y = tracks[0].last().unwrap().y;
            if last_y - y > 0.025 {
                if tracks[0].len() > best[0].len() {
                    best = tracks;
                }
                tracks = [Vec::new(), Vec::new()];
                started = false;
            }
        }
    }
    if tracks[0].len() > best[0].len() {
        best = tracks;
    }
    if best
        .iter()
        .all(|t| t.len() >= 9 && t[0].y - t.last().unwrap().y > 0.13)
    {
        Some(best)
    } else {
        None
    }
}

fn extended_arm(
    points: &[Vec3],
    shoulder: Vec3,
    sign: f32,
    hip_y: f32,
) -> Result<Vec<Vec3>, String> {
    let candidates: Vec<Vec3> = points
        .iter()
        .copied()
        .filter(|p| {
            sign * p.x > sign * shoulder.x + 0.02 && p.y > hip_y + 0.035 && p.y < shoulder.y + 0.18
        })
        .collect();
    if candidates.len() < 6 {
        return Err(
            "Could not distinguish the arms from the torso; use a separated rest pose".into(),
        );
    }
    let far = candidates
        .iter()
        .map(|p| sign * p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let tips: Vec<Vec3> = candidates
        .iter()
        .copied()
        .filter(|p| sign * p.x > far - 0.025)
        .collect();
    let tip = Vec3::new(
        median(tips.iter().map(|p| p.x).collect()),
        median(tips.iter().map(|p| p.y).collect()),
        median(tips.iter().map(|p| p.z).collect()),
    );
    let direction = (tip - shoulder).normalize();
    let length = tip.distance(shoulder);
    let mut line = vec![shoulder];
    for k in 1..12 {
        let target = k as f32 / 12.0 * length;
        let group: Vec<Vec3> = candidates
            .iter()
            .copied()
            .filter(|p| {
                let along = (*p - shoulder).dot(direction);
                (along - target).abs() < 0.025
                    && (*p - shoulder - direction * along).length() < 0.11
            })
            .collect();
        if !group.is_empty() {
            line.push(Vec3::new(
                median(group.iter().map(|p| p.x).collect()),
                median(group.iter().map(|p| p.y).collect()),
                median(group.iter().map(|p| p.z).collect()),
            ));
        }
    }
    line.push(tip);
    Ok(line)
}

/// Fit the standard 27-joint Kimodo body rig; caller supplies welded world geometry.
pub fn fit(
    points: &[Vec3],
    triangles: &[[usize; 3]],
    up: Option<Vec3>,
    forward: Option<Vec3>,
) -> Result<Skeleton, String> {
    if points.len() < 12 || triangles.is_empty() {
        return Err("A humanoid triangle mesh is required".into());
    }
    if points.iter().any(|p| !p.is_finite()) {
        return Err("Character positions must be finite".into());
    }
    if triangles
        .iter()
        .any(|t| t.iter().any(|i| *i >= points.len()))
    {
        return Err("Triangle index exceeds the character vertex count".into());
    }
    let mut notes = Vec::new();
    let up = up.unwrap_or_else(|| choose_up(points, &mut notes));
    let mut frame = make_frame(points, up, forward)?;
    let mut local: Vec<Vec3> = points.iter().map(|p| frame.project(*p)).collect();
    if forward.is_none() {
        if facing_sign(&local) < 0.0 {
            frame.forward = -frame.forward;
            frame.left = -frame.left;
        }
        local = points.iter().map(|p| frame.project(*p)).collect();
        notes.push("Facing inferred from head silhouette, then toes relative to shins; symmetric shapes remain ambiguous".into());
    }
    let slices: Vec<(f32, Vec<Section>)> = (2..96)
        .map(|k| {
            // Offset avoids coplanar cuts at quantized source-vertex coordinates.
            let y = k as f32 * 0.01 + 0.000137;
            (y, section(&local, triangles, y))
        })
        .collect();
    let at = |y: f32| -> &[Section] {
        &slices
            .iter()
            .min_by(|a, b| (a.0 - y).abs().total_cmp(&(b.0 - y).abs()))
            .unwrap()
            .1
    };
    let core = |y: f32| -> Vec3 {
        torso(at(y)).map_or(Vec3::new(0.0, y, 0.0), |s| Vec3::new(0.0, y, s.center.z))
    };

    let mut highest_legs = None;
    for (y, sections) in slices.iter().filter(|(y, _)| *y >= 0.12 && *y < 0.58) {
        let pair = leg_pair(sections);
        // The first midline connection is the crotch or an enclosing garment.
        // Do not relabel disconnected vest details or hands as thighs above it.
        let joined = sections.iter().any(|s| {
            s.lo.x < -0.015
                && s.hi.x > 0.015
                && s.width() > 0.06
                && s.perimeter > 0.10
                && pair.is_none_or(|p| {
                    s.lo.x <= p[1].center.x + 0.025
                        && s.hi.x >= p[0].center.x - 0.025
                        && s.area_proxy() >= 0.7 * (p[0].area_proxy() + p[1].area_proxy())
                })
        });
        if joined {
            break;
        }
        if let Some(pair) = pair {
            // Hands at hip height are too far apart to represent thighs.
            if pair[0].center.x - pair[1].center.x < 0.30 {
                highest_legs = Some((*y, [pair[0].center, pair[1].center]));
            }
        }
    }
    let covers_legs = highest_legs.is_none_or(|(y, _)| y < 0.30);
    let leg_y = if covers_legs {
        0.50
    } else {
        (highest_legs.unwrap().0 + 0.028).clamp(0.34, 0.56)
    };
    let hips_y = leg_y + 0.045;
    let hips = core(hips_y);
    if covers_legs {
        notes.push("Connected lower garment hides the thighs: pelvis and knees use a standing-human prior; keep cloth coherent".into());
    } else {
        notes.push("Leg centers and crotch fitted from separated surface cross-sections".into());
    }
    let mut joints = vec![Vec3::ZERO; NAMES.len()];
    joints[0] = hips;
    for (side, sign) in [1.0, -1.0].iter().enumerate() {
        let base = if side == 0 { 12 } else { 22 };
        let feet: Vec<Vec3> = local
            .iter()
            .copied()
            .filter(|p| p.y < 0.06 && sign * p.x > 0.015)
            .collect();
        if feet.len() < 3 {
            return Err("Could not distinguish two feet near the floor".into());
        }
        let (lo, hi) = bounds(&feet);
        let ankle_x = median(feet.iter().map(|p| p.x).collect());
        let ankle = Vec3::new(ankle_x, 0.055, lo.z + 0.28 * (hi.z - lo.z));
        let hip = if covers_legs {
            Vec3::new(sign * 0.055, leg_y, hips.z)
        } else {
            let p = highest_legs.unwrap().1[side];
            Vec3::new(p.x, leg_y, p.z)
        };
        let knee_y = 0.5 * (leg_y + ankle.y);
        let knee = if covers_legs {
            hip.lerp(ankle, 0.5) + Vec3::Z * 0.006
        } else {
            leg_pair(at(knee_y)).map_or(hip.lerp(ankle, 0.5), |p| {
                Vec3::new(p[side].center.x, knee_y, p[side].center.z + 0.003)
            })
        };
        joints[base] = hip;
        joints[base + 1] = knee;
        joints[base + 2] = ankle;
        joints[base + 3] = Vec3::new(ankle_x, 0.018, lo.z + 0.72 * (hi.z - lo.z));
        joints[base + 4] = Vec3::new(ankle_x, 0.012, hi.z - 0.002);
    }

    let tracks = arm_tracks(&slices);
    let neck_candidate = slices
        .iter()
        .filter(|(y, _)| *y > 0.74 && *y < 0.90)
        .filter_map(|(y, s)| torso(s).map(|s| (*y, s)))
        .min_by(|a, b| {
            (a.1.width() + 1.5 * (a.0 - 0.835).abs())
                .total_cmp(&(b.1.width() + 1.5 * (b.0 - 0.835).abs()))
        });
    let nominal_neck = neck_candidate.map_or(0.84, |(y, _)| y);
    let shoulder_y = nominal_neck - 0.065;
    let chest_y = shoulder_y - 0.026;
    let neck_y = nominal_neck.max(shoulder_y + 0.035);
    joints[1] = core(hips_y + 0.15 * (chest_y - hips_y));
    joints[2] = core(hips_y + 0.50 * (chest_y - hips_y));
    joints[3] = core(chest_y);
    joints[4] = core(neck_y);
    joints[5] = core(neck_y + 0.48 * (1.0 - neck_y));
    joints[6] = core(0.992);

    for (side, sign) in [1.0, -1.0].iter().enumerate() {
        let base = if side == 0 { 7 } else { 17 };
        let chest = torso(at(chest_y));
        let torso_half = chest.map_or(0.11, |s| if *sign > 0.0 { s.hi.x } else { -s.lo.x });
        let shoulder = tracks.as_ref().map_or_else(
            || {
                Vec3::new(
                    sign * (torso_half * 0.82).clamp(0.06, 0.17),
                    shoulder_y,
                    core(shoulder_y).z,
                )
            },
            |t| {
                let p = t[side][0];
                let q = t[side][t[side].len().min(5) - 1];
                let rise = shoulder_y - p.y;
                let gradient = if (p.y - q.y).abs() > 0.015 {
                    (p - q) / (p.y - q.y)
                } else {
                    Vec3::Y
                };
                let extrapolated = p + gradient * rise;
                // Deltoid centers approach the torso above the separated arm cut.
                let x = (sign * extrapolated.x)
                    .clamp((torso_half * 0.60).max(0.045), torso_half.max(0.065));
                Vec3::new(sign * x, shoulder_y, extrapolated.z)
            },
        );
        let line = if let Some(t) = &tracks {
            let mut line = vec![shoulder];
            line.extend(t[side].iter().copied());
            let last = *line.last().unwrap();
            let hand: Vec<Vec3> = local
                .iter()
                .copied()
                .filter(|p| {
                    sign * p.x > 0.07
                        && sign * p.x > sign * last.x - 0.025
                        && p.y < last.y + 0.012
                        && p.distance(last) < 0.065
                })
                .collect();
            if let Some(tip) = hand.iter().min_by(|a, b| a.y.total_cmp(&b.y)) {
                line.push(*tip);
            }
            line
        } else {
            extended_arm(&local, shoulder, *sign, hips_y)?
        };
        joints[base] = Vec3::new(sign * 0.023, chest_y + 0.012, shoulder.z);
        joints[base + 1] = shoulder;
        joints[base + 2] = polyline_point(&line, 0.43);
        joints[base + 3] = polyline_point(&line, 0.78);
        joints[base + 4] = *line.last().unwrap();
    }
    notes.push(
        if tracks.is_some() {
            "Arm chains traced through separated A-pose/arms-down cross-sections"
        } else {
            "Arm chains fitted from lateral centerlines; separated T/A-pose required"
        }
        .into(),
    );

    if joints.iter().any(|p| !p.is_finite()) || joints[4].y <= joints[0].y + 0.10 {
        return Err("The fitted anatomy is inconsistent with a standing humanoid".into());
    }
    for i in 1..joints.len() {
        if joints[i].distance(joints[PARENTS[i] as usize]) < 0.002 {
            return Err(format!("Fitted bone {} has negligible length", NAMES[i]));
        }
    }
    Ok(Skeleton {
        names: NAMES.iter().map(|s| (*s).into()).collect(),
        parents: PARENTS.to_vec(),
        joints: joints.into_iter().map(|p| frame.world(p)).collect(),
        forward: frame.forward,
        up: frame.up,
        height: frame.height,
        notes,
        covers_legs,
    })
}

#[cfg(test)]
#[path = "anatomy_tests.rs"]
mod tests;
