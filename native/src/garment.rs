//! Split existing garment/foot vertex roles while preserving every rest triangle.
//! The free lower fabric follows the pelvis; this does not simulate fabric dynamics.

use crate::{anatomy::Skeleton, glb::Geometry, weights::SkinWeights};
use glam::Vec3;
use serde_json::json;

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn garment_weights(
    point: Vec3,
    skeleton: &Skeleton,
    hips: usize,
    joints: [u16; 4],
    weights: [f32; 4],
) -> ([u16; 4], [f32; 4]) {
    let waist = skeleton.joints[hips].dot(skeleton.up);
    let strength = 1.0
        - smoothstep(
            waist - 0.10 * skeleton.height,
            waist - 0.035 * skeleton.height,
            point.dot(skeleton.up),
        );
    if strength <= 0.0 {
        return (joints, weights);
    }
    let mut dense = vec![0.0; skeleton.joints.len()];
    for k in 0..4 {
        dense[joints[k] as usize] += weights[k] * (1.0 - strength);
    }
    dense[hips] += strength;
    let mut order: Vec<usize> = (0..dense.len()).collect();
    order.sort_unstable_by(|&a, &b| dense[b].total_cmp(&dense[a]).then(a.cmp(&b)));
    let sum: f32 = order[..4].iter().map(|&i| dense[i]).sum();
    let mut output_joints = [0; 4];
    let mut output_weights = [0.0; 4];
    for k in 0..4 {
        output_joints[k] = order[k] as u16;
        output_weights[k] = dense[order[k]] / sum;
    }
    (output_joints, output_weights)
}

fn anatomical_foot_weights(
    point: Vec3,
    skeleton: &Skeleton,
    original_joints: [u16; 4],
    original_weights: [f32; 4],
) -> ([u16; 4], [f32; 4]) {
    let mut best = None;
    for side in ["Left", "Right"] {
        let Some(foot) = skeleton
            .names
            .iter()
            .position(|n| n == &format!("{side}Foot"))
        else {
            continue;
        };
        let Some(toe) = skeleton
            .names
            .iter()
            .position(|n| n == &format!("{side}ToeBase"))
        else {
            continue;
        };
        let ankle = skeleton.joints[foot];
        let tip = skeleton.joints[toe];
        let axis = tip - ankle;
        let t = ((point - ankle).dot(axis) / axis.length_squared().max(1e-12)).clamp(0.0, 1.0);
        let distance = point.distance(ankle + t * axis);
        if point.dot(skeleton.up) > ankle.dot(skeleton.up) + 0.055 * skeleton.height
            || distance > 0.075 * skeleton.height
        {
            continue;
        }
        if best.is_none_or(|(_, d)| distance < d) {
            best = Some((foot, distance));
        }
    }
    match best {
        // A real ankle/foot seam must move with its anatomical foot, not the robe.
        Some((foot, _)) => ([foot as u16; 4], [1.0, 0.0, 0.0, 0.0]),
        None => (original_joints, original_weights),
    }
}

/// Append only dual-role duplicates within each primitive, preserving its original prefix.
/// Original triangle order, corners, positions and attribute source mappings are retained.
pub fn separate(
    geometry: &mut Geometry,
    skeleton: &Skeleton,
    skin: &mut SkinWeights,
    roles: &[bool],
) -> Result<(), String> {
    if !skeleton.covers_legs {
        return Ok(());
    }
    if skin.joints.len() != geometry.points.len() || skin.weights.len() != geometry.points.len() {
        return Err("Garment separation requires one skin row per source vertex".into());
    }
    if roles.len() != geometry.triangles.len() {
        return Err("Garment separation requires one role per source triangle".into());
    }
    if skeleton.joints.len() < 4
        || skin
            .joints
            .iter()
            .flatten()
            .any(|&j| j as usize >= skeleton.joints.len())
        || geometry
            .triangles
            .iter()
            .flatten()
            .any(|&i| i >= geometry.points.len())
    {
        return Err("Invalid skeleton or triangle indices before garment separation".into());
    }
    let hips = skeleton
        .names
        .iter()
        .position(|n| n == "Hips")
        .ok_or("Garment attachment needs Hips")?;
    let source_points = std::mem::take(&mut geometry.points);
    let source_joints = std::mem::take(&mut skin.joints);
    let source_weights = std::mem::take(&mut skin.weights);
    let mut all_triangles = Vec::with_capacity(geometry.triangles.len());
    let mut face_offset = 0;
    let mut duplicated = 0;
    let mut clothing_faces = 0;
    let mut supported_foot_vertices = 0;
    for primitive in &mut geometry.primitives {
        let start = primitive.offset;
        let count = primitive.count;
        let end = start + count;
        if end > source_points.len()
            || primitive.source_vertices.len() != count
            || primitive.normals.as_ref().is_some_and(|n| n.len() != count)
            || primitive
                .texcoords
                .as_ref()
                .is_some_and(|uv| uv.len() != count)
        {
            return Err("Primitive attributes disagree before garment separation".into());
        }
        let face_end = face_offset + primitive.triangles.len();
        if face_end > roles.len() {
            return Err("Primitive face counts disagree with the geometry".into());
        }
        let clothing = &roles[face_offset..face_end];
        face_offset = face_end;
        let mut vertex_roles = vec![0_u8; count];
        for (triangle, &is_clothing) in primitive.triangles.iter().zip(clothing) {
            for &index in triangle {
                if index as usize >= count {
                    return Err("Garment face index exceeds its primitive".into());
                }
                vertex_roles[index as usize] |= if is_clothing { 2 } else { 1 };
            }
        }
        let offset = geometry.points.len();
        geometry
            .points
            .extend_from_slice(&source_points[start..end]);
        skin.joints.extend_from_slice(&source_joints[start..end]);
        skin.weights.extend_from_slice(&source_weights[start..end]);
        let mut clothing_index: Vec<u32> = (0..count as u32).collect();
        for (index, &role) in vertex_roles.iter().enumerate() {
            let source_index = start + index;
            if role == 3 {
                let local = geometry.points.len() - offset;
                if local > u32::MAX as usize {
                    return Err("Too many garment vertices for glTF indices".into());
                }
                clothing_index[index] = local as u32;
                geometry.points.push(source_points[source_index]);
                let (joints, weights) = garment_weights(
                    source_points[source_index],
                    skeleton,
                    hips,
                    source_joints[source_index],
                    source_weights[source_index],
                );
                skin.joints.push(joints);
                skin.weights.push(weights);
                if let Some(normals) = &mut primitive.normals {
                    normals.push(normals[index]);
                }
                if let Some(texcoords) = &mut primitive.texcoords {
                    texcoords.push(texcoords[index]);
                }
                primitive
                    .source_vertices
                    .push(primitive.source_vertices[index]);
                let (joints, weights) = anatomical_foot_weights(
                    source_points[source_index],
                    skeleton,
                    source_joints[source_index],
                    source_weights[source_index],
                );
                skin.joints[offset + index] = joints;
                skin.weights[offset + index] = weights;
                if joints != source_joints[source_index] || weights != source_weights[source_index]
                {
                    supported_foot_vertices += 1;
                }
                duplicated += 1;
            } else if role == 2 {
                let (joints, weights) = garment_weights(
                    source_points[source_index],
                    skeleton,
                    hips,
                    source_joints[source_index],
                    source_weights[source_index],
                );
                skin.joints[offset + index] = joints;
                skin.weights[offset + index] = weights;
            } else if role == 1 {
                // Palette refinement can reveal ankle vertices whose diffused rows
                // still carry robe weights. Support the whole foot, including these
                // body-only vertices; moving only its seam causes delamination.
                let (joints, weights) = anatomical_foot_weights(
                    source_points[source_index],
                    skeleton,
                    source_joints[source_index],
                    source_weights[source_index],
                );
                skin.joints[offset + index] = joints;
                skin.weights[offset + index] = weights;
                if joints != source_joints[source_index] || weights != source_weights[source_index]
                {
                    supported_foot_vertices += 1;
                }
            }
        }
        for (triangle, &is_clothing) in primitive.triangles.iter_mut().zip(clothing) {
            if is_clothing {
                for index in triangle.iter_mut() {
                    *index = clothing_index[*index as usize];
                }
                clothing_faces += 1;
            }
            all_triangles.push(triangle.map(|i| offset + i as usize));
        }
        primitive.offset = offset;
        primitive.count = geometry.points.len() - offset;
    }
    if face_offset != roles.len() {
        return Err("Unmapped faces remain after garment separation".into());
    }
    geometry.triangles = all_triangles;
    skin.report["garment_seam"] = json!({
        "duplicated_vertices": duplicated, "garment_faces": clothing_faces,
        "source_vertices": source_points.len(), "output_vertices": geometry.points.len(),
        "rest_triangle_geometry_preserved": true,
        "supported_foot_vertices": supported_foot_vertices,
        "foot_policy": "body surfaces in bounded anatomical foot capsules follow Foot rigidly; garment surfaces are excluded",
        "policy": "free lower garment follows pelvis; anatomical feet remain articulated; no cloth collision simulation",
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glb::Primitive;

    #[test]
    fn seam_changes_weight_roles_without_changing_any_rest_triangle_or_attribute_source() {
        let mut points = Vec::new();
        let mut faces = Vec::new();
        for y in [0.055, 0.54] {
            for i in 0..16 {
                let a = std::f32::consts::TAU * i as f32 / 16.0;
                points.push(Vec3::new(0.14 * a.cos(), y, 0.07 * a.sin()));
            }
        }
        for i in 0..16 {
            let next = (i + 1) % 16;
            faces.push([i, next, i + 16]);
            faces.push([next, next + 16, i + 16]);
        }
        for (side, shared) in [(1.0, 0), (-1.0, 8)] {
            let mut rings = [[0; 16]; 2];
            for (ring, y) in [0.0, 0.055].into_iter().enumerate() {
                for (i, vertex) in rings[ring].iter_mut().enumerate() {
                    if ring == 1 && i == 0 {
                        *vertex = shared;
                        continue;
                    }
                    let a = std::f32::consts::TAU * i as f32 / 16.0;
                    *vertex = points.len();
                    points.push(Vec3::new(
                        side * (0.105 + 0.035 * a.cos()),
                        y,
                        0.055 * a.sin(),
                    ));
                }
            }
            for i in 0..16 {
                let next = (i + 1) % 16;
                faces.push([rings[0][i], rings[0][next], rings[1][i]]);
                faces.push([rings[0][next], rings[1][next], rings[1][i]]);
            }
        }
        let count = points.len();
        let before_points = points.clone();
        let before_faces = faces.clone();
        let normals: Vec<Vec3> = points
            .iter()
            .map(|p| Vec3::new(p.x, 0.1, p.z).normalize())
            .collect();
        let before_normals = normals.clone();
        let texcoords: Vec<[f32; 2]> = points.iter().map(|p| [p.x + 0.5, p.y]).collect();
        let before_texcoords = texcoords.clone();
        let mut geometry = Geometry {
            points,
            triangles: faces.clone(),
            primitives: vec![Primitive {
                node: 0,
                source: json!({}),
                offset: 0,
                count,
                normals: Some(normals),
                texcoords: Some(texcoords),
                triangles: faces.iter().map(|t| t.map(|i| i as u32)).collect(),
                source_vertices: (0..count).collect(),
            }],
        };
        let mut joints = vec![Vec3::ZERO; 27];
        joints[0] = Vec3::new(0.0, 0.54, 0.0);
        joints[12] = Vec3::new(0.055, 0.5, 0.0);
        joints[22] = Vec3::new(-0.055, 0.5, 0.0);
        joints[14] = Vec3::new(0.105, 0.04, 0.0);
        joints[24] = Vec3::new(-0.105, 0.04, 0.0);
        joints[15] = Vec3::new(0.105, 0.02, 0.10);
        joints[25] = Vec3::new(-0.105, 0.02, 0.10);
        let skeleton = Skeleton {
            names: crate::anatomy::NAMES.iter().map(|n| (*n).into()).collect(),
            parents: crate::anatomy::PARENTS.to_vec(),
            joints,
            up: Vec3::Y,
            forward: Vec3::Z,
            height: 1.0,
            covers_legs: true,
            notes: Vec::new(),
        };
        let mut skin = SkinWeights {
            // Simulate diffusion contamination on a continuous robe/ankle mesh.
            joints: vec![[0; 4]; count],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; count],
            report: json!({}),
        };
        let roles =
            crate::cloth::lower_garment_faces(&geometry.points, &geometry.triangles, &skeleton);
        separate(&mut geometry, &skeleton, &mut skin, &roles).unwrap();
        assert!(
            geometry.points.len() > count,
            "A real role seam must duplicate shared vertices"
        );
        assert_eq!(&geometry.points[..count], before_points.as_slice());
        assert_eq!(geometry.triangles.len(), before_faces.len());
        for (old, new) in before_faces.iter().zip(&geometry.triangles) {
            for k in 0..3 {
                assert_eq!(before_points[old[k]], geometry.points[new[k]]);
            }
        }
        let primitive = &geometry.primitives[0];
        for (i, &source) in primitive.source_vertices.iter().enumerate() {
            assert_eq!(geometry.points[i], before_points[source]);
            assert_eq!(
                primitive.normals.as_ref().unwrap()[i],
                before_normals[source]
            );
            assert_eq!(
                primitive.texcoords.as_ref().unwrap()[i],
                before_texcoords[source]
            );
        }
        for &foot_role in &[0, 8] {
            assert_ne!(skin.joints[foot_role][0], 0);
        }
        for (triangle, is_garment) in before_faces.iter().zip(&roles) {
            if !is_garment {
                for &vertex in triangle {
                    assert!(matches!(skin.joints[vertex][0], 14 | 24));
                    assert_eq!(skin.weights[vertex], [1.0, 0.0, 0.0, 0.0]);
                }
            }
        }
        assert!(skin.joints[count..].iter().any(|j| j[0] == 0));
        assert!(
            skin.weights
                .iter()
                .all(|w| (w.iter().sum::<f32>() - 1.0).abs() < 1e-6)
        );
    }
}
