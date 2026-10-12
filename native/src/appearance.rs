//! Refine only ambiguous robe/foot contacts with palettes learned from the same mesh.
//! Colors resolve surface ownership; they do not supply anatomical ground truth.

use crate::{
    anatomy::Skeleton,
    glb::{Geometry, Glb, Primitive},
    mesh::Surface,
};
use glam::Vec3;
use image::{ImageDecoder, ImageReader, RgbImage};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    io::Cursor,
};

const MAX_PIXELS: u64 = 16 * 1024 * 1024;

struct Texture {
    pixels: RgbImage,
    wrap_s: u64,
    wrap_t: u64,
    factor: Vec3,
}

fn texture(glb: &Glb, primitive: &Primitive) -> Result<Texture, String> {
    let material = primitive.source["material"]
        .as_u64()
        .ok_or("No base-color material")?;
    let pbr = &glb.json["materials"][material as usize]["pbrMetallicRoughness"];
    let info = &pbr["baseColorTexture"];
    let index = info["index"].as_u64().ok_or("No base-color texture")?;
    if info["texCoord"].as_u64().unwrap_or(0) != 0 {
        return Err("Base-color texture requires an unavailable UV set".into());
    }
    if info["extensions"].get("KHR_texture_transform").is_some() {
        return Err("Transformed base-color UVs require geometry fallback".into());
    }
    let source = &glb.json["textures"][index as usize];
    let image_index = source["source"]
        .as_u64()
        .ok_or("Unsupported base-color texture source")?;
    let image = &glb.json["images"][image_index as usize];
    let view_index = image["bufferView"]
        .as_u64()
        .ok_or("Base-color image is not embedded")?;
    let view = &glb.json["bufferViews"][view_index as usize];
    if view["buffer"].as_u64().unwrap_or(0) != 0 {
        return Err("Base-color image uses an unavailable buffer".into());
    }
    let start = view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let length = view["byteLength"]
        .as_u64()
        .ok_or("Base-color image has no byte length")? as usize;
    let end = start
        .checked_add(length)
        .ok_or("Base-color image length overflows")?;
    let bytes = glb
        .binary
        .get(start..end)
        .ok_or("Base-color image exceeds its buffer")?;
    // Check dimensions before decoding: a decoder's allocation hint is not a hard bound.
    let (width, height) = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("Cannot identify base-color image: {e}"))?
        .into_dimensions()
        .map_err(|e| format!("Unsupported base-color image: {e}"))?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > MAX_PIXELS {
        return Err("Base-color image exceeds the 16 MP / 64 MiB RGBA decode bound".into());
    }
    let decoder = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("Cannot identify base-color image: {e}"))?
        .into_decoder()
        .map_err(|e| format!("Unsupported base-color image: {e}"))?;
    if decoder.total_bytes() > 64 * 1024 * 1024 {
        return Err("Base-color image bit depth exceeds the 64 MiB decoded pixel bound".into());
    }
    drop(decoder);
    let pixels = image::load_from_memory(bytes)
        .map_err(|e| format!("Cannot decode base-color image: {e}"))?
        .to_rgb8();
    let sampler = source["sampler"]
        .as_u64()
        .map(|i| &glb.json["samplers"][i as usize]);
    let wrap_s = sampler.and_then(|s| s["wrapS"].as_u64()).unwrap_or(10497);
    let wrap_t = sampler.and_then(|s| s["wrapT"].as_u64()).unwrap_or(10497);
    if ![33071, 33648, 10497].contains(&wrap_s) || ![33071, 33648, 10497].contains(&wrap_t) {
        return Err("Unsupported base-color texture wrap mode".into());
    }
    let factor = Vec3::from_array(std::array::from_fn(|i| {
        pbr["baseColorFactor"][i].as_f64().unwrap_or(1.0) as f32
    }));
    if !factor.is_finite() || factor.min_element() < 0.0 {
        return Err("Invalid base-color factor".into());
    }
    Ok(Texture {
        pixels,
        wrap_s,
        wrap_t,
        factor,
    })
}

fn wrapped(uv: f32, mode: u64) -> f32 {
    match mode {
        33071 => uv.clamp(0.0, 1.0),
        33648 => 1.0 - (uv.rem_euclid(2.0) - 1.0).abs(),
        _ => uv.rem_euclid(1.0),
    }
}

fn median(values: &mut [Vec3]) -> Vec3 {
    Vec3::from_array(std::array::from_fn(|channel| {
        values.sort_unstable_by(|a, b| a[channel].total_cmp(&b[channel]));
        values[values.len() / 2][channel]
    }))
}

fn chromaticity(color: Vec3) -> Option<Vec3> {
    let sum = color.element_sum();
    (color.is_finite() && color.min_element() >= 0.0 && sum > 0.12).then(|| color / sum)
}

impl Texture {
    fn face_color(&self, uv: [[f32; 2]; 3]) -> Option<Vec3> {
        if uv.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let weights = [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]];
        let mut samples = weights.map(|bary| {
            let u = (0..3).map(|i| uv[i][0] * bary[i]).sum();
            let v = (0..3).map(|i| uv[i][1] * bary[i]).sum();
            let x = (wrapped(u, self.wrap_s) * self.pixels.width() as f32) as u32;
            let y = (wrapped(v, self.wrap_t) * self.pixels.height() as f32) as u32;
            let rgb = self.pixels.get_pixel(
                x.min(self.pixels.width() - 1),
                y.min(self.pixels.height() - 1),
            );
            Vec3::from_array(rgb.0.map(|c| c as f32 / 255.0)) * self.factor
        });
        // Three interior probes reduce atlas boundary and individual texel artifacts.
        chromaticity(median(&mut samples))
    }
}

struct Palettes {
    garment: Vec3,
    foot: Vec3,
    separation: f32,
}

impl Palettes {
    fn learn(garment: &mut [Vec3], foot: &mut [Vec3]) -> Option<Self> {
        if garment.len() < 6 || foot.len() < 6 {
            return None;
        }
        let garment = median(garment);
        let foot = median(foot);
        let separation = garment.distance(foot);
        (separation >= 0.035).then_some(Self {
            garment,
            foot,
            separation,
        })
    }

    fn role(&self, color: Vec3) -> Option<bool> {
        let garment = color.distance(self.garment);
        let foot = color.distance(self.foot);
        // Abstain on unseen colors or palettes with an insufficient decision margin.
        if garment.min(foot) > (0.8 * self.separation).max(0.06) {
            return None;
        }
        let margin = 0.15 * self.separation;
        if garment + margin < foot {
            Some(true)
        } else if foot + margin < garment {
            Some(false)
        } else {
            None
        }
    }
}

fn connected_garment(
    geometry: &Geometry,
    original: &[bool],
    predictions: &[Option<bool>],
) -> Vec<bool> {
    let surface = Surface::new(&geometry.points, &geometry.triangles);
    let mut edges = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (face, triangle) in geometry.triangles.iter().enumerate() {
        let t = triangle.map(|i| surface.original_to_welded[i]);
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if a != b {
                edges.entry((a.min(b), a.max(b))).or_default().push(face);
            }
        }
    }
    let mut neighbors = vec![Vec::new(); original.len()];
    for faces in edges.values() {
        for &a in faces {
            for &b in faces {
                if a != b {
                    neighbors[a].push(b);
                }
            }
        }
    }
    let allowed: Vec<_> = original
        .iter()
        .zip(predictions)
        .map(|(&old, &vote)| vote.unwrap_or(old))
        .collect();
    let mut connected = vec![false; original.len()];
    let mut queue = VecDeque::new();
    for (i, &old) in original.iter().enumerate() {
        if old && allowed[i] {
            connected[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for &other in &neighbors[i] {
            if allowed[other] && !connected[other] {
                connected[other] = true;
                queue.push_back(other);
            }
        }
    }
    connected
}

/// Learn garment/sole palettes from geometric seeds and refine only ankle-height contacts.
/// Missing UVs, unsupported images and indistinguishable palettes retain geometric roles.
pub fn refine_roles(
    glb: &Glb,
    geometry: &Geometry,
    skeleton: &Skeleton,
    roles: &mut [bool],
) -> Result<Value, String> {
    if roles.len() != geometry.triangles.len() {
        return Err("Appearance refinement requires one role per source triangle".into());
    }
    if !skeleton.covers_legs {
        return Ok(json!({"status":"not_applicable", "reason":"Leg surfaces are distinct"}));
    }
    if geometry.points.is_empty() || !skeleton.height.is_finite() || skeleton.height <= 0.0 {
        return Err("Appearance refinement requires finite, nonempty geometry".into());
    }
    let floor = geometry
        .points
        .iter()
        .map(|p| p.dot(skeleton.up))
        .fold(f32::INFINITY, f32::min);
    let hip = skeleton
        .names
        .iter()
        .position(|n| n == "Hips")
        .ok_or("Appearance refinement needs Hips")?;
    let waist = (skeleton.joints[hip].dot(skeleton.up) - floor) / skeleton.height;
    let ankle = skeleton
        .names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.ends_with("Foot"))
        .map(|(i, _)| (skeleton.joints[i].dot(skeleton.up) - floor) / skeleton.height)
        .fold(0.0_f32, f32::max);
    let contact_max = (ankle + 0.10).min(waist - 0.08);
    let mut heights = vec![0.0; roles.len()];
    for (i, triangle) in geometry.triangles.iter().enumerate() {
        if triangle.iter().any(|&index| index >= geometry.points.len()) {
            return Err("Invalid triangle index before appearance refinement".into());
        }
        heights[i] = triangle
            .iter()
            .map(|&v| geometry.points[v].dot(skeleton.up))
            .sum::<f32>()
            / 3.0;
        heights[i] = (heights[i] - floor) / skeleton.height;
    }
    let mut colors = vec![None; roles.len()];
    let mut fallback = Vec::new();
    let mut face_offset = 0;
    let mut decoded = 0;
    for (primitive_index, primitive) in geometry.primitives.iter().enumerate() {
        let face_end = face_offset + primitive.triangles.len();
        if face_end > roles.len() {
            return Err("Primitive face counts disagree before appearance refinement".into());
        }
        let Some(uv) = &primitive.texcoords else {
            fallback.push(json!({"primitive":primitive_index,"reason":"No TEXCOORD_0; geometric roles retained"}));
            face_offset = face_end;
            continue;
        };
        if uv.len() != primitive.count {
            return Err("UV count disagrees with its primitive".into());
        }
        let texture = match texture(glb, primitive) {
            Ok(value) => value,
            Err(reason) => {
                fallback.push(json!({"primitive":primitive_index,"reason":reason}));
                face_offset = face_end;
                continue;
            }
        };
        decoded += 1;
        for (i, triangle) in primitive.triangles.iter().enumerate() {
            if triangle.iter().any(|&index| index as usize >= uv.len()) {
                return Err("Invalid UV triangle index before appearance refinement".into());
            }
            let global = face_offset + i;
            let y = heights[global];
            let seed =
                (roles[global] && y >= 0.12 && y <= waist - 0.08) || (!roles[global] && y <= 0.03);
            if seed || (y >= 0.03 && y <= contact_max) {
                colors[global] = texture.face_color(triangle.map(|v| uv[v as usize]));
            }
        }
        face_offset = face_end;
    }
    if face_offset != roles.len() {
        return Err("Unmapped faces before appearance refinement".into());
    }
    let mut garment_samples = Vec::new();
    let mut foot_samples = Vec::new();
    for (i, color) in colors.iter().enumerate() {
        let Some(color) = *color else {
            continue;
        };
        if roles[i] && heights[i] >= 0.12 && heights[i] <= waist - 0.08 {
            garment_samples.push(color);
        }
        if !roles[i] && heights[i] <= 0.03 {
            foot_samples.push(color);
        }
    }
    let before = roles.iter().filter(|&&r| r).count();
    let garment_seed_count = garment_samples.len();
    let foot_seed_count = foot_samples.len();
    let Some(palette) = Palettes::learn(&mut garment_samples, &mut foot_samples) else {
        return Ok(
            json!({"status":"geometry_fallback", "reason":"Insufficient or indistinguishable garment/sole palettes",
            "garment_seed_faces":garment_seed_count,"foot_seed_faces":foot_seed_count,
            "decoded_primitive_textures":decoded,"fallback_primitives":fallback,
            "garment_faces_before":before,"garment_faces_after":before,"changed_faces":0}),
        );
    };
    let predictions: Vec<_> = colors
        .iter()
        .enumerate()
        .map(|(i, color)| {
            if heights[i] < 0.03 || heights[i] > contact_max {
                None
            } else {
                color.and_then(|c| palette.role(c))
            }
        })
        .collect();
    let connected = connected_garment(geometry, roles, &predictions);
    let mut added = 0;
    let mut removed = 0;
    let mut disconnected = 0;
    let mut ambiguous = 0;
    for (i, prediction) in predictions.iter().enumerate() {
        if heights[i] < 0.03 || heights[i] > contact_max {
            continue;
        }
        match prediction {
            Some(true) if !roles[i] && connected[i] => {
                roles[i] = true;
                added += 1;
            }
            Some(true) if !roles[i] => disconnected += 1,
            Some(false) if roles[i] => {
                roles[i] = false;
                removed += 1;
            }
            None => ambiguous += 1,
            _ => {}
        }
    }
    Ok(
        json!({"status":"appearance_refined", "garment_faces_before":before,
        "garment_faces_after":before+added-removed,"added_garment_faces":added,"removed_garment_faces":removed,
        "changed_faces":added+removed,"ambiguous_contact_faces":ambiguous,
        "disconnected_positive_faces_retained_as_body":disconnected,
        "garment_seed_faces":garment_seed_count,"foot_seed_faces":foot_seed_count,
        "garment_chromaticity":palette.garment.to_array(),"sole_chromaticity":palette.foot.to_array(),
        "palette_separation":palette.separation,"contact_height_fraction":[0.03,contact_max],
        "decoded_primitive_textures":decoded,"fallback_primitives":fallback,
        "policy":"learned chromaticity resolves only connected lower-garment contacts; unseen or ambiguous colors retain geometric ownership"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palettes_follow_recoloring_and_abstain_without_color_separation() {
        for (cloth, shoe) in [
            (Vec3::new(0.12, 0.25, 0.8), Vec3::new(0.2, 0.65, 0.1)),
            (Vec3::new(0.8, 0.1, 0.15), Vec3::splat(0.5)),
        ] {
            let mut positive = vec![chromaticity(cloth).unwrap(); 8];
            let mut negative = vec![chromaticity(shoe).unwrap(); 8];
            let palette = Palettes::learn(&mut positive, &mut negative).unwrap();
            assert_eq!(palette.role(chromaticity(cloth * 0.4).unwrap()), Some(true));
            assert_eq!(palette.role(chromaticity(shoe * 0.6).unwrap()), Some(false));
            assert_eq!(palette.role((palette.garment + palette.foot) * 0.5), None);
        }
        assert!(
            Palettes::learn(
                &mut [Vec3::splat(1.0 / 3.0); 8],
                &mut [Vec3::splat(1.0 / 3.0); 8]
            )
            .is_none()
        );
    }

    #[test]
    fn pale_body_islands_are_not_turned_into_garments_without_surface_connection() {
        let geometry = Geometry {
            points: vec![
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::X + Vec3::Y,
                Vec3::Z,
                Vec3::Z + Vec3::X,
                Vec3::Z + Vec3::Y,
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2], [4, 5, 6]],
            primitives: Vec::new(),
        };
        let actual = connected_garment(
            &geometry,
            &[true, false, false],
            &[None, Some(true), Some(true)],
        );
        assert_eq!(actual, vec![true, true, false]);
    }
}
