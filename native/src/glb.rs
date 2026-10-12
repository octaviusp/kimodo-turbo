//! Preserve source GLB payloads while baking mesh transforms and adding a skin.
use crate::{anatomy::Skeleton, weights::SkinWeights};
use glam::{Mat4, Vec3};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};

type Result<T> = std::result::Result<T, String>;

pub struct Glb {
    pub json: Value,
    pub binary: Vec<u8>,
    pub original: Vec<u8>,
}

pub struct Primitive {
    pub node: usize,
    pub source: Value,
    pub offset: usize,
    pub count: usize,
    pub normals: Option<Vec<Vec3>>,
    pub triangles: Vec<[u32; 3]>,
    pub source_vertices: Vec<usize>,
    pub texcoords: Option<Vec<[f32; 2]>>,
}

pub struct Geometry {
    pub points: Vec<Vec3>,
    pub triangles: Vec<[usize; 3]>,
    pub primitives: Vec<Primitive>,
}

impl Glb {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        let word = |i: usize| -> Result<u32> {
            bytes
                .get(i..i + 4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .ok_or_else(|| "Truncated GLB header".into())
        };
        if word(0)? != 0x46546c67 || word(4)? != 2 || word(8)? as usize != bytes.len() {
            return Err("Expected a complete glTF 2.0 binary file".into());
        }
        let mut cursor = 12;
        let mut document = None;
        let mut binary = None;
        while cursor < bytes.len() {
            let length = word(cursor)? as usize;
            let kind = word(cursor + 4)?;
            let chunk = bytes
                .get(cursor + 8..cursor + 8 + length)
                .ok_or("Truncated GLB chunk")?;
            match kind {
                0x4e4f534a if document.is_none() => {
                    document = Some(serde_json::from_slice(chunk).map_err(|e| e.to_string())?)
                }
                0x004e4942 if binary.is_none() => binary = Some(chunk.to_vec()),
                _ => return Err("Unsupported or duplicate GLB chunk".into()),
            }
            cursor += 8 + length;
        }
        let json: Value = document.ok_or("Missing GLB JSON")?;
        gltf::Gltf::from_slice(&bytes).map_err(|e| format!("Invalid glTF: {e}"))?;
        if json["buffers"].as_array().is_none_or(|a| a.len() != 1)
            || json["buffers"][0].get("uri").is_some()
        {
            return Err("Use a self-contained GLB with one embedded buffer".into());
        }
        if let Some(extensions) = json["extensionsRequired"].as_array() {
            for extension in extensions {
                if matches!(
                    extension.as_str(),
                    Some("KHR_draco_mesh_compression" | "EXT_meshopt_compression")
                ) {
                    return Err("Decompress the mesh before rigging".into());
                }
            }
        }
        let mut binary = binary.ok_or("Missing GLB binary buffer")?;
        let size = json["buffers"][0]["byteLength"]
            .as_u64()
            .ok_or("Missing buffer length")? as usize;
        if size > binary.len() {
            return Err("Buffer length exceeds BIN chunk".into());
        }
        binary.truncate(size);
        Ok(Self {
            json,
            binary,
            original: bytes,
        })
    }

    pub fn geometry(&self) -> Result<Geometry> {
        let document =
            gltf::Gltf::from_slice(&self.original).map_err(|e| format!("Invalid glTF: {e}"))?;
        let nodes: Vec<_> = document.nodes().collect();
        let scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .ok_or("No scene")?;
        let mut world = vec![None; nodes.len()];
        fn visit(node: gltf::Node, parent: Mat4, worlds: &mut [Option<Mat4>]) -> Result<()> {
            if worlds[node.index()].is_some() {
                return Err("Node reused or cyclic scene graph".into());
            }
            let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
            if !m.is_finite() || m.determinant().abs() < 1e-12 {
                return Err("Singular or nonfinite node transform".into());
            }
            worlds[node.index()] = Some(m);
            for child in node.children() {
                visit(child, m, worlds)?;
            }
            Ok(())
        }
        for root in scene.nodes() {
            visit(root, Mat4::IDENTITY, &mut world)?;
        }
        let mut geometry = Geometry {
            points: vec![],
            triangles: vec![],
            primitives: vec![],
        };
        for node in nodes {
            let Some(world) = world[node.index()] else {
                continue;
            };
            let Some(mesh) = node.mesh() else {
                continue;
            };
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err("Only triangle primitives can be rigged".into());
                }
                if primitive.morph_targets().next().is_some() {
                    return Err("Bake morph targets before re-rigging".into());
                }
                let reader = primitive
                    .reader(|buffer| (buffer.index() == 0).then_some(self.binary.as_slice()));
                let positions: Vec<_> = reader
                    .read_positions()
                    .ok_or("Primitive lacks POSITION")?
                    .map(|p| world.transform_point3(Vec3::from(p)))
                    .collect();
                if positions.iter().any(|v| !v.is_finite()) {
                    return Err("Nonfinite vertex positions".into());
                }
                let offset = geometry.points.len();
                let count = positions.len();
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|i| i.into_u32().collect())
                    .unwrap_or_else(|| (0..count as u32).collect());
                if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= count)
                {
                    return Err("Invalid triangle indices".into());
                }
                let triangles: Vec<_> = indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| {
                        if world.determinant() < 0.0 {
                            [t[0], t[2], t[1]]
                        } else {
                            [t[0], t[1], t[2]]
                        }
                    })
                    .collect();
                let normal_matrix = world.inverse().transpose();
                let normals = reader.read_normals().map(|values| {
                    values
                        .map(|n| {
                            normal_matrix
                                .transform_vector3(Vec3::from(n))
                                .normalize_or_zero()
                        })
                        .collect()
                });
                let texcoords = reader
                    .read_tex_coords(0)
                    .map(|values| values.into_f32().collect());
                geometry
                    .triangles
                    .extend(triangles.iter().map(|t| t.map(|i| offset + i as usize)));
                geometry.points.extend(positions);
                geometry.primitives.push(Primitive {
                    node: node.index(),
                    source: self.json["meshes"][mesh.index()]["primitives"][primitive.index()]
                        .clone(),
                    offset,
                    count,
                    normals,
                    triangles,
                    source_vertices: (0..count).collect(),
                    texcoords,
                });
            }
        }
        if geometry.points.len() < 30 {
            return Err("Too few vertices for a humanoid surface".into());
        }
        Ok(geometry)
    }

    pub fn has_author_skin(&self) -> bool {
        self.json["skins"].as_array().is_some_and(|skins| {
            !skins.is_empty()
                && skins.iter().all(|s| {
                    !s["name"].as_str().unwrap_or("").contains("autorig")
                        && s["joints"].as_array().is_some_and(|j| j.len() >= 15)
                })
        })
    }

    fn accessor(
        &mut self,
        data: &[u8],
        component: u32,
        kind: &str,
        count: usize,
        target: Option<u32>,
        bounds: Option<(Vec3, Vec3)>,
    ) -> usize {
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
        let offset = self.binary.len();
        self.binary.extend_from_slice(data);
        if !self.json["bufferViews"].is_array() {
            self.json["bufferViews"] = json!([]);
        }
        let views = self.json["bufferViews"].as_array_mut().unwrap();
        let index = views.len();
        let mut view = json!({"buffer":0,"byteOffset":offset,"byteLength":data.len()});
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        views.push(view);
        if !self.json["accessors"].is_array() {
            self.json["accessors"] = json!([]);
        }
        let accessors = self.json["accessors"].as_array_mut().unwrap();
        let id = accessors.len();
        let mut accessor =
            json!({"bufferView":index,"componentType":component,"count":count,"type":kind});
        if let Some((min, max)) = bounds {
            accessor["min"] = json!(min.to_array());
            accessor["max"] = json!(max.to_array());
        }
        accessors.push(accessor);
        id
    }

    fn duplicate_attribute(&mut self, source: usize, vertices: &[usize]) -> Result<usize> {
        let accessor = self.json["accessors"][source].clone();
        if accessor.get("sparse").is_some() {
            return Err("Bake sparse vertex attributes before separating a garment seam".into());
        }
        let component = accessor["componentType"]
            .as_u64()
            .ok_or("Missing attribute component type")? as u32;
        let component_size = match component {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err("Unsupported vertex attribute component".into()),
        };
        let kind = accessor["type"].as_str().ok_or("Missing attribute type")?;
        let components = match kind {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => return Err("Unsupported vertex attribute shape".into()),
        };
        let size = component_size * components;
        let count = accessor["count"]
            .as_u64()
            .ok_or("Missing attribute count")? as usize;
        let view_id = accessor["bufferView"]
            .as_u64()
            .ok_or("Missing vertex attribute buffer")? as usize;
        let view = &self.json["bufferViews"][view_id];
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize
            + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = view["byteStride"].as_u64().unwrap_or(size as u64) as usize;
        let mut data = Vec::with_capacity(vertices.len() * size);
        for &index in vertices {
            if index >= count {
                return Err("Garment seam refers to a missing vertex attribute".into());
            }
            let offset = start + index * stride;
            data.extend_from_slice(
                self.binary
                    .get(offset..offset + size)
                    .ok_or("Truncated vertex attribute")?,
            );
        }
        let result = self.accessor(&data, component, kind, vertices.len(), Some(34962), None);
        for field in ["normalized", "min", "max"] {
            if let Some(value) = accessor.get(field) {
                self.json["accessors"][result][field] = value.clone();
            }
        }
        Ok(result)
    }

    pub fn bind(
        &mut self,
        geometry: &Geometry,
        skeleton: &Skeleton,
        skin: &SkinWeights,
    ) -> Result<()> {
        let obsolete: BTreeSet<usize> = self.json["skins"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|s| s["joints"].as_array().into_iter().flatten())
            .filter_map(|v| v.as_u64().map(|i| i as usize))
            .collect();
        let first = self.json["nodes"].as_array().ok_or("No nodes")?.len();
        let joint_nodes: Vec<_> = (first..first + skeleton.joints.len()).collect();
        for (i, &point) in skeleton.joints.iter().enumerate() {
            let parent = skeleton.parents[i];
            let translation = point
                - if parent >= 0 {
                    skeleton.joints[parent as usize]
                } else {
                    Vec3::ZERO
                };
            let children: Vec<_> = skeleton
                .parents
                .iter()
                .enumerate()
                .filter(|(_, p)| **p == i as i32)
                .map(|(j, _)| first + j)
                .collect();
            let mut node = json!({"name":skeleton.names[i],"translation":translation.to_array()});
            if !children.is_empty() {
                node["children"] = json!(children);
            }
            self.json["nodes"].as_array_mut().unwrap().push(node);
        }
        let matrices: Vec<u8> = skeleton
            .joints
            .iter()
            .flat_map(|&p| Mat4::from_translation(-p).to_cols_array())
            .flat_map(f32::to_le_bytes)
            .collect();
        let inverse_bind =
            self.accessor(&matrices, 5126, "MAT4", skeleton.joints.len(), None, None);
        self.json["skins"] = json!([{"name":"kimodo-surface-autorig-v1","joints":joint_nodes,"skeleton":first,"inverseBindMatrices":inverse_bind}]);
        self.json.as_object_mut().unwrap().remove("animations");
        let mesh_nodes: BTreeSet<_> = geometry.primitives.iter().map(|p| p.node).collect();
        let mut meshes = vec![];
        for &node_id in &mesh_nodes {
            let mut primitives = vec![];
            for primitive in geometry.primitives.iter().filter(|p| p.node == node_id) {
                let range = primitive.offset..primitive.offset + primitive.count;
                let points = &geometry.points[range.clone()];
                let min = points
                    .iter()
                    .fold(Vec3::splat(f32::INFINITY), |a, &p| a.min(p));
                let max = points
                    .iter()
                    .fold(Vec3::splat(f32::NEG_INFINITY), |a, &p| a.max(p));
                let bytes: Vec<u8> = points
                    .iter()
                    .flat_map(|p| p.to_array())
                    .flat_map(f32::to_le_bytes)
                    .collect();
                let position = self.accessor(
                    &bytes,
                    5126,
                    "VEC3",
                    points.len(),
                    Some(34962),
                    Some((min, max)),
                );
                let joints: Vec<u8> = skin.joints[range.clone()]
                    .iter()
                    .flatten()
                    .flat_map(|&i| i.to_le_bytes())
                    .collect();
                let weights: Vec<u8> = skin.weights[range.clone()]
                    .iter()
                    .flatten()
                    .flat_map(|&w| w.to_le_bytes())
                    .collect();
                let joint_accessor =
                    self.accessor(&joints, 5123, "VEC4", points.len(), Some(34962), None);
                let weight_accessor =
                    self.accessor(&weights, 5126, "VEC4", points.len(), Some(34962), None);
                let indices: Vec<u8> = primitive
                    .triangles
                    .iter()
                    .flatten()
                    .flat_map(|&i| i.to_le_bytes())
                    .collect();
                let index_accessor = self.accessor(
                    &indices,
                    5125,
                    "SCALAR",
                    primitive.triangles.len() * 3,
                    Some(34963),
                    None,
                );
                let mut output = primitive.source.clone();
                if primitive
                    .source_vertices
                    .iter()
                    .enumerate()
                    .any(|(i, &source)| i != source)
                {
                    let attributes = output["attributes"]
                        .as_object()
                        .ok_or("Missing vertex attributes")?
                        .clone();
                    for (name, accessor) in attributes {
                        if matches!(
                            name.as_str(),
                            "POSITION"
                                | "NORMAL"
                                | "TANGENT"
                                | "JOINTS_0"
                                | "JOINTS_1"
                                | "WEIGHTS_0"
                                | "WEIGHTS_1"
                        ) {
                            continue;
                        }
                        let source = accessor
                            .as_u64()
                            .ok_or("Invalid vertex attribute reference")?
                            as usize;
                        output["attributes"][&name] =
                            json!(self.duplicate_attribute(source, &primitive.source_vertices)?);
                    }
                }
                output["attributes"]["POSITION"] = json!(position);
                output["attributes"]["JOINTS_0"] = json!(joint_accessor);
                output["attributes"]["WEIGHTS_0"] = json!(weight_accessor);
                output["attributes"]
                    .as_object_mut()
                    .unwrap()
                    .remove("JOINTS_1");
                output["attributes"]
                    .as_object_mut()
                    .unwrap()
                    .remove("WEIGHTS_1");
                output["indices"] = json!(index_accessor);
                if let Some(normals) = &primitive.normals {
                    let bytes: Vec<u8> = normals
                        .iter()
                        .flat_map(|p| p.to_array())
                        .flat_map(f32::to_le_bytes)
                        .collect();
                    output["attributes"]["NORMAL"] = json!(self.accessor(
                        &bytes,
                        5126,
                        "VEC3",
                        normals.len(),
                        Some(34962),
                        None
                    ));
                }
                output["attributes"]
                    .as_object_mut()
                    .unwrap()
                    .remove("TANGENT");
                primitives.push(output);
            }
            let mesh_id = meshes.len();
            meshes.push(json!({"primitives":primitives}));
            let node = &mut self.json["nodes"][node_id];
            node["mesh"] = json!(mesh_id);
            node["skin"] = json!(0);
            for field in ["matrix", "translation", "rotation", "scale"] {
                node.as_object_mut().unwrap().remove(field);
            }
        }
        self.json["meshes"] = json!(meshes);
        for node in self.json["nodes"].as_array_mut().unwrap() {
            if let Some(children) = node.get_mut("children").and_then(Value::as_array_mut) {
                children.retain(|i| {
                    i.as_u64()
                        .is_none_or(|i| !mesh_nodes.contains(&(i as usize)))
                });
            }
        }
        let scene = self.json["scene"].as_u64().unwrap_or(0) as usize;
        let mut roots: Vec<usize> = self.json["scenes"][scene]["nodes"]
            .as_array()
            .ok_or("No scene roots")?
            .iter()
            .filter_map(|i| i.as_u64().map(|i| i as usize))
            .filter(|i| !mesh_nodes.contains(i))
            .collect();
        roots.extend(mesh_nodes);
        roots.push(first);
        self.json["scenes"][scene]["nodes"] = json!(roots);
        self.json["buffers"][0]["byteLength"] = json!(self.binary.len());
        self.json["asset"]["generator"] = json!("kimodo-autorig 0.1.0");
        // Retired joints must not collide with the new canonical names in GLTFLoader.
        if !obsolete.is_empty() {
            let nodes = self.json["nodes"].as_array().unwrap();
            let mut remap = vec![None; nodes.len()];
            let mut retained = vec![];
            for (i, node) in nodes.iter().enumerate() {
                if !obsolete.contains(&i) {
                    remap[i] = Some(retained.len());
                    retained.push(node.clone());
                }
            }
            for node in &mut retained {
                if let Some(children) = node.get_mut("children").and_then(Value::as_array_mut) {
                    *children = children
                        .iter()
                        .filter_map(|v| {
                            v.as_u64().and_then(|i| remap[i as usize]).map(|i| json!(i))
                        })
                        .collect();
                }
            }
            for scene in self.json["scenes"].as_array_mut().unwrap() {
                if let Some(nodes) = scene.get_mut("nodes").and_then(Value::as_array_mut) {
                    *nodes = nodes
                        .iter()
                        .filter_map(|v| {
                            v.as_u64().and_then(|i| remap[i as usize]).map(|i| json!(i))
                        })
                        .collect();
                }
            }
            let skin = &mut self.json["skins"][0];
            skin["joints"] = json!(
                joint_nodes
                    .iter()
                    .map(|&i| remap[i].unwrap())
                    .collect::<Vec<_>>()
            );
            skin["skeleton"] = json!(remap[first].unwrap());
            self.json["nodes"] = json!(retained);
        }
        Ok(())
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut json = serde_json::to_vec(&self.json).map_err(|e| e.to_string())?;
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        let mut binary = self.binary.clone();
        while !binary.len().is_multiple_of(4) {
            binary.push(0);
        }
        let length = 12 + 8 + json.len() + 8 + binary.len();
        let mut data = Vec::with_capacity(length);
        for value in [0x46546c67, 2, length as u32, json.len() as u32, 0x4e4f534a] {
            data.extend(value.to_le_bytes());
        }
        data.extend(json);
        for value in [binary.len() as u32, 0x004e4942] {
            data.extend(value.to_le_bytes());
        }
        data.extend(binary);
        gltf::Gltf::from_slice(&data)
            .map_err(|e| format!("Generated GLB validation failed: {e}"))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let temp = path.with_extension("glb.tmp");
        fs::write(&temp, data).map_err(|e| e.to_string())?;
        fs::rename(temp, path).map_err(|e| e.to_string())
    }
}
