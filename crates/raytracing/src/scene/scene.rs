use std::{collections::HashMap, ops::Range, path::Path};

use crate::{geometry::{Matrix4x4, Mesh, Shape, Transform, Vec3}, lights::Light, materials::Material, scene::primitive::{AggregatePrimitive, BasicPrimitive, Primitive, PrimitiveIndex, TransformIndex, TransformPrimitive}};

use super::camera::{Camera, RenderTile};

pub struct Scene {
    pub camera: Camera,
    
    transforms: Vec<Transform>,

    primitives: Vec<Primitive>,
    root_primitive: PrimitiveIndex,

    lights: Vec<Light>,

    materials: Vec<Material>
}

const HEIGHT: usize = 600;

impl Scene {
    pub fn from_gltf_file(filepath: &Path, render_tile: Option<RenderTile>) -> anyhow::Result<Scene> {
        let (document, buffers, _images) = gltf::import(filepath)?;
        let scene_gltf = document.default_scene().unwrap();
        let mut camera: Option<Camera> = None;  
        let height: usize = HEIGHT;
        
        // detect gltf level instancing (i.e. two nodes with the same "mesh" attribute)
        let mut instancing_map: HashMap<usize, Range<PrimitiveIndex>> = HashMap::new();

        let mut transforms: Vec<Transform> = Vec::new();
        let mut primitives: Vec<Primitive> = Vec::new();
        let mut lights: Vec<Light> = Vec::new();

        let mut root_primitive_children: Vec<PrimitiveIndex> = Vec::new();

        let mut materials: Vec<Material> = Vec::new();
        let mut material_emissions: Vec<Vec3> = Vec::new();
        for material in document.materials() {
            let diffuse_color = material.pbr_metallic_roughness().base_color_factor();
            let diffuse = Material::Diffuse { 
                albedo: [diffuse_color[0], diffuse_color[1], diffuse_color[2]].into()
            };

            materials.push(diffuse);

            let emission: Vec3 = material.emissive_factor().into();
            material_emissions.push(emission);
        }
        
        for node in scene_gltf.nodes() {
            if node.camera().is_some() {
                camera = Some(Camera::from_gltf_camera_node(&node, height, render_tile));
            }

            if node.mesh().is_some() {
                let gltf_mesh = node.mesh().unwrap();

                let mut transform_matrix = Matrix4x4 { data: node.transform().matrix() };
                transform_matrix.transpose();
                let transform = Transform::from(transform_matrix);

                
                // check if this mesh is instanced
                if let Some(range) = instancing_map.get(&gltf_mesh.index()) {
                    for i in range.clone() {
                        let transform_primitive = Primitive::Transform(TransformPrimitive {
                            primitive: i,
                            transform: transform.clone(),
                        });
                        let transform_primitive_idx = primitives.len() as PrimitiveIndex;
                        primitives.push(transform_primitive);
                        root_primitive_children.push(transform_primitive_idx);
                    }
                    continue;
                }

                let start = primitives.len() as PrimitiveIndex;
                for gltf_primitive in gltf_mesh.primitives() {
                    let primitive_id = primitives.len() as u32;
                    let material_idx = gltf_primitive.material().index().unwrap_or(0) as u32;

                    let rt_mesh = Mesh::from_gltf_primitive(gltf_primitive, &buffers);
                    let mesh_shape = Shape::TriangleMesh(rt_mesh);
                    
                    let area_light_idx = if material_emissions[material_idx as usize] != Vec3::zero() {
                        let area_light = Light::from_emissive_geometry(primitive_id, material_emissions[material_idx as usize]);
                        let area_light_idx = lights.len() as u32;
                        lights.push(area_light);
                        Some(area_light_idx)
                    } else { None };

                    let mesh_primitive = Primitive::Basic(BasicPrimitive {
                        shape: mesh_shape,
                        material: material_idx,
                        area_light: area_light_idx,
                    });

                    primitives.push(mesh_primitive);
                };
                let end = primitives.len() as PrimitiveIndex;
            
                // record the range of this mesh in the instancing map
                instancing_map.insert(gltf_mesh.index(), start..end);

                // create transform primitives for them
                for gltf_primitive in start..end {
                    let primitive_id = gltf_primitive as PrimitiveIndex;
                    
                    let transform_primitive = Primitive::Transform(TransformPrimitive {
                        primitive: primitive_id,
                        transform: transform.clone(),
                    });

                    let transform_primitive_idx = primitives.len() as PrimitiveIndex;
                    primitives.push(transform_primitive);
                    root_primitive_children.push(transform_primitive_idx);
                }
            }

            if node.light().is_some() {
                let punctual_light = Light::from_gltf_punctual_light(&node);
                lights.push(punctual_light);
            }
        }

        let root_primitive = Primitive::Aggregate(AggregatePrimitive {
            children: root_primitive_children
        });
        let root_primitive_idx = primitives.len() as u32;
        primitives.push(root_primitive);

        Ok(Scene {
            lights,
            transforms,
            primitives,
            root_primitive: root_primitive_idx,
            camera: camera.expect("Scene must have camera"),
            materials
        })
    }

    pub fn test_scene() -> Scene {
        todo!()
    }
}