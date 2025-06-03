use std::path::Path;

use crate::{geometry::{Matrix4x4, Mesh, Transform, Vec3}, lights::Light, materials::Material};

use super::camera::{Camera, RenderTile};

pub struct Scene {
    pub camera: Camera,
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,

    pub materials: Vec<Material>
}

const HEIGHT: usize = 600;

impl Scene {
    pub fn from_file(filepath: &Path, render_tile: Option<RenderTile>) -> anyhow::Result<Scene> {
        let (document, buffers, _images) = gltf::import(filepath)?;
        let scene_gltf = document.default_scene().unwrap();
        let mut camera: Option<Camera> = None;  
        let height: usize = HEIGHT;
        let mut meshes: Vec<Mesh> = Vec::new();
        let mut lights: Vec<Light> = Vec::new();

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
                let mut mesh = Mesh::from_gltf_mesh(&gltf_mesh, &buffers);
                
                let mut mesh_transform_matrix = Matrix4x4 { data: node.transform().matrix() };
                mesh_transform_matrix.transpose();
                let mesh_transform = Transform::from(mesh_transform_matrix);

                // convert to world space
                mesh.apply_transform(&mesh_transform);

                // if it's an emitter, then add an area light
                let prim_id = meshes.len() as u32;
                let emission = material_emissions[mesh.material_idx as usize];
                if emission != Vec3::zero() {
                    mesh.light_idx = Some(lights.len() as u32);

                    let area_light = Light::from_emissive_geometry(prim_id, emission);
                    lights.push(area_light);
                }

                meshes.push(mesh);
            }
            if node.light().is_some() {
                let punctual_light = Light::from_gltf_punctual_light(&node);
                lights.push(punctual_light);
            }
        }

        println!("{:?}", lights);
        Ok(Scene {
            lights,
            meshes,
            camera: camera.unwrap(),
            materials
        })
    }
}