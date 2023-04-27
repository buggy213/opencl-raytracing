use std::path::Path;

use gltf::{camera::{Projection, Perspective}, Node};

use crate::{geometry::{Transform, Mesh, Matrix4x4}};

use super::camera::Camera;

pub struct Scene {
    pub camera: Camera,
    pub mesh: (Mesh, Transform)
}

const HEIGHT: usize = 512;

impl Scene {
    pub fn from_file(filepath: &Path) -> anyhow::Result<Scene> {
        let (scene_gltf, buffers, images) = gltf::import(filepath)?;
        let scene_gltf = scene_gltf.default_scene().unwrap();
        assert!(scene_gltf.nodes().len() == 2);
        let mut camera: Option<Camera> = None;
        let height: usize = HEIGHT;
        let mut mesh: Option<Mesh> = None;
        let mut mesh_transform: Option<Transform> = None;

        for node in scene_gltf.nodes() {
            if node.camera().is_some() {
                camera = Some(Camera::from_gltf_camera_node(&node, height));
            }
            if node.mesh().is_some() {
                let gltf_mesh = node.mesh().unwrap();
                mesh = Some(Mesh::from_gltf_mesh(&gltf_mesh, &buffers));
                let mut mesh_transform_matrix = Matrix4x4 { data: node.transform().matrix() };
                mesh_transform_matrix.transpose();
                mesh_transform = Some(Transform::from(mesh_transform_matrix));
            }
        }

        
        Ok(Scene {
            camera: camera.unwrap(),
            mesh: (mesh.unwrap(), mesh_transform.unwrap())
        })
    }
}