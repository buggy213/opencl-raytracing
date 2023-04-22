use std::path::Path;

use gltf::camera::{Projection, Perspective};

use crate::{geometry::{Transform, Mesh, Matrix4x4}, create_perspective_transform};

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
        let mut camera;
        let height = HEIGHT;
        let mut width;
        let mut mesh;
        let mut mesh_transform;
        let mut nodes_iter = scene_gltf.nodes();

        let node = nodes_iter.next().unwrap();
        let gltf_camera = node.camera().unwrap();
        match gltf_camera.projection() {
            Projection::Perspective(perspective) => {
                width = (HEIGHT as f32 * perspective.aspect_ratio().unwrap()) as usize;
                let mut camera_to_world_matrix = Matrix4x4{ data: node.transform().matrix() };
                camera_to_world_matrix.transpose();
                let camera_position = node.transform().decomposed().0;
                let world_to_camera_matrix = camera_to_world_matrix.invert().unwrap();
                let world_to_camera = Transform::from(world_to_camera_matrix);
                println!("{:?}", world_to_camera);
                let camera_to_raster = create_perspective_transform(-perspective.zfar().unwrap(),-perspective.znear(), perspective.yfov().to_degrees(), width, height);
                let world_to_raster = world_to_camera.compose(camera_to_raster);
                camera = Camera {
                    camera_position,
                    world_to_raster,
                    raster_width: width,
                    raster_height: height,
                }
            },
            _ => panic!("unsupported projection type")
        }

        let node = nodes_iter.next().unwrap();
        let gltf_mesh = node.mesh().unwrap();
        mesh = Mesh::from_gltf_mesh(&gltf_mesh, &buffers);
        let mut mesh_transform_matrix = Matrix4x4 { data: node.transform().matrix() };
        mesh_transform_matrix.transpose();
        println!("{:?}", mesh_transform_matrix);
        mesh_transform = Transform::from(mesh_transform_matrix);

        println!("{}, {}, {:?}", camera.raster_height, camera.raster_width, camera.world_to_raster);

        Ok(Scene {
            camera,
            mesh: (mesh, mesh_transform)
        })
    }
}