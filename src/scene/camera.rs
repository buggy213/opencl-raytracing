use gltf::camera::{Projection, Perspective};

use crate::geometry::{Transform, Matrix4x4, Vec3, Vec2};

pub struct Camera {
    pub is_perspective: bool,
    pub camera_position: [f32; 3],
    pub world_to_raster: Transform,
    pub raster_width: usize,
    pub raster_height: usize,
    pub near_clip: f32,
    pub far_clip: f32
}

const DEFAULT_FAR_CLIP: f32 = 1000.0;

impl Camera {
    fn create_screen_to_raster_transform(width: usize, height: usize, screen_space_top_left: Vec3, screen_space_bottom_right: Vec3) -> Transform {
        let screen_to_zero: Transform = Transform::translate(-screen_space_top_left); // ignore depth
        let scaling_x: f32 = (screen_space_bottom_right - screen_space_top_left).x();
        let scaling_y: f32 = (screen_space_bottom_right - screen_space_top_left).y();
        let screen_to_ndc: Transform = 
            screen_to_zero.compose(Transform::scale(Vec3(1.0 / scaling_x, 1.0 / scaling_y, 1.0)));
        let screen_to_raster: Transform = 
            screen_to_ndc.compose(Transform::scale(Vec3(width as f32, height as f32, 1.0)));
        return screen_to_raster;
    }
    

    // Creates transform from camera space to raster space through screen space
    // fov given in degrees
    fn create_perspective_transform(far_clip: f32, near_clip: f32, yfov: f32, width: usize, height: usize) -> Transform {
        let persp: Matrix4x4 = Matrix4x4::create(
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, far_clip / (far_clip - near_clip), - (far_clip * near_clip) / (far_clip - near_clip), 
            0.0, 0.0, 1.0, 0.0
        );

        let persp: Transform = Transform::from(persp);
        let wide: bool = width >= height;
        let fov: f32 = if wide { yfov * (width as f32 / height as f32) } else { yfov };
        let invt: f32 = 1.0 / f32::tan(fov.to_radians() / 2.0);
        let fov_scale: Transform = Transform::scale(Vec3(-invt, invt, 1.0)); // not sure why flipping x coordinate is required, but oh well
        // if image is wide, screen space x ranges from -1 to 1 and screen space y ranges from -k to k where k is proportionally smaller
        // if image is tall, screen space x ranges from -k to k and screen space y ranges from -1 to 1 where k is proportionally smaller
        let screen_space_top_left: Vec3 = if wide { Vec3(-1.0, -(height as f32 / width as f32), 0.0) } 
            else { Vec3(-(width as f32 / height as f32), -1.0, 0.0) };
        let screen_space_bottom_right: Vec3 = if wide { Vec3(1.0, height as f32 / width as f32, 0.0) } 
            else { Vec3(width as f32 / height as f32, 1.0, 0.0) };
        let screen_to_raster = Self::create_screen_to_raster_transform(width, height, screen_space_top_left, screen_space_bottom_right);
        return persp.compose(fov_scale).compose(screen_to_raster);
    }
    
    fn create_orthographic_transform(far_clip: f32, near_clip: f32, width: usize, height: usize, screen_space_width: f32, screen_space_height: f32) -> Transform {
        let translate: Transform = Transform::translate(Vec3(0.0, 0.0, -near_clip));
        let scale: Transform = Transform::scale(Vec3(1.0, 1.0, 1.0 / (far_clip - near_clip)));
        let screen_space_top_left: Vec3 = Vec3(-screen_space_width / 2.0, -screen_space_height / 2.0, 0.0);
        let screen_space_bottom_right: Vec3 = Vec3(screen_space_width / 2.0, screen_space_height / 2.0, 0.0);
        let screen_to_raster: Transform = Self::create_screen_to_raster_transform(width, height, screen_space_top_left, screen_space_bottom_right);
        return translate.compose(scale).compose(screen_to_raster);
    }

    pub fn from_gltf_camera_node(camera_node: &gltf::Node, raster_height: usize) -> Camera {
        let gltf_camera: gltf::Camera = camera_node.camera().unwrap();
        let camera_position: [f32; 3] = camera_node.transform().decomposed().0;
        let mut camera_to_world_matrix: Matrix4x4 = Matrix4x4{ data: camera_node.transform().matrix() };
        camera_to_world_matrix.transpose();
        let world_to_camera_matrix: Matrix4x4 = camera_to_world_matrix.invert().unwrap();
        let world_to_camera: Transform = Transform::from(world_to_camera_matrix);
        let is_perspective;
        let (world_to_raster, raster_width) = 
        match gltf_camera.projection() {
            Projection::Perspective(perspective) => {
                let width: usize = (raster_height as f32 * perspective.aspect_ratio().unwrap()) as usize;
                let camera_to_raster: Transform = Self::create_perspective_transform(
                    -perspective.zfar().unwrap_or(DEFAULT_FAR_CLIP),
                    -perspective.znear(), 
                    perspective.yfov().to_degrees(), 
                    width, 
                    raster_height
                );
                let world_to_raster: Transform = world_to_camera.compose(camera_to_raster);
                is_perspective = true;
                (world_to_raster, width)
            },
            Projection::Orthographic(orthographic) => {
                let screen_space_width: f32 = orthographic.xmag();
                let screen_space_height: f32 = orthographic.ymag();
                println!("ssheight={}, sswidth={}", screen_space_height, screen_space_width);
                let width: usize = (raster_height as f32 * screen_space_width / screen_space_height) as usize;
                let camera_to_raster: Transform = Self::create_orthographic_transform(
                    -orthographic.zfar(), 
                    -orthographic.znear(), 
                    width, 
                    raster_height, 
                    screen_space_width, // idk why i need to flip this but i do
                    -screen_space_height
                );
                let world_to_raster: Transform = world_to_camera.compose(camera_to_raster);
                is_perspective = false;
                (world_to_raster, width)
            }
        };

        Camera {
            is_perspective,
            camera_position,
            world_to_raster,
            raster_width,
            raster_height,
            near_clip: 0.01,
            far_clip: 1000.0
        }
    }
}