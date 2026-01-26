use gltf::camera::Projection;

use crate::geometry::{Matrix4x4, Quaternion, Transform, Vec3};

#[derive(Debug, Clone, Copy)]
pub enum CameraType {
    Orthographic {
        screen_space_width: f32,
        screen_space_height: f32,
    },
    PinholePerspective {
        yfov: f32, // radians
    },
    ThinLensPerspective {
        yfov: f32,            // radians
        aperture_radius: f32, // lens radius in world units
        focal_distance: f32,  // distance to focal plane in camera space
    },
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub camera_position: Vec3,
    pub camera_rotation: Quaternion,

    pub camera_type: CameraType,
    pub raster_width: usize,
    pub raster_height: usize,
    pub near_clip: f32,
    pub far_clip: f32,

    pub world_to_raster: Transform,

    pub camera_to_world: Transform,
    pub raster_to_camera: Transform,
}

const DEFAULT_FAR_CLIP: f32 = 1000.0;

impl Camera {
    fn create_screen_to_raster_transform(
        width: usize,
        height: usize,
        screen_space_top_left: Vec3,
        screen_space_bottom_right: Vec3,
    ) -> Transform {
        let screen_to_zero: Transform = Transform::translate(-screen_space_top_left); // ignore depth
        let scaling_x: f32 = (screen_space_bottom_right - screen_space_top_left).x();
        let scaling_y: f32 = (screen_space_bottom_right - screen_space_top_left).y();
        let screen_to_ndc: Transform = screen_to_zero.compose(Transform::scale(Vec3(
            1.0 / scaling_x,
            1.0 / scaling_y,
            1.0,
        )));
        let screen_to_raster: Transform =
            screen_to_ndc.compose(Transform::scale(Vec3(width as f32, height as f32, 1.0)));

        screen_to_raster
    }

    // Creates transform from camera space to raster space through screen space
    // fov given in radians
    fn create_perspective_transform(
        far_clip: f32,
        near_clip: f32,
        yfov: f32,
        width: usize,
        height: usize,
    ) -> Transform {
        #[rustfmt::skip]
        let persp: Matrix4x4 = Matrix4x4::create(
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, far_clip / (far_clip - near_clip), - (far_clip * near_clip) / (far_clip - near_clip), 
            0.0, 0.0, 1.0, 0.0
        );

        let persp: Transform = Transform::from(persp);
        let wide: bool = width >= height;
        let fov: f32 = if wide {
            yfov * (width as f32 / height as f32)
        } else {
            yfov
        };
        let invt: f32 = 1.0 / f32::tan(fov / 2.0);
        let fov_scale: Transform = Transform::scale(Vec3(-invt, -invt, 1.0)); // flip both X and Y to match raster convention (Y=0 at top)
                                                                             // if image is wide, screen space x ranges from -1 to 1 and screen space y ranges from -k to k where k is proportionally smaller
                                                                             // if image is tall, screen space x ranges from -k to k and screen space y ranges from -1 to 1 where k is proportionally smaller
        let screen_space_top_left: Vec3 = if wide {
            Vec3(-1.0, -(height as f32 / width as f32), 0.0)
        } else {
            Vec3(-(width as f32 / height as f32), -1.0, 0.0)
        };
        let screen_space_bottom_right: Vec3 = if wide {
            Vec3(1.0, height as f32 / width as f32, 0.0)
        } else {
            Vec3(width as f32 / height as f32, 1.0, 0.0)
        };
        let screen_to_raster = Self::create_screen_to_raster_transform(
            width,
            height,
            screen_space_top_left,
            screen_space_bottom_right,
        );
        return persp.compose(fov_scale).compose(screen_to_raster);
    }

    fn create_orthographic_transform(
        far_clip: f32,
        near_clip: f32,
        width: usize,
        height: usize,
        screen_space_width: f32,
        screen_space_height: f32,
    ) -> Transform {
        let translate: Transform = Transform::translate(Vec3(0.0, 0.0, -near_clip));
        let scale: Transform = Transform::scale(Vec3(1.0, 1.0, 1.0 / (far_clip - near_clip)));
        let screen_space_top_left: Vec3 =
            Vec3(-screen_space_width / 2.0, -screen_space_height / 2.0, 0.0);
        let screen_space_bottom_right: Vec3 =
            Vec3(screen_space_width / 2.0, screen_space_height / 2.0, 0.0);
        let screen_to_raster: Transform = Self::create_screen_to_raster_transform(
            width,
            height,
            screen_space_top_left,
            screen_space_bottom_right,
        );
        return translate.compose(scale).compose(screen_to_raster);
    }

    pub fn from_gltf_camera_node(camera_node: &gltf::Node, raster_height: usize) -> Camera {
        let gltf_camera: gltf::Camera = camera_node.camera().unwrap();
        let (camera_position, camera_rotation, _) = camera_node.transform().decomposed();

        let mut camera_to_world_matrix: Matrix4x4 = Matrix4x4 {
            data: camera_node.transform().matrix(),
        };
        camera_to_world_matrix.transpose();
        let camera_to_world: Transform = Transform::from(camera_to_world_matrix);
        let world_to_camera_matrix: Matrix4x4 = camera_to_world_matrix.invert().unwrap();
        let world_to_camera: Transform = Transform::from(world_to_camera_matrix);
        let camera_type;
        let (world_to_raster, raster_width, camera_to_raster) = match gltf_camera.projection() {
            Projection::Perspective(perspective) => {
                let width: usize =
                    (raster_height as f32 * perspective.aspect_ratio().unwrap()) as usize;
                let camera_to_raster: Transform = Self::create_perspective_transform(
                    -perspective.zfar().unwrap_or(DEFAULT_FAR_CLIP),
                    -perspective.znear(),
                    perspective.yfov(),
                    width,
                    raster_height,
                );
                let world_to_raster: Transform = world_to_camera.compose(camera_to_raster.clone());
                camera_type = CameraType::PinholePerspective {
                    yfov: perspective.yfov(),
                };
                (world_to_raster, width, camera_to_raster)
            }
            Projection::Orthographic(orthographic) => {
                let screen_space_width: f32 = orthographic.xmag();
                let screen_space_height: f32 = orthographic.ymag();

                let width: usize =
                    (raster_height as f32 * screen_space_width / screen_space_height) as usize;
                let camera_to_raster: Transform = Self::create_orthographic_transform(
                    -orthographic.zfar(),
                    -orthographic.znear(),
                    width,
                    raster_height,
                    screen_space_width, // idk why i need to flip this but i do
                    -screen_space_height,
                );
                let world_to_raster: Transform = world_to_camera.compose(camera_to_raster.clone());
                camera_type = CameraType::Orthographic {
                    screen_space_width,
                    screen_space_height,
                };
                (world_to_raster, width, camera_to_raster)
            }
        };

        let raster_to_camera = camera_to_raster.invert();

        Camera {
            camera_position: camera_position.into(),
            camera_rotation: camera_rotation.into(),
            camera_type,
            raster_width,
            raster_height,
            near_clip: 0.01,
            far_clip: 1000.0,
            world_to_raster,
            camera_to_world,
            raster_to_camera,
        }
    }

    // creates camera w/ lookat transform + specified raster parameters
    pub fn lookat_camera_perspective(
        camera_position: Vec3,
        target: Vec3,
        up: Vec3,
        swap_handedness: bool,

        yfov: f32, // radians
        raster_width: usize,
        raster_height: usize,
    ) -> Camera {
        let near_clip = 0.01;
        let far_clip = 1000.0;

        let camera_to_raster = Camera::create_perspective_transform(
            far_clip,
            near_clip,
            yfov,
            raster_width,
            raster_height,
        );

        let camera_to_world = Transform::look_at(camera_position, target, up, swap_handedness);
        let world_to_camera = camera_to_world.invert();
        let raster_to_camera = camera_to_raster.invert();

        Camera {
            camera_position,
            camera_rotation: Quaternion::from_rotation_matrix(camera_to_world.forward),
            camera_type: CameraType::PinholePerspective { yfov },
            raster_width,
            raster_height,
            near_clip,
            far_clip,
            world_to_raster: world_to_camera.compose(camera_to_raster),
            camera_to_world,
            raster_to_camera,
        }
    }

    pub fn lookat_camera_orthographic(
        camera_position: Vec3,
        target: Vec3,
        up: Vec3,
        swap_handedness: bool,

        raster_width: usize,
        raster_height: usize,
        raster_to_screen_ratio: f32,
    ) -> Camera {
        let near_clip = 0.01;
        let far_clip = 1000.0;

        let screen_space_width = (raster_width as f32) * raster_to_screen_ratio;
        let screen_space_height = (raster_height as f32) * raster_to_screen_ratio;

        let camera_to_raster = Camera::create_orthographic_transform(
            far_clip,
            near_clip,
            raster_width,
            raster_height,
            screen_space_width,
            screen_space_height,
        );

        let camera_to_world = Transform::look_at(camera_position, target, up, swap_handedness);
        let world_to_camera = camera_to_world.invert();
        let raster_to_camera = camera_to_raster.invert();

        Camera {
            camera_position,
            camera_rotation: Quaternion::from_rotation_matrix(camera_to_world.forward),
            camera_type: CameraType::Orthographic {
                screen_space_width,
                screen_space_height,
            },
            raster_width,
            raster_height,
            near_clip,
            far_clip,
            world_to_raster: world_to_camera.compose(camera_to_raster),
            camera_to_world,
            raster_to_camera,
        }
    }

    // creates a thin-lens perspective camera for depth-of-field effects
    pub fn lookat_camera_thin_lens_perspective(
        camera_position: Vec3,
        target: Vec3,
        up: Vec3,
        swap_handedness: bool,

        yfov: f32, // radians
        raster_width: usize,
        raster_height: usize,
        aperture_radius: f32,  // lens radius in world units
        focal_distance: f32,   // distance to focal plane
    ) -> Camera {
        let near_clip = 0.01;
        let far_clip = 1000.0;

        let camera_to_raster = Camera::create_perspective_transform(
            far_clip,
            near_clip,
            yfov,
            raster_width,
            raster_height,
        );

        let camera_to_world = Transform::look_at(camera_position, target, up, swap_handedness);
        let world_to_camera = camera_to_world.invert();
        let raster_to_camera = camera_to_raster.invert();

        Camera {
            camera_position,
            camera_rotation: Quaternion::from_rotation_matrix(camera_to_world.forward),
            camera_type: CameraType::ThinLensPerspective {
                yfov,
                aperture_radius,
                focal_distance,
            },
            raster_width,
            raster_height,
            near_clip,
            far_clip,
            world_to_raster: world_to_camera.compose(camera_to_raster),
            camera_to_world,
            raster_to_camera,
        }
    }
}

#[test]
fn test_coordinate_system() {
    todo!("write a test to ensure coordinate system matches expectations");
}