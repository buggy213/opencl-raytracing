use std::{ptr::null_mut, cmp::max};

use cl3::{memory::CL_MEM_READ_WRITE, types::CL_TRUE};
use opencl3::{memory::Buffer, context::Context, command_queue::CommandQueue};

use crate::geometry::Vec3;

#[derive(Debug)]
#[repr(C)]
pub enum Light {
    PointLight {
        position: Vec3,
        intensity: Vec3
    }
}

impl Light {
    pub fn from_gltf_light(light: &gltf::Node) -> Light {
        let light_properties = light.light().unwrap();
        assert!(matches!(light_properties.kind(), gltf::khr_lights_punctual::Kind::Point)); // only point lights for now...
        let intensity: Vec3 = <[f32; 3] as Into<Vec3>>::into(light_properties.color()) * light_properties.intensity();
        let (light_position, _, _) = light.transform().decomposed(); // T * R * S

        Light::PointLight { position: light_position.into(), intensity }
    }

    pub fn to_cl_lights(lights: &[Light], context: &Context, command_queue: &CommandQueue) -> Buffer<Light> {
        unsafe {
            let mut lights_buffer: Buffer<Light> = Buffer::create(
                context, 
                CL_MEM_READ_WRITE, 
                max(1, lights.len()), 
                null_mut()
            ).expect("failed to create lights buffer");
            if lights.len() > 0 {
                // vortex driver does not like empty writes
                command_queue.enqueue_write_buffer(
                    &mut lights_buffer, 
                    CL_TRUE, 
                    0, 
                    lights, 
                    &[]
                ).expect("failed to write to lights buffer");
            }
            lights_buffer
        }
    }
}