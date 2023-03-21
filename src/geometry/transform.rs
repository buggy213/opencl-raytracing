use std::{ptr::null_mut, mem::size_of};

use cl3::{memory::CL_MEM_READ_WRITE, types::CL_TRUE};
use opencl3::{memory::Buffer, context::Context, command_queue::CommandQueue};

use super::{Matrix4x4, Vec3};

#[repr(C)]
#[derive(Debug)]
pub struct Transform {
    forward: Matrix4x4,
    inverse: Matrix4x4
}

impl Transform {
    pub fn identity() -> Self {
        Transform { forward: Matrix4x4::identity(), inverse: Matrix4x4::identity() }
    }

    pub fn translate(direction: Vec3) -> Self {
        Transform { 
            forward: Matrix4x4::create(1.0, 0.0, 0.0, direction.0, 
                                       0.0, 1.0, 0.0, direction.1, 
                                       0.0, 0.0, 1.0, direction.2, 
                                       0.0, 0.0, 0.0, 1.0), 
            inverse: Matrix4x4::create(1.0, 0.0, 0.0, -direction.0, 
                                       0.0, 1.0, 0.0, -direction.1, 
                                       0.0, 0.0, 1.0, -direction.2, 
                                       0.0, 0.0, 0.0, 1.0), 
        }
    }

    pub fn scale(scale: Vec3) -> Self {
        Transform {
            forward: Matrix4x4::create(scale.0, 0.0, 0.0, 0.0, 
                                       0.0, scale.1, 0.0, 0.0, 
                                       0.0, 0.0, scale.2, 0.0, 
                                       0.0, 0.0, 0.0, 1.0), 
            inverse: Matrix4x4::create(1.0 / scale.0, 0.0, 0.0, 0.0, 
                                       0.0, 1.0 / scale.1, 0.0, 0.0, 
                                       0.0, 0.0, 1.0 / scale.2, 0.0, 
                                       0.0, 0.0, 0.0, 1.0), 
        }
    }

    pub fn compose(&self, other: Transform) -> Self {
        // OTHER matmul SELF for forward direction
        // SELF.INVERSE matmul OTHER for inverse
        Transform { 
            forward: Matrix4x4::matmul(other.forward, self.forward),
            inverse: Matrix4x4::matmul(self.inverse, other.inverse)
        }
    }

    pub fn to_opencl_buffer(&self, context: &Context, command_queue: &CommandQueue) -> Buffer<f32> {
        let mut buffer: Buffer<f32>;
        unsafe {
            buffer = Buffer::create(
                context, 
                CL_MEM_READ_WRITE, 
                32, 
                null_mut()
            ).expect("failed to create opencl buffer for transform");
            
            let flat_slice = (&self.forward).into();
            command_queue.enqueue_write_buffer(
                &mut buffer, 
                CL_TRUE, 
                0, 
                flat_slice, 
                &[]
            ).expect("failed to write transform buffer");

            let flat_slice = (&self.inverse).into();
            command_queue.enqueue_write_buffer(
                &mut buffer, 
                CL_TRUE, 
                16 * size_of::<f32>(), 
                flat_slice, 
                &[]
            ).expect("failed to write transform buffer");
        };

        buffer
    }
}

impl From<Matrix4x4> for Transform {
    fn from(value: Matrix4x4) -> Self {
        Transform {
            forward: value,
            inverse: Matrix4x4::invert(&value).expect("failed to invert matrix")
        }
    }
}

