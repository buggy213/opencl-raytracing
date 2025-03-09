use std::{ptr::null_mut, mem::size_of};


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

    pub fn apply_point(&self, point: Vec3) -> Vec3 {
        self.forward.apply_point(point)
    }

    pub fn invert_inplace(&mut self) {
        let tmp = self.inverse;
        self.inverse = self.forward;
        self.forward = tmp;
    }

    pub fn invert(&self) -> Transform {
        Transform { forward: self.inverse, inverse: self.forward }
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

