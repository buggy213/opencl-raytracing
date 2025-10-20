use super::{Matrix4x4, Vec3};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Transform {
    pub forward: Matrix4x4,
    pub inverse: Matrix4x4
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

    pub fn apply_inverse_point(&self, point: Vec3) -> Vec3 {
        self.inverse.apply_point(point)
    }

    pub fn apply_vector(&self, vector: Vec3) -> Vec3 {
        self.forward.apply_vector(vector)
    }

    pub fn apply_inverse_vector(&self, vector: Vec3) -> Vec3 {
        self.inverse.apply_vector(vector)
    }

    pub fn apply_normal(&self, normal: Vec3) -> Vec3 {
        // inverse transpose times n is the correct approach. see 
        // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html
        
        self.inverse.apply_vector_transposed(normal)
    }

    pub fn invert_inplace(&mut self) {
        std::mem::swap(&mut self.forward, &mut self.inverse);
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

impl Transform {
    pub fn look_at(camera_pos: Vec3, target_pos: Vec3, up: Vec3) -> Transform {
        // affine transform
        let a41 = 0.0;
        let a42 = 0.0;
        let a43 = 0.0;
        let a44 = 1.0;
        
        // translation
        let a14 = camera_pos.x();
        let a24 = camera_pos.y();
        let a34 = camera_pos.z();

        // (+z)-forward
        let view_direction_world = (target_pos - camera_pos).unit();
        let a13 = view_direction_world.x();
        let a23 = view_direction_world.y();
        let a33 = view_direction_world.z();

        // x from (y cross z)
        let left = Vec3::cross(up, view_direction_world).unit();
        let a11 = left.x();
        let a21 = left.y();
        let a31 = left.z();

        // y from (z cross x)
        let up = Vec3::cross(view_direction_world, left);
        let a12 = up.x();
        let a22 = up.y();
        let a32 = up.z();
        
        let camera_to_world = Matrix4x4::create(
            a11, a12, a13, a14, 
            a21, a22, a23, a24, 
            a31, a32, a33, a34, 
            a41, a42, a43, a44
        );

        Transform { 
            forward: camera_to_world, 
            inverse: camera_to_world.invert().expect("look-at transform should be invertible") 
        }
    }
}

