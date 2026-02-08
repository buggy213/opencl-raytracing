use std::{
    ops::{Index, IndexMut},
    slice::from_raw_parts,
};

use super::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix4x4 {
    // row-major
    pub data: [[f32; 4]; 4],
}

// idk if this is sound at all lmao
impl<'a> From<&'a Matrix4x4> for &'a [f32] {
    fn from(val: &'a Matrix4x4) -> Self {
        unsafe { from_raw_parts(val.data.as_ptr() as *const f32, 16) }
    }
}

impl Index<usize> for Matrix4x4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index / 4][index % 4]
    }
}

impl IndexMut<usize> for Matrix4x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index / 4][index % 4]
    }
}

impl Matrix4x4 {
    pub fn identity() -> Self {
        Matrix4x4 {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    pub fn create(a11: f32, a12: f32, a13: f32, a14: f32,
                  a21: f32, a22: f32, a23: f32, a24: f32,
                  a31: f32, a32: f32, a33: f32, a34: f32,
                  a41: f32, a42: f32, a43: f32, a44: f32) -> Self {
        Matrix4x4 {
            data: [[a11, a12, a13, a14],
                   [a21, a22, a23, a24],
                   [a31, a32, a33, a34],
                   [a41, a42, a43, a44]]
        }
    }

    pub fn create_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Self {
        #[rustfmt::skip]
        Matrix4x4::create(
            x.0, y.0, z.0, 0.0,
            x.1, y.1, z.1, 0.0,
            x.2, y.2, z.2, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    pub fn invert(&self) -> Option<Self> {
        let mut inv = Matrix4x4::identity();
        inv[0] = self[5] * self[10] * self[15]
            - self[5] * self[11] * self[14]
            - self[9] * self[6] * self[15]
            + self[9] * self[7] * self[14]
            + self[13] * self[6] * self[11]
            - self[13] * self[7] * self[10];

        inv[4] = -self[4] * self[10] * self[15]
            + self[4] * self[11] * self[14]
            + self[8] * self[6] * self[15]
            - self[8] * self[7] * self[14]
            - self[12] * self[6] * self[11]
            + self[12] * self[7] * self[10];

        inv[8] = self[4] * self[9] * self[15]
            - self[4] * self[11] * self[13]
            - self[8] * self[5] * self[15]
            + self[8] * self[7] * self[13]
            + self[12] * self[5] * self[11]
            - self[12] * self[7] * self[9];

        inv[12] = -self[4] * self[9] * self[14]
            + self[4] * self[10] * self[13]
            + self[8] * self[5] * self[14]
            - self[8] * self[6] * self[13]
            - self[12] * self[5] * self[10]
            + self[12] * self[6] * self[9];

        inv[1] = -self[1] * self[10] * self[15]
            + self[1] * self[11] * self[14]
            + self[9] * self[2] * self[15]
            - self[9] * self[3] * self[14]
            - self[13] * self[2] * self[11]
            + self[13] * self[3] * self[10];

        inv[5] = self[0] * self[10] * self[15]
            - self[0] * self[11] * self[14]
            - self[8] * self[2] * self[15]
            + self[8] * self[3] * self[14]
            + self[12] * self[2] * self[11]
            - self[12] * self[3] * self[10];

        inv[9] = -self[0] * self[9] * self[15]
            + self[0] * self[11] * self[13]
            + self[8] * self[1] * self[15]
            - self[8] * self[3] * self[13]
            - self[12] * self[1] * self[11]
            + self[12] * self[3] * self[9];

        inv[13] = self[0] * self[9] * self[14]
            - self[0] * self[10] * self[13]
            - self[8] * self[1] * self[14]
            + self[8] * self[2] * self[13]
            + self[12] * self[1] * self[10]
            - self[12] * self[2] * self[9];

        inv[2] = self[1] * self[6] * self[15]
            - self[1] * self[7] * self[14]
            - self[5] * self[2] * self[15]
            + self[5] * self[3] * self[14]
            + self[13] * self[2] * self[7]
            - self[13] * self[3] * self[6];

        inv[6] = -self[0] * self[6] * self[15]
            + self[0] * self[7] * self[14]
            + self[4] * self[2] * self[15]
            - self[4] * self[3] * self[14]
            - self[12] * self[2] * self[7]
            + self[12] * self[3] * self[6];

        inv[10] = self[0] * self[5] * self[15]
            - self[0] * self[7] * self[13]
            - self[4] * self[1] * self[15]
            + self[4] * self[3] * self[13]
            + self[12] * self[1] * self[7]
            - self[12] * self[3] * self[5];

        inv[14] = -self[0] * self[5] * self[14]
            + self[0] * self[6] * self[13]
            + self[4] * self[1] * self[14]
            - self[4] * self[2] * self[13]
            - self[12] * self[1] * self[6]
            + self[12] * self[2] * self[5];

        inv[3] = -self[1] * self[6] * self[11]
            + self[1] * self[7] * self[10]
            + self[5] * self[2] * self[11]
            - self[5] * self[3] * self[10]
            - self[9] * self[2] * self[7]
            + self[9] * self[3] * self[6];

        inv[7] = self[0] * self[6] * self[11]
            - self[0] * self[7] * self[10]
            - self[4] * self[2] * self[11]
            + self[4] * self[3] * self[10]
            + self[8] * self[2] * self[7]
            - self[8] * self[3] * self[6];

        inv[11] = -self[0] * self[5] * self[11]
            + self[0] * self[7] * self[9]
            + self[4] * self[1] * self[11]
            - self[4] * self[3] * self[9]
            - self[8] * self[1] * self[7]
            + self[8] * self[3] * self[5];

        inv[15] = self[0] * self[5] * self[10]
            - self[0] * self[6] * self[9]
            - self[4] * self[1] * self[10]
            + self[4] * self[2] * self[9]
            + self[8] * self[1] * self[6]
            - self[8] * self[2] * self[5];

        let mut det = self[0] * inv[0] + self[1] * inv[4] + self[2] * inv[8] + self[3] * inv[12];

        if det == 0.0 {
            return None;
        }

        det = 1.0 / det;
        for i in 0..16 {
            inv[i] *= det;
        }

        Some(inv)
    }

    pub fn det(self) -> f32 {
        let a = self[5] * self[10] * self[15]
            - self[5] * self[11] * self[14]
            - self[9] * self[6] * self[15]
            + self[9] * self[7] * self[14]
            + self[13] * self[6] * self[11]
            - self[13] * self[7] * self[10];

        let b = -self[4] * self[10] * self[15]
            + self[4] * self[11] * self[14]
            + self[8] * self[6] * self[15]
            - self[8] * self[7] * self[14]
            - self[12] * self[6] * self[11]
            + self[12] * self[7] * self[10];

        let c = self[4] * self[9] * self[15]
            - self[4] * self[11] * self[13]
            - self[8] * self[5] * self[15]
            + self[8] * self[7] * self[13]
            + self[12] * self[5] * self[11]
            - self[12] * self[7] * self[9];

        let d = -self[4] * self[9] * self[14]
            + self[4] * self[10] * self[13]
            + self[8] * self[5] * self[14]
            - self[8] * self[6] * self[13]
            - self[12] * self[5] * self[10]
            + self[12] * self[6] * self[9];

        let frac_one_det = self[0] * a + self[1] * b + self[2] * c + self[3] * d;
        1.0 / frac_one_det
    }

    pub fn matmul(a: Matrix4x4, b: Matrix4x4) -> Self {
        let mut m = Matrix4x4::identity();
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for k in 0..4 {
                    dot += a.data[i][k] * b.data[k][j]
                }
                m.data[i][j] = dot;
            }
        }
        m
    }

    pub fn transpose(&mut self) {
        for i in 0..4 {
            for j in 0..i {
                let tmp = self.data[i][j];
                self.data[i][j] = self.data[j][i];
                self.data[j][i] = tmp;
            }
        }
    }

    pub fn transposed(&self) -> Matrix4x4 {
        let mut me = *self;
        me.transpose();
        me
    }

    pub fn translation(direction: Vec3) -> Matrix4x4 {
        let mut me = Self::identity();
        me.data[0][3] = direction.0;
        me.data[1][3] = direction.1;
        me.data[2][3] = direction.2;

        me
    }

    // rotate theta clockwise about v, assuming right-handed coordinate system
    pub fn rotation(theta: f32, v: Vec3) -> Matrix4x4 {
        // rotation matrix
        let a14 = 0.0;
        let a24 = 0.0;
        let a34 = 0.0;
        let a41 = 0.0;
        let a42 = 0.0;
        let a43 = 0.0;
        let a44 = 1.0;

        let rotate_u_about_v = |u: Vec3| {
            let v_c = v * Vec3::dot(u, v);
            let v1 = u - v;
            let v2 = Vec3::cross(v, v1);

            v_c + v1 * f32::cos(theta) + v2 * f32::sin(theta)
        };

        let x_under_rotation = rotate_u_about_v(Vec3(1.0, 0.0, 0.0));
        let a11 = x_under_rotation.x();
        let a21 = x_under_rotation.y();
        let a31 = x_under_rotation.z();

        let y_under_rotation = rotate_u_about_v(Vec3(0.0, 1.0, 0.0));
        let a12 = y_under_rotation.x();
        let a22 = y_under_rotation.y();
        let a32 = y_under_rotation.z();

        let z_under_rotation = rotate_u_about_v(Vec3(0.0, 0.0, 1.0));
        let a13 = z_under_rotation.x();
        let a23 = z_under_rotation.y();
        let a33 = z_under_rotation.z();

        #[rustfmt::skip]
        Matrix4x4::create(
            a11, a12, a13, a14,
            a21, a22, a23, a24,
            a31, a32, a33, a34,
            a41, a42, a43, a44
        )
    }

    pub fn scale(scale: Vec3) -> Matrix4x4 {
        let mut me = Self::identity();
        me.data[0][0] = scale.0;
        me.data[1][1] = scale.1;
        me.data[2][2] = scale.2;

        me
    }
}

impl Matrix4x4 {
    pub fn apply_point(&self, p: Vec3) -> Vec3 {
        let a = self.data[0][0] * p.0
            + self.data[0][1] * p.1
            + self.data[0][2] * p.2
            + self.data[0][3] * 1.0;
        let b = self.data[1][0] * p.0
            + self.data[1][1] * p.1
            + self.data[1][2] * p.2
            + self.data[1][3] * 1.0;
        let c = self.data[2][0] * p.0
            + self.data[2][1] * p.1
            + self.data[2][2] * p.2
            + self.data[2][3] * 1.0;
        let d = self.data[3][0] * p.0
            + self.data[3][1] * p.1
            + self.data[3][2] * p.2
            + self.data[3][3] * 1.0;
        Vec3(a / d, b / d, c / d)
    }

    pub fn apply_vector(&self, v: Vec3) -> Vec3 {
        let a = self.data[0][0] * v.0 + self.data[0][1] * v.1 + self.data[0][2] * v.2;
        let b = self.data[1][0] * v.0 + self.data[1][1] * v.1 + self.data[1][2] * v.2;
        let c = self.data[2][0] * v.0 + self.data[2][1] * v.1 + self.data[2][2] * v.2;
        Vec3(a, b, c)
    }

    // used for calculation of normals
    pub fn apply_vector_transposed(&self, v: Vec3) -> Vec3 {
        let a = self.data[0][0] * v.0 + self.data[1][0] * v.1 + self.data[2][0] * v.2;
        let b = self.data[0][1] * v.0 + self.data[1][1] * v.1 + self.data[2][1] * v.2;
        let c = self.data[0][2] * v.0 + self.data[1][2] * v.1 + self.data[2][2] * v.2;
        Vec3(a, b, c)
    }
}

// both `From` implementations assume row major
impl From<[f32; 16]> for Matrix4x4 {
    fn from(value: [f32; 16]) -> Self {
        Matrix4x4 {
            data: [
                [value[0], value[1], value[2], value[3]],
                [value[4], value[5], value[6], value[7]],
                [value[8], value[9], value[10], value[11]],
                [value[12], value[13], value[14], value[15]],
            ],
        }
    }
}

impl From<[[f32; 4]; 4]> for Matrix4x4 {
    fn from(value: [[f32; 4]; 4]) -> Self {
        Self { data: value }
    }
}