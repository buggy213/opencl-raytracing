use std::{ops::{Index, IndexMut, Add}, slice::from_raw_parts};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix4x4 {
    data: [[f32; 4]; 4]
}

// idk if this is sound at all lmao
impl <'a> Into<&'a [f32]> for &'a Matrix4x4 {
    fn into(self) -> &'a [f32] {
        unsafe {
            from_raw_parts(self.data.as_ptr() as *const f32, 16)
        }
    }
}

impl Index<usize> for Matrix4x4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index / 4][index % 4];
    }
}

impl IndexMut<usize> for Matrix4x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[index / 4][index % 4];
    }
}

impl Matrix4x4 {
    pub fn identity() -> Self {
        Matrix4x4 { 
            data: [[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]] 
        }
    }

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

    pub fn invert(&self) -> Option<Self> {
        let mut inv = Matrix4x4::identity();
        inv[0] = self[5]  * self[10] * self[15] - 
                self[5]  * self[11] * self[14] - 
                self[9]  * self[6]  * self[15] + 
                self[9]  * self[7]  * self[14] +
                self[13] * self[6]  * self[11] - 
                self[13] * self[7]  * self[10];

        inv[4] = -self[4]  * self[10] * self[15] + 
                self[4]  * self[11] * self[14] + 
                self[8]  * self[6]  * self[15] - 
                self[8]  * self[7]  * self[14] - 
                self[12] * self[6]  * self[11] + 
                self[12] * self[7]  * self[10];

        inv[8] = self[4]  * self[9] * self[15] - 
                self[4]  * self[11] * self[13] - 
                self[8]  * self[5] * self[15] + 
                self[8]  * self[7] * self[13] + 
                self[12] * self[5] * self[11] - 
                self[12] * self[7] * self[9];

        inv[12] = -self[4]  * self[9] * self[14] + 
                self[4]  * self[10] * self[13] +
                self[8]  * self[5] * self[14] - 
                self[8]  * self[6] * self[13] - 
                self[12] * self[5] * self[10] + 
                self[12] * self[6] * self[9];

        inv[1] = -self[1]  * self[10] * self[15] + 
                self[1]  * self[11] * self[14] + 
                self[9]  * self[2] * self[15] - 
                self[9]  * self[3] * self[14] - 
                self[13] * self[2] * self[11] + 
                self[13] * self[3] * self[10];

        inv[5] = self[0]  * self[10] * self[15] - 
                self[0]  * self[11] * self[14] - 
                self[8]  * self[2] * self[15] + 
                self[8]  * self[3] * self[14] + 
                self[12] * self[2] * self[11] - 
                self[12] * self[3] * self[10];

        inv[9] = -self[0]  * self[9] * self[15] + 
                self[0]  * self[11] * self[13] + 
                self[8]  * self[1] * self[15] - 
                self[8]  * self[3] * self[13] - 
                self[12] * self[1] * self[11] + 
                self[12] * self[3] * self[9];

        inv[13] = self[0]  * self[9] * self[14] - 
                self[0]  * self[10] * self[13] - 
                self[8]  * self[1] * self[14] + 
                self[8]  * self[2] * self[13] + 
                self[12] * self[1] * self[10] - 
                self[12] * self[2] * self[9];

        inv[2] = self[1]  * self[6] * self[15] - 
                self[1]  * self[7] * self[14] - 
                self[5]  * self[2] * self[15] + 
                self[5]  * self[3] * self[14] + 
                self[13] * self[2] * self[7] - 
                self[13] * self[3] * self[6];

        inv[6] = -self[0]  * self[6] * self[15] + 
                self[0]  * self[7] * self[14] + 
                self[4]  * self[2] * self[15] - 
                self[4]  * self[3] * self[14] - 
                self[12] * self[2] * self[7] + 
                self[12] * self[3] * self[6];

        inv[10] = self[0]  * self[5] * self[15] - 
                self[0]  * self[7] * self[13] - 
                self[4]  * self[1] * self[15] + 
                self[4]  * self[3] * self[13] + 
                self[12] * self[1] * self[7] - 
                self[12] * self[3] * self[5];

        inv[14] = -self[0]  * self[5] * self[14] + 
                self[0]  * self[6] * self[13] + 
                self[4]  * self[1] * self[14] - 
                self[4]  * self[2] * self[13] - 
                self[12] * self[1] * self[6] + 
                self[12] * self[2] * self[5];

        inv[3] = -self[1] * self[6] * self[11] + 
                self[1] * self[7] * self[10] + 
                self[5] * self[2] * self[11] - 
                self[5] * self[3] * self[10] - 
                self[9] * self[2] * self[7] + 
                self[9] * self[3] * self[6];

        inv[7] = self[0] * self[6] * self[11] - 
                self[0] * self[7] * self[10] - 
                self[4] * self[2] * self[11] + 
                self[4] * self[3] * self[10] + 
                self[8] * self[2] * self[7] - 
                self[8] * self[3] * self[6];

        inv[11] = -self[0] * self[5] * self[11] + 
                self[0] * self[7] * self[9] + 
                self[4] * self[1] * self[11] - 
                self[4] * self[3] * self[9] - 
                self[8] * self[1] * self[7] + 
                self[8] * self[3] * self[5];

        inv[15] = self[0] * self[5] * self[10] - 
                self[0] * self[6] * self[9] - 
                self[4] * self[1] * self[10] + 
                self[4] * self[2] * self[9] + 
                self[8] * self[1] * self[6] - 
                self[8] * self[2] * self[5];

        let mut det = self[0] * inv[0] + self[1] * inv[4] + self[2] * inv[8] + self[3] * inv[12];

        if det == 0.0 {
            return None
        }
        
        det = 1.0 / det;
        for i in 0..16 {
            inv[i] *= det;
        }
        
        Some(inv)
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
}