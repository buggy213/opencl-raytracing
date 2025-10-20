use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::geometry::{Matrix4x4, Transform, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Quaternion(pub f32, pub Vec3);

impl Quaternion {
    pub fn a(self) -> f32 {
        self.0
    }

    pub fn b(self) -> f32 {
        self.1.x()
    }

    pub fn c(self) -> f32 {
        self.1.y()
    }

    pub fn d(self) -> f32 {
        self.1.z()
    }

    pub fn real(self) -> f32 {
        self.0
    }

    pub fn pure(self) -> Vec3 {
        self.1
    }
}

impl Add for Quaternion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quaternion(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl AddAssign for Quaternion {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;

    fn sub(self, rhs: Self) -> Self::Output {
        Quaternion(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl SubAssign for Quaternion {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

// scalar multiplication
impl Mul<f32> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: f32) -> Self::Output {
        Quaternion(self.0 * rhs, self.1 * rhs)
    }
}

impl MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl Div<f32> for Quaternion {
    type Output = Quaternion;

    fn div(self, rhs: f32) -> Self::Output {
        Quaternion(self.0 / rhs, self.1 / rhs)
    }
}

impl DivAssign<f32> for Quaternion {
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;

    fn neg(self) -> Self::Output {
        Quaternion(-self.0, -self.1)
    }
}

// multiplication is defined on the division ring of quaternions
impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Self::Output {
        Quaternion(
            self.0 + rhs.0 - Vec3::dot(self.1, rhs.1),
            self.0 * rhs.1 + rhs.0 * self.1 + Vec3::cross(self.1, rhs.1)
        )
    }
}

impl MulAssign<Quaternion> for Quaternion {
    fn mul_assign(&mut self, rhs: Quaternion) {
        *self = *self * rhs;
    }
}

impl Quaternion {
    pub fn conj(self) -> Self {
        Quaternion(self.0, -self.1)
    }

    pub fn norm2(self) -> f32 {
        self.0 * self.0 + self.1.square_magnitude()
    }

    pub fn norm(self) -> f32 {
        self.norm2().sqrt()
    }

    pub fn normalized(self) -> Quaternion {
        Quaternion(self.0 / self.norm(), self.1 / self.norm())
    }

    pub fn normalize(&mut self) {
        self.0 /= self.norm();
        self.1 /= self.norm();
    }

    pub fn inverse(self) -> Quaternion {
        self.conj() / self.norm2()
    }

    pub fn dot(a: Quaternion, b: Quaternion) -> f32 {
        a.0 * b.0 + Vec3::dot(a.1, b.1)
    }
}

impl Quaternion {
    pub fn rotate(self, x: Vec3) -> Vec3 {
        let conjugation = self * Quaternion(0.0, x) * self.inverse();
        conjugation.pure()
    }
}

impl From<Quaternion> for Matrix4x4 {
    fn from(value: Quaternion) -> Self {
        let v = value.pure().unit();

        // real component = cos(1/2 * theta)
        let theta = 2.0 * f32::acos(value.0);

        Matrix4x4::rotation(theta, v)
    }
}

impl From<Quaternion> for Transform {
    fn from(value: Quaternion) -> Self {
        let rotation_matrix: Matrix4x4 = value.into();
        let inverse = rotation_matrix.transposed();
        
        Transform {
            forward: rotation_matrix,
            inverse,
        }
    }
}