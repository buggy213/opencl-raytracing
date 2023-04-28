use std::{
    ops::{
        self, 
        MulAssign
    }
};

use rand::random;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
pub struct Vec3(pub f32, pub f32, pub f32);

impl Vec3 {
    pub fn x(&self) -> f32 {
        self.0
    }

    pub fn y(&self) -> f32 {
        self.1
    }

    pub fn z(&self) -> f32 {
        self.2
    }

    pub fn square_magnitude(&self) -> f32 {
        self.0 * self.0
            + self.1 * self.1
            + self.2 * self.2
    }

    pub fn length(&self) -> f32 {
        self.square_magnitude().sqrt()
    }

    pub fn dot(a: Vec3, b: Vec3) -> f32 {
        a.0 * b.0 + a.1 * b.1 + a.2 * b.2
    }

    pub fn cross(u: Vec3, v: Vec3) -> Vec3 {
        Vec3(u.1 * v.2 - u.2 * v.1,
            u.2 * v.0 - u.0 * v.2,
            u.0 * v.1 - u.1 * v.0)
    }

    pub fn normalize(u: &mut Vec3) {
        *u /= u.length()
    }

    pub fn normalized(u: Vec3) -> Vec3 {
        u / u.length()
    }

    pub fn random_vec() -> Vec3 {
        Vec3(random(), random(), random())
    }

    pub fn near_zero(&self) -> bool {
        const EPSILON: f32 = 1e-8;
        self.0.abs() < EPSILON && self.1.abs() < EPSILON && self.2.abs() < EPSILON
    }

    pub fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        let clamped_t = t.clamp(0.0, 1.0);
        a * clamped_t + (1.0 - clamped_t) * b
    }

    pub fn elementwise_min(a: Vec3, b: Vec3) -> Vec3 {
        Vec3(f32::min(a.0, b.0), f32::min(a.1, b.1), f32::min(a.2, b.2))
    }

    pub fn elementwise_max(a: Vec3, b: Vec3) -> Vec3 {
        Vec3(f32::max(a.0, b.0), f32::max(a.1, b.1), f32::max(a.2, b.2))
    }
}

impl ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl ops::MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
    }
}

impl ops::DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self.mul_assign(1.0 / rhs);
    }
}

impl ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3(self.0 + rhs.0, 
            self.1 + rhs.1,
            self.2 + rhs.2)
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}

impl ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3(self.0 - rhs.0, 
            self.1 - rhs.1,
            self.2 - rhs.2)
    }
}

// element-wise product
impl ops::Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Self) -> Self::Output {
        Vec3(self.0 * rhs.0, 
            self.1 * rhs.1,
            self.2 * rhs.2)
    }
}

impl ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec3(self.0 * rhs, 
            self.1 * rhs,
            self.2 * rhs)
    }
}

impl ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl ops::Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(value: [f32; 3]) -> Self {
        Vec3(value[0], value[1], value[2])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
pub struct Vec3u(pub u32, pub u32, pub u32);