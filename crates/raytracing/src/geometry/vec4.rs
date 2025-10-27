use std::ops::{
    self, 
    MulAssign
};

use rand::random;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
pub struct Vec4(pub f32, pub f32, pub f32, pub f32);

impl Vec4 {
    pub fn r(&self) -> f32 {
        self.0
    }

    pub fn g(&self) -> f32 {
        self.1
    }

    pub fn b(&self) -> f32 {
        self.2
    }

    pub fn a(&self) -> f32 {
        self.2
    }

    pub fn square_magnitude(&self) -> f32 {
        self.0 * self.0
            + self.1 * self.1
            + self.2 * self.2
            + self.3 * self.3
    }

    pub fn length(&self) -> f32 {
        self.square_magnitude().sqrt()
    }

    pub fn dot(a: Vec4, b: Vec4) -> f32 {
        a.0 * b.0 + a.1 * b.1 + a.2 * b.2 + a.3 * b.3
    }

    pub fn normalize(u: &mut Vec4) {
        *u /= u.length()
    }

    pub fn normalized(u: Vec4) -> Vec4 {
        u / u.length()
    }

    pub fn unit(self) -> Vec4 {
        Vec4::normalized(self)
    }

    pub fn random_vec() -> Vec4 {
        Vec4(random(), random(), random(), random())
    }

    pub fn near_zero(&self) -> bool {
        const EPSILON: f32 = 1e-8;
        self.0.abs() < EPSILON && self.1.abs() < EPSILON && self.2.abs() < EPSILON && self.3.abs() < EPSILON
    }

    pub fn lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        let clamped_t = t.clamp(0.0, 1.0);
        a * clamped_t + (1.0 - clamped_t) * b
    }

    pub fn elementwise_min(a: Vec4, b: Vec4) -> Vec4 {
        Vec4(f32::min(a.0, b.0), f32::min(a.1, b.1), f32::min(a.2, b.2), f32::min(a.3, b.3))
    }

    pub fn elementwise_max(a: Vec4, b: Vec4) -> Vec4 {
        Vec4(f32::max(a.0, b.0), f32::max(a.1, b.1), f32::max(a.2, b.2), f32::max(a.3, b.3))
    }

    pub fn zero() -> Vec4 {
        Vec4(0.0, 0.0, 0.0, 0.0)
    }
}

impl ops::AddAssign for Vec4 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
        self.3 += rhs.3
    }
}

impl ops::MulAssign for Vec4 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
        self.1 *= rhs.1;
        self.2 *= rhs.2;
        self.3 *= rhs.3
    }
}

impl ops::MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
        self.3 *= rhs;
    }
}

impl ops::DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, rhs: f32) {
        self.mul_assign(1.0 / rhs);
    }
}

impl ops::Add for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Self) -> Self::Output {
        Vec4(self.0 + rhs.0, 
            self.1 + rhs.1,
            self.2 + rhs.2,
            self.3 + rhs.3)
    }
}

impl ops::Neg for Vec4 {
    type Output = Vec4;
    fn neg(self) -> Self::Output {
        Vec4(-self.0, -self.1, -self.2, -self.3)
    }
}

impl ops::Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec4(self.0 - rhs.0, 
            self.1 - rhs.1,
            self.2 - rhs.2,
            self.3 - rhs.3)
    }
}

impl ops::SubAssign for Vec4 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
        self.2 -= rhs.2;
        self.3 -= rhs.3;
    }
}

// element-wise product
impl ops::Mul for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: Self) -> Self::Output {
        Vec4(self.0 * rhs.0, 
            self.1 * rhs.1,
            self.2 * rhs.2,
            self.3 * rhs.3)
    }
}

impl ops::Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec4(self.0 * rhs, 
            self.1 * rhs,
            self.2 * rhs,
            self.3 * rhs)
    }
}

impl ops::Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        rhs * self
    }
}

impl ops::Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from(value: [f32; 4]) -> Self {
        Vec4(value[0], value[1], value[2], value[3])
    }
}
