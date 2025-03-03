use std::ops::{self, MulAssign};

use rand::random;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd)]
pub struct Vec2(pub f32, pub f32);

impl Vec2 {
    pub fn x(&self) -> f32 {
        self.0
    }

    pub fn y(&self) -> f32 {
        self.1
    }

    pub fn square_magnitude(&self) -> f32 {
        self.0 * self.0
            + self.1 * self.1
    }

    pub fn length(&self) -> f32 {
        self.square_magnitude().sqrt()
    }

    pub fn dot(a: Vec2, b: Vec2) -> f32 {
        a.0 * b.0 + a.1 * b.1
    }

    pub fn normalize(u: &mut Vec2) {
        *u /= u.length()
    }

    pub fn normalized(u: Vec2) -> Vec2 {
        u / u.length()
    }

    pub fn random_vec() -> Vec2 {
        Vec2(random(), random())
    }

    pub fn near_zero(&self) -> bool {
        const EPSILON: f32 = 1e-8;
        self.0.abs() < EPSILON && self.1.abs() < EPSILON
    }

    pub fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        let clamped_t = t.clamp(0.0, 1.0);
        a * clamped_t + (1.0 - clamped_t) * b
    }

    pub fn elementwise_min(a: Vec2, b: Vec2) -> Vec2 {
        Vec2(f32::min(a.0, b.0), f32::min(a.1, b.1))
    }

    pub fn elementwise_max(a: Vec2, b: Vec2) -> Vec2 {
        Vec2(f32::max(a.0, b.0), f32::max(a.1, b.1))
    }
}

impl ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl ops::MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl ops::DivAssign<f32> for Vec2 {
    fn div_assign(&mut self, rhs: f32) {
        self.mul_assign(1.0 / rhs);
    }
}

impl ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2(
            self.0 + rhs.0, 
            self.1 + rhs.1,
        )
    }
}

impl ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Self::Output {
        Vec2(-self.0, -self.1)
    }
}

impl ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2(
            self.0 - rhs.0, 
            self.1 - rhs.1,
        )
    }
}

// element-wise product
impl ops::Mul for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: Self) -> Self::Output {
        Vec2(
            self.0 * rhs.0, 
            self.1 * rhs.1,
        )
    }
}

impl ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f32) -> Self::Output {
        Vec2(
            self.0 * rhs, 
            self.1 * rhs,
        )
    }
}

impl ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

impl ops::Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}