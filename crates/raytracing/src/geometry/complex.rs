use std::{fmt::Display, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};

#[derive(Debug, Clone, Copy)]
pub struct Complex(pub f32, pub f32);

impl Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}+{}i", self.0, self.1)
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, rhs: Self) -> Self::Output {
        Complex(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl Add<f32> for Complex {
    type Output = Complex;

    fn add(self, rhs: f32) -> Self::Output {
        Complex(self.0 + rhs, self.1)
    }
}

impl AddAssign<f32> for Complex {
    fn add_assign(&mut self, rhs: f32) {
        self.0 += rhs;
    }
}

impl Add<Complex> for f32 {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Self::Output {
        rhs + self
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl Sub<f32> for Complex {
    type Output = Complex;

    fn sub(self, rhs: f32) -> Self::Output {
        Complex(self.0 - rhs, self.1)
    }
}

impl SubAssign<f32> for Complex {
    fn sub_assign(&mut self, rhs: f32) {
        self.0 -= rhs;
    }
}

impl Sub<Complex> for f32 {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Self::Output {
        -rhs + self
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex(self.0 * rhs.0 - self.1 * rhs.1, self.1 * rhs.0 + self.0 * rhs.1)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<f32> for Complex {
    type Output = Complex;

    fn mul(self, rhs: f32) -> Self::Output {
        Complex(self.0 * rhs, self.1 * rhs)
    }
}

impl MulAssign<f32> for Complex {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
        self.1 *= rhs;
    }
}

impl Mul<Complex> for f32 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        rhs * self
    }
}

impl Div for Complex {
    type Output = Complex;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.0 * rhs.0 + rhs.1 * rhs.1;
        Complex(
            (self.0 * rhs.0 + self.1 * rhs.1) / denom,
            (self.1 * rhs.0 - self.0 * rhs.1) / denom
        )
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Div<f32> for Complex {
    type Output = Complex;

    fn div(self, rhs: f32) -> Self::Output {
        Complex(self.0 / rhs, self.1 / rhs)
    }
}

impl DivAssign<f32> for Complex {
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
        self.1 /= rhs;
    }
}

impl Div<Complex> for f32 {
    type Output = Complex;

    fn div(self, rhs: Complex) -> Self::Output {
        Complex(self, 0.0) / rhs
    }
}

impl Neg for Complex {
    type Output = Complex;

    fn neg(self) -> Self::Output {
        Complex(-self.0, -self.1)
    }
}

impl From<f32> for Complex {
    fn from(value: f32) -> Self {
        Complex(value, 0.0)
    }
}

impl Complex {
    pub fn square_magnitude(self) -> f32 {
        self.0 * self.0 + self.1 * self.1
    }

    pub fn modulus(self) -> f32 {
        self.square_magnitude().sqrt()
    }

    // outputs: modulus, angle (in radians, from -pi to pi)
    pub fn to_polar(self) -> (f32, f32) {
        (self.modulus(), f32::atan2(self.1, self.0))
    }
}
