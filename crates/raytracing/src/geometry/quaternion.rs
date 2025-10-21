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

    #[allow(non_snake_case, reason = "math convention")]
    pub fn from_rotation_matrix(rotation: Matrix4x4) -> Quaternion {
        /* https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        the explanation on wikipedia is a bit confusing, so i'll explain it here too
        basically, a unit quaternion corresponds to the rotation matrix
        \begin{pmatrix}
            1   - 2y^2 - 2z^2 & 2xy - 2zw         & 2xz + 2yw          \\
            2xy + 2zw         & 1   - 2x^2 - 2z^2 & 2yz - 2xw          \\
            2xz - 2yw         & 2yz + 2xw         & 1   - 2x^2 - 2y^2
        \end{pmatrix}

        we want to find w, x, y, z given this matrix. the straightforward approach
        is to notice that the trace equals 4w^2 - 1 using the normalization condition
        w^2 + x^2 + y^2 + z^2 = 1, and we can solve for the positive root to get w.
        note that flipping the sign of w, x, y, and z represents the same rotation since
        every term is of degree 0 (constant) or 2; we fix this degree of freedom by choosing
        the positive root!

        to find x, we can notice Q_11 - Q_22 - Q_33 = 4x^2 - 1, and solve for x. however, 
        we need to ensure that x has the right sign; Q_32 - Q_23 = 4xw, and we chose w to be
        positive, so this fixes the sign of x. finding y and z is similar.

        doing this approach requires 4 square roots, which is expensive. it is possible to directly
        use the skew-symmetry of the matrix to do better. specifically, if we define r = sqrt(1 + tr Q),
        s = 1/(2r) and w = (1/2)r, then we saw before that Q_32 - Q_23 = 4xw, so 
        x = (Q_32 - Q_23)/(4w) = (Q_32 - Q_23)s. We can derive similar expressions for y and z. 

        however, there is still a problem - this can have numerical instability when tr Q \approx -1,
        and so we divide by almost 0 when calculating s. this corresponds to cos(\theta/2) \approx 0, or
        \theta \approx \pi (180 degree rotation).

        suppose Q_11 is the largest diagonal entry, then r = sqrt(1 + Q_11 - Q_22 - Q_33) is not small.
        this is actually not super trivial to justify, but apparently it's true?
        furthermore, 1 + Q_11 - Q_22 - Q_33 = 4x^2, so x = r/2. we can then use the same approach of
        difference of skew-symmetric elements to find w, y, and z, while avoiding numerical instability.

        the same can be done for Q_22, Q_33 being the largest.
        */

        let Q_11 = rotation.data[0][0];
        let Q_12 = rotation.data[0][1];
        let Q_13 = rotation.data[0][2];
        let Q_21 = rotation.data[1][0];
        let Q_22 = rotation.data[1][1];
        let Q_23 = rotation.data[1][2];
        let Q_31 = rotation.data[2][0];
        let Q_32 = rotation.data[2][1];
        let Q_33 = rotation.data[2][2];

        let trace = Q_11 + Q_22 + Q_33;
        

        let w: f32;
        let x: f32;
        let y: f32;
        let z: f32;

        if trace > 0.0 {
            let r = (1.0 + trace).sqrt();
            let s = 1.0 / (2.0 * r);
            w = 0.5 * r;
            x = (Q_32 - Q_23) * s;
            y = (Q_13 - Q_31) * s;
            z = (Q_21 - Q_12) * s;
        }
        else if Q_11 >= Q_22 && Q_11 >= Q_33 {
            let r = (1.0 + Q_11 - Q_22 - Q_33).sqrt();
            let s = 1.0 / (2.0 * r);
            w = (Q_32 - Q_23) * s;
            x = 0.5 * r;
            y = (Q_12 + Q_21) * s;
            z = (Q_13 + Q_31) * s;
        }
        else if Q_22 >= Q_11 && Q_22 >= Q_33 {
            let r = (1.0 - Q_11 + Q_22 - Q_33).sqrt();
            let s = 1.0 / (2.0 * r);
            w = (Q_13 - Q_31) * s;
            x = (Q_12 + Q_21) * s;
            y = 0.5 * r;
            z = (Q_23 + Q_32) * s;
        }
        else {
            let r = (1.0 - Q_11 - Q_22 + Q_33).sqrt();
            let s = 1.0 / (2.0 * r);
            w = (Q_21 - Q_12) * s;
            x = (Q_13 + Q_31) * s;
            y = (Q_23 + Q_32) * s;
            z = 0.5 * r;
        }

        Quaternion(w, Vec3(x, y, z))
    }
}

impl Quaternion {
    pub fn rotate(self, x: Vec3) -> Vec3 {
        let conjugation = self * Quaternion(0.0, x) * self.inverse();
        conjugation.pure()
    }
}

impl From<[f32; 4]> for Quaternion {
    fn from(value: [f32; 4]) -> Self {
        Quaternion(value[0], Vec3(value[1], value[2], value[3]))
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