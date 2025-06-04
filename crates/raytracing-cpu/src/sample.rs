use std::f32;

use raytracing::geometry::{Vec2, Vec3};

pub(crate) fn sample_uniform() -> f32 {
    rand::random_range(0.0 .. 1.0)
}

pub(crate) fn sample_uniform2() -> Vec2 {
    Vec2(sample_uniform(), sample_uniform())
}

pub(crate) fn sample_unit_disk() -> Vec2 {
    let u = sample_uniform2();
    let r = f32::sqrt(u.0);
    let theta = 2.0 * f32::consts::PI * u.1;
    Vec2(r * f32::cos(theta), r * f32::sin(theta))
}

pub(crate) fn sample_cosine_hemisphere() -> (Vec3, f32) {
    let d = sample_unit_disk();
    let z = f32::sqrt(1.0 - d.0 * d.0 - d.1 * d.1);
    let h = Vec3(d.0, d.1, z);
    (h, z / f32::consts::PI)   
}