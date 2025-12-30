use std::{cell::RefCell, f32, ops::Range};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use raytracing::geometry::{Vec2, Vec3};

// TODO: this should really be replaced with proper "dependency injection" of 
// sampler, like pbrt
thread_local! {
    pub(crate) static RNG: RefCell<rand_chacha::ChaCha8Rng> = RefCell::new(ChaCha8Rng::seed_from_u64(0))
}

pub fn set_seed(seed: u64) {
    RNG.with_borrow_mut(|rng| {
        *rng = ChaCha8Rng::seed_from_u64(seed);
    })
}

pub(crate) fn sample_uniform() -> f32 {
    if cfg!(debug_assertions) {
        RNG.with_borrow_mut(|rng| {
            (rng.next_u32() as f64 / u32::MAX as f64) as f32
        })
    }
    else {
        rand::random_range(0.0 .. 1.0)
    }
}

pub(crate) fn sample_integer(range: Range<u32>) -> u32 {
    if cfg!(debug_assertions) {
        RNG.with_borrow_mut(|rng| {
            // so, this is not really very good but
            // should be good enough for debugging
            let width = range.end - range.start;
            range.start + rng.next_u32() % width
        })
    }
    else {
        rand::random_range(range)
    }
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