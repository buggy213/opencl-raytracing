use std::{f32, hash::Hasher, ops::Range};

use num::Integer;
use rand::Rng;
use raytracing::geometry::{Vec2, Vec3};
use rustc_hash::FxHasher;

#[derive(Debug, Clone)]
pub(crate) enum CpuSampler {
    Independent {
        seed: u64,
        rng: rand_pcg::Pcg32,
    },

    Stratified {
        jitter: bool,
        x_strata: u32,
        y_strata: u32,

        seed: u64,
        rng: rand_pcg::Pcg32,

        dimension: u32,
        sample_index: u32,
    }
}

impl CpuSampler {
    const SAMPLE_SPACING: u64 = 1 << 16;

    pub(crate) fn from_sampler(sampler: &raytracing::sampling::Sampler, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(42);

        let mut hash = FxHasher::default();
        hash.write_u64(seed);
        let seed = hash.finish();
        
        match sampler {
            raytracing::sampling::Sampler::Independent => {
                Self::Independent { 
                    seed,
                    rng: rand_pcg::Pcg32::new(seed, 0)
                }
            },
            raytracing::sampling::Sampler::Stratified { jitter, x_strata, y_strata } => {
                Self::Stratified { 
                    jitter: *jitter, 
                    x_strata: *x_strata, 
                    y_strata: *y_strata, 
                    
                    seed, 
                    rng: rand_pcg::Pcg32::new(seed, 0), 
                    
                    dimension: 0, 
                    sample_index: 0 
                }
            },
        }
    }

    // unlike pbrt, we don't expose starting dimension; sampling using `rand` is opaque - not possible to know
    // how many values were consumed by RNG. it seems like a pretty niche use case anyways. it's still possible
    // to start at a particular `sample_index`
    pub(crate) fn start_sample(&mut self, p: (u32, u32), sample_index: u64) {
        let mut hash = FxHasher::default();
        hash.write_u32(p.0);
        hash.write_u32(p.1);
        hash.write_u32(sample_index as u32);
        let stream = hash.finish();

        match self {
            CpuSampler::Independent { rng, seed, .. } => {
                *rng = rand_pcg::Pcg32::new(*seed, stream);
                rng.advance(sample_index * Self::SAMPLE_SPACING);
            },
            CpuSampler::Stratified { seed, rng, dimension, sample_index: sampler_sample_index, .. } => {
                *rng = rand_pcg::Pcg32::new(*seed, stream);
                rng.advance(sample_index * Self::SAMPLE_SPACING);
                
                *dimension = 0;
                *sampler_sample_index = sample_index as u32;
            }
        }
    }

    pub(crate) fn sample_uniform(&mut self) -> f32 {
        match self {
            CpuSampler::Independent { rng, .. } => {
                rng.sample(rand::distr::StandardUniform)
            },
            CpuSampler::Stratified { 
                jitter, 
                x_strata: x_samples, 
                y_strata: y_samples, 
                rng, 
                seed ,
                dimension,
                sample_index,
            } => {
                let total_samples = (*x_samples) * (*y_samples);
                
                let mut hasher = FxHasher::default();
                hasher.write_u32(*dimension);
                hasher.write_u64(*seed);
                let hash = hasher.finish() as u32;
                
                let strata = permute(*sample_index, total_samples, hash);
                let delta = if *jitter { 
                    rng.sample(rand::distr::StandardUniform) 
                } else {
                    0.5
                };

                *dimension += 1;

                (strata as f32 + delta) / total_samples as f32
            }
        }
    }

    pub(crate) fn sample_u32(&mut self, range: Range<u32>) -> u32 {
        match self {
            CpuSampler::Independent { rng, .. } => {
                rng.random_range(range)
            },
            CpuSampler::Stratified { .. } => {
                // hard to properly stratify, since set of values is discrete, so just
                // use float valued
                let u = self.sample_uniform();
                let offset = u * (range.end - range.start) as f32;
                
                range.start + offset as u32
            }
        }
    }

    pub(crate) fn sample_uniform2(&mut self) -> Vec2 {
        match self {
            CpuSampler::Independent { rng, .. } => {
                let first = rng.sample(rand::distr::StandardUniform);
                let second = rng.sample(rand::distr::StandardUniform);
                Vec2(first, second)
            },
            CpuSampler::Stratified { 
                jitter, 
                x_strata: x_samples, 
                y_strata: y_samples, 
                seed, 
                rng, 
                dimension, 
                sample_index 
            } => {
                let mut hasher = FxHasher::default();
                hasher.write_u32(*dimension);
                hasher.write_u64(*seed);
                let hash = hasher.finish() as u32;

                let total_samples = (*x_samples) * (*y_samples);
                let strata = permute(*sample_index, total_samples, hash);

                *dimension += 2;

                let (y, x) = strata.div_rem(x_samples);
                let (dx, dy) = if *jitter {
                    (rng.sample(rand::distr::StandardUniform), rng.sample(rand::distr::StandardUniform))  
                }
                else {
                    (0.5, 0.5)
                };
                
                Vec2(
                    (x as f32 + dx) / (*x_samples as f32),
                    (y as f32 + dy) / (*y_samples as f32)
                )
            },
        }
        
    }
}

pub(crate) fn sample_unit_disk(u: Vec2) -> Vec2 {
    let r = f32::sqrt(u.0);
    let theta = 2.0 * f32::consts::PI * u.1;
    Vec2(r * f32::cos(theta), r * f32::sin(theta))
}

pub(crate) fn sample_unit_disk_concentric(u: Vec2) -> Vec2 {
    let u_offset = 2.0 * u - Vec2(1.0, 1.0);
    if u_offset == Vec2::zero() {
        return Vec2::zero();
    }

    let (theta, r) = if f32::abs(u_offset.0) > f32::abs(u_offset.1) {
        (f32::consts::FRAC_PI_4 * (u_offset.1 / u_offset.0), u_offset.0)
    }
    else {
        (f32::consts::FRAC_PI_2 - f32::consts::FRAC_PI_4 * (u_offset.0 / u_offset.1), u_offset.1)
    };

    r * Vec2(f32::cos(theta), f32::sin(theta))
}

pub(crate) fn sample_cosine_hemisphere(u: Vec2) -> (Vec3, f32) {
    let d = sample_unit_disk(u);
    let z = f32::sqrt(1.0 - d.0 * d.0 - d.1 * d.1);
    let h = Vec3(d.0, d.1, z);
    (h, z / f32::consts::PI)
}

// https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
// https://andrew-helmer.github.io/permute/
pub(crate) fn permute(mut index: u32, length: u32, seed: u32) -> u32 {
    let mask = length.next_power_of_two() - 1;
    loop {
        index ^= seed;
        index *= 0xe170893d;
        index ^= seed >> 16;
        index ^= (index & mask) >> 4;
        index ^= seed >> 8;
        index *= 0x0929eb3f;
        index ^= seed >> 23;
        index ^= (index & mask) >> 1;
        index *= 1 | seed >> 27;
        index *= 0x6935fa69;
        index ^= (index & mask) >> 11; 
        index *= 0x74dcb303;
        index ^= (index & mask) >> 2; 
        index *= 0x9e501cc3;
        index ^= (index & mask) >> 2; 
        index *= 0xc860a3df;
        index &= mask;
        index ^= index >> 5;

        if index < length {
            return (index + seed) % length;
        }
    }
}

#[test]
fn test_permute() {
    // determinism: same inputs yield same outputs
    let a = permute(5, 17, 0xdeadbeef);
    let b = permute(5, 17, 0xdeadbeef);
    assert_eq!(a, b, "permute should be deterministic for same inputs");

    // range and uniqueness: for a variety of lengths, outputs for indices 0..length-1
    // should be in [0, length) and form a permutation (unique values)
    let lengths = [1u32, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 97];
    for &len in &lengths {
        let mut seen = std::collections::HashSet::with_capacity(len as usize);
        for i in 0..len {
            let v = permute(i, len, 0x1234_5678);
            assert!(v < len, "permute returned out-of-range value {} for length {}", v, len);
            seen.insert(v);
        }
        assert_eq!(seen.len() as u32, len, "permute did not produce a full permutation for length {}", len);
    }
}