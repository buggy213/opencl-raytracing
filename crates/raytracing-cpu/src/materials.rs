use std::f32;

use raytracing::{geometry::Vec3, materials::Material};

use crate::sample::sample_cosine_hemisphere;

#[derive(Debug)]
pub(crate) struct BsdfSample {
    pub(crate) wi: Vec3,
    pub(crate) bsdf: Vec3,
    pub(crate) pdf: f32
}
pub(crate) trait CpuMaterial {
    fn get_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3;
    fn sample_bsdf(&self, wo: Vec3) -> BsdfSample;
    fn is_delta_bsdf(&self) -> bool;
}

impl CpuMaterial for Material {
    // local coordinates, wo and wi both in upper hemisphere
    fn get_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        match self {
            Material::Diffuse { albedo } => {
                *albedo / f32::consts::PI
            },
            Material::Dielectric { eta } => todo!(),
            Material::Conductor { eta, k } => todo!()
        }
    }

    fn sample_bsdf(&self, wo: Vec3) -> BsdfSample {
        match self {
            Material::Diffuse { albedo } => {
                // cosine-weighted hemisphere sampling
                let (wi, pdf) = sample_cosine_hemisphere();
                BsdfSample {
                    wi,
                    bsdf: *albedo / f32::consts::PI,
                    pdf,
                }
            },
            Material::Dielectric { eta } => {
                todo!()
            },
            Material::Conductor { eta, k } => todo!(),
        }
    }

    // if bsdf only has deltas, then doing light sampling is not worthwhile
    fn is_delta_bsdf(&self) -> bool {
        match self {
            Material::Diffuse { .. } => false,
            Material::Dielectric { .. } => true,
            Material::Conductor { .. } => true,
        }
    }
}