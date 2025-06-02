use std::f32;

use raytracing::{geometry::Vec3, materials::Material};

pub(crate) trait CpuMaterial {
    fn get_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3;
}

impl CpuMaterial for Material {
    // local coordinates, wo and wi both in upper hemisphere
    fn get_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        match self {
            Material::Diffuse { albedo } => {
                let unnormalized: Vec3 = albedo.into();
                unnormalized / f32::consts::PI
            },
            Material::Dielectric { eta } => todo!(),
            Material::Conductor { eta, k } => todo!(),
        }
    }
}