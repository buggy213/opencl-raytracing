use std::f32;

use raytracing::{geometry::Vec3, materials::Material};

pub(crate) trait CpuMaterial {
    fn get_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3;
    fn sample_bsdf(&self, wo: Vec3) -> (Vec3, f32); // wi + pdf
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

    fn sample_bsdf(&self, wo: Vec3) -> (Vec3, f32) {
        todo!()
    }
}