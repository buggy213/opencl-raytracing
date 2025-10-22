use crate::geometry::{Complex, Vec3};

#[derive(Debug)]
pub enum Material {
    Diffuse { 
        albedo: Vec3
    },

    // perfectly smooth dielectric material / conductor material
    // bsdf are purely delta functions
    SmoothDielectric { 
        eta: f32 
    },
    SmoothConductor { 
        eta: Complex
    },
}