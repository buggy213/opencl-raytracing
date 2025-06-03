use crate::geometry::Vec3;

pub enum Material {
    Diffuse { 
        albedo: Vec3
    },
    Dielectric { 
        eta: f32 
    },
    Conductor { 
        eta: f32, 
        k: f32 
    },
}