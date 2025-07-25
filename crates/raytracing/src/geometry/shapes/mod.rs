use crate::geometry::{Mesh, Vec3};

pub mod mesh;

pub enum Shape {
    TriangleMesh(Mesh),
    Sphere {
        center: Vec3,
        radius: f32
    }
    
}