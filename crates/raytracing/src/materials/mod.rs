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

mod image;
mod texture;

pub use image::Image;
pub use image::ImageId;
pub use texture::Texture;
pub use texture::TextureId;
pub use texture::WrapMode;
pub use texture::FilterMode;
pub use texture::TextureSampler;