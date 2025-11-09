
#[derive(Debug)]
pub enum Material {
    Diffuse { 
        // only RGB is used, alpha is ignored
        albedo: TextureId
    },

    // perfectly smooth dielectric material / conductor materials
    // bsdfs are purely delta functions
    SmoothDielectric {
        // only R is used
        eta: TextureId
    },
    SmoothConductor {
        // R is the real component, G is the imaginary component
        eta: TextureId
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