
#[derive(Debug)]
pub enum Material {
    Diffuse { 
        // only RGB is used, alpha is ignored
        albedo: TextureId
    },

    // perfectly smooth dielectric material / conductor materials
    // bsdfs are purely delta functions
    SmoothDielectric {
        // only R is used, real component of refractive index
        eta: TextureId
    },
    SmoothConductor {
        // R is the real component, G is the imaginary component
        eta: TextureId
    },

    // 
    RoughDielectric {
        // same interpretation as for SmoothDielectric
        eta: TextureId
    },
    RoughConductor {
        // same interpretation as for SmoothConductor
        eta: TextureId,

        // parameterization of Trowbridge-Reitz microfacet distribution
        // R component corresponds to roughness along tangent direction
        // G component corresponds to roughness along bitangent direction
        // these are converted into alpha_x, alpha_y by taking the square root
        roughness: TextureId
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