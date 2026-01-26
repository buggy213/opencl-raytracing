#[derive(Debug)]
pub enum Material {
    Diffuse {
        // only RGB is used, alpha is ignored
        albedo: TextureId,
    },

    // perfectly smooth dielectric material / conductor materials
    // bsdfs are purely delta functions
    SmoothDielectric {
        // only R is used, real component of refractive index
        eta: TextureId,
    },
    SmoothConductor {
        // real component (alpha ignored)
        eta: TextureId,
        // imaginary component (alpha ignored)
        kappa: TextureId,
    },

    RoughDielectric {
        // same interpretation as for SmoothDielectric
        eta: TextureId,

        // parameterization of Trowbridge-Reitz microfacet distribution
        // R component corresponds to roughness along tangent direction
        // G component corresponds to roughness along bitangent direction
        // these are converted into alpha_x, alpha_y by taking the square root
        roughness: TextureId,
    },
    RoughConductor {
        // same interpretation as for SmoothConductor
        eta: TextureId,
        kappa: TextureId,

        // same interpretation as for RoughDielectric
        roughness: TextureId,
    },

    // diffuse, covered by a variable thickness / albedo layer with dielectric interface on surface
    CoatedDiffuse {
        diffuse_albedo: TextureId,
        
        dielectric_eta: TextureId,
        dielectric_roughness: Option<TextureId>,

        thickness: TextureId,
        coat_albedo: TextureId,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialId(pub u32);

mod image;
mod texture;

pub use image::Image;
pub use image::ImageId;
pub use texture::FilterMode;
pub use texture::Texture;
pub use texture::TextureId;
pub use texture::TextureSampler;
pub use texture::WrapMode;
