use raytracing::{geometry::Vec4, materials::{Image, Texture, TextureId, TextureSampler}, scene::Scene};

struct CpuMipmap {
    mips: Vec<image::DynamicImage>
}

// generally, images can just be used straight from the scene representation
// but, we need to prepare mipmaps for images that require it 
// (i.e. a texture using trilinear filtering references some image)
// for now, we'll just support bilinear interpolation, but set up interface so 
// that adding mipmaps later is easy

// Wrapper around scene representation of textures / images to add sampling and
// (TODO) mipmap support
#[derive(Debug, Clone)]
pub(crate) struct CpuTextures<'scene> {
    scene_textures: &'scene [Texture],
    scene_images: &'scene [Image]
}

impl CpuTextures<'_> {
    pub(crate) fn new(scene: &Scene) -> CpuTextures<'_> {
        CpuTextures { 
            scene_textures: &scene.textures, 
            scene_images: &scene.images 
        }
    }

    fn sample_image_texture(
        image: &Image,
        u: f32,
        v: f32,
        sampler: TextureSampler,
    ) -> Vec4 {
        let w = image.width() as f32;
        let h = image.height() as f32;

        let u = sampler.wrap.apply(u);
        let v = sampler.wrap.apply(v);

        let x = u * w - 0.5;
        let y = v * h - 0.5;
        
        match sampler.filter {
            raytracing::materials::FilterMode::Nearest => {
                let x = f32::clamp(f32::round(x), 0.0, w - 1.0) as u32;
                let y = f32::clamp(f32::round(y), 0.0, h - 1.0) as u32;
                
                image.get_pixel(x, y)
            },
            raytracing::materials::FilterMode::Bilinear => {
                let x0 = f32::clamp(f32::floor(x), 0.0, w - 1.0) as u32;
                let x1 = f32::clamp(f32::ceil(x), 0.0, w - 1.0) as u32;
                let y0 = f32::clamp(f32::floor(y), 0.0, h - 1.0) as u32;
                let y1 = f32::clamp(f32::ceil(y), 0.0, h - 1.0) as u32;

                let x_frac = f32::clamp(f32::fract(x), 0.0, 1.0);
                let y_frac = f32::clamp(f32::fract(x), 0.0, 1.0);

                let p00 = image.get_pixel(x0, y0);
                let p01 = image.get_pixel(x1, y0);
                let p10 = image.get_pixel(x0, y1);
                let p11 = image.get_pixel(x1, y1);

                let u0 = p00 * (1.0 - x_frac) + p01 * x_frac;
                let u1 = p10 * (1.0 - x_frac) + p11 * x_frac;

                u0 * (1.0 - y_frac) + u1 * y_frac
            },
            raytracing::materials::FilterMode::Trilinear => todo!("mipmaps"),
        }
    }

    pub(crate) fn sample(
        &self, 
        texture_id: TextureId,
        u: f32,
        v: f32,
    ) -> Vec4 {
        let tex = &self.scene_textures[texture_id.0 as usize];
        match tex {
            Texture::ImageTexture { image, sampler } => {
                let image = &self.scene_images[image.0 as usize];
                Self::sample_image_texture(image, u, v, *sampler)
            },
            Texture::ConstantTexture { value } => *value,
            Texture::ScaleTexture { a, b } => {
                let a_val = self.sample(*a, u, v);
                let b_val = self.sample(*b, u, v);
                a_val * b_val
            },
            Texture::MixTexture { a, b, c } => {
                let one = Vec4(1.0, 1.0, 1.0, 1.0);
                let c_val = self.sample(*c, u, v);
                
                let b_val = if c_val == Vec4::zero() {
                    Vec4::zero()
                } else { 
                    self.sample(*b, u, v)
                };

                let a_val = if c_val == one {
                    Vec4::zero()
                } else {
                    self.sample(*a, u, v)
                };

                (one - c_val) * a_val + c_val * b_val
            },
        }
    }
}