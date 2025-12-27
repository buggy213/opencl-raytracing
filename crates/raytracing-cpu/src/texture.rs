use raytracing::{geometry::Vec4, materials::{Image, Texture, TextureId, TextureSampler}, scene::Scene};

use crate::materials::MaterialEvalContext;

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
        eval_ctx: &MaterialEvalContext,
        sampler: TextureSampler,
    ) -> Vec4 {
        let uv = eval_ctx.uv;
        let u = uv.u();
        let v = uv.v();

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
                let y_frac = f32::clamp(f32::fract(y), 0.0, 1.0);

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
        eval_ctx: &MaterialEvalContext,
    ) -> Vec4 {
        let uv = eval_ctx.uv;
        let u = uv.u();
        let v = uv.v();

        let tex = &self.scene_textures[texture_id.0 as usize];
        match tex {
            Texture::ImageTexture { image, sampler } => {
                let image = &self.scene_images[image.0 as usize];
                Self::sample_image_texture(image, eval_ctx, *sampler)
            },
            Texture::ConstantTexture { value } => *value,
            Texture::CheckerTexture { color1, color2 } => {
                // "repeat" texture wrapping is natural for checkered textures
                let u = u - f32::floor(u);
                let v = v - f32::floor(v);
                let dudx = eval_ctx.dudx;
                let dudy = eval_ctx.dudy;
                let dvdx = eval_ctx.dvdx;
                let dvdy = eval_ctx.dvdy;
                if dudx == 0.0 && dvdx == 0.0 || dudy == 0.0 && dvdy == 0.0 {
                    // don't bother antialiasing, since it's being point sampled in at least one direction anyways
                    if (u > 0.5) != (v > 0.5) {
                        *color1
                    }
                    else {
                        *color2
                    }
                }
                else {
                    // antialiasing of checker pattern based on gaussian filter
                    // very ad-hoc and arguably too expensive
                    let sampling_rate_x = f32::sqrt(dudx * dudx + dvdx * dvdx);
                    let sampling_rate_y = f32::sqrt(dudy * dudy + dvdy * dvdy);
                    let sampling_rate = f32::max(sampling_rate_x, sampling_rate_y);
                    let sigma = 0.1 * sampling_rate;
                    
                    let a = if u < 0.25 {
                        u
                    }
                    else if u < 0.75 {
                        -(u - 0.5)
                    }
                    else {
                        u - 1.0
                    };

                    let b = if v < 0.25 {
                        v
                    }
                    else if v < 0.75 {
                        -(v - 0.5)
                    }
                    else {
                        v - 1.0
                    };

                    let x_z = a / (f32::sqrt(2.0) * sigma);
                    let y_z = b / (f32::sqrt(2.0) * sigma);

                    let x_factor = 0.5 * (1.0 + f32::erf(x_z));
                    let y_factor = 0.5 * (1.0 + f32::erf(y_z));
                    let x_factor = if v > 0.5 { x_factor } else { 1.0 - x_factor };
                    let y_factor = if u > 0.5 { y_factor } else { 1.0 - y_factor };
                    let factor = x_factor * y_factor;
                    let color1 = *color1;
                    let color2 = *color2;

                    factor * color1 + (1.0 - factor) * color2
                }
            }
            Texture::ScaleTexture { a, b } => {
                let a_val = self.sample(*a, eval_ctx);
                let b_val = self.sample(*b, eval_ctx);
                a_val * b_val
            },
            Texture::MixTexture { a, b, c } => {
                let one = Vec4(1.0, 1.0, 1.0, 1.0);
                let c_val = self.sample(*c, eval_ctx);
                
                let b_val = if c_val == Vec4::zero() {
                    Vec4::zero()
                } else { 
                    self.sample(*b, eval_ctx)
                };

                let a_val = if c_val == one {
                    Vec4::zero()
                } else {
                    self.sample(*a, eval_ctx)
                };

                (one - c_val) * a_val + c_val * b_val
            },
        }
    }
}