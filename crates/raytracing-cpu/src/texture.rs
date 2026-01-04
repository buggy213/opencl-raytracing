use raytracing::{geometry::Vec4, materials::{FilterMode, Image, Texture, TextureId, TextureSampler}, scene::Scene};

use crate::materials::MaterialEvalContext;

mod mipmap {
    use image::{ColorType, DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba, buffer::ConvertBuffer, imageops, metadata::Cicp};
    use raytracing::materials::Image;
    use tracing::warn;

    #[derive(Debug, Clone)]
    pub(crate) struct CpuMipmap {
        pub(crate) mip0: Image,
        pub(crate) mips: Vec<Image>,
    }

    // `image` crate is "opinionated" about the image formats it supports in the standard container
    // we need to do repeated resizing at higher precision f32 to avoid artifacts, even for grayscale / grayscale+alpha
    // images, so this type serves as a simple wrapper for f32 images only
    enum DynamicImageFloat {
        LF(ImageBuffer<Luma<f32>, Vec<f32>>),
        LaF(ImageBuffer<LumaA<f32>, Vec<f32>>),
        RgbF(ImageBuffer<Rgb<f32>, Vec<f32>>),
        RgbaF(ImageBuffer<Rgba<f32>, Vec<f32>>),
    }

    impl From<ImageBuffer<Luma<f32>, Vec<f32>>> for DynamicImageFloat {
        fn from(value: ImageBuffer<Luma<f32>, Vec<f32>>) -> Self {
            Self::LF(value)
        }
    }

    impl From<ImageBuffer<LumaA<f32>, Vec<f32>>> for DynamicImageFloat {
        fn from(value: ImageBuffer<LumaA<f32>, Vec<f32>>) -> Self {
            Self::LaF(value)
        }
    }

    impl From<ImageBuffer<Rgb<f32>, Vec<f32>>> for DynamicImageFloat {
        fn from(value: ImageBuffer<Rgb<f32>, Vec<f32>>) -> Self {
            Self::RgbF(value)
        }
    }

    impl From<ImageBuffer<Rgba<f32>, Vec<f32>>> for DynamicImageFloat {
        fn from(value: ImageBuffer<Rgba<f32>, Vec<f32>>) -> Self {
            Self::RgbaF(value)
        }
    }

    impl DynamicImageFloat {
        fn width(&self) -> u32 {
            match self {
                DynamicImageFloat::LF(image_buffer) => image_buffer.width(),
                DynamicImageFloat::LaF(image_buffer) => image_buffer.width(),
                DynamicImageFloat::RgbF(image_buffer) => image_buffer.width(),
                DynamicImageFloat::RgbaF(image_buffer) => image_buffer.width(),
            }
        }

        fn height(&self) -> u32 {
            match self {
                DynamicImageFloat::LF(image_buffer) => image_buffer.height(),
                DynamicImageFloat::LaF(image_buffer) => image_buffer.height(),
                DynamicImageFloat::RgbF(image_buffer) => image_buffer.height(),
                DynamicImageFloat::RgbaF(image_buffer) => image_buffer.height(),
            }
        }

        fn resize_exact(&self, width: u32, height: u32) -> DynamicImageFloat {
            match self {
                DynamicImageFloat::LF(image_buffer) => {
                    imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into()
                }
                DynamicImageFloat::LaF(image_buffer) => {
                    imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into()
                }
                DynamicImageFloat::RgbF(image_buffer) => {
                    imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into()
                }
                DynamicImageFloat::RgbaF(image_buffer) => {
                    imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into()
                }
            }
        }

        fn cast(&self, ty: ColorType) -> DynamicImage {
            match self {
                DynamicImageFloat::LF(image_buffer) => match ty {
                    ColorType::L8 => DynamicImage::ImageLuma8(image_buffer.convert()),
                    ColorType::L16 => DynamicImage::ImageLuma16(image_buffer.convert()),
                    _ => panic!("bad cast to DynamicImage"),
                },
                DynamicImageFloat::LaF(image_buffer) => match ty {
                    ColorType::La8 => DynamicImage::ImageLumaA8(image_buffer.convert()),
                    ColorType::La16 => DynamicImage::ImageLumaA16(image_buffer.convert()),
                    _ => panic!("bad cast to DynamicImage"),
                },
                DynamicImageFloat::RgbF(image_buffer) => match ty {
                    ColorType::Rgb8 => DynamicImage::ImageRgb8(image_buffer.convert()),
                    ColorType::Rgb16 => DynamicImage::ImageRgb16(image_buffer.convert()),
                    ColorType::Rgb32F => DynamicImage::ImageRgb32F(image_buffer.convert()),
                    _ => panic!("bad cast to DynamicImage"),
                },
                DynamicImageFloat::RgbaF(image_buffer) => match ty {
                    ColorType::Rgba8 => DynamicImage::ImageRgba8(image_buffer.convert()),
                    ColorType::Rgba16 => DynamicImage::ImageRgba16(image_buffer.convert()),
                    ColorType::Rgba32F => DynamicImage::ImageRgba32F(image_buffer.convert()),
                    _ => panic!("bad cast to DynamicImage"),
                },
            }
        }
    }

    pub(crate) fn generate_mips(base: &image::DynamicImage) -> CpuMipmap {
        // ensure float, power-of-two, linear values before starting
        if base.color_space() != Cicp::SRGB_LINEAR {
            warn!("`generate_mips` called for non-linear image data");
        }

        let original_color_type = base.color();

        let base: DynamicImageFloat = match base.color() {
            image::ColorType::L8 => base.to::<Luma<f32>>().into(),
            image::ColorType::La8 => base.to::<LumaA<f32>>().into(),
            image::ColorType::Rgb8 => base.to_rgb32f().into(),
            image::ColorType::Rgba8 => base.to_rgba32f().into(),
            image::ColorType::L16 => base.to::<Luma<f32>>().into(),
            image::ColorType::La16 => base.to::<LumaA<f32>>().into(),
            image::ColorType::Rgb16 => base.to_rgb32f().into(),
            image::ColorType::Rgba16 => base.to_rgba32f().into(),
            image::ColorType::Rgb32F => base.to_rgb32f().into(),
            image::ColorType::Rgba32F => base.to_rgba32f().into(),
            _ => unimplemented!("unsupported dynamic image format"),
        };

        let power_of_two = base.width().is_power_of_two() && base.height().is_power_of_two();
        let square = base.width() == base.height();
        let base = if !power_of_two || !square {
            warn!("image has non-square, non-power-of-two dimensions, resizing");
            let w = base.width().next_power_of_two();
            let h = base.height().next_power_of_two();
            let s = u32::max(w, h);

            base.resize_exact(s, s)
        } else {
            base
        };

        // generate pyramid
        let levels = base.width().ilog2();
        let mip0 = base.cast(original_color_type).into();
        let mut pyramid = Vec::with_capacity(levels as usize);
        
        let mut current = base;
        while current.width() > 1 && current.height() > 1 {
            let next = current.resize_exact(current.width() / 2, current.height() / 2);
            pyramid.push(next.cast(original_color_type).into());
            current = next;
        }

        CpuMipmap {
            mip0,
            mips: pyramid,
        }
    }

    #[test]
    fn test_mipmap_generation() {
        let crate_path = env!("CARGO_MANIFEST_DIR");
        let test_path = std::path::Path::new(crate_path).join("test");

        let img =
            Image::load_from_path(&test_path.join("mip_test.jpg")).expect("failed to load image");

        img.buffer
            .save(&test_path.join("output_base.png"))
            .expect("failed to save base image");

        let mipmap = generate_mips(&img.buffer);

        mipmap.mip0.save(&test_path.join("output_base_mip.png"))
            .expect("failed to save mip0");

        for (i, mip) in mipmap.mips.iter().enumerate() {
            let filename = format!("output_mip_{}.png", i);
            mip.save(&test_path.join(filename))
                .expect("failed to save mip image");
        }
    }
}

pub(crate) use mipmap::{
    CpuMipmap, generate_mips
};

// generally, images can just be used straight from the scene representation
// but, we need to prepare mipmaps for images that require it 
// (i.e. a texture using trilinear filtering references some image)
// for now, we'll just support bilinear interpolation, but set up interface so 
// that adding mipmaps later is easy

// Wrapper around scene representation of textures / images to add sampling and
// mipmap support
#[derive(Debug, Clone)]
pub(crate) struct CpuTextures<'scene> {
    scene_textures: &'scene [Texture],
    scene_images: &'scene [Image],
    scene_image_mipmaps: Vec<Option<CpuMipmap>>
}

impl CpuTextures<'_> {
    pub(crate) fn new(scene: &Scene) -> CpuTextures<'_> {
        let mut scene_image_mipmaps: Vec<Option<CpuMipmap>> = vec![None; scene.images.len()];
        for texture in &scene.textures {
            match texture {
                Texture::ImageTexture { image, sampler } => {
                    if matches!(sampler.filter, FilterMode::Trilinear) {
                        let image_buffer = &scene.images[image.0 as usize].buffer;

                        if scene_image_mipmaps[image.0 as usize].is_none() {
                            let mipmap = generate_mips(image_buffer);
                            scene_image_mipmaps[image.0 as usize] = Some(mipmap);
                        }
                    }
                }
                _ => ()
            }
        }

        CpuTextures { 
            scene_textures: &scene.textures, 
            scene_images: &scene.images,
            scene_image_mipmaps, 
        }
    }

    fn point_sample(image: &Image, u: f32, v: f32) -> Vec4 {
        let w = image.width() as f32;
        let h = image.height() as f32;

        let x = u * w - 0.5;
        let y = v * h - 0.5;

        let x = f32::clamp(f32::round(x), 0.0, w - 1.0) as u32;
        let y = f32::clamp(f32::round(y), 0.0, h - 1.0) as u32;
        
        image.get_pixel(x, y)
    }

    fn bilerp_sample(image: &Image, u: f32, v: f32) -> Vec4 {
        let w = image.width() as f32;
        let h = image.height() as f32;

        let x = u * w - 0.5;
        let y = v * h - 0.5;

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
    }

    fn sample_image_texture(
        image: &Image,
        image_mipmap: Option<&CpuMipmap>,
        eval_ctx: &MaterialEvalContext,
        sampler: TextureSampler,
    ) -> Vec4 {
        let uv = eval_ctx.uv;
        let u = uv.u();
        let v = uv.v();

        let u = sampler.wrap.apply(u);
        let v = sampler.wrap.apply(v);

        
        match sampler.filter {
            raytracing::materials::FilterMode::Nearest => {
                Self::point_sample(image, u, v)
            },
            raytracing::materials::FilterMode::Bilinear => {
                Self::bilerp_sample(image, u, v)
            },
            raytracing::materials::FilterMode::Trilinear => {
                let (dudx, dudy, dvdx, dvdy) = (
                    eval_ctx.dudx,
                    eval_ctx.dudy,
                    eval_ctx.dvdx,
                    eval_ctx.dvdy
                );

                let dx = f32::sqrt(dudx * dudx + dvdx * dvdx);
                let dy = f32::sqrt(dudy * dudy + dvdy * dvdy);

                let mipmap = image_mipmap.expect(
                    "trilinear filter mode requires mipmap"
                );

                let mip0 = &mipmap.mip0;
                let mips = &mipmap.mips;

                // TODO: anisotropic mipmaps?
                // if sampling rate is faster than every half pixel, then all frequencies in mip0 
                // can be reconstructed (theoretically) so just do bilinear filtering.
                let larger_derivative = f32::max(dx, dy);
                let half_pixel = 1.0 / (2.0 * mip0.width() as f32);

                let mip_level = f32::log2(larger_derivative / half_pixel);
                let max_mip_level = mips.len() as f32;
                let lower = f32::floor(f32::clamp(mip_level, 0.0, max_mip_level)) as u32;
                let upper = f32::ceil(f32::clamp(mip_level, 0.0, max_mip_level)) as u32;

                let lower_mip = if lower == 0 {
                    &mipmap.mip0
                } else {
                    &mipmap.mips[(lower - 1) as usize]
                };

                let upper_mip = if upper == 0 {
                    &mipmap.mip0
                } else {
                    &mipmap.mips[(upper - 1) as usize]
                };

                let t = f32::fract(mip_level);
                let a = Self::bilerp_sample(lower_mip, u, v);
                let b = Self::bilerp_sample(upper_mip, u, v);

                t * b + (1.0 - t) * a
            },
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
                let image_mipmap = self.scene_image_mipmaps[image.0 as usize].as_ref();
                let image = &self.scene_images[image.0 as usize];
                Self::sample_image_texture(image, image_mipmap, eval_ctx, *sampler)
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
