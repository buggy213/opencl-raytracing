//! For now, we only support 4-channel images. The actual underlying image data might
//! not actually have 4 channels; the remaining channels are treated as all 0.
//! 
//! I really don't like this current implementation - will want to revisit this in the future
//! maybe a custom Image handling facility is better?

use std::path::Path;

use anyhow::Context;
use image::{buffer::ConvertBuffer, imageops, metadata::Cicp, ColorType, ConvertColorOptions, DynamicImage, ImageBuffer, ImageReader, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use num::ToPrimitive;
use tracing::warn;

#[derive(Debug)]
pub struct Image {
    buffer: image::DynamicImage,
    mips: Vec<image::DynamicImage>,
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
            DynamicImageFloat::LF(image_buffer) => imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into(),
            DynamicImageFloat::LaF(image_buffer) => imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into(),
            DynamicImageFloat::RgbF(image_buffer) => imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into(),
            DynamicImageFloat::RgbaF(image_buffer) => imageops::resize(image_buffer, width, height, imageops::FilterType::Lanczos3).into(),
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

impl Image {
    pub fn depth(&self) -> u32 {
        match &self.buffer.color() {
            image::ColorType::L8 => 1,
            image::ColorType::La8 => 2,
            image::ColorType::Rgb8 => 3,
            image::ColorType::Rgba8 => 4,
            image::ColorType::L16 => 1,
            image::ColorType::La16 => 2,
            image::ColorType::Rgb16 => 3,
            image::ColorType::Rgba16 => 4,
            image::ColorType::Rgb32F => 3,
            image::ColorType::Rgba32F => 4,
            _ => {
                unimplemented!("unsupported dynamic image format")
            },
        }    
    }

    fn get_pixel_channel<P: Pixel>(
        image_buffer: &ImageBuffer<P, Vec<P::Subpixel>>,
        x: u32,
        y: u32,
        c: u32
    ) -> f32 {
        let a = image_buffer.get_pixel(x, y).channels()[c as usize].to_f32()
            .expect("all image data should be representable as f32");
        let b = P::Subpixel::DEFAULT_MAX_VALUE.to_f32()
            .expect("all image data should be representable as f32");

        a / b
    }

    pub fn get_channel(&self, x: u32, y: u32, c: u32) -> f32 {
        if c >= self.depth() {
            return 0.0;
        }

        match &self.buffer {
            image::DynamicImage::ImageLuma8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageLumaA8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgb8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgba8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageLuma16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageLumaA16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgb16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgba16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgb32F(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            image::DynamicImage::ImageRgba32F(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            },
            _ => {
                unimplemented!("unsupported dynamic image format")
            },
        }
    }

    pub fn get_channel_mip(&self, x: u32, y: u32, c: u32, mip: u32) -> f32 {
        if c >= self.depth() {
            return 0.0;
        }

        let mip = if mip as usize >= self.mips.len() {
            warn!("out-of-bounds mip level, returning value from highest mip");
            self.mips.len() - 1
        }
        else {
            mip as usize
        };

        match &self.mips[mip] {
            DynamicImage::ImageLuma8(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageLumaA8(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgb8(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgba8(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageLuma16(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageLumaA16(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgb16(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgba16(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgb32F(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            DynamicImage::ImageRgba32F(image_buffer) => Self::get_pixel_channel(image_buffer, x, y, c),
            _ => unreachable!("mipmap generation code cannot create unknown formats"),
        }
    }

    fn generate_mips(base: &image::DynamicImage) -> Vec<image::DynamicImage> {
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
            _ => unimplemented!(
                "unsupported dynamic image format"
            ),
        };
        
        let base = if !base.width().is_power_of_two() || !base.height().is_power_of_two() {
            warn!("image has non-power-of-two dimensions, resizing");
            let w = base.width().next_power_of_two();
            let h = base.height().next_power_of_two();
            
            base.resize_exact(w, h)
        } else {
            base
        };

        // generate pyramid
        let levels = 1 + u32::max(base.width().ilog2(), base.height().ilog2());
        let mut pyramid = Vec::with_capacity(levels as usize);
        pyramid.push(base.cast(original_color_type));

        let mut current = base;
        while current.width() > 1 && current.height() > 1 {
            let next = current.resize_exact(current.width() / 2, current.height() / 2);
            pyramid.push(next.cast(original_color_type));
            current = next;
        }

        pyramid
    }

    pub fn load_from_path(path: &Path) -> anyhow::Result<Self> {
        let image_reader = ImageReader::open(path)
            .map_err(anyhow::Error::from)
            .with_context(|| "failed while loading image")?;
        let mut image_data = image_reader.decode()
            .map_err(anyhow::Error::from)
            .with_context(|| "failed while decoding image")?;
        
        // ensure values are linear
        let conversion_result = 
            image_data.apply_color_space(Cicp::SRGB_LINEAR, ConvertColorOptions::default());

        if let Err(conversion_err) = conversion_result {
            warn!("Unable to convert to linear values for image {} (reason: {})", path.display(), conversion_err);
        }

        let mips = Image::generate_mips(&image_data);
        
        Ok(Self {
            buffer: image_data,
            mips
        })
    }
}

#[cfg(test)]
mod test {
    use crate::materials::Image;

    #[test]
    fn test_mipmap_generation() {
        let crate_path = env!("CARGO_MANIFEST_DIR");
        let test_path = std::path::Path::new(crate_path)
            .join("test");

        let img = Image::load_from_path(&test_path.join("mip_test.jpg"))
            .expect("failed to load image");
        
        img.buffer
            .save(&test_path.join("output_base.png"))
            .expect("failed to save base image");
        for (i, mip) in img.mips.iter().enumerate() {
            let filename = format!("output_mip_{}.png", i);
            mip.save(&test_path.join(filename)).expect("failed to save mip image");
        }
    }
}