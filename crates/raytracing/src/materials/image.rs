//! For now, we only support 4-channel images. The actual underlying image data might
//! not actually have 4 channels; the remaining channels are treated as all 0.
//! This is also an intentionally limited interface, only encapsulating common behavior
//! across different platforms (color space conversion)
//!
//! I really don't like this current implementation - will want to revisit this in the future
//! maybe a custom Image handling facility is better?

use std::path::Path;

use anyhow::Context;
use image::{
    metadata::Cicp, ConvertColorOptions, DynamicImage,
    ImageBuffer, ImageReader, Luma, LumaA, Pixel, Primitive, Rgb, Rgba,
};
use num::ToPrimitive;
use tracing::warn;

use crate::geometry::Vec4;

#[derive(Debug, Clone)]
pub struct Image {
    pub buffer: image::DynamicImage,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageId(pub u32);

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
            }
        }
    }

    pub fn width(&self) -> u32 {
        self.buffer.width()
    }

    pub fn height(&self) -> u32 {
        self.buffer.height()
    }

    fn get_pixel_channel<P: Pixel>(
        image_buffer: &ImageBuffer<P, Vec<P::Subpixel>>,
        x: u32,
        y: u32,
        c: u32,
    ) -> f32 {
        let a = image_buffer.get_pixel(x, y).channels()[c as usize]
            .to_f32()
            .expect("all image data should be representable as f32");
        let b = P::Subpixel::DEFAULT_MAX_VALUE
            .to_f32()
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
            }
            image::DynamicImage::ImageLumaA8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgb8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgba8(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageLuma16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageLumaA16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgb16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgba16(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgb32F(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            image::DynamicImage::ImageRgba32F(image_buffer) => {
                Self::get_pixel_channel(image_buffer, x, y, c)
            }
            _ => {
                unimplemented!("unsupported dynamic image format")
            }
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> Vec4 {
        Vec4(
            self.get_channel(x, y, 0),
            self.get_channel(x, y, 1),
            self.get_channel(x, y, 2),
            self.get_channel(x, y, 3),
        )
    }

    pub fn load_from_path(path: &Path) -> anyhow::Result<Self> {
        let image_reader = ImageReader::open(path)
            .map_err(anyhow::Error::from)
            .with_context(|| "failed while loading image")?;
        let mut image_data = image_reader
            .decode()
            .map_err(anyhow::Error::from)
            .with_context(|| "failed while decoding image")?;

        // ensure values are linear
        let conversion_result =
            image_data.apply_color_space(Cicp::SRGB_LINEAR, ConvertColorOptions::default());

        if let Err(conversion_err) = conversion_result {
            warn!(
                "Unable to convert to linear values for image {} (reason: {})",
                path.display(),
                conversion_err
            );
        }

        Ok(Self { buffer: image_data })
    }

    pub fn load_from_gltf_image(gltf_image: gltf::image::Data) -> Self {
        // Note: unwrapping here is ok, since gltf_image is internally doing the inverse of this
        let dynamic_image: DynamicImage = match gltf_image.format {
            gltf::image::Format::R8 => ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels,
            )
            .unwrap()
            .into(),
            gltf::image::Format::R8G8 => ImageBuffer::<LumaA<u8>, Vec<u8>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels,
            )
            .unwrap()
            .into(),
            gltf::image::Format::R8G8B8 => ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels,
            )
            .unwrap()
            .into(),
            gltf::image::Format::R8G8B8A8 => ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                gltf_image.pixels,
            )
            .unwrap()
            .into(),
            gltf::image::Format::R16 => ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
            gltf::image::Format::R16G16 => ImageBuffer::<LumaA<u16>, Vec<u16>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
            gltf::image::Format::R16G16B16 => ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
            gltf::image::Format::R16G16B16A16 => ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
            gltf::image::Format::R32G32B32FLOAT => ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
            gltf::image::Format::R32G32B32A32FLOAT => ImageBuffer::<Rgba<f32>, Vec<f32>>::from_raw(
                gltf_image.width,
                gltf_image.height,
                bytemuck::cast_vec(gltf_image.pixels),
            )
            .unwrap()
            .into(),
        };

        Self {
            buffer: dynamic_image,
        }
    }

    pub fn save<Q: AsRef<Path>>(&self, path: Q) -> Result<(), image::ImageError> {
        self.buffer.save(path)
    }
}

impl From<image::DynamicImage> for Image {
    fn from(value: image::DynamicImage) -> Self {
        Self { buffer: value }
    }
}
