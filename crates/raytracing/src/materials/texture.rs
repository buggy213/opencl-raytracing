//! For now, we only support 4-channel textures. The actual underlying image data might
//! not actually have 4 channels; the remaining channels are treated as all 0's if accessed

use crate::geometry::Vec4;

#[derive(Debug)]
pub enum FilterMode {
    Nearest, Bilinear, Trilinear
}

#[derive(Debug)]
pub enum WrapMode {
    Repeat, Mirror, Clamp
}

#[derive(Debug)]
pub struct TextureSampler {
    filter: FilterMode,
    wrap: WrapMode
}

#[derive(Debug)]
pub enum Texture {
    ImageTexture,
    ConstantTexture {
        value: Vec4
    },

}

impl Texture {
    pub fn sample(&self, u: f32, v: f32) -> Vec4 {
        match self {
            Texture::ImageTexture => todo!(),
            Texture::ConstantTexture { value } => *value,
        }
    }
}

