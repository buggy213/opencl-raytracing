//! For now, we only support 4-channel textures. The actual underlying image data might
//! not actually have 4 channels; the remaining channels are treated as all 0's if accessed

use std::fmt::Display;

use gltf::texture::WrappingMode;

use crate::{geometry::Vec4, materials::image::ImageId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest, Bilinear, Trilinear
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    Repeat, Mirror, Clamp
}

impl From<WrappingMode> for WrapMode {
    fn from(value: WrappingMode) -> Self {
        match value {
            WrappingMode::ClampToEdge => Self::Clamp,
            WrappingMode::MirroredRepeat => Self::Mirror,
            WrappingMode::Repeat => Self::Repeat,
        }
    }
}

impl Display for WrapMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WrapMode::Repeat => f.write_str("repeat"),
            WrapMode::Mirror => f.write_str("mirror"),
            WrapMode::Clamp => f.write_str("clamp"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureSampler {
    pub filter: FilterMode,
    pub wrap: WrapMode
}

pub struct TextureId(pub u32);

#[derive(Debug)]
pub enum Texture {
    ImageTexture {
        image: ImageId,
        sampler: TextureSampler
    },
    ConstantTexture {
        value: Vec4
    },
}

