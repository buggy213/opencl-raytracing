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

impl WrapMode {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            WrapMode::Repeat => {
                let frac = f32::fract(x);
                if frac < 0.0 {
                    1.0 + frac
                }
                else {
                    frac
                }
            },
            WrapMode::Mirror => {
                let frac = f32::fract(x);
                let repeat = if frac < 0.0 {
                    1.0 + frac
                }
                else {
                    frac
                };
                
                let floor = f32::floor(x) as i32;
                if i32::rem_euclid(floor, 2) == 1 {
                    1.0 - repeat
                }
                else {
                    repeat
                }
            },
            WrapMode::Clamp => f32::clamp(x, 0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureSampler {
    pub filter: FilterMode,
    pub wrap: WrapMode
}

#[derive(Debug, Clone, Copy)]
pub struct TextureId(pub u32);

#[derive(Debug)]
pub enum Texture {
    // "Base" textures
    ImageTexture {
        image: ImageId,
        sampler: TextureSampler
    },
    ConstantTexture {
        value: Vec4
    },

    // "Derived" textures

    // output = a * b at each point
    ScaleTexture {
        a: TextureId,
        b: TextureId
    },

    // output = mix(a, b, c) at each point
    MixTexture {
        a: TextureId,
        b: TextureId,
        c: TextureId
    },
}

