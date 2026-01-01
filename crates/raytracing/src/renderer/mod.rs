use bitflags::bitflags;

use crate::geometry::{Vec2, Vec3};

// AOV pass is separated from main render; the intention is that most AOVs are cheap to compute
// and thus spending one extra ray is basically trivial compared to the cost of computing the main
// "beauty" result. should try to ensure the AOV matches the first actual sample taken within that pixel
// TODO: implement more deterministic sampler so that can actually be done
// TODO: light path expression support? looks cool but is certainly not cheap to compute.
bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AOVFlags: u32 {
        const BEAUTY = 1 << 0;
        const NORMALS = 1 << 1;
        const UV_COORDS = 1 << 2;
        // ray differentials, uv derivatives, hit-point derivatives, bvh traversal depth
        // material properties? might be useful for materials themselves to be able to define
        // (blender has ability to do fully custom AOV from shader graph, looks cool)

        const DEBUG =
            AOVFlags::NORMALS.bits()
            | AOVFlags::UV_COORDS.bits();

        const FIRST_HIT_AOVS =
            AOVFlags::NORMALS.bits()
            | AOVFlags::UV_COORDS.bits();
    }
}

#[derive(Debug)]
pub struct RenderOutput {
    pub width: u32,
    pub height: u32,

    pub beauty: Option<Vec<Vec3>>,
    pub normals: Option<Vec<Vec3>>,
    pub uv: Option<Vec<Vec2>>,
}

impl RenderOutput {
    pub fn new(width: u32, height: u32) -> Self {
        RenderOutput {
            width,
            height,
            beauty: None,
            normals: None,
            uv: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RaytracerSettings {
    pub max_ray_depth: u32,
    pub light_sample_count: u32,
    pub samples_per_pixel: u32,
    pub accumulate_bounces: bool,

    pub outputs: AOVFlags,
}

impl Default for RaytracerSettings {
    fn default() -> Self {
        Self {
            max_ray_depth: 8,
            light_sample_count: 4,
            samples_per_pixel: 32,
            accumulate_bounces: true,

            outputs: AOVFlags::BEAUTY,
        }
    }
}
