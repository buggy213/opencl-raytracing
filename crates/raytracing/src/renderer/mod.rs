use bitflags::bitflags;

use crate::{geometry::{Vec2, Vec3}, sampling::Sampler};

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
        
        // mip-level also depends on resolution of underlying image too; if textures
        // are not uniformly sized then directly comparing mip-levels is not really valid.
        // it might make sense to create a "normalized" mip-level which takes this into account
        const MIP_LEVEL = 1 << 3;

        // ray differentials, uv derivatives, hit-point derivatives, bvh traversal depth
        // material properties? might be useful for materials themselves to be able to define
        // (blender has ability to do fully custom AOV from shader graph, looks cool)

        const DEBUG =
            AOVFlags::NORMALS.bits()
            | AOVFlags::UV_COORDS.bits()
            | AOVFlags::MIP_LEVEL.bits();

        const FIRST_HIT_AOVS =
            AOVFlags::NORMALS.bits()
            | AOVFlags::UV_COORDS.bits()
            | AOVFlags::MIP_LEVEL.bits();
    }
}

#[derive(Debug)]
pub struct RenderOutput {
    pub width: u32,
    pub height: u32,

    pub beauty: Option<Vec<Vec3>>,
    pub normals: Option<Vec<Vec3>>,
    pub uv: Option<Vec<Vec2>>,
    pub mip_level: Option<Vec<f32>>,
}

impl RenderOutput {
    pub fn new(width: u32, height: u32) -> Self {
        RenderOutput {
            width,
            height,
            beauty: None,
            normals: None,
            uv: None,
            mip_level: None
        }
    }
}

#[derive(Debug, Clone)]
pub struct RaytracerSettings {
    pub max_ray_depth: u32,
    pub accumulate_bounces: bool,

    pub light_sample_count: u32,
    pub samples_per_pixel: u32,
    pub seed: Option<u64>,
    pub sampler: Sampler,

    pub outputs: AOVFlags,

    pub antialias_primary_rays: bool,
    pub antialias_secondary_rays: bool,
}

impl Default for RaytracerSettings {
    fn default() -> Self {
        Self {
            max_ray_depth: 8,
            accumulate_bounces: true,

            light_sample_count: 4,
            samples_per_pixel: 32,
            seed: None,
            sampler: Sampler::Independent,

            outputs: AOVFlags::BEAUTY,

            antialias_primary_rays: true,
            antialias_secondary_rays: true,
        }
    }
}
