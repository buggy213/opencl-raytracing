#[derive(Debug, Clone, Copy)]
pub struct RaytracerSettings {
    pub max_ray_depth: u32,
    pub light_sample_count: u32,
    pub samples_per_pixel: u32,
    pub accumulate_bounces: bool,

    pub debug_normals: bool,
}