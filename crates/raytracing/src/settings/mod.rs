mod aovs;

pub use aovs::AOVFlags;

#[derive(Debug, Clone, Copy)]
pub struct RaytracerSettings {
    pub max_ray_depth: u32,
    pub light_sample_count: u32,
    pub samples_per_pixel: u32,
    pub accumulate_bounces: bool,

    pub debug_normals: bool,
}

impl Default for RaytracerSettings {
    fn default() -> Self {
        Self { 
            max_ray_depth: 8, 
            light_sample_count: 4, 
            samples_per_pixel: 32, 
            accumulate_bounces: true, 
            
            debug_normals: false
        }
    }
}