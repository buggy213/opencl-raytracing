use crate::geometry::Transform;

pub struct Camera {
    pub camera_position: [f32; 3],
    pub world_to_raster: Transform,
    pub raster_width: usize,
    pub raster_height: usize
}