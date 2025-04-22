use super::vec3::Vec3;

/// Axis-aligned bounding box
/// Defined by 2 points
#[derive(Clone, Copy, Default, Debug)]
pub struct AABB {
    pub minimum: Vec3,
    pub maximum: Vec3
}

impl AABB {
    pub fn new(minimum: Vec3, maximum: Vec3) -> AABB {
        AABB { minimum, maximum }
    }

    pub fn from_bounds(
        min_x0: f32, min_x1: f32,
        min_y0: f32, min_y1: f32,
        min_z0: f32, min_z1: f32,
        max_x0: f32, max_x1: f32,
        max_y0: f32, max_y1: f32,
        max_z0: f32, max_z1: f32,
    ) -> AABB {
        AABB { 
            minimum: Vec3(
                f32::min(min_x0, min_x1),
                f32::min(min_y0, min_y1),
                f32::min(min_z0, min_z1)
            ), 
            maximum: Vec3(
                f32::max(max_x0, max_x1),
                f32::max(max_y0, max_y1),
                f32::max(max_z0, max_z1)
            ) 
        }
    }

    /// Returns a box which surrounds both a and b
    pub fn surrounding_box(a: AABB, b: AABB) -> AABB {
        AABB { 
            minimum: Vec3(
                f32::min(a.minimum.0, b.minimum.0),
                f32::min(a.minimum.1, b.minimum.1),
                f32::min(a.minimum.2, b.minimum.2)
            ), 
            maximum: Vec3(
                f32::max(a.maximum.0, b.maximum.0),
                f32::max(a.maximum.1, b.maximum.1),
                f32::max(a.maximum.2, b.maximum.2)
            ) 
        }
    }
}

use crate::macros;
macro_rules! from_points {
    ($($y:expr),+) => (
        let minimum = variadic_min_comparator!(Vec3::elementwise_min, $($y),+);
        let maximum = variadic_max_comparator!(Vec3::elementwise_max, $($y),+);
        AABB {
            minimum, maximum
        }
    );
}