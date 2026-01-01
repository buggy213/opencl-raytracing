use crate::geometry::Transform;

use super::vec3::Vec3;

/// Axis-aligned bounding box
/// Defined by 2 points
#[derive(Clone, Copy, Default, Debug)]
pub struct AABB {
    pub minimum: Vec3,
    pub maximum: Vec3,
}

#[macro_export]
macro_rules! from_points {
    ($($y:expr),+) => (
        {
            let minimum = crate::macros::variadic_min_comparator!(Vec3::elementwise_min, $($y),+);
            let maximum = crate::macros::variadic_max_comparator!(Vec3::elementwise_max, $($y),+);
            AABB {
                minimum, maximum
            }
        }
    );
}

impl AABB {
    pub fn center(&self) -> Vec3 {
        (self.maximum + self.minimum) / 2.0
    }

    pub fn radius(&self) -> f32 {
        (self.maximum - self.center()).length()
    }
}

impl AABB {
    pub fn new(minimum: Vec3, maximum: Vec3) -> AABB {
        AABB { minimum, maximum }
    }
    #[rustfmt::skip]
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
                f32::min(min_z0, min_z1),
            ),
            maximum: Vec3(
                f32::max(max_x0, max_x1),
                f32::max(max_y0, max_y1),
                f32::max(max_z0, max_z1),
            ),
        }
    }

    /// Returns a box which surrounds both a and b
    pub fn surrounding_box(a: AABB, b: AABB) -> AABB {
        AABB {
            minimum: Vec3(
                f32::min(a.minimum.0, b.minimum.0),
                f32::min(a.minimum.1, b.minimum.1),
                f32::min(a.minimum.2, b.minimum.2),
            ),
            maximum: Vec3(
                f32::max(a.maximum.0, b.maximum.0),
                f32::max(a.maximum.1, b.maximum.1),
                f32::max(a.maximum.2, b.maximum.2),
            ),
        }
    }

    pub fn transform_aabb(a: AABB, t: &Transform) -> AABB {
        let v0 = a.minimum;
        let v1 = a.maximum;

        let x000 = t.apply_point(Vec3(v0.0, v0.1, v0.2));
        let x001 = t.apply_point(Vec3(v0.0, v0.1, v1.2));
        let x010 = t.apply_point(Vec3(v0.0, v1.1, v0.2));
        let x011 = t.apply_point(Vec3(v0.0, v1.1, v1.2));
        let x100 = t.apply_point(Vec3(v1.0, v0.1, v0.2));
        let x101 = t.apply_point(Vec3(v1.0, v0.1, v1.2));
        let x110 = t.apply_point(Vec3(v1.0, v1.1, v0.2));
        let x111 = t.apply_point(Vec3(v1.0, v1.1, v1.2));

        from_points!(x000, x001, x010, x011, x100, x101, x110, x111)
    }

    pub fn infinite() -> AABB {
        AABB {
            minimum: Vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            maximum: Vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        }
    }
}
