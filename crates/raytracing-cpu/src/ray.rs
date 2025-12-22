use raytracing::geometry::{Transform, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    pub fn transform(ray: Ray, transform: &Transform) -> Ray {
        Ray {
            origin: transform.apply_point(ray.origin),
            direction: transform.apply_vector(ray.direction),
        }
    }
}