use raytracing::geometry::{Transform, Vec3};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Ray {
    pub(crate) origin: Vec3,
    pub(crate) direction: Vec3,
}

impl Ray {
    pub(crate) fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    pub(crate) fn transform(ray: Ray, transform: &Transform) -> Ray {
        Ray {
            origin: transform.apply_point(ray.origin),
            direction: transform.apply_vector(ray.direction),
        }
    }
}

// RayDifferentials used for better texture aliasing for primary rays
// (see PBRT 4ed 10.1)
// TODO: consider adding RayDifferentials for specular extensions 
// to primary ray as well
#[derive(Clone, Copy, Debug)]
pub(crate) struct RayDifferentials {
    pub(crate) x_origin: Vec3,
    pub(crate) y_origin: Vec3,
    pub(crate) x_direction: Vec3,
    pub(crate) y_direction: Vec3
}
