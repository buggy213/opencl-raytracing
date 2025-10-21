use std::ops::Range;

use raytracing::geometry::{Mesh, Shape, Transform, Vec3, Vec3u, AABB};

use crate::ray::Ray;

// return range of t at which ray intersects an AABB (possibly including negative values of t), 
// or None if no intersection
pub(crate) fn intersect_aabb(
    aabb: AABB, 
    ray: Ray
) -> Option<Range<f32>> {
    let a = (aabb.minimum.x() - ray.origin.x()) / ray.direction.x();
    let b = (aabb.maximum.x() - ray.origin.x()) / ray.direction.x();
    let t0_x = f32::min(a, b);
    let t1_x = f32::max(a, b);

    let c = (aabb.minimum.y() - ray.origin.y()) / ray.direction.y();
    let d = (aabb.maximum.y() - ray.origin.y()) / ray.direction.y();
    let t0_y = f32::min(c, d);
    let t1_y = f32::max(c, d);

    let e = (aabb.minimum.z() - ray.origin.z()) / ray.direction.z();
    let f = (aabb.maximum.z() - ray.origin.z()) / ray.direction.z();
    let t0_z = f32::min(e, f);
    let t1_z = f32::max(e, f);

    let t0 = f32::max(f32::max(t0_x, t0_y), t0_z);
    let t1 = f32::min(f32::min(t1_x, t1_y), t1_z);

    if t0 <= t1 {
        Some(t0..t1)
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IntersectResult {
    pub(crate) t: f32,
    pub(crate) point: Vec3,
    pub(crate) normal: Vec3,
}

pub(crate) fn intersect_shape(
    w_ray: Ray, // bvh-space
    t_min: f32,
    t_max: f32,
    o2w_transform: &Transform, // object-to-bvh
    shape: &Shape, // object-space
    prim_index: u32,
) -> Option<IntersectResult> /* bvh-space */ {
    let w2o_transform = o2w_transform.invert();
    let o_ray = Ray::transform(w_ray, &w2o_transform);

    let s = o_ray.direction.length() / w_ray.direction.length();
    
    let Some(o_intersection_result) = (match shape {
        Shape::TriangleMesh(mesh) => {
            ray_mesh_intersect(mesh, prim_index, o_ray, t_min, t_max)
        },
        Shape::Sphere { center, radius } => {
            ray_sphere_intersect(*center, *radius, o_ray, t_min, t_max)
        },
    }) else {
        return None;
    };

    let w_point = o2w_transform.apply_point(o_intersection_result.point);
    let w_normal = o2w_transform.apply_normal(o_intersection_result.normal).unit();

    let w_intersection_result = IntersectResult {
        t: o_intersection_result.t * s,
        point: w_point,
        normal: w_normal,
    };

    Some(w_intersection_result)
}


fn ray_sphere_intersect(
    center: Vec3,
    radius: f32,
    o_ray: Ray, // object-space
    t_min: f32,
    t_max: f32,
) -> Option<IntersectResult> /* object-space */ {
    let origin_minus_center = o_ray.origin - center;

    let a = o_ray.direction.square_magnitude();
    let b = 2.0 * Vec3::dot(o_ray.direction, origin_minus_center);
    let c = origin_minus_center.square_magnitude() - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    let (mut t1, mut t2) = if discriminant < 0.0 {
        return None
    }
    else if discriminant == 0.0 {
        let t = -b / (2.0 * a);
        (t, t)
    }
    else {
        // more numerically stable than normal quadratic formula
        let q = -0.5 * (b + b.signum() * discriminant.sqrt());
        (q / a, c / q)
    };

    if t1 > t2 {
        std::mem::swap(&mut t1, &mut t2);
    }

    let t = if t1 >= t_min && t1 <= t_max {
        t1
    }
    else if t2 >= t_min && t2 <= t_max {
        t2
    }
    else {
        return None
    };

    let result = IntersectResult {
        t,
        point: o_ray.at(t),
        normal: (o_ray.at(t) - center) / radius,
    };
    
    Some(result)
}

fn ray_mesh_intersect(
    mesh: &Mesh, tri_index: u32,
    o_ray: Ray, // object-space
    t_min: f32,
    t_max: f32,
) -> Option<IntersectResult> /* object-space */ {
    let Vec3u(i0, i1, i2) = mesh.tris[tri_index as usize];
    let p0 = mesh.vertices[i0 as usize];
    let p1 = mesh.vertices[i1 as usize];
    let p2 = mesh.vertices[i2 as usize];
    let n0 = mesh.normals[i0 as usize];
    let n1 = mesh.normals[i1 as usize];
    let n2 = mesh.normals[i2 as usize];
    
    ray_triangle_intersect(p0, p1, p2, n0, n1, n2, o_ray, t_min, t_max)
}

#[allow(non_snake_case)] // conventional symbols
fn ray_triangle_intersect(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    n0: Vec3,
    n1: Vec3,
    n2: Vec3,
    o_ray: Ray,
    t_min: f32,
    t_max: f32,
) -> Option<IntersectResult> {
    // Moller-Trumbore algorithm
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    
    let P = Vec3::cross(o_ray.direction, e2);
    let denom = Vec3::dot(P, e1);

    if denom == 0.0 {
        return None;
    }

    let T = o_ray.origin - p0;
    let u = Vec3::dot(P, T) / denom;
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let Q = Vec3::cross(T, e1);
    let v = Vec3::dot(Q, o_ray.direction) / denom;

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = Vec3::dot(Q, e2) / denom;
    if t < t_min || t > t_max {
        return None;
    }

    let w = 1.0 - u - v;
    let n = Vec3::normalized(w * n0 + u * n1 + v * n2);
    
    let result = IntersectResult {
        t,
        point: o_ray.at(t),
        normal: n
    };

    Some(result)
}