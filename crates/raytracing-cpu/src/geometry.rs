use raytracing::geometry::Vec3;

use crate::ray::Ray;

pub(crate) fn ray_triangle_intersect(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    ray: Ray,
) -> Option<Vec3> {
    // Moller-Trumbore algorithm
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    
    let P = Vec3::cross(ray.direction, e2);
    let denom = Vec3::dot(P, e1);

    if denom == 0.0 {
        return None;
    }

    let T = ray.origin - p0;
    let u = Vec3::dot(P, T) / denom;
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let Q = Vec3::cross(T, e1);
    let v = Vec3::dot(Q, ray.direction) / denom;

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = Vec3::dot(Q, e2) / denom;
    let tuv = Vec3(t, u, v);
    return Some(tuv);
}