use std::{f32, ops::Range};

use raytracing::geometry::{AABB, Mesh, Shape, Transform, Vec2, Vec3, Vec3u};

use crate::ray::Ray;

// from a *normalized* z, produce some x and y s.t. {x,y,z} is a valid right-handed coordinate system
pub(crate) fn make_orthonormal_basis(z: Vec3) -> (Vec3, Vec3) {
    let a = if f32::abs(z.2) < 0.8 {
        Vec3(0.0, 0.0, 1.0)
    }
    else {
        Vec3(0.0, 1.0, 0.0)
    };

    let x = Vec3::cross(a, z).unit();
    let y = Vec3::cross(z, x);

    (x, y)
}

#[test]
fn test_make_orthonormal_basis() {
    let z = Vec3(0.262, -0.151, 0.370).unit();
    let (x, y) = make_orthonormal_basis(z);
    assert!(f32::abs(1.0 - x.length()) < 1.0e-8);
    assert!(f32::abs(1.0 - y.length()) < 1.0e-8);
    let x_cross_y = Vec3::cross(x, y);
    assert!((z - x_cross_y).length() < 1.0e-6)
}

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
    pub(crate) uv: Vec2,
    pub(crate) point: Vec3,
    pub(crate) normal: Vec3,
    // TODO: add tangent vector

    pub(crate) dpdu: Vec3,
    pub(crate) dpdv: Vec3,
}

pub(crate) fn intersect_shape(
    w_ray: Ray, // bvh-space
    t_min: f32, // bvh-space
    t_max: f32, // bvh-space
    o2w_transform: &Transform, // object-to-bvh
    shape: &Shape, // object-space
    prim_index: u32,
) -> Option<IntersectResult> /* bvh-space */ {
    let w2o_transform = o2w_transform.invert();
    let o_ray = Ray::transform(w_ray, &w2o_transform);

    /* Note: w2o_transform is affine, and can be decomposed into TRS,  
       if there is any scaling, it will scale distances between points 
       and the direction vector of the ray, so the t computed in object space
       will be the same as the t in bvh-space (the same logic applies to min/max bounds as well)
    */
    
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
    let w_dpdu = o2w_transform.apply_vector(o_intersection_result.dpdu);
    let w_dpdv = o2w_transform.apply_vector(o_intersection_result.dpdv);

    let w_intersection_result = IntersectResult {
        t: o_intersection_result.t,
        uv: o_intersection_result.uv,
        point: w_point,
        normal: w_normal,

        dpdu: w_dpdu,
        dpdv: w_dpdv
    };

    Some(w_intersection_result)
}


fn ray_sphere_intersect(
    center: Vec3, // object-space
    radius: f32, // object-space
    o_ray: Ray, // object-space
    t_min: f32, // object-space
    t_max: f32, // object-space
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

    // we take the convention of u being azimuthal angle (phi), 
    // v being polar angle (theta) with z-up
    let point = o_ray.at(t);
    
    let r_cos_theta = point.z();
    let theta = f32::acos(r_cos_theta / radius);
    let r_sin_theta_cos_phi = point.x();
    let cos_phi = r_sin_theta_cos_phi / (radius * f32::sin(theta));
    let r_sin_theta_sin_phi = point.y();
    let sin_phi = r_sin_theta_sin_phi / (radius * f32::sin(theta));

    let phi = if point.y() > 0.0 {
        f32::acos(cos_phi)
    } else {
        2.0 * f32::consts::PI - f32::acos(cos_phi)
    };
    
    let uv = Vec2(
        phi / (2.0 * f32::consts::PI),
        theta / f32::consts::PI 
    );

    let dpdu = Vec3(
        -2.0 * f32::consts::PI * point.y(),
        2.0 * f32::consts::PI * point.x(),
        0.0
    );

    let sin_theta = f32::sin(theta);
    let dpdv = f32::consts::PI * Vec3(
        point.z() * cos_phi,
        point.z() * sin_phi,
        -radius * sin_theta
    );

    let result = IntersectResult {
        t,
        uv,
        point,
        normal: (point - center) / radius,

        dpdu,
        dpdv,
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

    let (t, u, v) = ray_triangle_intersect(p0, p1, p2, o_ray, t_min, t_max)?;

    let w = 1.0 - u - v;
    let n = Vec3::normalized(w * n0 + u * n1 + v * n2);
    let (uv, dpdu, dpdv) = {
        let (uv0, uv1, uv2) = if mesh.uvs.is_empty() {
            // if uvs are not provided, then using these values as uv coords 
            // is fine since it won't be used to access textures
            (Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.0, 1.0))
        } else {
            (mesh.uvs[i0 as usize], mesh.uvs[i1 as usize], mesh.uvs[i2 as usize])
        };

        let uv = w * uv0 + u * uv1 + v * uv2;
        
        // see pbrt 4ed equation 6.7 and surrounding text
        let duv02 = uv0 - uv2;
        let duv12 = uv1 - uv2;
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = duv02.u() * duv12.v() - duv02.v() * duv12.u();

        let degenerate_uv = f32::abs(determinant) < 1.0e-9;
        
        if degenerate_uv {
            (uv, Vec3::zero(), Vec3::zero())
        }
        else {
            let inv_det = 1.0 / determinant;

            let dpdu = inv_det * (duv12.v() * dp02 - duv02.v() * dp12);
            let dpdv = inv_det * (duv02.u() * dp12 - duv12.u() * dp02);

            (uv, dpdu, dpdv)
        }
    };
    
    Some(IntersectResult {
        t,
        uv,
        point: o_ray.at(t),
        normal: n,

        dpdu,
        dpdv,
    })
}

#[allow(non_snake_case)] // conventional symbols
fn ray_triangle_intersect(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    
    o_ray: Ray,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32)> {
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

    Some((t, u, v))
}