use raytracing::{accel::bvh2::{LinearizedBVHNode, TriPtr}, geometry::{Mesh, Vec3, AABB}};

use crate::{geometry::ray_triangle_intersect, ray::Ray};

// return t at which ray intersects an AABB, or infinity if no intersection
fn intersect_aabb(aabb: AABB, ray: Ray) -> f32 {
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

    if t0 <= t1 && t0 >= 0.0 {
        t0
    } else {
        f32::INFINITY
    }
}

pub(crate) struct HitInfo {
    pub(crate) t: f32,
    pub(crate) barycentric: Vec3,
    pub(crate) point: Vec3,
    pub(crate) tangent: Vec3,
    pub(crate) normal: Vec3,
    pub(crate) material_idx: u32,
}

pub(crate) struct BVHData<'bvh> {
    pub(crate) nodes: &'bvh [LinearizedBVHNode],
    pub(crate) meshes: &'bvh [Mesh],
    pub(crate) indices: &'bvh [TriPtr]
}

pub(crate) fn traverse_bvh(
    ray: Ray,
    t_min: f32,
    mut t_max: f32, 
    bvh: &BVHData<'_>,
    early_exit: bool
) -> Option<HitInfo> {
    let mut node = &bvh.nodes[0];
    let mut stack: Vec<usize> = Vec::with_capacity(48);
    let mut hit_info: Option<HitInfo> = None;
    loop {
        if node.tri_count > 0 {
            // leaf node
            for i in 0..node.tri_count {
                let tri_ptr_idx = node.left_first + i;
                let tri_ptr = bvh.indices[tri_ptr_idx as usize];
                let mesh = &bvh.meshes[tri_ptr.geom_id as usize];
                let vertex_indices = tri_ptr.verts;
                let p0 = mesh.vertices[vertex_indices.0 as usize];
                let p1 = mesh.vertices[vertex_indices.1 as usize];
                let p2 = mesh.vertices[vertex_indices.2 as usize];

                let result = ray_triangle_intersect(
                    p0, p1, p2, ray, t_min, t_max
                );

                if let Some(tuv) = result {
                    let (t, u, v) = (tuv.0, tuv.1, tuv.2);
                    if t < t_min || t > t_max {
                        continue;
                    }
                    
                    // barycentric interpolation of normal
                    let w = 1.0 - u - v;
                    let n0 = mesh.normals[vertex_indices.0 as usize];
                    let n1 = mesh.normals[vertex_indices.1 as usize];
                    let n2 = mesh.normals[vertex_indices.2 as usize];

                    let normal = Vec3::normalized(w * n0 + u * n1 + v * n2);
                    
                    let material_idx = mesh.material_idx as u32;

                    hit_info = Some(HitInfo { 
                        t, 
                        barycentric: Vec3(u, v, 1.0 - u - v), 
                        point: ray.at(t), 
                        tangent: Vec3::zero(), // TODO: fix
                        normal,
                        material_idx
                    });

                    t_max = t;

                    if early_exit {
                        return hit_info;
                    }
                }
            }

            if let Some(prev) = stack.pop() {
                node = &bvh.nodes[prev];
            }
            else {
                break;
            }
        }
        else {
            // inner node
            let left_idx = node.left_first as usize;
            let left = &bvh.nodes[left_idx];
            let right = &bvh.nodes[left_idx + 1];

            let t_left = intersect_aabb(left.aabb(), ray);
            let t_right = intersect_aabb(right.aabb(), ray);

            let (
                t_closer, 
                t_farther, 
                closer, 
                farther
            ) = if t_left < t_right {
                (t_left, t_right, left_idx, left_idx + 1)
            } else {
                (t_right, t_left, left_idx + 1, left_idx)
            };

            if f32::is_infinite(t_closer) {
                if let Some(prev) = stack.pop() {
                    node = &bvh.nodes[prev];
                }
                else {
                    break;
                }
            }
            else {
                node = &bvh.nodes[closer];
                if f32::is_finite(t_farther) {
                    stack.push(farther);
                }
            }

        }
    }

    hit_info
} 