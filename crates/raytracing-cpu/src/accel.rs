use std::ops::Range;

use raytracing::{accel::bvh2::{LinearizedBVHNode, TriPtr}, geometry::{Mesh, Vec3, AABB}};

use crate::{geometry::ray_triangle_intersect, ray::Ray};

// return range of t at which ray intersects an AABB, or infinity if no intersection
fn intersect_aabb(aabb: AABB, ray: Ray) -> Option<Range<f32>> {
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

pub(crate) struct HitInfo {
    pub(crate) t: f32,
    pub(crate) barycentric: Vec3,
    pub(crate) point: Vec3,
    pub(crate) tangent: Vec3,
    pub(crate) normal: Vec3,

    pub(crate) material_idx: u32,
    pub(crate) light_idx: Option<u32>,
}

pub(crate) struct BVHData<'bvh> {
    pub(crate) nodes: &'bvh [LinearizedBVHNode],
    pub(crate) meshes: &'bvh [Mesh],
    pub(crate) indices: &'bvh [TriPtr]
}

struct BVHTraversalStackEntry {
    node: usize,
    t_min: f32,
    t_max: f32,
}


// Find the closest intersection of ray against bvh data in the range t_min..t_max
// (or any, if early_exit is true)
pub(crate) fn traverse_bvh(
    ray: Ray,
    t_min: f32,
    t_max: f32, 
    bvh: &BVHData<'_>,
    early_exit: bool,
) -> Option<HitInfo> {
    let t_min = t_min;
    let mut closest_t = t_max;

    let mut stack: Vec<BVHTraversalStackEntry> = Vec::with_capacity(48);
    let root = BVHTraversalStackEntry { 
        node: 0, // root is node 0 by construction
        t_min: f32::NEG_INFINITY, // TODO: bvh doesn't contain bounding box of whole scene 
        t_max: f32::INFINITY 
    };
    stack.push(root);
    
    let mut hit_info: Option<HitInfo> = None;

    loop {
        let mut node_idx = None;
        while let Some(prev) = stack.pop() {
            if prev.t_min > closest_t || prev.t_max < t_min {
                continue;
            }
            else {
                node_idx = Some(prev.node);
                break;
            }
        }

        let node = if let Some(node_idx) = node_idx {
            &bvh.nodes[node_idx]
        }
        else {
            break
        };

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
                    p0, p1, p2, ray
                );

                if let Some(tuv) = result {
                    let (t, u, v) = (tuv.0, tuv.1, tuv.2);
                    if t < t_min || t > closest_t {
                        continue;
                    }
                    
                    // barycentric interpolation of normal
                    let w = 1.0 - u - v;
                    let n0 = mesh.normals[vertex_indices.0 as usize];
                    let n1 = mesh.normals[vertex_indices.1 as usize];
                    let n2 = mesh.normals[vertex_indices.2 as usize];

                    let normal = Vec3::normalized(w * n0 + u * n1 + v * n2);
                    
                    let material_idx = mesh.material_idx;
                    let light_idx = mesh.light_idx;

                    hit_info = Some(HitInfo { 
                        t, 
                        barycentric: Vec3(u, v, 1.0 - u - v), 
                        point: ray.at(t), 
                        tangent: Vec3::zero(), // TODO: fix
                        normal,

                        material_idx,
                        light_idx
                    });

                    closest_t = t;

                    if early_exit {
                        return hit_info;
                    }
                }
            }
        }
        else {
            // inner node
            let left_idx = node.left_first as usize;
            let left = &bvh.nodes[left_idx];
            let right = &bvh.nodes[left_idx + 1];

            let t_left = intersect_aabb(left.aabb(), ray);
            let t_right = intersect_aabb(right.aabb(), ray);
            
            match (t_left, t_right) {
                (None, None) => {
                    // no intersection with either child, pop stack on next loop iteration
                },
                (None, Some(r)) => {
                    let r = BVHTraversalStackEntry {
                        node: left_idx + 1,
                        t_min: r.start,
                        t_max: r.end,
                    };
                    stack.push(r);
                },
                (Some(l), None) => {
                    let l = BVHTraversalStackEntry {
                        node: left_idx,
                        t_min: l.start,
                        t_max: l.end
                    };
                    stack.push(l);
                },
                (Some(mut l), Some(mut r)) => {
                    if r.start < l.start {
                        std::mem::swap(&mut l, &mut r);
                    }
                    let l = BVHTraversalStackEntry {
                        node: left_idx,
                        t_min: l.start,
                        t_max: l.end
                    };
                    let r = BVHTraversalStackEntry {
                        node: left_idx + 1,
                        t_min: r.start,
                        t_max: r.end,
                    };

                    // prefer traversing closer node
                    stack.push(r);
                    stack.push(l);
                },
            }
        }
    }

    hit_info
} 