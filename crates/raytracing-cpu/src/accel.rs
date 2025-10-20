use std::ops::Range;

use raytracing::{accel::bvh2::PrimPtr, geometry::{Transform, Vec2, Vec3, AABB}, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

use crate::geometry::{self, intersect_aabb};
use crate::{ray::Ray, scene::CPUAccelerationStructures};

// all fields in world-space
pub(crate) struct HitInfo {
    pub(crate) t: f32,
    pub(crate) uv: Vec2,
    pub(crate) point: Vec3,
    pub(crate) normal: Vec3,

    pub(crate) material_idx: u32,
    pub(crate) light_idx: Option<u32>,
}

#[derive(Debug, Clone)]
struct TraversalStackEntry {
    // aggregate_index indexes into scene description, while
    // bvh_index indexes into CPUAccelerationStructures
    // both should correspond to the same thing
    aggregate_index: AggregatePrimitiveIndex,
    bvh_index: u32,

    // index into LinearizedBVH
    node_index: u32,

    // which child is being considered (0 or 1 for internal nodes, 0 up to # of children for leaf nodes)
    progress: u32,
}

// Every BVH is traversed in its local coordinate frame we store this info 
// out-of-line of traversal code / traversal stack for simplicity
#[derive(Debug, Clone)]
struct LocalTraversalContext {
    transform: Transform,
    local_ray: Ray,
    scaling_factor: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct CPUTraversalContext<'scene> {
    pub(crate) scene: &'scene Scene,
    pub(crate) acceleration_structures: &'scene CPUAccelerationStructures,
}


impl<'scene> CPUTraversalContext<'scene> {
    pub(crate) fn new(scene: &'scene Scene, acceleration_structures: &'scene CPUAccelerationStructures)
        -> CPUTraversalContext<'scene> {
        CPUTraversalContext { 
            scene, 
            acceleration_structures,
        }
    }
}


// Find the closest intersection of ray against bvh data in the range t_min..t_max
// (or any, if early_exit is true)
pub(crate) fn traverse_bvh(
    ray: Ray,
    t_min: f32,
    t_max: f32, 
    traversal_context: CPUTraversalContext,
    early_exit: bool,
) -> Option<HitInfo> {
    // t_min, closest_t are in root coordinate frame
    let mut closest_t = t_max;

    let CPUTraversalContext { 
        scene, 
        acceleration_structures,
    } = traversal_context;

    let mut stack: Vec<TraversalStackEntry> = Vec::with_capacity(48);
    
    // check intersection with root
    let root_bvh_index = acceleration_structures.root_bvh_index();
    let root_bounds = acceleration_structures.bvhs[root_bvh_index].bounds();

    let Some(_) = intersect_aabb(root_bounds, ray) else {
        return None;
    };

    let root_entry = TraversalStackEntry { 
        aggregate_index: scene.root_index(),
        bvh_index: acceleration_structures.root_bvh_index() as u32,
        node_index: 0, // root is node 0 by construction
        
        progress: 0
    };

    stack.push(root_entry);

    let mut local_traversal_ctxs: Vec<Option<LocalTraversalContext>> = vec![None; acceleration_structures.bvhs.len()];
    local_traversal_ctxs[root_bvh_index] = Some(LocalTraversalContext {
        transform: Transform::identity(),
        local_ray: ray,
        scaling_factor: 1.0,
    });

    let mut hit_info: Option<HitInfo> = None;

    loop {
        let Some(top_of_stack) = stack.last() else {
            break
        };

        let stack_idx = stack.len() - 1;
        let TraversalStackEntry {
            aggregate_index,
            bvh_index,
            node_index,
            progress,
        } = *top_of_stack;

        let local_traversal_context = local_traversal_ctxs[bvh_index as usize]
            .as_ref()
            .expect("uninitialized local traversal context");

        let root_to_local_transform = local_traversal_context.transform.clone();
        let local_to_root_transform = root_to_local_transform.invert();
        let local_ray = local_traversal_context.local_ray;
        let local_t_min = t_min / local_traversal_context.scaling_factor;
        let local_closest_t = closest_t / local_traversal_context.scaling_factor;
        let local_bvh = &acceleration_structures.bvhs[bvh_index as usize];
        let node = &local_bvh.nodes[node_index as usize];

        use raytracing::accel::bvh2::DepthFirstLinearizedBVHNode::{Internal, Leaf};
        match node {
            Leaf { prim_ptr_offset, prim_count, .. } => {
                let progress = stack[stack_idx].progress;
                let prim_idx = *prim_ptr_offset + progress;
                let PrimPtr {
                    geom_id,
                    prim_id,
                } = acceleration_structures.bvhs[bvh_index as usize].prim_ptrs[prim_idx as usize];

                let (primitive_index, transform) = 
                    scene.get_descendant(aggregate_index, geom_id as usize);

                match scene.get_primitive(primitive_index) {
                    Primitive::Basic(basic_primitive) => {
                        let intersect_result = geometry::intersect_shape(
                            local_ray, 
                            local_t_min, 
                            local_closest_t, 
                            &transform, 
                            &basic_primitive.shape, 
                            prim_id
                        );

                        if let Some(intersect_result) = intersect_result {
                            let global_t = intersect_result.t * local_traversal_context.scaling_factor;
                            let global_point = local_to_root_transform.apply_point(intersect_result.point);
                            let global_normal = local_to_root_transform.apply_normal(intersect_result.normal);

                            hit_info = Some(HitInfo { 
                                t: global_t, 
                                uv: Vec2(0.0, 0.0), // TODO: uvs 
                                point: global_point, 
                                normal: global_normal, 
                                material_idx: basic_primitive.material, 
                                light_idx: basic_primitive.area_light 
                            });
                            
                            closest_t = global_t;
                            
                            if early_exit {
                                return hit_info;
                            }
                        }

                        stack[stack_idx].progress += 1;
                        if stack[stack_idx].progress == prim_count.get() {
                            stack.pop();
                        }
                    },
                    Primitive::Aggregate(_) => {
                        // check if we intersect the bounding box of aggregate
                        let sub_bvh_index = prim_id;
                        let sub_bvh_aabb = acceleration_structures.bvhs[sub_bvh_index as usize].bounds();
                        
                        // make sure it is in our local frame
                        let inverse_transform = transform.invert();
                        let sub_bvh_aabb = AABB::transform_aabb(sub_bvh_aabb, &inverse_transform);

                        stack[stack_idx].progress += 1;
                        if stack[stack_idx].progress == prim_count.get() {
                            stack.pop();
                        }

                        match geometry::intersect_aabb(sub_bvh_aabb, local_ray) {
                            Some(Range { start, .. }) if start > local_closest_t => {
                                // initialize local traversal context and push stack
                                if local_traversal_ctxs[sub_bvh_index as usize].is_none() {
                                    let root_to_sub_bvh = root_to_local_transform.compose(transform);
                                    let new_local_ray = Ray::transform(ray, &root_to_sub_bvh);
                                    let new_scaling_factor = new_local_ray.direction.length() / ray.direction.length(); 
                                    local_traversal_ctxs[sub_bvh_index as usize] = Some(LocalTraversalContext { 
                                        transform: root_to_sub_bvh, 
                                        local_ray: new_local_ray, 
                                        scaling_factor: new_scaling_factor
                                    });
                                }
                                stack.push(TraversalStackEntry { 
                                    aggregate_index: primitive_index.try_into().unwrap(), 
                                    bvh_index: sub_bvh_index, 
                                    node_index: 0, // root is 0 by construction 
                                    progress: 0
                                })
                            }
                            _ => () // didn't intersect, move on
                        }
                    },
                    Primitive::Transform(_) => 
                        unreachable!("transforms should be flattened"),
                }
            },
            Internal { right_child_offset, .. } => {
                if progress == 0 {
                    // do left child
                    let left_child_index = node_index + 1;
                    let left_child_bounds = local_bvh.nodes[left_child_index as usize].bounds();
                    match geometry::intersect_aabb(left_child_bounds, local_ray) {
                        Some(Range { start, .. }) if start > local_closest_t => {
                            stack.push(TraversalStackEntry { 
                                aggregate_index, 
                                bvh_index, 
                                node_index: left_child_index, 
                                progress: 0 
                            });
                        }
                        _ => () // didn't intersect, move on
                    }
                    stack[stack_idx].progress += 1;
                }
                else {
                    // pop and do right child
                    stack.pop();
                    let right_child_bounds = local_bvh.nodes[*right_child_offset as usize].bounds();
                    match geometry::intersect_aabb(right_child_bounds, local_ray) {
                        Some(Range { start, .. }) if start > local_closest_t => {
                            stack.push(TraversalStackEntry { 
                                aggregate_index, 
                                bvh_index, 
                                node_index: *right_child_offset, 
                                progress: 0 
                            });
                        }
                        _ => () // didn't intersect, move on
                    }
                }
            },
        }
    }

    hit_info
} 