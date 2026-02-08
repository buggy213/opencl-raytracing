use embree4::{
    bvh::{BVHBuildArguments, BVHCallbacks, BuiltBVH},
    Bounds, BuildPrimitive, Device, BVH,
};
use std::{collections::VecDeque, num::NonZero, ptr::null};
use tracing::warn;

use crate::{
    geometry::{Mesh, Shape, Transform, Vec3, AABB},
    macros::{variadic_max_comparator, variadic_min_comparator},
};

#[derive(Debug)]
pub enum BVHNode {
    // we keep pointers instead of references in internal BVHNode to avoid lifetime
    // complications; requires a small amount of unsafe
    Inner {
        left: *const BVHNode,
        right: *const BVHNode,
        left_aabb: AABB,
        right_aabb: AABB,
    },
    Leaf {
        // 8 bvh primitives in a leaf at most
        prim_count: u32,
        prim_indices: [u32; 8],
        geom_indices: [u32; 8],
    },
}

impl From<Bounds> for AABB {
    fn from(value: Bounds) -> Self {
        AABB {
            minimum: Vec3(value.lower_x, value.lower_y, value.lower_z),
            maximum: Vec3(value.upper_x, value.upper_y, value.upper_z),
        }
    }
}

impl Default for BVHNode {
    fn default() -> Self {
        BVHNode::Inner {
            left: null(),
            right: null(),
            left_aabb: AABB::default(),
            right_aabb: AABB::default(),
        }
    }
}

pub struct BVH2<'device> {
    handle: BuiltBVH<'device, BVHNode>,
}

impl BVH2<'_> {
    fn create_node_bvh2(child_count: u32) -> BVHNode {
        assert!(
            child_count == 2,
            "Embree tried to create a BVH2 node with >2 children"
        );
        BVHNode::Inner {
            left: null(),
            right: null(),
            left_aabb: AABB::default(),
            right_aabb: AABB::default(),
        }
    }

    fn create_leaf_bvh2(primitives: &[BuildPrimitive]) -> BVHNode {
        let prim_indices = [0, 1, 2, 3, 4, 5, 6, 7]
            .map(|i| primitives.get(i).map(|p| p.primID).unwrap_or_default());

        let geom_indices = [0, 1, 2, 3, 4, 5, 6, 7]
            .map(|i| primitives.get(i).map(|p| p.geomID).unwrap_or_default());

        BVHNode::Leaf {
            prim_count: primitives.len() as u32,
            prim_indices,
            geom_indices,
        }
    }

    fn set_node_bounds_bvh2(node: &mut BVHNode, bounds: &[&Bounds]) {
        assert!(bounds.len() == 2);
        match node {
            BVHNode::Inner {
                left_aabb,
                right_aabb,
                ..
            } => {
                *left_aabb = (*bounds[0]).into();
                *right_aabb = (*bounds[1]).into();
            }
            _ => unreachable!("Embree should not confuse leaf and inner nodes"),
        }
    }

    fn set_node_children_bvh2(node: &mut BVHNode, children: &[&BVHNode]) {
        assert!(children.len() == 2);
        match node {
            BVHNode::Inner { left, right, .. } => {
                *left = children[0] as *const BVHNode;
                *right = children[1] as *const BVHNode;
            }
            _ => unreachable!("Embree should not confuse leaf and inner nodes"),
        }
    }

    const BVH2_CALLBACKS: BVHCallbacks<BVHNode> = BVHCallbacks::new(
        BVH2::create_node_bvh2,
        BVH2::create_leaf_bvh2,
        BVH2::set_node_bounds_bvh2,
        BVH2::set_node_children_bvh2,
    );
}

impl<'device> BVH2<'device> {
    fn create(device: &'device Device, primitives: Vec<BuildPrimitive>) -> BVH2<'device> {
        let bvh = BVH::new(device);

        let bvh_arguments = BVHBuildArguments::new()
            .max_leaf_size(8)
            .register_callbacks(&BVH2::BVH2_CALLBACKS)
            .set_primitives(primitives.as_slice());

        let build_result = bvh.build(bvh_arguments);

        BVH2 {
            handle: build_result,
        }
    }
}

// unsafe accessors
impl BVH2<'_> {
    pub fn root(&self) -> &BVHNode {
        self.handle.root()
    }
}

impl BVHNode {
    pub fn left(&self) -> Option<&BVHNode> {
        match self {
            BVHNode::Inner { left, .. } => (!left.is_null()).then(|| unsafe { &**left }),
            BVHNode::Leaf { .. } => None,
        }
    }

    pub fn right(&self) -> Option<&BVHNode> {
        match self {
            BVHNode::Inner { right, .. } => (!right.is_null()).then(|| unsafe { &**right }),
            BVHNode::Leaf { .. } => None,
        }
    }
}

pub struct BVH2Builder {
    primitives: Vec<BuildPrimitive>,
}

impl Default for BVH2Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl BVH2Builder {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
        }
    }

    pub fn add_shape(&mut self, shape: &Shape, o2w_transform: &Transform, geom_index: u32) {
        match shape {
            Shape::TriangleMesh(mesh) => self.add_mesh(mesh, o2w_transform, geom_index),
            Shape::Sphere { center, radius } => {
                self.add_sphere(*center, *radius, o2w_transform, geom_index)
            }
        }
    }

    pub fn add_sub_bvh(
        &mut self,
        bounds: AABB,
        o2w_transform: &Transform,
        geom_index: u32,
        prim_index: u32,
    ) {
        self.add_bvh(bounds, o2w_transform, geom_index, prim_index);
    }

    pub fn build<'device>(self, device: &'device Device) -> BVH2<'device> {
        BVH2::create(device, self.primitives)
    }
}

impl BVH2Builder {
    fn add_tri(&mut self, p0: Vec3, p1: Vec3, p2: Vec3, tri_index: u32, geom_index: u32) {
        let min = variadic_min_comparator!(Vec3::elementwise_min, p0, p1, p2);
        let max = variadic_max_comparator!(Vec3::elementwise_max, p0, p1, p2);
        let primitive = BuildPrimitive {
            lower_x: min.0,
            lower_y: min.1,
            lower_z: min.2,
            geomID: geom_index,
            upper_x: max.0,
            upper_y: max.1,
            upper_z: max.2,
            primID: tri_index,
        };

        self.primitives.push(primitive);
    }

    fn add_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        o2w_transform: &Transform,
        geom_index: u32,
    ) {
        let aabb = AABB::new(
            center - Vec3(radius, radius, radius),
            center + Vec3(radius, radius, radius),
        );

        let transformed_aabb = AABB::transform_aabb(aabb, o2w_transform);

        let primitive = BuildPrimitive {
            lower_x: transformed_aabb.minimum.x(),
            lower_y: transformed_aabb.minimum.y(),
            lower_z: transformed_aabb.minimum.z(),
            geomID: geom_index,
            upper_x: transformed_aabb.maximum.x(),
            upper_y: transformed_aabb.maximum.y(),
            upper_z: transformed_aabb.maximum.z(),
            primID: 0,
        };

        self.primitives.push(primitive);
    }

    fn add_mesh(&mut self, mesh: &Mesh, o2w_transform: &Transform, geom_index: u32) {
        for (tri_index, tri) in mesh.tris.iter().cloned().enumerate() {
            let (t0, t1, t2) = (tri.0, tri.1, tri.2);
            let p0 = o2w_transform.apply_point(mesh.vertices[t0 as usize]);
            let p1 = o2w_transform.apply_point(mesh.vertices[t1 as usize]);
            let p2 = o2w_transform.apply_point(mesh.vertices[t2 as usize]);
            self.add_tri(p0, p1, p2, tri_index as u32, geom_index);
        }
    }

    fn add_bvh(
        &mut self,
        bounds: AABB,
        o2w_transform: &Transform,
        geom_index: u32,
        prim_index: u32,
    ) {
        let transformed_aabb = AABB::transform_aabb(bounds, o2w_transform);

        let primitive = BuildPrimitive {
            lower_x: transformed_aabb.minimum.x(),
            lower_y: transformed_aabb.minimum.y(),
            lower_z: transformed_aabb.minimum.z(),
            geomID: geom_index,
            upper_x: transformed_aabb.maximum.x(),
            upper_y: transformed_aabb.maximum.y(),
            upper_z: transformed_aabb.maximum.z(),
            primID: prim_index,
        };

        self.primitives.push(primitive);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrimPtr {
    pub geom_id: u32,
    pub prim_id: u32,
}

#[repr(C)]
#[derive(Debug)]
// Optimized for GPU traversal (breadth-first)
pub struct BreadthFirstLinearizedBVHNode {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    // if tri_count is 0, then this node is an internal node, and the value represents
    // index of left child (right child is subsequent element).
    // otherwise, this node is a leaf node, and the value represents the index of the
    // first PrimPtr
    pub left_first: u32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
    pub prim_count: u32,
}

impl BreadthFirstLinearizedBVHNode {
    pub fn bounds(&self) -> AABB {
        AABB {
            minimum: Vec3(self.min_x, self.min_y, self.min_z),
            maximum: Vec3(self.max_x, self.max_y, self.max_z),
        }
    }
}

pub struct BreadthFirstLinearizedBVH {
    pub nodes: Vec<BreadthFirstLinearizedBVHNode>,
    pub prim_ptrs: Vec<PrimPtr>,
}

impl BreadthFirstLinearizedBVH {
    pub fn linearize_bvh(bvh: &BVH2) -> BreadthFirstLinearizedBVH {
        // perform BFS over BVH, make primitives pointed to by leaf nodes contiguous
        let mut queue: VecDeque<(&BVHNode, AABB)> = VecDeque::new();
        let node = bvh.root();
        
        let root_aabb = match node {
            BVHNode::Inner {
                left_aabb,
                right_aabb,
                ..
            } => {
                AABB::surrounding_box(*left_aabb, *right_aabb)
            }
            BVHNode::Leaf { .. } => {
                unimplemented!("BVH root is a leaf node");
            }
        };

        let mut bvh_nodes: Vec<BreadthFirstLinearizedBVHNode> = Vec::new();
        let mut contiguous_prims: Vec<PrimPtr> = Vec::new();
        queue.push_back((node, root_aabb));
        while !queue.is_empty() {
            let (node, aabb) = queue.pop_front().unwrap();
            match node {
                BVHNode::Inner {
                    left_aabb,
                    right_aabb,
                    ..
                } => {
                    let left_idx = bvh_nodes.len() + queue.len() + 1;
                    queue.push_back((node.left().unwrap(), *left_aabb));
                    queue.push_back((node.right().unwrap(), *right_aabb));
                    bvh_nodes.push(BreadthFirstLinearizedBVHNode {
                        min_x: aabb.minimum.0,
                        min_y: aabb.minimum.1,
                        min_z: aabb.minimum.2,
                        left_first: left_idx as u32,
                        max_x: aabb.maximum.0,
                        max_y: aabb.maximum.1,
                        max_z: aabb.maximum.2,
                        prim_count: 0,
                    });
                }
                BVHNode::Leaf {
                    prim_count,
                    prim_indices,
                    geom_indices,
                } => {
                    bvh_nodes.push(BreadthFirstLinearizedBVHNode {
                        min_x: aabb.minimum.0,
                        min_y: aabb.minimum.1,
                        min_z: aabb.minimum.2,
                        left_first: contiguous_prims.len() as u32,
                        max_x: aabb.maximum.0,
                        max_y: aabb.maximum.1,
                        max_z: aabb.maximum.2,
                        prim_count: *prim_count,
                    });

                    for i in 0..*prim_count {
                        let geom_id = geom_indices[i as usize];
                        let prim_id = prim_indices[i as usize];
                        let prim_ptr = PrimPtr { geom_id, prim_id };
                        contiguous_prims.push(prim_ptr);
                    }
                }
            }
        }

        BreadthFirstLinearizedBVH {
            nodes: bvh_nodes,
            prim_ptrs: contiguous_prims,
        }
    }

    pub fn root(&self) -> &BreadthFirstLinearizedBVHNode {
        &self.nodes[0]
    }

    pub fn bounds(&self) -> AABB {
        self.root().bounds()
    }
}

// Optimized for CPU traversal (depth-first)
#[derive(Debug)]
pub enum DepthFirstLinearizedBVHNode {
    Internal {
        bounds: AABB,
        right_child_offset: u32,
    },
    Leaf {
        bounds: AABB,
        prim_ptr_offset: u32,
        prim_count: NonZero<u32>,
    },
}

impl DepthFirstLinearizedBVHNode {
    pub fn bounds(&self) -> AABB {
        match self {
            DepthFirstLinearizedBVHNode::Internal { bounds, .. }
            | DepthFirstLinearizedBVHNode::Leaf { bounds, .. } => *bounds,
        }
    }
}

#[derive(Debug)]
pub struct DepthFirstLinearizedBVH {
    pub nodes: Vec<DepthFirstLinearizedBVHNode>,
    pub prim_ptrs: Vec<PrimPtr>,
}

enum NodeProgress {
    None,
    ProcessedLeft { my_index: usize },
}

impl DepthFirstLinearizedBVH {
    pub fn linearize_bvh(bvh: &BVH2) -> DepthFirstLinearizedBVH {
        let mut stack: Vec<(&BVHNode, AABB, NodeProgress)> = Vec::new();

        let node = bvh.root();
        
        let root_aabb = match node {
            BVHNode::Inner {
                left_aabb,
                right_aabb,
                ..
            } => {
                AABB::surrounding_box(*left_aabb, *right_aabb)
            }
            BVHNode::Leaf { .. } => {
                warn!("BVH root is a leaf node, setting bounds to infinity");

                AABB::infinite()
            }
        };

        stack.push((node, root_aabb, NodeProgress::None));

        let mut bvh_nodes: Vec<DepthFirstLinearizedBVHNode> = Vec::new();
        let mut contiguous_prims: Vec<PrimPtr> = Vec::new();

        while !stack.is_empty() {
            let (node, aabb, progress) = stack.last_mut().unwrap();
            match node {
                BVHNode::Inner {
                    left_aabb,
                    right_aabb,
                    ..
                } => {
                    match progress {
                        NodeProgress::None => {
                            let index = bvh_nodes.len();
                            bvh_nodes.push(DepthFirstLinearizedBVHNode::Internal {
                                bounds: *aabb,
                                right_child_offset: 0, // filled in later
                            });

                            *progress = NodeProgress::ProcessedLeft { my_index: index };

                            let left_child = node.left().unwrap();
                            let left_child_bounds = *left_aabb;
                            stack.push((left_child, left_child_bounds, NodeProgress::None));
                        }
                        NodeProgress::ProcessedLeft { my_index } => {
                            let index = bvh_nodes.len();

                            match &mut bvh_nodes[*my_index] {
                                DepthFirstLinearizedBVHNode::Internal {
                                    right_child_offset,
                                    ..
                                } => *right_child_offset = index as u32,
                                DepthFirstLinearizedBVHNode::Leaf { .. } => unreachable!(),
                            }

                            let right_child = node.right().unwrap();
                            let right_child_bounds = *right_aabb;

                            stack.pop();
                            stack.push((right_child, right_child_bounds, NodeProgress::None));
                        }
                    }
                }
                BVHNode::Leaf {
                    prim_count,
                    prim_indices,
                    geom_indices,
                } => {
                    bvh_nodes.push(DepthFirstLinearizedBVHNode::Leaf {
                        bounds: *aabb,
                        prim_ptr_offset: contiguous_prims.len() as u32,
                        prim_count: NonZero::new(*prim_count)
                            .expect("prim_count is definitely not zero"),
                    });

                    for i in 0..*prim_count {
                        let geom_id = geom_indices[i as usize];
                        let prim_id = prim_indices[i as usize];
                        let prim_ptr = PrimPtr { geom_id, prim_id };
                        contiguous_prims.push(prim_ptr);
                    }

                    stack.pop();
                }
            }
        }

        DepthFirstLinearizedBVH {
            nodes: bvh_nodes,
            prim_ptrs: contiguous_prims,
        }
    }

    pub fn bounds(&self) -> AABB {
        self.nodes[0].bounds()
    }
}
