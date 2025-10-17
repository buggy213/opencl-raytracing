use std::{collections::VecDeque, ptr::null};
use embree4::{bvh::{BVHBuildArguments, BVHCallbacks, BuiltBVH}, Bounds, BuildPrimitive, Device, BVH};

use crate::{geometry::{Mesh, Shape, Vec3, AABB}, macros::{variadic_max_comparator, variadic_min_comparator}};

#[derive(Debug)]
pub enum BVHNode {
    // we keep pointers instead of references in internal BVHNode to avoid lifetime 
    // complications; requires a small amount of unsafe
    Inner { 
        left: *const BVHNode,
        right: *const BVHNode,
        left_aabb: AABB,
        right_aabb: AABB
    },
    Leaf {
        // 8 bvh primitives in a leaf at most
        prim_count: u32, 
        prim_indices: [u32; 8],
        geom_indices: [u32; 8]
    }
}

impl From<Bounds> for AABB {
    fn from(value: Bounds) -> Self {
        AABB {
            minimum: Vec3(value.lower_x, value.lower_y, value.lower_z),
            maximum: Vec3(value.upper_x, value.upper_y, value.upper_z)
        }
    }
}


impl Default for BVHNode {
    fn default() -> Self {
        BVHNode::Inner { left: null(), right: null(), left_aabb: AABB::default(), right_aabb: AABB::default() }
    }
}

pub struct BVH2<'device> {
    handle: BuiltBVH<'device, BVHNode>, 
}

impl BVH2<'_> {
    fn create_node_bvh2(child_count: u32) -> BVHNode {
        assert!(child_count == 2, "Embree tried to create a BVH2 node with >2 children");
        BVHNode::Inner { 
            left: null(), 
            right: null(), 
            left_aabb: AABB::default(),
            right_aabb: AABB::default()
        }
    }

    fn create_leaf_bvh2(primitives: &[BuildPrimitive]) -> BVHNode {
        let prim_indices = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| 
            primitives.get(i).map(|p| p.primID).unwrap_or_default()
        );

        let geom_indices = [0, 1, 2, 3, 4, 5, 6, 7].map(|i|
            primitives.get(i).map(|p| p.geomID).unwrap_or_default()
        );

        BVHNode::Leaf { 
            prim_count: primitives.len() as u32, 
            prim_indices,
            geom_indices
        }
    }

    fn set_node_bounds_bvh2(node: &mut BVHNode, bounds: &[&Bounds]) {
        assert!(bounds.len() == 2);
        match node {
            BVHNode::Inner { left_aabb, right_aabb, .. } => {
                *left_aabb = (*bounds[0]).into();
                *right_aabb = (*bounds[1]).into();
            },
            _ => unreachable!("Embree should not confuse leaf and inner nodes"),
        }
    }

    fn set_node_children_bvh2(node: &mut BVHNode, children: &[&BVHNode]) {
        assert!(children.len() == 2);
        match node {
            BVHNode::Inner { left, right, .. } => {
                *left = children[0] as *const BVHNode;
                *right = children[1] as *const BVHNode;
            },
            _ => unreachable!("Embree should not confuse leaf and inner nodes")
        }
    }

    const BVH2_CALLBACKS: BVHCallbacks<BVHNode> = BVHCallbacks::new(
        BVH2::create_node_bvh2,
        BVH2::create_leaf_bvh2,
        BVH2::set_node_bounds_bvh2,
        BVH2::set_node_children_bvh2
    );
}

impl<'device> BVH2<'device> {
    fn create(device: &'device Device, primitives: Vec<BuildPrimitive>) -> BVH2<'device> {
        let bvh = BVH::new(device);
        
        let bvh_arguments = 
        BVHBuildArguments::new()
            .max_leaf_size(8)
            .register_callbacks(&BVH2::BVH2_CALLBACKS)
            .set_primitives(primitives.as_slice());
        
        let build_result = bvh.build(bvh_arguments);
        
        BVH2 { 
            handle: build_result
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


struct BVH2Builder {
    primitives: Vec<BuildPrimitive>
}

impl BVH2Builder {
    pub fn new() -> Self {
        Self { primitives: Vec::new() }
    }

    pub fn add_shape(&mut self, shape: &Shape, geom_index: u32) {
        match shape {
            Shape::TriangleMesh(mesh) => self.add_mesh(mesh, geom_index),
            Shape::Sphere { center, radius } => self.add_sphere(*center, *radius, geom_index),
        }
    }

    pub fn add_sub_bvh(&mut self, bounds: AABB, geom_index: u32) {
        self.add_bvh(bounds, geom_index);
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
            primID: tri_index 
        };

        self.primitives.push(primitive);
    }

    fn add_sphere(&mut self, center: Vec3, radius: f32, geom_index: u32) {
        let primitive = BuildPrimitive {
            lower_x: center.x() - radius,
            lower_y: center.y() - radius,
            lower_z: center.z() - radius,
            geomID: geom_index,
            upper_x: center.x() + radius,
            upper_y: center.y() + radius,
            upper_z: center.z() + radius,
            primID: 0,
        };

        self.primitives.push(primitive);
    }

    fn add_mesh(&mut self, mesh: &Mesh, geom_index: u32) {
        for (tri_index, tri) in mesh.tris.iter().cloned().enumerate() {
            let (t0, t1, t2) = (tri.0, tri.1, tri.2);
            let p0 = mesh.vertices[t0 as usize];
            let p1 = mesh.vertices[t1 as usize];
            let p2 = mesh.vertices[t2 as usize];
            self.add_tri(p0, p1, p2, tri_index as u32 , geom_index);
        }
    }

    fn add_bvh(&mut self, bounds: AABB, geom_index: u32) {
        let primitive = BuildPrimitive {
            lower_x: bounds.minimum.x(),
            lower_y: bounds.minimum.y(),
            lower_z: bounds.minimum.z(),
            geomID: geom_index,
            upper_x: bounds.maximum.x(),
            upper_y: bounds.maximum.y(),
            upper_z: bounds.maximum.z(),
            primID: 0
        };

        self.primitives.push(primitive);
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct LinearizedBVHNode {
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
    pub prim_count: u32
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrimPtr {
    pub geom_id: u32,
    pub prim_id: u32,
}

pub struct LinearizedBVH {
    pub nodes: Vec<LinearizedBVHNode>,
    pub prim_ptrs: Vec<PrimPtr>,
}

impl LinearizedBVHNode {
    pub fn linearize_bvh_mesh(bvh: &BVH2) -> LinearizedBVH {
        // perform BFS over BVH, make primitives pointed to by leaf nodes contiguous
        let mut queue: VecDeque<(&BVHNode, AABB)> = VecDeque::new();
        let node = bvh.root();
        let root_aabb;

        match node {
            BVHNode::Inner { left_aabb, right_aabb, .. } => {
                root_aabb = AABB::surrounding_box(*left_aabb, *right_aabb);
            },
            BVHNode::Leaf { .. } => {
                unimplemented!("BVH root is a leaf node");
            }
        }

        let mut bvh_nodes: Vec<LinearizedBVHNode> = Vec::new();
        let mut contiguous_prims: Vec<PrimPtr> = Vec::new();
        queue.push_back((node, root_aabb));
        while !queue.is_empty() {
            let (node, aabb) = queue.pop_front().unwrap();
            match node {
                BVHNode::Inner { left_aabb, right_aabb, .. } => {
                    let left_idx = bvh_nodes.len() + queue.len() + 1;
                    queue.push_back((node.left().unwrap(), *left_aabb));
                    queue.push_back((node.right().unwrap(), *right_aabb));
                    bvh_nodes.push(LinearizedBVHNode { 
                        min_x: aabb.minimum.0, 
                        min_y: aabb.minimum.1, 
                        min_z: aabb.minimum.2, 
                        left_first: left_idx as u32, 
                        max_x: aabb.maximum.0, 
                        max_y: aabb.maximum.1, 
                        max_z: aabb.maximum.2, 
                        prim_count: 0
                    });
                },
                BVHNode::Leaf { prim_count, prim_indices, geom_indices } => {
                    bvh_nodes.push(LinearizedBVHNode { 
                        min_x: aabb.minimum.0, 
                        min_y: aabb.minimum.1, 
                        min_z: aabb.minimum.2, 
                        left_first: contiguous_prims.len() as u32, 
                        max_x: aabb.maximum.0, 
                        max_y: aabb.maximum.1, 
                        max_z: aabb.maximum.2, 
                        prim_count: *prim_count
                    });

                    
                    for i in 0..*prim_count {
                        let geom_id = geom_indices[i as usize];
                        let prim_id = prim_indices[i as usize];
                        let prim_ptr = PrimPtr { 
                            geom_id, 
                            prim_id,
                        };
                        contiguous_prims.push(prim_ptr);
                    }
                },
            }
        }
        
        LinearizedBVH {
            nodes: bvh_nodes,
            prim_ptrs: contiguous_prims,
        }
    }

    pub fn aabb(&self) -> AABB {
        AABB { minimum: Vec3(self.min_x, self.min_y, self.min_z), maximum: Vec3(self.max_x, self.max_y, self.max_z) }
    }
}