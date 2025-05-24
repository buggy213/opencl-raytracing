use std::{collections::VecDeque, ffi::c_void, mem::{align_of, size_of}, ptr::{null, null_mut}, slice::from_raw_parts};
use embree4::{bvh::{BVHBuildArguments, BVHCallbacks, BuiltBVH}, Bounds, BuildPrimitive, Device, BVH};

use crate::{geometry::{Mesh, Vec3, Vec3u, AABB}, macros::{variadic_max_comparator, variadic_min_comparator}};

#[derive(Debug)]
pub enum BVHNode {
    Inner {
        left: *const BVHNode,
        right: *const BVHNode,
        left_aabb: AABB,
        right_aabb: AABB
    },
    Leaf { // 8 triangles in a leaf at most
        tri_count: u32, 
        tri_indices: [u32; 8],
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

pub struct BVH2 {
    handle: BuiltBVH<BVHNode>, 
}

impl BVH2 {
    fn create_node_bvh2(child_count: u32) -> BVHNode {
        assert!(child_count == 2);
        BVHNode::Inner { 
            left: null(), 
            right: null(), 
            left_aabb: AABB::default(),
            right_aabb: AABB::default()
        }
    }

    fn create_leaf_bvh2(primitives: &[BuildPrimitive]) -> BVHNode {
        let indices = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| 
            primitives.get(i).map(|p| p.primID).unwrap_or_default()
        );

        let geom_indices = [0, 1, 2, 3, 4, 5, 6, 7].map(|i|
            primitives.get(i).map(|p| p.geomID).unwrap_or_default()
        );

        BVHNode::Leaf { 
            tri_count: primitives.len() as u32, 
            tri_indices: indices,
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

impl BVH2 {
    fn primitive_from_triangle(p0: Vec3, p1: Vec3, p2: Vec3, tri_index: u32, geom_index: u32) -> BuildPrimitive {
        let min = variadic_min_comparator!(Vec3::elementwise_min, p0, p1, p2);
        let max = variadic_max_comparator!(Vec3::elementwise_max, p0, p1, p2);
        BuildPrimitive { 
            lower_x: min.0, 
            lower_y: min.1, 
            lower_z: min.2, 
            geomID: geom_index, 
            upper_x: max.0, 
            upper_y: max.1, 
            upper_z: max.2, 
            primID: tri_index 
        }
    }

    fn primitives_from_meshes(meshes: &[Mesh]) -> Vec<BuildPrimitive> {
        let mut primitives = Vec::new();
        for (i, mesh) in meshes.iter().enumerate() {
            for (j, tri) in mesh.tris.iter().cloned().enumerate() {
                let (t0, t1, t2) = (tri.0, tri.1, tri.2);
                let p0 = mesh.vertices[t0 as usize];
                let p1 = mesh.vertices[t1 as usize];
                let p2 = mesh.vertices[t2 as usize];
                primitives.push(BVH2::primitive_from_triangle(p0, p1, p2, j as u32, i as u32));
            }
        }

        primitives
    }

    fn sanity_check(root_node: *const BVHNode, top: bool) -> (i32, i32, i32, i32) {
        let node = unsafe { root_node.read() };
        let inner_nodes;
        let leaf_nodes;
        let tris;
        let max_depth;
        match node {
            BVHNode::Inner { left, right, left_aabb, right_aabb } =>  {
                let (left_inner_nodes, left_leaf_nodes, left_tris, left_max_depth) = BVH2::sanity_check(left, false);
                let (right_inner_nodes, right_leaf_nodes, right_tris, right_max_depth) = BVH2::sanity_check(right, false);
                inner_nodes = 1 + left_inner_nodes + right_inner_nodes;
                leaf_nodes = left_leaf_nodes + right_leaf_nodes;
                tris = left_tris + right_tris;
                max_depth = i32::max(left_max_depth, right_max_depth) + 1;
                eprintln!("height={}, left_aabb={:?}, right_aabb={:?}", max_depth, left_aabb, right_aabb);
            },
            BVHNode::Leaf { tri_count, .. } => {
                inner_nodes = 0;
                leaf_nodes = 1;
                tris = tri_count as i32;
                max_depth = 1;
            },
        }

        if top {
            println!("inner nodes={} leaf nodes={} tris={}, max_depth={}", inner_nodes, leaf_nodes, tris, max_depth);
        }

        return (inner_nodes, leaf_nodes, tris, max_depth);
    }

    pub fn create(device: &Device, meshes: &[Mesh]) -> BVH2 {
        let bvh = BVH::new(device);
        // println!("{}", device.get_error());

        let primitives: Vec<BuildPrimitive> = BVH2::primitives_from_meshes(meshes);
        let bvh_arguments = 
        BVHBuildArguments::new()
            .max_leaf_size(8)
            .register_callbacks(&BVH2::BVH2_CALLBACKS)
            .set_primitives(primitives.as_slice());
        
        let build_result = bvh.build(bvh_arguments);
        // BVH2::sanity_check(build_result, true);
        BVH2 { 
            handle: build_result
        }
    }
}

// unsafe accessors
impl BVH2 {
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

#[repr(C)]
#[derive(Debug)]
pub struct LinearizedBVHNode {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    pub left_first: u32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    pub tri_count: u32
}

#[derive(Debug, Clone, Copy)]
pub struct TriPtr {
    pub geom_id: u32,
    pub verts: Vec3u
}

impl LinearizedBVHNode {
    pub fn linearize_bvh_mesh(bvh: &BVH2, meshes: &[Mesh]) -> (Vec<TriPtr>, Vec<LinearizedBVHNode>) {
        // perform BFS over BVH, make triangles pointed to by leaf nodes contiguous
        let pessimal_capacity = meshes.iter().map(|mesh| mesh.tris.len() / 8).sum();
        let mut queue: VecDeque<(&BVHNode, AABB)> = VecDeque::with_capacity(pessimal_capacity);
        let node = bvh.root();
        let root_aabb;

        match node {
            BVHNode::Inner { left_aabb, right_aabb, .. } => {
                root_aabb = AABB::surrounding_box(*left_aabb, *right_aabb);
            },
            BVHNode::Leaf { tri_count, tri_indices, geom_indices } => {
                unimplemented!("BVH root is a leaf node");
            }
        }

        let mut bvh_nodes: Vec<LinearizedBVHNode> = Vec::new();
        let mut contiguous_tris: Vec<TriPtr> = Vec::new();
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
                        tri_count: 0
                    });
                },
                BVHNode::Leaf { tri_count, tri_indices, geom_indices } => {
                    bvh_nodes.push(LinearizedBVHNode { 
                        min_x: aabb.minimum.0, 
                        min_y: aabb.minimum.1, 
                        min_z: aabb.minimum.2, 
                        left_first: contiguous_tris.len() as u32, 
                        max_x: aabb.maximum.0, 
                        max_y: aabb.maximum.1, 
                        max_z: aabb.maximum.2, 
                        tri_count: *tri_count
                    });

                    
                    for i in 0..*tri_count {
                        let geom_id = geom_indices[i as usize] as usize;
                        let mesh = &meshes[geom_id];
                        let mesh_tri = mesh.tris[tri_indices[i as usize] as usize];
                        let tri_ptr = TriPtr { 
                            geom_id: geom_id as u32, 
                            verts: mesh_tri 
                        };
                        contiguous_tris.push(tri_ptr);
                    }
                },
            }
        }
        
        (contiguous_tris, bvh_nodes)
    }

    pub fn aabb(&self) -> AABB {
        AABB { minimum: Vec3(self.min_x, self.min_y, self.min_z), maximum: Vec3(self.max_x, self.max_y, self.max_z) }
    }
}