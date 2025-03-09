use std::{mem::{size_of, align_of}, ptr::null_mut, ffi::c_void, slice::from_raw_parts};

use embree4_sys::{bvh::{BVHCallbacks, BVHProgressCallback}, rtcThreadLocalAlloc, BVH, RTCBounds, Device, bvh::BVHBuildArguments, RTCBuildQuality, RTCBuildPrimitive};

use crate::{geometry::{Vec3, AABB, Mesh}, macros::{variadic_min_comparator, variadic_max_comparator}};

#[repr(C)]
pub enum BVHNode {
    Inner {
        left: *mut BVHNode,
        right: *mut BVHNode,
        left_aabb: AABB,
        right_aabb: AABB
    },
    Leaf { // 8 triangles in a leaf at most
        tri_count: u32, 
        tri_indices: [u32; 8]
    }
}

impl From<RTCBounds> for AABB {
    fn from(value: RTCBounds) -> Self {
        AABB {
            minimum: Vec3(value.lower_x, value.lower_y, value.lower_z),
            maximum: Vec3(value.upper_x, value.upper_y, value.upper_z)
        }
    }
}

struct BVH2Functions;
impl BVHCallbacks for BVH2Functions {
    #[allow(non_snake_case, unused_variables)]
    unsafe extern "C" fn unsafe_create_node(
        allocator: embree4_sys::RTCThreadLocalAllocator,
        childCount: std::ffi::c_uint,
        userPtr: *mut std::ffi::c_void
    ) -> *mut std::ffi::c_void {
        let new_node = rtcThreadLocalAlloc(
            allocator, 
            size_of::<BVHNode>(), 
        align_of::<BVHNode>()
        ) as *mut BVHNode;

        new_node.write(BVHNode::Inner { 
            left: null_mut(), 
            right: null_mut(), 
            left_aabb: AABB::default(),
            right_aabb: AABB::default()
        });

        return new_node as *mut c_void
    }

    #[allow(non_snake_case, unused_variables)]
    unsafe extern "C" fn unsafe_create_leaf(
        allocator: embree4_sys::RTCThreadLocalAllocator,
        primitives: *const embree4_sys::RTCBuildPrimitive,
        primitiveCount: usize,
        userPtr: *mut std::ffi::c_void
    ) -> *mut std::ffi::c_void {
        let new_leaf = rtcThreadLocalAlloc(
            allocator, 
            size_of::<BVHNode>(), 
            align_of::<BVHNode>()
        ) as *mut BVHNode;
        
        let mut tris: [u32; 8] = [0; 8];
        let primitives = from_raw_parts(primitives, primitiveCount);
        for (i, primitive) in primitives.iter().enumerate() {
            tris[i] = primitive.primID;
        }

        new_leaf.write(BVHNode::Leaf { 
            tri_count: primitiveCount as u32, 
            tri_indices: tris
        });

        return new_leaf as *mut c_void;
    }

    #[allow(non_snake_case, unused_variables)]
    unsafe extern "C" fn unsafe_set_node_bounds(
        nodePtr: *mut std::ffi::c_void,
        bounds: *mut *const embree4_sys::RTCBounds,
        childCount: std::ffi::c_uint,
        userPtr: *mut std::ffi::c_void
    ) {
        let node = (nodePtr as *mut BVHNode).as_mut().unwrap_unchecked();
        let bounds = from_raw_parts(bounds, childCount as usize);
        if let BVHNode::Inner { left_aabb, right_aabb, .. } = node {
            *left_aabb = AABB::from(*bounds[0]);
            *right_aabb = AABB::from(*bounds[1]);
        }
    }

    #[allow(non_snake_case, unused_variables)]
    unsafe extern "C" fn unsafe_set_node_children(
        nodePtr: *mut std::ffi::c_void,
        children: *mut *mut std::ffi::c_void,
        childCount: std::ffi::c_uint,
        userPtr: *mut std::ffi::c_void
    ) {
        let node = (nodePtr as *mut BVHNode).as_mut().unwrap_unchecked();
        let children = from_raw_parts(children as *mut *mut BVHNode, childCount as usize);
        if let BVHNode::Inner { left, right, .. } = node {
            *left = children[0];
            *right = children[1];
        }
    }
}

impl BVHProgressCallback for BVH2Functions {
    fn progress(n: f64) -> bool {
        println!("BVH constructing {:.3}", n);
        true
    }
}

pub struct BVH2 {
    _handle: BVH, // needed for RAII reasons
    root: *mut BVHNode // drop is handled by dropping handle, which will deallocate things
}

impl BVH2 {
    fn primitive_from_triangle(p0: Vec3, p1: Vec3, p2: Vec3, tri_index: u32) -> RTCBuildPrimitive {
        let min = variadic_min_comparator!(Vec3::elementwise_min, p0, p1, p2);
        let max = variadic_max_comparator!(Vec3::elementwise_max, p0, p1, p2);
        RTCBuildPrimitive { 
            lower_x: min.0, 
            lower_y: min.1, 
            lower_z: min.2, 
            geomID: 0, 
            upper_x: max.0, 
            upper_y: max.1, 
            upper_z: max.2, 
            primID: tri_index 
        }
    }

    fn primitives_from_mesh(mesh: &Mesh) -> Vec<RTCBuildPrimitive> {
        let mut primitives = Vec::new();
        for (i, tri) in mesh.tris.iter().cloned().enumerate() {
            let (t0, t1, t2) = (tri.0, tri.1, tri.2);
            let p0 = mesh.vertices[t0 as usize];
            let p1 = mesh.vertices[t1 as usize];
            let p2 = mesh.vertices[t2 as usize];
            primitives.push(BVH2::primitive_from_triangle(p0, p1, p2, i as u32));
        }

        primitives
    }

    fn sanity_check(root_node: *mut BVHNode, top: bool) -> (i32, i32, i32, i32) {
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
            BVHNode::Leaf { tri_count, tri_indices } => {
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

    pub fn create(device: &Device, mesh: &Mesh) -> BVH2 {
        let bvh = BVH::new(device);
        // println!("{}", device.get_error());

        let mut primitives = BVH2::primitives_from_mesh(mesh);
        let bvh_arguments = 
            BVHBuildArguments::default()
            .set_max_leaf_size(8)
            .quality(RTCBuildQuality::RTC_BUILD_QUALITY_MEDIUM)
            .register_callbacks::<BVH2Functions>()
            .register_progress_callback::<BVH2Functions>()
            .set_primitives(&mut primitives);
        let build_result = bvh.build(bvh_arguments).expect("failed to build bvh") as *mut BVHNode;
        // BVH2::sanity_check(build_result, true);
        BVH2 { _handle: bvh, root: build_result }
    }
}

