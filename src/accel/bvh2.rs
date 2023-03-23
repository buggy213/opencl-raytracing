use std::{mem::{size_of, align_of}, ptr::null_mut, ffi::c_void, slice::from_raw_parts};

use embree4_sys::{bvh::BVHCallbacks, rtcThreadLocalAlloc};

use crate::geometry::Vec3;

#[repr(C)]
pub enum BVHNode {
    Inner {
        left: *mut BVHNode,
        right: *mut BVHNode,
        left_min: Vec3,
        left_max: Vec3,
        right_min: Vec3,
        right_max: Vec3
    },
    Leaf {
        tri_indices: [usize; 3]
    }
}

struct BVH2Functions;
impl BVHCallbacks for BVH2Functions {
    unsafe extern "C" fn unsafe_create_node(
        allocator: embree4_sys::RTCThreadLocalAllocator,
        _childCount: std::ffi::c_uint,
        _userPtr: *mut std::ffi::c_void
    ) -> *mut std::ffi::c_void {
        let new_node = rtcThreadLocalAlloc(
            allocator, 
            size_of::<BVHNode>(), 
        align_of::<BVHNode>()
        ) as *mut BVHNode;

        new_node.write(BVHNode::Inner { 
            left: null_mut(), 
            right: null_mut(), 
            left_min: Vec3::default(), 
            left_max: Vec3::default(), 
            right_min: Vec3::default(), 
            right_max: Vec3::default() 
        });

        return new_node as *mut c_void
    }

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


        return new_leaf as *mut c_void;
    }

    unsafe extern "C" fn unsafe_set_node_bounds(
        nodePtr: *mut std::ffi::c_void,
        bounds: *mut *const embree4_sys::RTCBounds,
        childCount: std::ffi::c_uint,
        _userPtr: *mut std::ffi::c_void
    ) {
        let node = (nodePtr as *mut BVHNode).as_mut().unwrap_unchecked();
        let bounds = from_raw_parts(bounds, childCount as usize);
        if let BVHNode::Inner { left_min, left_max, right_min, right_max, .. } = node {
            *left_min = Vec3((*bounds[0]).lower_x, (*bounds[0]).lower_y, (*bounds[0]).lower_z);
            *left_max = Vec3((*bounds[0]).upper_x, (*bounds[0]).upper_y, (*bounds[0]).upper_z);
            *right_min = Vec3((*bounds[1]).lower_x, (*bounds[1]).lower_y, (*bounds[1]).lower_z);
            *right_max = Vec3((*bounds[1]).upper_x, (*bounds[1]).upper_y, (*bounds[1]).upper_z);
        }
    }

    unsafe extern "C" fn unsafe_set_node_children(
        nodePtr: *mut std::ffi::c_void,
        children: *mut *mut std::ffi::c_void,
        childCount: std::ffi::c_uint,
        _userPtr: *mut std::ffi::c_void
    ) {
        let node = (nodePtr as *mut BVHNode).as_mut().unwrap_unchecked();
        let children = from_raw_parts(children as *mut *mut BVHNode, childCount as usize);
        if let BVHNode::Inner { left, right, .. } = node {
            *left = children[0];
            *right = children[1];
        }
    }
}