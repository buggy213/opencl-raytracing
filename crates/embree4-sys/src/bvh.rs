use std::{mem::size_of, ptr::null_mut, ffi::{c_void, c_uint}};

use anyhow::anyhow;

use crate::*;

pub struct BVH {
    pub(crate) handle: RTCBVH 
}
pub type BVHBuildArguments = RTCBuildArguments;

pub trait BVHCallbacks {
    unsafe extern "C" fn unsafe_create_node(
        allocator: RTCThreadLocalAllocator,
        childCount: c_uint,
        userPtr: *mut c_void
    ) -> *mut c_void;

    unsafe extern "C" fn unsafe_create_leaf(
        allocator: RTCThreadLocalAllocator,
        primitives: *const RTCBuildPrimitive,
        primitiveCount: usize,
        userPtr: *mut c_void
    ) -> *mut c_void;

    unsafe extern "C" fn unsafe_set_node_bounds(
        nodePtr: *mut c_void,
        bounds: *mut *const RTCBounds,
        childCount: c_uint,
        userPtr: *mut c_void
    );

    unsafe extern "C" fn unsafe_set_node_children(
        nodePtr: *mut c_void,
        children: *mut *mut c_void,
        childCount: c_uint,
        userPtr: *mut c_void
    );
}

pub trait BVHProgressCallback {
    fn progress(n: f64) -> bool;
    unsafe extern "C" fn unsafe_build_progress(
        userPtr: *mut c_void,
        n: f64
    ) -> bool {
        Self::progress(n)
    }
}

impl BVHBuildArguments {
    pub fn quality(mut self, quality: RTCBuildQuality::Type) -> BVHBuildArguments {
        self.buildQuality = quality;
        self
    }

    fn set_bvh(mut self, bvh: &BVH) -> BVHBuildArguments {
        self.bvh = bvh.handle;
        self
    }

    pub fn register_callbacks<T>(
        mut self
    ) -> BVHBuildArguments where T: BVHCallbacks {
        self.createLeaf = Some(T::unsafe_create_leaf);
        self.createNode = Some(T::unsafe_create_node);
        self.setNodeBounds = Some(T::unsafe_set_node_bounds);
        self.setNodeChildren = Some(T::unsafe_set_node_children);
        self
    }

    pub fn set_primitives(mut self, primitives: &mut [RTCBuildPrimitive]) -> BVHBuildArguments {
        self.primitives = primitives.as_mut_ptr();
        self.primitiveCount = primitives.len();
        self.primitiveArrayCapacity = primitives.len();
        self
    }

    fn valid(&self) -> bool {
        self.createLeaf.is_some() && self.createNode.is_some() && 
        self.setNodeChildren.is_some() && self.setNodeBounds.is_some() &&
        !self.primitives.is_null() && !self.bvh.is_null() &&
        self.primitiveCount > 0
    }

    pub fn set_max_leaf_size(mut self, max_leaf_size: u32) -> BVHBuildArguments {
        self.maxLeafSize = max_leaf_size;
        self
    }

    pub fn register_progress_callback<T> (
        mut self
    ) -> BVHBuildArguments where T: BVHProgressCallback {
        self.buildProgress = Some(T::unsafe_build_progress);
        self
    }
}



// rtcDefaultBuildArguments is inlined, so have to reimplement here.
impl Default for BVHBuildArguments {
    fn default() -> BVHBuildArguments {
        BVHBuildArguments { 
            byteSize: size_of::<BVHBuildArguments>(), 
            buildQuality: RTCBuildQuality::RTC_BUILD_QUALITY_MEDIUM, 
            buildFlags: RTCBuildFlags::RTC_BUILD_FLAG_NONE, 
            maxBranchingFactor: 2, 
            maxDepth: 32, 
            sahBlockSize: 1, 
            minLeafSize: 1, 
            maxLeafSize: RTCBuildConstants_RTC_BUILD_MAX_PRIMITIVES_PER_LEAF as u32, 
            traversalCost: 1.0, 
            intersectionCost: 1.0, 
            bvh: null_mut(), 
            primitives: null_mut(), 
            primitiveCount: 0, 
            primitiveArrayCapacity: 0, 
            createNode: None, 
            setNodeChildren: None, 
            setNodeBounds: None, 
            createLeaf: None, 
            splitPrimitive: None, 
            buildProgress: None, 
            userPtr: null_mut() 
        }
    }
}

impl BVH {
    pub fn new(device: &Device) -> BVH {
        BVH {
            handle: unsafe {
                rtcNewBVH(device.handle)
            }
        }
    }

    pub fn build(&self, args: BVHBuildArguments) -> anyhow::Result<*mut c_void> {
        let args = args.set_bvh(self);
        if !args.valid() {
            return Err(anyhow!("invalid bvh construction arguments"))
        }
        unsafe {
            let build_result = rtcBuildBVH(&args);
            return Ok(build_result);
        }
    }
}

impl Drop for BVH {
    fn drop(&mut self) {
        unsafe { 
            rtcReleaseBVH(self.handle);
        }
    }
}