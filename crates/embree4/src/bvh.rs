use std::{ffi::{c_uint, c_void}, fmt::Debug, mem::size_of, ptr::{null_mut, slice_from_raw_parts}};

use embree4_sys::{rtcBuildBVH, rtcNewBVH, rtcReleaseBVH, rtcThreadLocalAlloc, RTCBounds, RTCBuildArguments, RTCBuildConstants_RTC_BUILD_MAX_PRIMITIVES_PER_LEAF, RTCBuildFlags::RTC_BUILD_FLAG_NONE, RTCBuildPrimitive, RTCBuildQuality::RTC_BUILD_QUALITY_MEDIUM, RTCThreadLocalAllocator, RTCBVH};

use crate::*;

pub struct BVH {
    handle: RTCBVH
}

pub struct BuiltBVH<NodeType> {
    _handle: BVH, // kept around for RAII
    root: *const NodeType
}

type CreateNodeFunction<NodeType> = fn(u32) -> NodeType;
type CreateLeafFunction<NodeType> = fn(&[RTCBuildPrimitive]) -> NodeType;
type SetNodeBoundsFunction<NodeType> = fn(&mut NodeType, &[&RTCBounds]);
type SetNodeChildrenFunction<NodeType> = fn(&mut NodeType, &[&NodeType]);
type BuildProgressFunction = fn(f64);

// NodeType shouldn't be a reference
pub struct BVHBuildArguments<NodeType: 'static> {
    primitives: *mut RTCBuildPrimitive,
    primitive_count: Option<usize>,
    primitive_array_capacity: Option<usize>,

    // tree shape
    max_leaf_size: Option<u32>,
    max_branching_factor: Option<u32>,
    
    callbacks: Option<&'static BVHCallbacks<NodeType>>
}

pub struct BVHCallbacks<NodeType> {
    create_node: CreateNodeFunction<NodeType>,
    create_leaf: CreateLeafFunction<NodeType>,
    set_node_bounds: SetNodeBoundsFunction<NodeType>,
    set_node_children: SetNodeChildrenFunction<NodeType>,
}

impl<NodeType> BVHCallbacks<NodeType> {
    pub const fn new(
        create_node: CreateNodeFunction<NodeType>,
        create_leaf: CreateLeafFunction<NodeType>,
        set_node_bounds: SetNodeBoundsFunction<NodeType>,
        set_node_children: SetNodeChildrenFunction<NodeType>,
    ) -> Self {
        Self { create_node, create_leaf, set_node_bounds, set_node_children }
    }
}

extern "C" fn unsafe_create_node<NodeType: Debug>(
    allocator: RTCThreadLocalAllocator,
    childCount: c_uint,
    userPtr: *mut c_void
) -> *mut c_void {
    let (node, callbacks) = unsafe { 
        let node = rtcThreadLocalAlloc(allocator, size_of::<NodeType>(), align_of::<NodeType>());
        
        let node = node as *mut NodeType;
        let callbacks = &*(userPtr as *const BVHCallbacks<NodeType>);  
        
        (node, callbacks)
    };
    
    let new_node = (callbacks.create_node)(childCount);
    
    unsafe {
        node.write(new_node);
    }

    node as *mut NodeType as *mut c_void
}

extern "C" fn unsafe_create_leaf<NodeType: Debug>(
    allocator: RTCThreadLocalAllocator,
    primitives: *const RTCBuildPrimitive,
    primitiveCount: usize,
    userPtr: *mut c_void
) -> *mut c_void {
    let (leaf, primitives, callbacks) = unsafe { 
        let leaf = rtcThreadLocalAlloc(allocator, size_of::<NodeType>(), align_of::<NodeType>());
        
        let leaf = leaf as *mut NodeType;
        let primitives = &*slice_from_raw_parts(primitives, primitiveCount);
        let callbacks = &*(userPtr as *const BVHCallbacks<NodeType>);  
        
        (leaf, primitives, callbacks)
    };
    
    let new_leaf = (callbacks.create_leaf)(primitives);

    unsafe {
        leaf.write(new_leaf);
    }

    leaf as *mut NodeType as *mut c_void
}

extern "C" fn unsafe_set_node_bounds<NodeType: Debug>(
    nodePtr: *mut c_void,
    bounds: *mut *const RTCBounds,
    childCount: c_uint,
    userPtr: *mut c_void
) {
    let (node, bounds, callbacks) = unsafe {
        let node = &mut *(nodePtr as *mut NodeType);
        let bounds = &*slice_from_raw_parts(bounds, childCount as usize);

        // SAFETY: idk lol
        let bounds: &[&RTCBounds] = std::mem::transmute(bounds);
        
        let callbacks = &*(userPtr as *const BVHCallbacks<NodeType>);  

        (node, bounds, callbacks)
    };

    (callbacks.set_node_bounds)(node, bounds);
}

extern "C" fn unsafe_set_node_children<NodeType: Debug>(
    nodePtr: *mut c_void,
    children: *mut *mut c_void,
    childCount: c_uint,
    userPtr: *mut c_void
) {
    let (node, children, callbacks) = unsafe {
        let node = &mut *(nodePtr as *mut NodeType);
        let children = &*slice_from_raw_parts(children, childCount as usize);
        
        // SAFETY: idk lol
        let children: &[&NodeType] = std::mem::transmute(children);

        let callbacks = &*(userPtr as *const BVHCallbacks<NodeType>);

        (node, children, callbacks)
    };

    (callbacks.set_node_children)(node, children);

}


impl<NodeType: Debug> BVHBuildArguments<NodeType> {
    pub fn new() -> Self {
        Self {
            primitives: null_mut(),
            primitive_count: None,
            primitive_array_capacity: None,
            max_leaf_size: None,
            max_branching_factor: None,
            callbacks: None,
        }
    }

    pub fn register_callbacks(
        mut self,
        callbacks: &'static BVHCallbacks<NodeType>,
    ) -> Self {
        self.callbacks = Some(callbacks);
        self
    }

    pub fn max_leaf_size(mut self, max_leaf_size: u32) -> Self {
        self.max_leaf_size = Some(max_leaf_size);
        self
    }
    
    pub fn max_branching_factor(mut self, max_branching_factor: u32) -> Self {
        self.max_branching_factor = Some(max_branching_factor);
        self
    }

    pub fn set_primitives(mut self, primitives: &[BuildPrimitive]) -> BVHBuildArguments<NodeType> {
        // SAFETY: in BUILD_QUALITY_MEDIUM mode, primitives list is not mutated
        self.primitives = primitives.as_ptr() as *mut BuildPrimitive;
        self.primitive_count = Some(primitives.len());
        self.primitive_array_capacity = Some(primitives.len());
        self
    }

    fn build(self, bvh: &BVH) -> RTCBuildArguments {
        match self {
            BVHBuildArguments { 
                primitives, 
                primitive_count: Some(primitive_count), 
                primitive_array_capacity: Some(primitive_array_capacity), 
                max_leaf_size, 
                max_branching_factor, 
                callbacks: Some(callbacks) 
            } if !primitives.is_null() => {
                let rtc_build_args = RTCBuildArguments {
                    byteSize: size_of::<RTCBuildArguments>(),
                    buildQuality: RTC_BUILD_QUALITY_MEDIUM,
                    buildFlags: RTC_BUILD_FLAG_NONE,
                    maxBranchingFactor: max_branching_factor.unwrap_or_else(|| 2),
                    maxDepth: 32,
                    sahBlockSize: 1,
                    minLeafSize: 1,
                    maxLeafSize: max_leaf_size.unwrap_or_else(|| RTCBuildConstants_RTC_BUILD_MAX_PRIMITIVES_PER_LEAF as u32),
                    traversalCost: 1.0,
                    intersectionCost: 1.0,
                    bvh: bvh.handle,
                    primitives,
                    primitiveCount: primitive_count,
                    primitiveArrayCapacity: primitive_array_capacity,
                    createNode: Some(unsafe_create_node::<NodeType>),
                    setNodeChildren: Some(unsafe_set_node_children::<NodeType>),
                    setNodeBounds: Some(unsafe_set_node_bounds::<NodeType>),
                    createLeaf: Some(unsafe_create_leaf::<NodeType>),
                    splitPrimitive: None,
                    buildProgress: None,
                    userPtr: callbacks as *const BVHCallbacks<NodeType> as *mut c_void
                };

                rtc_build_args
            }
            _ => {
                todo!("Internal error: BVH builder missing fields. TODO: return error");
            }
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

    pub fn build<NodeType: Debug>(self, args: BVHBuildArguments<NodeType>) -> BuiltBVH<NodeType> {
        let embree_args = args.build(&self);
        let build_result = unsafe {
            rtcBuildBVH(&embree_args)
        };
        
        BuiltBVH { _handle: self, root: build_result as *const NodeType }
    }
}

impl<NodeType> BuiltBVH<NodeType> {
    pub fn root(&self) -> &NodeType {
        unsafe { &*(self.root) }
    } 
}

impl Drop for BVH {
    fn drop(&mut self) {
        eprintln!("Dropping BVH");
        unsafe { 
            rtcReleaseBVH(self.handle);
        }
    }
}