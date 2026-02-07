//! Most of the scene description conversion is kept within Rust
//! that calls into C++ in order to generate hierarchy of Geometry-AS / Instance-AS for OptiX.
//! We do it this way to avoid having to make the whole scene description #[repr(C)]

use raytracing::{geometry::Shape, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

use crate::optix::{self, OptixAccelerationStructure, Vec3SliceWrap, Vec3uSliceWrap};

// hooks for SBT to be constructed alongside the AS hierarchy. `visit` functions on a node 
// return corresponding SBT offset which will be used when constructing that node's parent.
pub(crate) trait SbtVisitor {
    fn visit_geometry_as(&mut self, shape: &Shape) -> u32;
    fn visit_instance_as(&mut self) -> u32;
}

fn make_leaf_geometry_as(
    ctx: optix::OptixDeviceContext, 
    shape: &Shape
) -> OptixAccelerationStructure {
    match shape {
        Shape::TriangleMesh(mesh) => {
            let vertices: Vec3SliceWrap = mesh.vertices.as_slice().into();

            #[allow(non_snake_case, reason = "match C++ API")]
            let (vertices, verticesLen) = (vertices.as_ptr(), vertices.len());
            
            let tris: Vec3uSliceWrap = mesh.tris.as_slice().into();

            #[allow(non_snake_case, reason = "match C++ API")]
            let (tris, trisLen) = (tris.as_ptr(), tris.len());

            // SAFETY: verticesLen and trisLen are valid lengths for vertices / tris
            unsafe {
                optix::makeMeshAccelerationStructure(
                    ctx, 
                    vertices, 
                    verticesLen, 
                    tris, 
                    trisLen
                )
            }
        },
        Shape::Sphere { center, radius } => {
            let center = optix::Vec3 {
                x: center.x(),
                y: center.y(),
                z: center.z(),
            };

            // SAFETY: makeSphereAccelerationStructure has no real safety requirements
            unsafe { 
                optix::makeSphereAccelerationStructure(ctx, center, *radius) 
            }
        },
    }
}

fn make_instance_as(
    ctx: optix::OptixDeviceContext,
    instances: &[OptixAccelerationStructure],
    transforms: &[optix::Matrix4x4],
    sbt_offsets: &[u32]
) -> OptixAccelerationStructure {
    assert!(instances.len() == transforms.len());
    
    // SAFETY: instances and transforms are valid pointers to arrays of OptixAccelerationStructure and Matrix4x4
    // and instances.len() == transforms.len()
    unsafe {
        optix::makeInstanceAccelerationStructure(
            ctx,
            instances.as_ptr(),
            transforms.as_ptr(),
            sbt_offsets.as_ptr(),
            instances.len()
        )
    }
}

pub(crate) fn prepare_optix_acceleration_structures(
    ctx: optix::OptixDeviceContext,
    scene: &Scene,
    sbt_visitor: &mut dyn SbtVisitor,
) -> OptixAccelerationStructure { 
    fn recursive_helper(
        ctx: optix::OptixDeviceContext,
        scene: &Scene,
        sbt_visitor: &mut dyn SbtVisitor, 
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> OptixAccelerationStructure {

        let descendants = scene.descendants_iter(aggregate_primitive_index);
        let mut descendant_acceleration_structures: Vec<OptixAccelerationStructure> = Vec::with_capacity(descendants.len());
        let mut descendant_transforms: Vec<optix::Matrix4x4> = Vec::with_capacity(descendants.len());
        let mut descendant_sbt_offsets: Vec<u32> = Vec::with_capacity(descendants.len());

        for (primitive_index, child_transform) in descendants {
            descendant_transforms.push(child_transform.forward.into());
            
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    let leaf_gas = make_leaf_geometry_as(ctx, &basic.shape);
                    let leaf_gas_sbt_offset = sbt_visitor.visit_geometry_as(&basic.shape);
                    descendant_acceleration_structures.push(leaf_gas);
                    descendant_sbt_offsets.push(leaf_gas_sbt_offset);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().expect("this should be an aggregate index");
                    let child_ias = recursive_helper(
                        ctx,
                        scene,
                        sbt_visitor,
                        aggregate_index
                    );
                    let child_ias_sbt_offset = sbt_visitor.visit_instance_as();
                    descendant_acceleration_structures.push(child_ias);
                }
                Primitive::Transform(_) => unreachable!("DescendantsIter should flatten transforms")
            }
        }

        make_instance_as(
            ctx, 
            &descendant_acceleration_structures, 
            &descendant_transforms,
            &descendant_sbt_offsets
        )
    }

    recursive_helper(
        ctx,
        scene,
        sbt_visitor,
        scene.root_index()
    )
}