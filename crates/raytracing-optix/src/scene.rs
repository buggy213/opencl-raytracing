//! Most of the scene description conversion is kept within Rust
//! that calls specific C++ in order to generate hierarchy of Geometry-AS / Instance-AS for OptiX.
//! We do it this way to avoid having to make the whole scene description #[repr(C)]

use raytracing::{geometry::Shape, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

use crate::optix::{self, OptixAccelerationStructure};

fn make_leaf_geometry_as(
    ctx: optix::OptixDeviceContext, 
    shape: &Shape
) -> OptixAccelerationStructure {
    match shape {
        Shape::TriangleMesh(mesh) => {
            let vertices = mesh.vertices.as_ptr() as *const optix::Vec3;
            #[allow(non_snake_case, reason = "match C++ API")]
            let verticesLen = mesh.vertices.len();

            let tris = mesh.tris.as_ptr() as *const optix::Vec3u;
            #[allow(non_snake_case, reason = "match C++ API")]
            let trisLen = mesh.tris.len();

            // SAFETY: verticesLen and trisLen are valid lengths for vertices / tris, 
            // and transform is valid pointer to 4x4 row-major matrix of floats
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
) -> OptixAccelerationStructure {
    assert!(instances.len() == transforms.len());
    
    // SAFETY: instances and transforms are valid pointers to arrays of OptixAccelerationStructure and Matrix4x4
    // and instances.len() == transforms.len()
    unsafe {
        optix::makeInstanceAccelerationStructure(
            ctx,
            instances.as_ptr(),
            transforms.as_ptr(),
            instances.len()
        )
    }
}


pub(crate) fn prepare_optix_acceleration_structures(
    ctx: optix::OptixDeviceContext,
    scene: &Scene
) -> OptixAccelerationStructure { 
    fn recursive_helper(
        ctx: optix::OptixDeviceContext,
        scene: &Scene, 
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> OptixAccelerationStructure {

        let descendants = scene.descendants_iter(aggregate_primitive_index);
        let mut descendant_acceleration_structures: Vec<OptixAccelerationStructure> = Vec::with_capacity(descendants.len());
        let mut descendant_transforms: Vec<optix::Matrix4x4> = Vec::with_capacity(descendants.len());

        for (primitive_index, child_transform) in descendants {
            descendant_transforms.push(child_transform.forward.into());
            
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    let leaf_gas = make_leaf_geometry_as(ctx, &basic.shape);
                    descendant_acceleration_structures.push(leaf_gas);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().expect("this should be an aggregate index");
                    let child_ias = recursive_helper(
                        ctx,
                        scene,
                        aggregate_index
                    );

                    descendant_acceleration_structures.push(child_ias);
                }
                Primitive::Transform(_) => unreachable!("DescendantsIter should flatten transforms")
            }
        }

        make_instance_as(
            ctx, 
            &descendant_acceleration_structures, 
            &descendant_transforms
        )
    }

    recursive_helper(
        ctx,
        scene,
        scene.root_index()
    )
}