//! Most of the scene description conversion is kept within Rust
//! that calls specific C++ in order to generate hierarchy of Geometry-AS / Instance-AS for OptiX.
//! We do it this way to avoid having to make the whole scene description #[repr(C)]

use raytracing::{geometry::{Shape, Transform}, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

use crate::optix::{self, OptixTraversableHandle};

fn make_leaf_geometry_as(
    ctx: optix::OptixDeviceContext, 
    transform: &Transform,
    shape: &Shape
) {
    match shape {
        Shape::TriangleMesh(mesh) => {
            let vertices = mesh.vertices.as_ptr();
            let verticesLen = mesh.vertices.len();

            let tris = mesh.tris.as_ptr();
            let trisLen = mesh.tris.len();

            // ensure that vertices / tris are in correct format for zero-copy transfer to C++
            const {
                assert!(std::mem::size_of::<raytracing::geometry::Vec3>() == 3 * std::mem::size_of::<f32>());
                assert!(std::mem::align_of::<raytracing::geometry::Vec3>() == std::mem::align_of::<f32>());
                assert!(std::mem::size_of::<raytracing::geometry::Vec3u>() == 3 * std::mem::size_of::<u32>());
                assert!(std::mem::align_of::<raytracing::geometry::Vec3u>() == std::mem::align_of::<u32>());
            }

            let vertices = vertices as *const f32;
            let tris = tris as *const u32;        
            let transform = transform.forward.data.as_ptr() as *const f32;

            // SAFETY: verticesLen and trisLen are valid lengths for vertices / tris, 
            // and transform is valid pointer to 4x4 row-major matrix of floats
            let mesh_as = unsafe {
                optix::makeMeshAccelerationStructure(
                    ctx, 
                    vertices, 
                    verticesLen, 
                    tris, 
                    trisLen, 
                    transform
                )
            };
        },
        Shape::Sphere { center, radius } => {
            let center = optix::Vec3 {
                x: center.x(),
                y: center.y(),
                z: center.z(),
            };

            // SAFETY: makeSphereAccelerationStructure has no real safety requirements
            let sphere_as = unsafe { 
                optix::makeSphereAccelerationStructure(ctx, center, *radius) 
            };
        },
    }
    todo!()
}


pub(crate) fn prepare_optix_acceleration_structures(
    ctx: optix::OptixDeviceContext,
    scene: &Scene
) {
    let mut bvhs: Vec<OptixTraversableHandle> = Vec::new();
    let mut bvh_transforms: Vec<Transform> = Vec::new();
    fn recursive_helper(
        ctx: optix::OptixDeviceContext,
        transform: Transform,
        bvhs: &mut Vec<OptixTraversableHandle>,
        bvh_transforms: &mut Vec<Transform>,
        scene: &Scene, 
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> usize {
        let bvh_index = bvhs.len();
        
        let descendants = scene.descendants_iter(aggregate_primitive_index);
        for (geom_index, (primitive_index, child_transform)) in descendants.enumerate() {
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    // bvh_builder.add_shape(&basic.shape, &child_transform, geom_index as u32);
                    
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().unwrap();
                    let sub_bvh_index = recursive_helper(
                        ctx,
                        child_transform.compose(transform.clone()),
                        bvhs, 
                        bvh_transforms,
                        scene, 
                        aggregate_index
                    );

                    // let bounds = bvhs[sub_bvh_index].bounds();
                    // bvh_builder.add_sub_bvh(bounds, &child_transform, geom_index as u32, sub_bvh_index as u32);
                }
                Primitive::Transform(_) => unreachable!("descendants iterator should flatten transforms")
            }
        }

        bvhs.push(todo!());
        bvh_transforms.push(transform.clone());

        bvh_index
    }

    recursive_helper(
        ctx,
        Transform::identity(), 
        &mut bvhs, 
        &mut bvh_transforms, 
        scene, 
        scene.root_index()
    );

    todo!()
}