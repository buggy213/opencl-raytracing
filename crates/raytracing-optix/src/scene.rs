//! Most of the scene description conversion is kept within Rust
//! that calls specific C++ in order to generate hierarchy of Geometry-AS / Instance-AS for OptiX.
//! We do it this way to avoid having to make the whole scene description #[repr(C)]

use std::marker::PhantomPinned;

use raytracing::{geometry::Transform, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

#[repr(C)]
struct OptixAccelerationStructure {
    _data: (),
    _marker: core::marker::PhantomData<(*mut u8, PhantomPinned)>
}

struct OptixAccelerationStructures {
    bvhs: Vec<*const OptixAccelerationStructure>
}

pub(crate) fn prepare_optix_acceleration_structures(scene: &Scene) -> OptixAccelerationStructures {
    let mut bvhs: Vec<*const OptixAccelerationStructure> = Vec::new();
    let mut bvh_transforms: Vec<Transform> = Vec::new();
    fn recursive_helper(
        transform: Transform,
        bvhs: &mut Vec<*const OptixAccelerationStructure>,
        bvh_transforms: &mut Vec<Transform>,
        scene: &Scene, 
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> usize {
        let bvh_index = bvhs.len();
        bvhs.push(std::ptr::null());
        bvh_transforms.push(transform.clone());

        let descendants = scene.descendants_iter(aggregate_primitive_index);
        for (geom_index, (primitive_index, child_transform)) in descendants.enumerate() {
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    // bvh_builder.add_shape(&basic.shape, &child_transform, geom_index as u32);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().unwrap();
                    let sub_bvh_index = recursive_helper(
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

        bvh_index
    }

    recursive_helper( 
        Transform::identity(), 
        &mut bvhs, 
        &mut bvh_transforms, 
        scene, 
        scene.root_index()
    );

    todo!()
}