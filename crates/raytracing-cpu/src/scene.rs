use raytracing::{accel::bvh2::{BVH2Builder, LinearizedBVH}, scene::{AggregatePrimitive, Primitive, Scene}};

// CPU-backend specific acceleration structures
struct CPUAccelerationStructures {
    root_bvh: LinearizedBVH,
    bvhs: Vec<LinearizedBVH>
}

// to prepare a scene for CPU render, must turn all AggregatePrimitives into BVHs using Embree
// this is done recursively to handle multi-level BVH
fn prepare_cpu_scene(scene: &Scene) -> CPUAccelerationStructures {
    let embree = embree4::Device::new();

    let mut bvh_builder = BVH2Builder::new();
    let descendants = scene.descendants_iter(scene.root());

    let mut bvhs: Vec<LinearizedBVH> = Vec::new();
    
    fn recursive_helper(
        embree: &embree4::Device,
        bvhs: &mut Vec<LinearizedBVH>, 
        scene: &Scene, 
        aggregate_primitive: &AggregatePrimitive
    ) -> usize {
        let mut bvh_builder = BVH2Builder::new();
        let descendants = scene.descendants_iter(aggregate_primitive);
        for (geom_index, (primitive, transform)) in descendants.enumerate() {
            match primitive {
                Primitive::Basic(basic) => {
                    bvh_builder.add_shape(&basic.shape, &transform, geom_index as u32);
                }
                Primitive::Aggregate(aggregate) => {
                    let sub_bvh_index = recursive_helper(embree, bvhs, scene, aggregate);
                    let bounds = bvhs[sub_bvh_index].bounds();
                    bvh_builder.add_sub_bvh(bounds, &transform, geom_index as u32, sub_bvh_index as u32);
                }
                Primitive::Transform(_) => unreachable!("descendants iterator should flatten transforms")
            }
        }

        let bvh = bvh_builder.build(embree);
        let bvh_index = bvhs.len();
        let bvh = LinearizedBVH::linearize_bvh(&bvh);
        bvhs.push(bvh);

        bvh_index
    }

    for (geom_index, (primitive, transform)) in descendants.enumerate() {
        match primitive {
            Primitive::Basic(basic) => {
                bvh_builder.add_shape(&basic.shape, &transform, geom_index as u32);
            }
            Primitive::Aggregate(aggregate) => {
                let sub_bvh_index = recursive_helper(&embree, &mut bvhs, scene, aggregate);
                let bounds = bvhs[sub_bvh_index].bounds();
                bvh_builder.add_sub_bvh(bounds, &transform, geom_index as u32, sub_bvh_index as u32);
            }
            Primitive::Transform(_) => unreachable!("descendants iterator should flatten transforms")
        }
    }


    let root_bvh = bvh_builder.build(&embree);
    let root_bvh = LinearizedBVH::linearize_bvh(&root_bvh);

    CPUAccelerationStructures { 
        root_bvh, 
        bvhs 
    }
}