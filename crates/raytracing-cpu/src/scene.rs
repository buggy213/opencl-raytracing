use raytracing::{accel::bvh2::{BVH2Builder, DepthFirstLinearizedBVH}, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

// CPU-backend specific acceleration structures
pub(crate) struct CPUAccelerationStructures {
    pub(crate) bvhs: Vec<DepthFirstLinearizedBVH>
}

// to prepare a scene for CPU render, must turn all AggregatePrimitives into BVHs using Embree
// this is done recursively to handle multi-level BVH
pub(crate) fn prepare_cpu_scene(scene: &Scene) -> CPUAccelerationStructures {
    let embree = embree4::Device::new();

    let mut bvhs: Vec<DepthFirstLinearizedBVH> = Vec::new();
    fn recursive_helper(
        embree: &embree4::Device,
        bvhs: &mut Vec<DepthFirstLinearizedBVH>, 
        scene: &Scene, 
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> usize {
        let mut bvh_builder = BVH2Builder::new();
        let descendants = scene.descendants_iter(aggregate_primitive_index);
        for (geom_index, (primitive_index, transform)) in descendants.enumerate() {
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    bvh_builder.add_shape(&basic.shape, &transform, geom_index as u32);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().unwrap();
                    let sub_bvh_index = recursive_helper(embree, bvhs, scene, aggregate_index);
                    let bounds = bvhs[sub_bvh_index].bounds();
                    bvh_builder.add_sub_bvh(bounds, &transform, geom_index as u32, sub_bvh_index as u32);
                }
                Primitive::Transform(_) => unreachable!("descendants iterator should flatten transforms")
            }
        }

        let bvh = bvh_builder.build(embree);
        let bvh_index = bvhs.len();
        let bvh = DepthFirstLinearizedBVH::linearize_bvh(&bvh);
        bvhs.push(bvh);

        bvh_index
    }

    recursive_helper(&embree, &mut bvhs, scene, scene.root_index());

    CPUAccelerationStructures {
        bvhs 
    }
}

impl CPUAccelerationStructures {
    pub(crate) fn root_bvh_index(&self) -> usize {
        // construction algorithm pushes root bvh last
        self.bvhs.len() - 1
    }
}