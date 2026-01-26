use raytracing::{accel::bvh2::{BVH2Builder, DepthFirstLinearizedBVH}, geometry::Transform, scene::{AggregatePrimitiveIndex, Primitive, Scene}};

// CPU-backend specific acceleration structures
#[derive(Debug)]
pub(crate) struct CpuAccelerationStructures {
    pub(crate) bvhs: Vec<DepthFirstLinearizedBVH>,

    // We cache the transforms from local BVH space to global space
    pub(crate) bvh_transforms: Vec<Transform>,
}

// to prepare a scene for CPU render, must turn all AggregatePrimitives into BVHs using Embree
// this is done recursively to handle multi-level BVH
pub(crate) fn prepare_cpu_acceleration_structures(scene: &Scene) -> CpuAccelerationStructures {
    let embree = embree4::Device::new();

    let mut bvhs: Vec<DepthFirstLinearizedBVH> = Vec::new();
    let mut bvh_transforms: Vec<Transform> = Vec::new();
    fn recursive_helper(
        embree: &embree4::Device,
        transform: Transform,
        bvhs: &mut Vec<DepthFirstLinearizedBVH>,
        bvh_transforms: &mut Vec<Transform>,
        scene: &Scene,
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> usize {
        let mut bvh_builder = BVH2Builder::new();
        let descendants = scene.descendants_iter(aggregate_primitive_index);
        for (geom_index, (primitive_index, child_transform)) in descendants.enumerate() {
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    bvh_builder.add_shape(&basic.shape, &child_transform, geom_index as u32);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().unwrap();
                    let sub_bvh_index = recursive_helper(
                        embree,
                        child_transform.compose(transform.clone()),
                        bvhs,
                        bvh_transforms,
                        scene,
                        aggregate_index
                    );
                    let bounds = bvhs[sub_bvh_index].bounds();
                    bvh_builder.add_sub_bvh(bounds, &child_transform, geom_index as u32, sub_bvh_index as u32);
                }
                Primitive::Transform(_) => unreachable!("descendants iterator should flatten transforms")
            }
        }

        let bvh = bvh_builder.build(embree);
        let bvh_index = bvhs.len();
        let bvh = DepthFirstLinearizedBVH::linearize_bvh(&bvh);
        bvhs.push(bvh);
        bvh_transforms.push(transform);

        bvh_index
    }

    recursive_helper(
        &embree,
        Transform::identity(),
        &mut bvhs,
        &mut bvh_transforms,
        scene,
        scene.root_index()
    );

    CpuAccelerationStructures {
        bvhs,
        bvh_transforms
    }
}

impl CpuAccelerationStructures {
    pub(crate) fn root_bvh_index(&self) -> usize {
        // construction algorithm pushes root bvh last
        self.bvhs.len() - 1
    }
}