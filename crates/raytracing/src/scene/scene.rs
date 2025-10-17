use std::{collections::HashMap, ops::Range, path::Path};

use crate::{geometry::{Matrix4x4, Mesh, Shape, Transform, Vec3}, lights::Light, materials::Material, scene::primitive::{AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex, Primitive, PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex}};

use super::camera::{Camera, RenderTile};

pub struct Scene {
    pub camera: Camera,

    primitives: Vec<Primitive>,
    root_primitive: AggregatePrimitiveIndex,

    pub lights: Vec<Light>,

    pub materials: Vec<Material>
}

const HEIGHT: usize = 600;

// Accessors for primitives
impl From<PrimitiveIndex> for usize {
    fn from(value: PrimitiveIndex) -> Self {
        match value {
            PrimitiveIndex::BasicPrimitiveIndex(BasicPrimitiveIndex(i))
            | PrimitiveIndex::TransformPrimitiveIndex(TransformPrimitiveIndex(i))
            | PrimitiveIndex::AggregatePrimitiveIndex(AggregatePrimitiveIndex(i)) => i as usize
        }
    }
}

impl Scene {
    pub fn primitive_index_from_usize(&self, raw_index: usize) -> PrimitiveIndex {
        match &self.primitives[raw_index] {
            Primitive::Basic(_) => BasicPrimitiveIndex(raw_index as u32).into(),
            Primitive::Transform(_) => TransformPrimitiveIndex(raw_index as u32).into(),
            Primitive::Aggregate(_) => AggregatePrimitiveIndex(raw_index as u32).into(),
        }
    }

    pub fn get_primitive(&self, primitive_index: PrimitiveIndex) -> &Primitive {
        let raw_index: usize = primitive_index.into();
        &self.primitives[raw_index]
    }

    pub fn get_basic_primitive(&self, basic_primitive_index: BasicPrimitiveIndex) -> &BasicPrimitive {
        let raw_index: usize = basic_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Basic(basic_primitive) => basic_primitive,
            _ => unreachable!("corrupted BasicPrimitiveIndex")
        }
    }

    pub fn get_transform_primitive(&self, transform_primitive_index: TransformPrimitiveIndex) -> &TransformPrimitive {
        let raw_index: usize = transform_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Transform(transform_primitive) => transform_primitive,
            _ => unreachable!("corrupted TransformPrimitiveIndex")
        }
    }

    pub fn get_aggregate_primitive(&self, aggregate_primitive_index: AggregatePrimitiveIndex) -> &AggregatePrimitive {
        let raw_index: usize = aggregate_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Aggregate(aggregate_primitive) => aggregate_primitive,
            _ => unreachable!("corrupted AggregatePrimitiveIndex")
        }
    }
}

// Helpers for traversing primitive graph

// Iterator over direct descendants of an AggregatePrimitive
struct DirectDescendantsIter<'scene> {
    scene: &'scene Scene,
    aggregate_primitive: &'scene AggregatePrimitive,
    index: usize,
}

// Iterator over leaf descendants of an AggregatePrimitive
struct DescendantsIter<'scene> {
    scene: &'scene Scene,
    aggregate_primitive: &'scene AggregatePrimitive,
    index: usize,
}

impl<'scene> Iterator for DirectDescendantsIter<'scene> {
    type Item = (&'scene Primitive, Transform);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.aggregate_primitive.children.len() { 
            return None;
        }

        let child_idx = self.aggregate_primitive.children[self.index];
        let direct_descendant = self.scene.get_primitive(child_idx);
        self.index += 1;

        Some((direct_descendant, Transform::identity()))
    }
}

impl<'scene> Iterator for DescendantsIter<'scene> {
    type Item = (&'scene Primitive, Transform);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.aggregate_primitive.children.len() {
            return None;
        }

        let child_idx = self.aggregate_primitive.children[self.index];
        let mut current = self.scene.get_primitive(child_idx);
        let mut transform = Transform::identity();

        loop {
            match current {
                Primitive::Basic(_) => break,
                Primitive::Aggregate(_) => break,
                Primitive::Transform(transform_primitive) => {
                    let child_idx = transform_primitive.primitive;
                    current = self.scene.get_primitive(child_idx);
                    transform = transform.compose(transform_primitive.transform.clone());
                },
                
            }
        }

        Some((current, transform))
    }
}

impl Scene {
    pub fn root(&self) -> &AggregatePrimitive {
        self.get_aggregate_primitive(self.root_primitive)
    }
    
    pub fn direct_descendants_iter<'scene>(&'scene self, aggregate_primitive: &'scene AggregatePrimitive) 
        -> DirectDescendantsIter<'scene> {
        DirectDescendantsIter { scene: self, aggregate_primitive, index: 0 }
    }

    pub fn descendants_iter<'scene>(&'scene self, aggregate_primitive: &'scene AggregatePrimitive)
        -> DescendantsIter<'scene> {
        DescendantsIter { scene: self, aggregate_primitive, index: 0 }
    }
}

impl Scene {
    pub fn from_gltf_file(filepath: &Path, render_tile: Option<RenderTile>) -> anyhow::Result<Scene> {
        let (document, buffers, _images) = gltf::import(filepath)?;
        let scene_gltf = document.default_scene().unwrap();
        let mut camera: Option<Camera> = None;  
        let height: usize = HEIGHT;
        
        // detect gltf level instancing (i.e. two nodes with the same "mesh" attribute)
        let mut instancing_map: HashMap<usize, Range<usize>> = HashMap::new();

        let mut primitives: Vec<Primitive> = Vec::new();
        let mut lights: Vec<Light> = Vec::new();

        let mut root_primitive_children: Vec<PrimitiveIndex> = Vec::new();

        let mut materials: Vec<Material> = Vec::new();
        let mut material_emissions: Vec<Vec3> = Vec::new();
        for material in document.materials() {
            let diffuse_color = material.pbr_metallic_roughness().base_color_factor();
            let diffuse = Material::Diffuse { 
                albedo: [diffuse_color[0], diffuse_color[1], diffuse_color[2]].into()
            };

            materials.push(diffuse);

            let emission: Vec3 = material.emissive_factor().into();
            material_emissions.push(emission);
        }
        
        for node in scene_gltf.nodes() {
            if node.camera().is_some() {
                camera = Some(Camera::from_gltf_camera_node(&node, height, render_tile));
            }

            if node.mesh().is_some() {
                let gltf_mesh = node.mesh().unwrap();

                let mut transform_matrix = Matrix4x4 { data: node.transform().matrix() };
                transform_matrix.transpose();
                let transform = Transform::from(transform_matrix);

                
                // check if this mesh is instanced
                if let Some(range) = instancing_map.get(&gltf_mesh.index()) {
                    for i in range.clone() {
                        let instance_primitive_idx = BasicPrimitiveIndex(i as u32);
                        let transform_primitive = Primitive::Transform(TransformPrimitive {
                            primitive: instance_primitive_idx.into(),
                            transform: transform.clone(),
                        });
                        let transform_primitive_idx = TransformPrimitiveIndex(primitives.len() as u32);
                        primitives.push(transform_primitive);
                        root_primitive_children.push(transform_primitive_idx.into());
                    }
                    continue;
                }

                let start = primitives.len();
                for gltf_primitive in gltf_mesh.primitives() {
                    let primitive_id = primitives.len() as u32;
                    let material_idx = gltf_primitive.material().index().unwrap_or(0) as u32;

                    let rt_mesh = Mesh::from_gltf_primitive(gltf_primitive, &buffers);
                    let mesh_shape = Shape::TriangleMesh(rt_mesh);
                    
                    let area_light_idx = if material_emissions[material_idx as usize] != Vec3::zero() {
                        let area_light = Light::from_emissive_geometry(primitive_id, material_emissions[material_idx as usize]);
                        let area_light_idx = lights.len() as u32;
                        lights.push(area_light);
                        Some(area_light_idx)
                    } else { None };

                    let mesh_primitive = Primitive::Basic(BasicPrimitive {
                        shape: mesh_shape,
                        material: material_idx,
                        area_light: area_light_idx,
                    });

                    primitives.push(mesh_primitive);
                };
                let end = primitives.len();
            
                // record the range of this mesh in the instancing map
                instancing_map.insert(gltf_mesh.index(), start..end);

                // create transform primitives for them
                for gltf_primitive in start..end {
                    let primitive_id = BasicPrimitiveIndex(gltf_primitive as u32);
                    
                    let transform_primitive = Primitive::Transform(TransformPrimitive {
                        primitive: primitive_id.into(),
                        transform: transform.clone(),
                    });

                    let transform_primitive_idx = TransformPrimitiveIndex(primitives.len() as u32);
                    primitives.push(transform_primitive);
                    root_primitive_children.push(transform_primitive_idx.into());
                }
            }

            if node.light().is_some() {
                let punctual_light = Light::from_gltf_punctual_light(&node);
                lights.push(punctual_light);
            }
        }

        let root_primitive = Primitive::Aggregate(AggregatePrimitive {
            children: root_primitive_children
        });
        let root_primitive_idx = AggregatePrimitiveIndex(primitives.len() as u32);
        primitives.push(root_primitive);

        Ok(Scene {
            lights,
            primitives,
            root_primitive: root_primitive_idx,
            camera: camera.expect("Scene must have camera"),
            materials
        })
    }

    pub fn test_scene() -> Scene {
        todo!()
    }
}
