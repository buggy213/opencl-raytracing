use std::{borrow::Cow, collections::HashMap, ops::Range, path::Path};

use tracing::warn;

use crate::{geometry::{Matrix4x4, Mesh, Shape, Transform, Vec3}, lights::Light, materials::{FilterMode, Image, ImageId, Material, Texture, TextureSampler, WrapMode}, scene::primitive::{AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex, Primitive, PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex}};

use super::camera::{Camera, RenderTile};

#[derive(Debug)]
pub struct Scene {
    pub camera: Camera,

    primitives: Vec<Primitive>,
    root_primitive: AggregatePrimitiveIndex,

    pub lights: Vec<Light>,

    pub materials: Vec<Material>,

    pub textures: Vec<Texture>,
    pub images: Vec<Image>,
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
pub struct DirectDescendantsIter<'scene> {
    scene: &'scene Scene,
    aggregate_primitive_index: AggregatePrimitiveIndex,
    index: usize,
}

// Iterator over leaf descendants of an AggregatePrimitive
pub struct DescendantsIter<'scene> {
    scene: &'scene Scene,
    aggregate_primitive_index: AggregatePrimitiveIndex,
    index: usize,
}

impl<'scene> Iterator for DirectDescendantsIter<'scene> {
    type Item = (PrimitiveIndex, Transform);

    fn next(&mut self) -> Option<Self::Item> {
        let aggregate_primitive = self.scene.get_aggregate_primitive(self.aggregate_primitive_index);
        if self.index >= aggregate_primitive.children.len() { 
            return None;
        }

        let result = self.scene.get_direct_descendant(self.aggregate_primitive_index, self.index);
        self.index += 1;

        Some(result)
    }
}

impl<'scene> Iterator for DescendantsIter<'scene> {
    type Item = (PrimitiveIndex, Transform);

    fn next(&mut self) -> Option<Self::Item> {
        let aggregate_primitive = self.scene.get_aggregate_primitive(self.aggregate_primitive_index);
        if self.index >= aggregate_primitive.children.len() {
            return None;
        }

        let result = self.scene.get_descendant(self.aggregate_primitive_index, self.index);
        self.index += 1;

        Some(result)
    }
}

impl Scene {
    pub fn root_index(&self) -> AggregatePrimitiveIndex {
        self.root_primitive
    }
    
    pub fn direct_descendants_iter<'scene>(&'scene self, aggregate_primitive_index: AggregatePrimitiveIndex) 
        -> DirectDescendantsIter<'scene> {
        DirectDescendantsIter { scene: self, aggregate_primitive_index, index: 0 }
    }

    pub fn descendants_iter<'scene>(&'scene self, aggregate_primitive_index: AggregatePrimitiveIndex)
        -> DescendantsIter<'scene> {
        DescendantsIter { scene: self, aggregate_primitive_index, index: 0 }
    }

    
    pub fn get_direct_descendant<'scene>(&'scene self, aggregate_primitive_index: AggregatePrimitiveIndex, index: usize) 
        -> (PrimitiveIndex, Transform) {
        (self.get_aggregate_primitive(aggregate_primitive_index).children[index], Transform::identity())
    }

    pub fn get_descendant<'scene>(&'scene self, aggregate_primitive_index: AggregatePrimitiveIndex, index: usize) 
        -> (PrimitiveIndex, Transform) {
        let mut current_idx = self.get_aggregate_primitive(aggregate_primitive_index).children[index];
        let mut transform = Transform::identity();

        loop {
            match self.get_primitive(current_idx) {
                Primitive::Basic(_) => break,
                Primitive::Aggregate(_) => break,
                Primitive::Transform(transform_primitive) => {
                    let child_idx = transform_primitive.primitive;
                    current_idx = child_idx;
                    transform = transform.compose(transform_primitive.transform.clone());
                },
                
            }
        }

        (current_idx, transform)
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

        let mut images: Vec<Image> = Vec::with_capacity(document.images().len());
        for image in document.images() {
            match image.source() {
                gltf::image::Source::View { view: _, mime_type: _ } => {
                    todo!("implement binary image deserialization")
                },
                gltf::image::Source::Uri { uri, mime_type: _ } => {
                    let img = Image::load_from_path(Path::new(uri))?;
                    images.push(img);
                },
            }
        }

        let mut textures: Vec<Texture> = Vec::with_capacity(document.textures().len());
        for texture in document.textures() {
            let image_id = ImageId(texture.source().index() as u32);
            let sampler = texture.sampler();
            
            let sampler_name = match (sampler.name(), sampler.index()) {
                (Some(name), _) => Cow::Borrowed(name),
                (None, Some(idx)) => {
                    Cow::Owned(idx.to_string())
                }
                (None, None) => Cow::Borrowed("default")
            };

            let wrap_s: WrapMode = sampler.wrap_s().into();
            let wrap_t: WrapMode = sampler.wrap_t().into();
            
            if wrap_s != wrap_t {
                warn!(
                    "gltf sampler ({}) defined with different wrap modes along s ({}) and t ({}), which we don't support. using {}",
                    sampler_name,
                    wrap_s,
                    wrap_t,
                    wrap_s
                );
            }

            let wrap = wrap_s;
            
            // TODO: actually respect filter mode from gltf (its model is somewhat different)
            let filter_mode = FilterMode::Nearest;

            let texture = Texture::ImageTexture { 
                image: image_id, 
                sampler: TextureSampler {
                    filter: filter_mode,
                    wrap,
                }
            };

            textures.push(texture);
        }

        let mut materials: Vec<Material> = Vec::with_capacity(document.materials().len());
        let mut material_emissions: Vec<Vec3> = Vec::with_capacity(document.materials().len());
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
                    let material_idx = gltf_primitive.material().index().unwrap_or(0) as u32;

                    let rt_mesh = Mesh::from_gltf_primitive(gltf_mesh.clone(), gltf_primitive, &buffers);
                    let mesh_shape = Shape::TriangleMesh(rt_mesh);
                    
                    let basic_primitive_idx = BasicPrimitiveIndex(primitives.len() as u32);

                    let area_light_idx = if material_emissions[material_idx as usize] != Vec3::zero() {
                        let area_light = Light::from_emissive_geometry(
                            basic_primitive_idx, 
                            material_emissions[material_idx as usize],
                            transform.clone(),
                        );
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
            materials,
            textures,
            images
        })
    }
}

struct SceneBuilder {
    camera: Option<Camera>,
    primitives: Vec<Primitive>,
    primitive_idxs: Vec<PrimitiveIndex>,
    materials: Vec<Material>,
}

impl SceneBuilder {
    fn new() -> Self {
        SceneBuilder { 
            camera: None, 
            primitives: Vec::new(), 
            primitive_idxs: Vec::new(), 
            materials: Vec::new() 
        }
    }

    fn add_lookat_camera(
        mut self,
        camera_position: Vec3,
        target: Vec3,
        raster_width: usize,
        raster_height: usize,
    ) -> Self {
        let camera = Camera::lookat_camera(
            camera_position, 
            target, 
            raster_width, 
            raster_height
        );

        self.camera = Some(camera);

        self
    }

    fn add_shape_at_position(
        mut self,
        shape: Shape,
        material: Material,
        position: Vec3,
    ) -> Self {
        let material_idx = self.materials.len() as u32;
        self.materials.push(material);

        let basic_primitive_idx = BasicPrimitiveIndex(self.primitives.len() as u32);
        let basic_primitive = BasicPrimitive {
            shape,
            material: material_idx,
            area_light: None,
        };

        self.primitives.push(basic_primitive.into());

        let transform_primitive_idx = TransformPrimitiveIndex(self.primitives.len() as u32);
        let transformed = TransformPrimitive {
            primitive: basic_primitive_idx.into(),
            transform: Transform::translate(position),
        };

        self.primitives.push(transformed.into());

        self.primitive_idxs.push(transform_primitive_idx.into());
        self
    }

    fn build(mut self) -> Scene {
        let root_primitive_idx = AggregatePrimitiveIndex(self.primitives.len() as u32);
        let root_primitive = AggregatePrimitive {
            children: self.primitive_idxs,
        };

        self.primitives.push(root_primitive.into());

        Scene { 
            camera: self.camera.expect("scene description incomplete"), 
            primitives: self.primitives, 
            root_primitive: root_primitive_idx, 
            lights: Vec::new(), 
            materials: self.materials,
            textures: Vec::new(),
            images: Vec::new(),
        }
    }
}

pub mod test_scenes {
    use crate::{geometry::{Shape, Vec3}, materials::Material, scene::Scene};

    use super::SceneBuilder;

    // super simple test scene with one sphere, no light
    pub fn test_scene() -> Scene {
        let scene_builder = SceneBuilder::new();
        
        let sphere = Shape::Sphere { 
            center: Vec3::zero(), 
            radius: 1.0
        };

        let sphere_material = Material::Diffuse { 
            albedo: Vec3(1.0, 0.0, 0.0) 
        };

        let scene = scene_builder
        .add_shape_at_position(
            sphere, 
            sphere_material,
            Vec3(0.0, 0.0, -3.0)
        )
        .add_lookat_camera(
            Vec3(0.0, 0.0, 0.0), 
            Vec3(0.0, 0.0, -3.0), 
            400, 
            400
        )
        .build();

        scene
    }
}