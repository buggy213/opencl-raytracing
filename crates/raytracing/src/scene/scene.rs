use crate::{
    geometry::{Shape, Transform, Vec3, Vec4},
    lights::{EnvironmentLight, Light},
    materials::{Image, ImageId, Material, Texture, TextureId},
    scene::primitive::{
        AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex,
        MaterialIndex, Primitive, PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex,
    },
};

use super::camera::Camera;

#[derive(Debug)]
pub struct Scene {
    pub camera: Camera,

    primitives: Vec<Primitive>,
    root_primitive: AggregatePrimitiveIndex,

    pub environment_light: Option<EnvironmentLight>,
    pub lights: Vec<Light>,

    pub materials: Vec<Material>,

    pub textures: Vec<Texture>,
    pub images: Vec<Image>,
}

// Accessors for primitives
impl From<PrimitiveIndex> for usize {
    fn from(value: PrimitiveIndex) -> Self {
        match value {
            PrimitiveIndex::BasicPrimitiveIndex(BasicPrimitiveIndex(i))
            | PrimitiveIndex::TransformPrimitiveIndex(TransformPrimitiveIndex(i))
            | PrimitiveIndex::AggregatePrimitiveIndex(AggregatePrimitiveIndex(i)) => i as usize,
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

    pub fn get_basic_primitive(
        &self,
        basic_primitive_index: BasicPrimitiveIndex,
    ) -> &BasicPrimitive {
        let raw_index: usize = basic_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Basic(basic_primitive) => basic_primitive,
            _ => unreachable!("corrupted BasicPrimitiveIndex"),
        }
    }

    pub fn get_transform_primitive(
        &self,
        transform_primitive_index: TransformPrimitiveIndex,
    ) -> &TransformPrimitive {
        let raw_index: usize = transform_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Transform(transform_primitive) => transform_primitive,
            _ => unreachable!("corrupted TransformPrimitiveIndex"),
        }
    }

    pub fn get_aggregate_primitive(
        &self,
        aggregate_primitive_index: AggregatePrimitiveIndex,
    ) -> &AggregatePrimitive {
        let raw_index: usize = aggregate_primitive_index.0 as usize;
        match &self.primitives[raw_index] {
            Primitive::Aggregate(aggregate_primitive) => aggregate_primitive,
            _ => unreachable!("corrupted AggregatePrimitiveIndex"),
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
        let aggregate_primitive = self
            .scene
            .get_aggregate_primitive(self.aggregate_primitive_index);
        if self.index >= aggregate_primitive.children.len() {
            return None;
        }

        let result = self
            .scene
            .get_direct_descendant(self.aggregate_primitive_index, self.index);
        self.index += 1;

        Some(result)
    }
}

impl<'scene> ExactSizeIterator for DirectDescendantsIter<'scene> {
    fn len(&self) -> usize {
        self.scene
            .get_aggregate_primitive(self.aggregate_primitive_index)
            .children
            .len()
    }
}

impl<'scene> Iterator for DescendantsIter<'scene> {
    type Item = (PrimitiveIndex, Transform);

    fn next(&mut self) -> Option<Self::Item> {
        let aggregate_primitive = self
            .scene
            .get_aggregate_primitive(self.aggregate_primitive_index);
        if self.index >= aggregate_primitive.children.len() {
            return None;
        }

        let result = self
            .scene
            .get_descendant(self.aggregate_primitive_index, self.index);
        self.index += 1;

        Some(result)
    }
}

impl<'scene> ExactSizeIterator for DescendantsIter<'scene> {
    fn len(&self) -> usize {
        self.scene
            .get_aggregate_primitive(self.aggregate_primitive_index)
            .children
            .len()
    }
}

impl Scene {
    pub fn root_index(&self) -> AggregatePrimitiveIndex {
        self.root_primitive
    }

    pub fn direct_descendants_iter<'scene>(
        &'scene self,
        aggregate_primitive_index: AggregatePrimitiveIndex,
    ) -> DirectDescendantsIter<'scene> {
        DirectDescendantsIter {
            scene: self,
            aggregate_primitive_index,
            index: 0,
        }
    }

    pub fn descendants_iter<'scene>(
        &'scene self,
        aggregate_primitive_index: AggregatePrimitiveIndex,
    ) -> DescendantsIter<'scene> {
        DescendantsIter {
            scene: self,
            aggregate_primitive_index,
            index: 0,
        }
    }

    pub fn get_direct_descendant<'scene>(
        &'scene self,
        aggregate_primitive_index: AggregatePrimitiveIndex,
        index: usize,
    ) -> (PrimitiveIndex, Transform) {
        (
            self.get_aggregate_primitive(aggregate_primitive_index)
                .children[index],
            Transform::identity(),
        )
    }

    pub fn get_descendant<'scene>(
        &'scene self,
        aggregate_primitive_index: AggregatePrimitiveIndex,
        index: usize,
    ) -> (PrimitiveIndex, Transform) {
        let mut current_idx = self
            .get_aggregate_primitive(aggregate_primitive_index)
            .children[index];
        let mut transform = Transform::identity();

        loop {
            match self.get_primitive(current_idx) {
                Primitive::Basic(_) => break,
                Primitive::Aggregate(_) => break,
                Primitive::Transform(transform_primitive) => {
                    let child_idx = transform_primitive.primitive;
                    current_idx = child_idx;
                    transform = transform.compose(transform_primitive.transform.clone());
                }
            }
        }

        (current_idx, transform)
    }
}

pub(super) mod gltf {
    use std::{borrow::Cow, collections::HashMap, ops::Range, path::Path};

    use tracing::warn;

    use crate::{
        geometry::{Matrix4x4, Mesh, Shape, Transform, Vec3, Vec4},
        lights::Light,
        materials::{
            FilterMode, Image, ImageId, Material, Texture, TextureId, TextureSampler, WrapMode,
        },
        scene::{
            Camera, Scene,
            primitive::{
                AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex,
                Primitive, PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex,
            },
        },
    };

    const HEIGHT: usize = 600;

    pub fn scene_from_gltf_file(filepath: &Path) -> anyhow::Result<Scene> {
        let (document, buffers, image_data) = gltf::import(filepath)?;
        let scene_gltf = document.default_scene().unwrap();
        let mut camera: Option<Camera> = None;
        let height: usize = HEIGHT;

        // detect gltf level instancing (i.e. two nodes with the same "mesh" attribute)
        let mut instancing_map: HashMap<usize, Range<usize>> = HashMap::new();

        let mut primitives: Vec<Primitive> = Vec::new();
        let mut lights: Vec<Light> = Vec::new();

        let mut root_primitive_children: Vec<PrimitiveIndex> = Vec::new();

        let mut images: Vec<Image> = Vec::with_capacity(document.images().len());
        for gltf_image in image_data {
            // annoyingly, gltf crate's import feature also uses the `image` crate under the hood
            // so, we are essentially round-tripping data for no reason, but oh well
            let image = Image::load_from_gltf_image(gltf_image);
            images.push(image);
        }

        let mut textures: Vec<Texture> = Vec::with_capacity(document.textures().len());
        for texture in document.textures() {
            let image_id = ImageId(texture.source().index() as u32);
            let sampler = texture.sampler();

            let sampler_name = match (sampler.name(), sampler.index()) {
                (Some(name), _) => Cow::Borrowed(name),
                (None, Some(idx)) => Cow::Owned(idx.to_string()),
                (None, None) => Cow::Borrowed("default"),
            };

            let wrap_s: WrapMode = sampler.wrap_s().into();
            let wrap_t: WrapMode = sampler.wrap_t().into();

            if wrap_s != wrap_t {
                warn!(
                    "gltf sampler ({}) defined with different wrap modes along s ({}) and t ({}), which we don't support. using {}",
                    sampler_name, wrap_s, wrap_t, wrap_s
                );
            }

            let wrap = wrap_s;

            let filter_mode = match (sampler.min_filter(), sampler.mag_filter()) {
                (None, None) => FilterMode::Nearest,
                (None, Some(gltf::texture::MagFilter::Nearest)) => FilterMode::Nearest,
                (None, Some(gltf::texture::MagFilter::Linear)) => FilterMode::Bilinear,
                (Some(gltf::texture::MinFilter::Nearest), _) => FilterMode::Nearest,
                (Some(gltf::texture::MinFilter::Linear), _) => FilterMode::Bilinear,
                (Some(gltf::texture::MinFilter::LinearMipmapLinear), _) => FilterMode::Trilinear,
                (Some(other_min_filter), _) => {
                    warn!("filter type {other_min_filter:?} not supported, falling back to nearest");
                    FilterMode::Nearest
                }
            };

            let texture = Texture::ImageTexture {
                image: image_id,
                sampler: TextureSampler {
                    filter: filter_mode,
                    wrap,
                },
            };

            textures.push(texture);
        }

        let mut materials: Vec<Material> = Vec::with_capacity(document.materials().len());
        let mut material_emissions: Vec<Vec3> = Vec::with_capacity(document.materials().len());
        for material in document.materials() {
            let material_name = match material.name() {
                Some(name) => Cow::Borrowed(name),
                None => match material.index() {
                    Some(material_id) => Cow::Owned(format!("{}", material_id)),
                    None => Cow::Borrowed("default"),
                },
            };

            let pbr_params = material.pbr_metallic_roughness();
            let base_color_tex = pbr_params.base_color_texture();
            let base_color_fac = pbr_params.base_color_factor();
            let metallic_roughness_tex = pbr_params.metallic_roughness_texture();

            let base_color_tex = if let Some(gltf_tex) = base_color_tex {
                if gltf_tex.tex_coord() != 0 {
                    warn!(
                        "material {} uses non-zero TEXCOORD attribute for base color texture, not supported yet",
                        material_name
                    );
                }

                let base_id = TextureId(gltf_tex.texture().index() as u32);

                if base_color_fac != [1.0, 1.0, 1.0, 1.0] {
                    let factor_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ConstantTexture {
                        value: base_color_fac.into(),
                    });
                    let scale_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ScaleTexture {
                        a: base_id,
                        b: factor_id,
                    });

                    scale_id
                } else {
                    base_id
                }
            } else {
                let id = TextureId(textures.len() as u32);
                textures.push(Texture::ConstantTexture {
                    value: base_color_fac.into(),
                });

                id
            };

            let metallic_roughness_tex = if let Some(gltf_tex) = metallic_roughness_tex {
                if gltf_tex.tex_coord() != 0 {
                    warn!(
                        "material {} uses non-zero TEXCOORD attribute for metallic-roughness texture, not supported yet",
                        material_name
                    );
                }

                let metallic_roughness_id = TextureId(gltf_tex.texture().index() as u32);
                let metallic = pbr_params.metallic_factor();
                let roughness = pbr_params.roughness_factor();

                if metallic != 1.0 || roughness != 1.0 {
                    let factor_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ConstantTexture {
                        value: Vec4(0.0, roughness, metallic, 0.0),
                    });
                    let scale_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ScaleTexture {
                        a: metallic_roughness_id,
                        b: factor_id,
                    });

                    scale_id
                } else {
                    metallic_roughness_id
                }
            } else {
                let metallic = pbr_params.metallic_factor();
                let roughness = pbr_params.roughness_factor();
                let id = TextureId(textures.len() as u32);
                textures.push(Texture::ConstantTexture {
                    value: Vec4(0.0, roughness, metallic, 0.0),
                });

                id
            };

            let pbr_material = todo!("find a more suitable material");

            materials.push(pbr_material);

            let emission: Vec3 = material.emissive_factor().into();
            material_emissions.push(emission);
        }

        for node in scene_gltf.nodes() {
            if node.camera().is_some() {
                camera = Some(Camera::from_gltf_camera_node(&node, height));
            }

            if node.mesh().is_some() {
                let gltf_mesh = node.mesh().unwrap();

                let mut transform_matrix = Matrix4x4 {
                    data: node.transform().matrix(),
                };
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
                        let transform_primitive_idx =
                            TransformPrimitiveIndex(primitives.len() as u32);
                        primitives.push(transform_primitive);
                        root_primitive_children.push(transform_primitive_idx.into());
                    }
                    continue;
                }

                let start = primitives.len();
                for gltf_primitive in gltf_mesh.primitives() {
                    let material_idx = gltf_primitive.material().index().unwrap_or(0) as u32;

                    let rt_mesh =
                        Mesh::from_gltf_primitive(gltf_mesh.clone(), gltf_primitive, &buffers);
                    let mesh_shape = Shape::TriangleMesh(rt_mesh);

                    let basic_primitive_idx = BasicPrimitiveIndex(primitives.len() as u32);

                    let area_light_idx =
                        if material_emissions[material_idx as usize] != Vec3::zero() {
                            let area_light = Light::from_emissive_geometry(
                                basic_primitive_idx,
                                material_emissions[material_idx as usize],
                                transform.clone(),
                            );
                            let area_light_idx = lights.len() as u32;
                            lights.push(area_light);
                            Some(area_light_idx)
                        } else {
                            None
                        };

                    let mesh_primitive = Primitive::Basic(BasicPrimitive {
                        shape: mesh_shape,
                        material: material_idx,
                        area_light: area_light_idx,
                    });

                    primitives.push(mesh_primitive);
                }
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

            if let Some(light) = node.light() {
                let punctual_light = Light::from_gltf_punctual_light(&node, &light);
                if let Some(punctual_light) = punctual_light {
                    lights.push(punctual_light);
                }
            }
        }

        let root_primitive = Primitive::Aggregate(AggregatePrimitive {
            children: root_primitive_children,
        });
        let root_primitive_idx = AggregatePrimitiveIndex(primitives.len() as u32);
        primitives.push(root_primitive);

        Ok(Scene {
            // gltf doesn't support environment lighting
            environment_light: None, 
            lights,
            primitives,
            root_primitive: root_primitive_idx,
            camera: camera.expect("Scene must have camera"),
            materials,
            textures,
            images,
        })
    }
}

pub(crate) struct SceneBuilder {
    camera: Option<Camera>,
    primitives: Vec<Primitive>,
    primitive_idxs: Vec<PrimitiveIndex>,
    environment_light: Option<EnvironmentLight>,
    lights: Vec<Light>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    images: Vec<Image>,
}

impl SceneBuilder {
    pub(crate) fn new() -> Self {
        SceneBuilder {
            camera: None,
            primitives: Vec::new(),
            primitive_idxs: Vec::new(),
            environment_light: None,
            lights: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            images: Vec::new(),
        }
    }

    pub(crate) fn add_camera(&mut self, camera: Camera) {
        self.camera = Some(camera);
    }

    pub(crate) fn add_environment_light(&mut self, environment_light: EnvironmentLight) {
        self.environment_light = Some(environment_light);
    }

    pub(crate) fn add_texture(&mut self, tex: Texture) -> TextureId {
        let texture_id = TextureId(self.textures.len() as u32);
        self.textures.push(tex);

        texture_id
    }

    pub(crate) fn add_constant_texture(&mut self, value: Vec4) -> TextureId {
        let const_texture = Texture::ConstantTexture { value };
        self.add_texture(const_texture)
    }

    pub(crate) fn add_material(&mut self, material: Material) -> MaterialIndex {
        let material_id = self.materials.len() as u32;
        self.materials.push(material);

        material_id
    }

    pub(crate) fn add_image(&mut self, image: Image) -> ImageId {
        let image_id = ImageId(self.images.len() as u32);
        self.images.push(image);

        image_id
    }

    pub(crate) fn add_shape_at_position(&mut self, shape: Shape, material_id: MaterialIndex, position: Vec3) {
        let basic_primitive_idx = BasicPrimitiveIndex(self.primitives.len() as u32);
        let basic_primitive = BasicPrimitive {
            shape,
            material: material_id,
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
    }

    pub(crate) fn add_shape_with_transform(
        &mut self,
        shape: Shape,
        material_id: MaterialIndex,
        transform: &Transform,
        area_light_radiance: Option<Vec3>,
    ) {
        let basic_primitive_idx = BasicPrimitiveIndex(self.primitives.len() as u32);

        let area_light_idx = if let Some(radiance) = area_light_radiance {
            let area_light = Light::DiffuseAreaLight {
                prim_id: basic_primitive_idx,
                radiance,
                transform: transform.clone(),
            };
            let idx = self.lights.len() as u32;
            self.lights.push(area_light);
            Some(idx)
        } else {
            None
        };

        let basic_primitive = BasicPrimitive {
            shape,
            material: material_id,
            area_light: area_light_idx,
        };
        self.primitives.push(basic_primitive.into());

        let transform_primitive_idx = TransformPrimitiveIndex(self.primitives.len() as u32);
        let transformed = TransformPrimitive {
            primitive: basic_primitive_idx.into(),
            transform: transform.clone(),
        };
        self.primitives.push(transformed.into());

        self.primitive_idxs.push(transform_primitive_idx.into());
    }

    pub(crate) fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    pub(crate) fn add_point_light(&mut self, position: Vec3, intensity: Vec3) {
        let point_light = Light::PointLight {
            position,
            intensity,
        };
        self.add_light(point_light);
    }

    pub(crate) fn build(mut self) -> Scene {
        let root_primitive_idx = AggregatePrimitiveIndex(self.primitives.len() as u32);
        let root_primitive = AggregatePrimitive {
            children: self.primitive_idxs,
        };

        self.primitives.push(root_primitive.into());

        Scene {
            camera: self.camera.expect("scene description incomplete"),
            primitives: self.primitives,
            root_primitive: root_primitive_idx,
            environment_light: self.environment_light, 
            lights: self.lights,
            materials: self.materials,
            textures: self.textures,
            images: self.images,
        }
    }
}

