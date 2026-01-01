use crate::{
    geometry::{Shape, Transform, Vec3, Vec4},
    lights::Light,
    materials::{Image, Material, Texture, TextureId},
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

            // TODO: actually respect filter mode from gltf (its model is somewhat different)
            let filter_mode = FilterMode::Nearest;

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

            let pbr_material = Material::GLTFMetallicRoughness {
                base_color: base_color_tex,
                metallic_roughness: metallic_roughness_tex,
            };

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

struct SceneBuilder {
    camera: Option<Camera>,
    primitives: Vec<Primitive>,
    primitive_idxs: Vec<PrimitiveIndex>,
    lights: Vec<Light>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
}

impl SceneBuilder {
    fn new() -> Self {
        SceneBuilder {
            camera: None,
            primitives: Vec::new(),
            primitive_idxs: Vec::new(),
            lights: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
        }
    }

    fn add_camera(&mut self, camera: Camera) {
        self.camera = Some(camera);
    }

    fn add_texture(&mut self, tex: Texture) -> TextureId {
        let texture_id = TextureId(self.textures.len() as u32);
        self.textures.push(tex);

        texture_id
    }

    fn add_constant_texture(&mut self, value: Vec4) -> TextureId {
        let const_texture = Texture::ConstantTexture { value };
        self.add_texture(const_texture)
    }

    fn add_material(&mut self, material: Material) -> MaterialIndex {
        let material_id = self.materials.len() as u32;
        self.materials.push(material);

        material_id
    }

    fn add_shape_at_position(&mut self, shape: Shape, material_id: MaterialIndex, position: Vec3) {
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

    fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }

    fn add_point_light(&mut self, position: Vec3, intensity: Vec3) {
        let point_light = Light::PointLight {
            position,
            intensity,
        };
        self.add_light(point_light);
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
            lights: self.lights,
            materials: self.materials,
            textures: self.textures,
            images: Vec::new(),
        }
    }
}

pub mod test_scenes {
    use crate::{
        geometry::{Mesh, Shape, Vec2, Vec3, Vec3u, Vec4},
        lights::Light,
        materials::{Material, Texture},
        renderer::{AOVFlags, RaytracerSettings},
        scene::{Camera, Scene},
    };

    use super::SceneBuilder;

    // helpers for basic meshes
    fn make_mesh(verts: &[Vec3], tris: &[Vec3u], normals: &[Vec3]) -> Mesh {
        Mesh {
            vertices: verts.to_vec(),
            tris: tris.to_vec(),
            normals: normals.to_vec(),
            uvs: Vec::new(),
        }
    }

    fn make_plane(a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3) -> Mesh {
        // ensure a, b, c, d are coplanar, counterclockwise
        let ab = b - a;
        let ac = c - a;
        let x = Vec3::cross(ab, ac).unit();

        let cd = d - c;
        let ca = -ac;
        let y = Vec3::cross(cd, ca).unit();

        assert!(
            (x - normal).near_zero(),
            "points not in plane defined by normal"
        );
        assert!(
            (y - normal).near_zero(),
            "points not in plane defined by normal"
        );

        make_mesh(
            &[a, b, c, d],
            &[Vec3u(0, 1, 2), Vec3u(2, 3, 0)],
            &[normal, normal, normal, normal],
        )
    }

    #[rustfmt::skip]
    fn make_cube(side_length: f32) -> Mesh {
        let h = side_length / 2.0;

        // 6 faces, each with 4 vertices and 2 triangles
        // Vertices are duplicated per face for correct flat-shaded normals
        let mut vertices = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut tris = Vec::with_capacity(12);

        // Face order: +X, -X, +Y, -Y, +Z, -Z
        // Each face defined CCW when looking at the face from outside

        // +X face (normal = +X)
        let base = vertices.len() as u32;

        vertices.extend_from_slice(&[
            Vec3(h, -h, -h),
            Vec3(h,  h, -h),
            Vec3(h,  h,  h),
            Vec3(h, -h,  h),
        ]);
        normals.extend_from_slice(&[Vec3(1.0, 0.0, 0.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        // -X face (normal = -X)
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            Vec3(-h,  h, -h),
            Vec3(-h, -h, -h),
            Vec3(-h, -h,  h),
            Vec3(-h,  h,  h),
        ]);
        normals.extend_from_slice(&[Vec3(-1.0, 0.0, 0.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        // +Y face (normal = +Y)
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            Vec3( h, h, -h),
            Vec3(-h, h, -h),
            Vec3(-h, h,  h),
            Vec3( h, h,  h),
        ]);
        normals.extend_from_slice(&[Vec3(0.0, 1.0, 0.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        // -Y face (normal = -Y)
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            Vec3(-h, -h, -h),
            Vec3( h, -h, -h),
            Vec3( h, -h,  h),
            Vec3(-h, -h,  h),
        ]);
        normals.extend_from_slice(&[Vec3(0.0, -1.0, 0.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        // +Z face (normal = +Z)
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            Vec3(-h, -h, h),
            Vec3( h, -h, h),
            Vec3( h,  h, h),
            Vec3(-h,  h, h),
        ]);
        normals.extend_from_slice(&[Vec3(0.0, 0.0, 1.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        // -Z face (normal = -Z)
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[
            Vec3( h, -h, -h),
            Vec3(-h, -h, -h),
            Vec3(-h,  h, -h),
            Vec3( h,  h, -h),
        ]);
        normals.extend_from_slice(&[Vec3(0.0, 0.0, -1.0); 4]);
        tris.push(Vec3u(base, base + 1, base + 2));
        tris.push(Vec3u(base, base + 2, base + 3));

        Mesh {
            vertices,
            tris,
            normals,
            uvs: Vec::new(),
        }
    }

    // super simple test scene with one sphere, no light
    pub fn sphere_scene() -> Scene {
        let mut scene_builder = SceneBuilder::new();

        let sphere = Shape::Sphere {
            center: Vec3::zero(),
            radius: 1.0,
        };

        let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

        let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

        scene_builder.add_shape_at_position(sphere, white_diffuse, Vec3(0.0, 0.0, -3.0));

        let camera = Camera::lookat_camera_perspective(
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, -3.0),
            Vec3(0.0, 1.0, 0.0),
            (45.0_f32).to_radians(),
            400,
            400,
        );
        scene_builder.add_camera(camera);

        let scene = scene_builder.build();

        scene
    }

    pub fn cube_scene() -> Scene {
        let mut scene_builder = SceneBuilder::new();

        let cube_mesh = make_cube(1.0);
        let cube = Shape::TriangleMesh(cube_mesh);

        let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

        let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

        scene_builder.add_shape_at_position(cube, white_diffuse, Vec3(0.0, 0.0, -3.0));

        let camera = Camera::lookat_camera_perspective(
            Vec3(1.0, 0.75, -1.0),
            Vec3(0.0, 0.0, -3.0),
            Vec3(0.0, 1.0, 0.0),
            (45.0_f32).to_radians(),
            400,
            400,
        );
        scene_builder.add_camera(camera);

        scene_builder.build()
    }

    pub fn cube_orthographic_scene() -> Scene {
        let mut scene_builder = SceneBuilder::new();

        let cube_mesh = make_cube(1.0);
        let cube = Shape::TriangleMesh(cube_mesh);

        let white = scene_builder.add_constant_texture(Vec4(1.0, 1.0, 1.0, 1.0));

        let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });

        scene_builder.add_shape_at_position(cube, white_diffuse, Vec3(0.0, 0.0, -3.0));

        let camera = Camera::lookat_camera_orthographic(
            Vec3(1.0, 0.75, -1.0),
            Vec3(0.0, 0.0, -3.0),
            Vec3(0.0, 1.0, 0.0),
            400,
            400,
            2.5 / 400.0,
        );

        scene_builder.add_camera(camera);

        scene_builder.build()
    }

    pub fn checkered_plane_scene() -> Scene {
        let mut scene_builder = SceneBuilder::new();

        let plane = {
            let mut plane = make_plane(
                Vec3(-100.0, -100.0, 0.1),
                Vec3(100.0, -100.0, 0.1),
                Vec3(100.0, 100.0, 0.1),
                Vec3(-100.0, 100.0, 0.1),
                Vec3(0.0, 0.0, 1.0),
            );

            plane.uvs = vec![
                Vec2(-500.0, -500.0),
                Vec2(500.0, -500.0),
                Vec2(500.0, 500.0),
                Vec2(-500.0, 500.0),
            ];

            plane
        };
        let plane_shape = Shape::TriangleMesh(plane);

        let checker_tex = Texture::CheckerTexture {
            color1: Vec4(0.0, 0.0, 0.0, 1.0),
            color2: Vec4(1.0, 1.0, 1.0, 1.0),
        };
        let checker_tex_id = scene_builder.add_texture(checker_tex);
        let checker_material = Material::Diffuse {
            albedo: checker_tex_id,
        };

        let checker_material_id = scene_builder.add_material(checker_material);

        scene_builder.add_shape_at_position(plane_shape, checker_material_id, Vec3::zero());

        // light from straight up
        let sun = Light::DirectionLight {
            direction: Vec3(0.0, 0.0, -1.0),
            radiance: Vec3(1000.0, 1000.0, 1000.0),
        };
        scene_builder.add_light(sun);

        // angle below +y axis
        let y_angle = (10.0_f32).to_radians();
        let lookat_dist = 1.0;

        let camera = Camera::lookat_camera_perspective(
            Vec3(0.0, 0.0, 0.22),
            Vec3(
                0.0,
                f32::cos(y_angle) * lookat_dist,
                0.22 - f32::sin(y_angle) * lookat_dist,
            ),
            Vec3(0.0, 0.0, 1.0),
            (40.0_f32).to_radians(),
            480,
            270,
        );
        scene_builder.add_camera(camera);

        scene_builder.build()
    }

    // template for cornell box, returns a SceneBuilder so other functions can add on top
    // note: this isn't the same dimensions as the real cornell box
    #[rustfmt::skip]
    fn cornell_box() -> SceneBuilder {
        let mut scene_builder = SceneBuilder::new();

        // Dimensions: width=2, height=1.5, depth=2.0, y-up
        let w = 2.0;
        let h = 1.5;
        let d = 2.0;

        // Box corners
        let left   = w / 2.0;
        let right  =  -w / 2.0;
        let bottom = 0.0;
        let top    = h;
        let back   = -d / 2.0;
        let front  =  d / 2.0;

        // Plane normals
        let up    = Vec3(0.0, 0.0, 1.0);
        let down  = Vec3(0.0, 0.0, -1.0);
        let leftn = Vec3(-1.0, 0.0, 0.0);
        let rightn= Vec3(1.0, 0.0, 0.0);
        let backn = Vec3(0.0, 1.0, 0.0);

        // Floor
        let floor = make_plane(
            Vec3(right, front, bottom),
            Vec3(right, back, bottom),
            Vec3(left, back, bottom),
            Vec3(left, front, bottom),
            up,
        );
        // Ceiling
        let ceiling = make_plane(
            Vec3(left, front, top),
            Vec3(left, back, top),
            Vec3(right, back, top),
            Vec3(right, front, top),
            down,
        );
        // Left wall
        let left_wall = make_plane(
            Vec3(left, front, bottom),
            Vec3(left, back, bottom),
            Vec3(left, back, top),
            Vec3(left, front, top),
            leftn,
        );
        // Right wall
        let right_wall = make_plane(
            Vec3(right, front, top),
            Vec3(right, back, top),
            Vec3(right, back, bottom),
            Vec3(right, front, bottom),
            rightn,
        );
        // Back wall
        let back_wall = make_plane(
            Vec3(right, back, top),
            Vec3(left, back, top),
            Vec3(left, back, bottom),
            Vec3(right, back, bottom),
            backn,
        );

        // Add planes to scene builder with colored walls
        let white = scene_builder.add_constant_texture(Vec4(0.6, 0.6, 0.6, 1.0));
        let red   = scene_builder.add_constant_texture(Vec4(0.6, 0.2, 0.2, 1.0));
        let blue  = scene_builder.add_constant_texture(Vec4(0.2, 0.2, 0.6, 1.0));

        let white_diffuse = scene_builder.add_material(Material::Diffuse { albedo: white });
        let red_diffuse   = scene_builder.add_material(Material::Diffuse { albedo: red });
        let blue_diffuse  = scene_builder.add_material(Material::Diffuse { albedo: blue });

        scene_builder.add_shape_at_position(Shape::TriangleMesh(floor), white_diffuse, Vec3(0.0, 0.0, 0.0));
        scene_builder.add_shape_at_position(Shape::TriangleMesh(ceiling), white_diffuse, Vec3(0.0, 0.0, 0.0));
        scene_builder.add_shape_at_position(Shape::TriangleMesh(left_wall), red_diffuse, Vec3(0.0, 0.0, 0.0));
        scene_builder.add_shape_at_position(Shape::TriangleMesh(right_wall), blue_diffuse, Vec3(0.0, 0.0, 0.0));
        scene_builder.add_shape_at_position(Shape::TriangleMesh(back_wall), white_diffuse, Vec3(0.0, 0.0, 0.0));

        // Camera looking into the box from the front
        let camera = Camera::lookat_camera_perspective(
            Vec3(0.0, front + 3.4, 0.4),
            Vec3(0.0, 0.0, h / 2.0),
            Vec3(0.0, 0.0, 1.0),
            (37.8_f32).to_radians(),
            500,
            500,
        );
        scene_builder.add_camera(camera);

        // Add a point light near the top center
        scene_builder.add_point_light(
            Vec3(0.0, 0.0, top - 0.1),    // slightly below the ceiling, centered
            Vec3(1000.0, 1000.0, 1000.0), // bright white light
        );

        scene_builder
    }

    // single dielectric sphere (ior = 1.5) in cornell box
    pub fn dielectric_scene() -> Scene {
        let mut cornell_box = cornell_box();

        let ior_texture = cornell_box.add_constant_texture(Vec4(1.5, 0.0, 0.0, 0.0));
        let dielectric_material =
            cornell_box.add_material(Material::SmoothDielectric { eta: ior_texture });
        cornell_box.add_shape_at_position(
            Shape::Sphere {
                center: Vec3::zero(),
                radius: 0.5,
            },
            dielectric_material,
            Vec3(0.0, 0.0, 0.75),
        );

        cornell_box.build()
    }

    // single "gold" sphere in cornell box
    // ior at red wavelengths is 0.13 + 4.10i
    // ior at green wavelengths is 0.43 + 2.46i
    // ior at blue wavelengths is 1.38 + 1.91i
    pub fn metal_scene() -> Scene {
        let mut cornell_box = cornell_box();

        let ior_texture = cornell_box.add_constant_texture(Vec4(0.13, 0.43, 1.38, 0.0));
        let kappa_texture = cornell_box.add_constant_texture(Vec4(4.10, 2.46, 1.91, 0.0));
        let metal_material = cornell_box.add_material(Material::SmoothConductor {
            eta: ior_texture,
            kappa: kappa_texture,
        });
        cornell_box.add_shape_at_position(
            Shape::Sphere {
                center: Vec3::zero(),
                radius: 0.5,
            },
            metal_material,
            Vec3(0.0, 0.0, 0.75),
        );

        cornell_box.build()
    }

    pub fn rough_metal_scene() -> Scene {
        let mut cornell_box = cornell_box();

        let ior_texture = cornell_box.add_constant_texture(Vec4(0.13, 0.43, 1.38, 0.0));
        let kappa_texture = cornell_box.add_constant_texture(Vec4(4.10, 2.46, 1.91, 0.0));
        let roughness_texture = cornell_box.add_constant_texture(Vec4(0.5, 0.5, 0.0, 0.0));
        let rough_conductor_material = cornell_box.add_material(Material::RoughConductor {
            eta: ior_texture,
            kappa: kappa_texture,
            roughness: roughness_texture,
        });

        cornell_box.add_shape_at_position(
            Shape::Sphere {
                center: Vec3::zero(),
                radius: 0.5,
            },
            rough_conductor_material,
            Vec3(0.0, 0.0, 0.75),
        );

        cornell_box.build()
    }

    pub fn rough_dielectric_scene() -> Scene {
        let mut cornell_box = cornell_box();

        let ior_texture = cornell_box.add_constant_texture(Vec4(1.5, 0.0, 0.0, 0.0));
        let roughness_texture = cornell_box.add_constant_texture(Vec4(0.5, 0.5, 0.0, 0.0));
        let rough_dielectric_material = cornell_box.add_material(Material::RoughDielectric {
            eta: ior_texture,
            roughness: roughness_texture,
        });
        cornell_box.add_shape_at_position(
            Shape::Sphere {
                center: Vec3::zero(),
                radius: 0.5,
            },
            rough_dielectric_material,
            Vec3(0.0, 0.0, 0.75),
        );

        cornell_box.build()
    }

    fn gltf_sphere_scene(base_color: Vec3, metallic: f32, roughness: f32) -> Scene {
        let mut cornell_box = cornell_box();

        let base_color_texture =
            cornell_box.add_constant_texture(Vec4(base_color.0, base_color.1, base_color.2, 0.0));
        let metallic_roughness_texture =
            cornell_box.add_constant_texture(Vec4(0.0, roughness, metallic, 0.0));
        let gltf_material = cornell_box.add_material(Material::GLTFMetallicRoughness {
            base_color: base_color_texture,
            metallic_roughness: metallic_roughness_texture,
        });

        cornell_box.add_shape_at_position(
            Shape::Sphere {
                center: Vec3::zero(),
                radius: 0.5,
            },
            gltf_material,
            Vec3(0.0, 0.0, 0.75),
        );

        cornell_box.build()
    }

    pub fn gltf_rough_metal() -> Scene {
        gltf_sphere_scene(Vec3(0.5, 0.5, 0.0), 1.0, 0.5)
    }

    pub fn gltf_rough_nonmetal() -> Scene {
        gltf_sphere_scene(Vec3(0.5, 0.5, 0.0), 0.0, 0.5)
    }

    pub fn gltf_smooth_metal() -> Scene {
        gltf_sphere_scene(Vec3(0.5, 0.5, 0.0), 1.0, 0.0)
    }

    pub fn gltf_smooth_nonmetal() -> Scene {
        gltf_sphere_scene(Vec3(0.5, 0.5, 0.0), 0.0, 0.0)
    }

    fn debug_normals_settings() -> RaytracerSettings {
        RaytracerSettings {
            outputs: AOVFlags::NORMALS,
            ..Default::default()
        }
    }

    pub struct TestScene {
        pub name: &'static str,
        pub scene_func: fn() -> Scene,
        pub settings_func: fn() -> RaytracerSettings,
    }

    pub const fn all_test_scenes() -> &'static [TestScene] {
        &[
            TestScene {
                name: "sphere",
                scene_func: sphere_scene,
                settings_func: debug_normals_settings,
            },
            TestScene {
                name: "cube",
                scene_func: cube_scene,
                settings_func: debug_normals_settings,
            },
            TestScene {
                name: "cube_orthographic",
                scene_func: cube_orthographic_scene,
                settings_func: debug_normals_settings,
            },
            TestScene {
                name: "checkered_plane",
                scene_func: checkered_plane_scene,
                // deliberately only 1 spp to exhibit aliasing
                settings_func: || RaytracerSettings {
                    samples_per_pixel: 1,
                    ..Default::default()
                },
            },
            TestScene {
                name: "dielectric",
                scene_func: dielectric_scene,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "metal",
                scene_func: metal_scene,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "rough_metal",
                scene_func: rough_metal_scene,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "rough_dielectric",
                scene_func: rough_dielectric_scene,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "gltf_rough_metal",
                scene_func: gltf_rough_metal,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "gltf_rough_nonmetal",
                scene_func: gltf_rough_nonmetal,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "gltf_smooth_metal",
                scene_func: gltf_smooth_metal,
                settings_func: RaytracerSettings::default,
            },
            TestScene {
                name: "gltf_smooth_nonmetal",
                scene_func: gltf_smooth_nonmetal,
                settings_func: RaytracerSettings::default,
            },
        ]
    }
}
