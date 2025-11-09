use std::{borrow::Cow, collections::HashMap, ops::Range, path::Path};

use tracing::warn;

use crate::{geometry::{Matrix4x4, Mesh, Shape, Transform, Vec3, Vec4}, lights::Light, materials::{FilterMode, Image, ImageId, Material, Texture, TextureId, TextureSampler, WrapMode}, scene::primitive::{AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex, MaterialIndex, Primitive, PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex}};

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
            let pbr_params = material.pbr_metallic_roughness();
            let diffuse_gltf_texture = pbr_params.base_color_texture();
            let diffuse_factor = pbr_params.base_color_factor();
            
            let albedo_texture = if let Some(gltf_tex) = diffuse_gltf_texture {
                if gltf_tex.tex_coord() != 0 {
                    let material_name = match material.name() {
                        Some(name) => Cow::Borrowed(name),
                        None => match material.index() {
                            Some(material_id) => Cow::Owned(format!("{}", material_id)),
                            None => Cow::Borrowed("default")
                        },
                    };

                    warn!("material {} uses non-zero TEXCOORD attribute, not supported yet", material_name);
                }
                
                let base_id = TextureId(gltf_tex.texture().index() as u32);
                
                if diffuse_factor != [1.0, 1.0, 1.0, 1.0] {
                    let factor_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ConstantTexture { value: diffuse_factor.into() });
                    let scale_id = TextureId(textures.len() as u32);
                    textures.push(Texture::ScaleTexture { a: base_id, b: factor_id });
                    
                    scale_id
                }
                else {
                    base_id
                }
            }
            else {
                let id = TextureId(textures.len() as u32);
                textures.push(Texture::ConstantTexture { 
                    value: diffuse_factor.into() 
                });

                id
            };
            
            let diffuse = Material::Diffuse { 
                albedo: albedo_texture
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

    fn add_lookat_camera(
        &mut self,
        camera_position: Vec3,
        target: Vec3,
        up: Vec3,
        yfov: f32, // radians
        raster_width: usize,
        raster_height: usize,
    ) {
        let camera = Camera::lookat_camera(
            camera_position, 
            target, 
            up,
            yfov,
            raster_width, 
            raster_height
        );

        self.camera = Some(camera);
    }

    fn add_constant_texture(
        &mut self,
        value: Vec4
    ) -> TextureId {
        let const_texture_id = TextureId(self.textures.len() as u32);
        let const_texture = Texture::ConstantTexture { value };
        self.textures.push(const_texture);

        const_texture_id
    }

    fn add_material(
        &mut self,
        material: Material
    ) -> MaterialIndex {
        let material_id = self.materials.len() as u32;
        self.materials.push(material);

        material_id
    }

    fn add_shape_at_position(
        &mut self,
        shape: Shape,
        material_id: MaterialIndex,
        position: Vec3,
    ) {
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

    fn add_point_light(
        &mut self,
        position: Vec3,
        intensity: Vec3
    ) {
        let point_light = Light::PointLight { position, intensity };
        self.lights.push(point_light);
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
        geometry::{Mesh, Shape, Vec3, Vec3u, Vec4}, 
        materials::Material, 
        scene::Scene
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
        
        assert!((x - normal).near_zero(), "points not in plane defined by normal");
        assert!((y - normal).near_zero(), "points not in plane defined by normal");

        make_mesh(
            &[a, b, c, d],
            &[Vec3u(0, 1, 2), Vec3u(2, 3, 0)],
            &[normal, normal, normal, normal]
        )
    }

    // super simple test scene with one sphere, no light
    pub fn sphere_scene() -> Scene {
        let mut scene_builder = SceneBuilder::new();
        
        let sphere = Shape::Sphere { 
            center: Vec3::zero(), 
            radius: 1.0
        };

        let white = scene_builder.add_constant_texture(
            Vec4(1.0, 1.0, 1.0, 1.0)
        );

        let white_diffuse = scene_builder.add_material(
            Material::Diffuse { albedo: white }
        );
        
        scene_builder.add_shape_at_position(
            sphere, 
            white_diffuse,
            Vec3(0.0, 0.0, -3.0)
        );
        scene_builder.add_lookat_camera(
            Vec3(0.0, 0.0, 0.0), 
            Vec3(0.0, 0.0, -3.0),
            Vec3(0.0, 1.0, 0.0),
        (45.0_f32).to_radians(),
            400, 
            400
        );
        
        let scene = scene_builder.build();

        scene
    }

    // template for cornell box, returns a SceneBuilder so other functions can add on top
    // note: this isn't the same dimensions as the real cornell box
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
            up
        );
        // Ceiling
        let ceiling = make_plane(
            Vec3(left, front, top),
            Vec3(left, back, top),
            Vec3(right, back, top),
            Vec3(right, front, top),
            down
        );
        // Left wall
        let left_wall = make_plane(
            Vec3(left, front, bottom),
            Vec3(left, back, bottom),
            Vec3(left, back, top),
            Vec3(left, front, top),
            leftn
        );
        // Right wall
        let right_wall = make_plane(
            Vec3(right, front, top),
            Vec3(right, back, top),
            Vec3(right, back, bottom),
            Vec3(right, front, bottom),
            rightn
        );
        // Back wall
        let back_wall = make_plane(
            Vec3(right, back, top),
            Vec3(left, back, top),
            Vec3(left, back, bottom),
            Vec3(right, back, bottom),
            backn
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
        scene_builder.add_lookat_camera(
            Vec3(0.0, front + 3.4, 0.4),
            Vec3(0.0, 0.0, h / 2.0),
            Vec3(0.0, 0.0, 1.0),
            (37.8_f32).to_radians(),
            500,
            500,
        );

        // Add a point light near the top center
        scene_builder.add_point_light(
            Vec3(0.0, 0.0, top - 0.1), // slightly below the ceiling, centered
            Vec3(1000.0, 1000.0, 1000.0),    // bright white light
        );

        scene_builder
    }

    // single dielectric sphere (ior = 1.5) in cornell box
    pub fn dielectric_scene() -> Scene {
        let cornell_box = cornell_box().build();
        cornell_box
    }

    // single metal sphere (ior = 2.0 + 2.0i) in cornell box
    pub fn metal_scene() -> Scene {
        todo!()
    }

    pub struct TestScene {
        pub name: &'static str,
        pub func: fn() -> Scene
    }

    pub const fn all_test_scenes() -> &'static [TestScene] {
        &[
            TestScene {
                name: "sphere",
                func: sphere_scene
            },
            TestScene {
                name: "dielectric",
                func: dielectric_scene
            },
            TestScene {
                name: "metal",
                func: metal_scene
            }
        ]
    }
}