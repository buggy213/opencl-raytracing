//! Most of the scene description conversion is kept within Rust
//! that calls into C++ in order to generate hierarchy of Geometry-AS / Instance-AS for OptiX.
//! We do it this way to avoid having to make the whole scene description #[repr(C)]

use std::{collections::HashMap, marker::PhantomData, os::raw::c_void};

use image::metadata::Cicp;
use raytracing::{geometry::Shape, lights::Light, materials::Material, scene::{AggregatePrimitiveIndex, BasicPrimitiveIndex, Primitive, Scene}};
use tracing::warn;

use crate::optix::{self, CudaArray, OptixAccelerationStructure, Texture, Texture_TextureVariant, Texture_TextureVariant_ConstantTexture, Texture_TextureVariant_ImageTexture, Vec3SliceWrap, Vec3uSliceWrap, makeCudaArray, makeCudaTexture};

// hooks for SBT to be constructed alongside the AS hierarchy. `visit` functions on a node 
// return corresponding SBT offset which will be used when constructing that node's parent.
pub(crate) trait SbtVisitor {
    fn visit_geometry_as(&mut self, geometry_data: optix::DeviceGeometryData, material: &Material, area_light: Option<u32>) -> u32;
    fn visit_instance_as(&mut self) -> u32;
}

fn geometry_data_from_shape(shape: &Shape) -> optix::DeviceGeometryData {
    match shape {
        Shape::TriangleMesh(mesh) => {
            let tris: Vec3uSliceWrap = mesh.tris.as_slice().into();
            let (tris, num_tris) = (tris.as_ptr(), tris.len());
            
            let vertices: Vec3SliceWrap = mesh.vertices.as_slice().into();
            let (vertices, num_vertices) = (vertices.as_ptr(), vertices.len());

            let normals = if mesh.normals.len() == mesh.vertices.len() {
                let normals: Vec3SliceWrap = mesh.normals.as_slice().into();
                normals.as_ptr()
            }
            else {
                std::ptr::null()
            };

            let uvs = if mesh.uvs.len() == mesh.vertices.len() {
                let uvs: optix::Vec2SliceWrap = mesh.uvs.as_slice().into();
                uvs.as_ptr()
            }
            else {
                std::ptr::null()
            };

            let host_geometry_data = optix::HostGeometryData {
                kind: optix::GeometryKind::TRIANGLE,
                num_tris,
                tris,
                num_vertices,
                vertices,
                normals,
                uvs,
            };

            unsafe {
                optix::uploadGeometryData(host_geometry_data)
            }
        },
        Shape::Sphere { .. } => {
            optix::DeviceGeometryData {
                kind: optix::GeometryKind::SPHERE,
                num_tris: 0,
                d_tris: std::ptr::null(),
                num_vertices: 0,
                d_vertices: std::ptr::null(),
                d_normals: std::ptr::null(),
                d_uvs: std::ptr::null(),
            }
        },
    }
}

fn make_leaf_geometry_as(
    ctx: optix::OptixDeviceContext, 
    shape: &Shape,
    device_geometry_data: optix::DeviceGeometryData,
) -> OptixAccelerationStructure {
    match shape {
        Shape::TriangleMesh(_) => {
            unsafe {
                optix::makeMeshAccelerationStructure(
                    ctx, 
                    device_geometry_data
                )
            }
        },
        Shape::Sphere { center, radius } => {
            let center = optix::Vec3 {
                x: center.x(),
                y: center.y(),
                z: center.z(),
            };

            // SAFETY: makeSphereAccelerationStructure has no real safety requirements
            unsafe { 
                optix::makeSphereAccelerationStructure(ctx, center, *radius) 
            }
        },
    }
}

fn make_instance_as(
    ctx: optix::OptixDeviceContext,
    instances: &[OptixAccelerationStructure],
    transforms: &[optix::Matrix4x4],
    sbt_offsets: &[u32]
) -> OptixAccelerationStructure {
    assert!(instances.len() == transforms.len());
    
    // SAFETY: instances and transforms are valid pointers to arrays of OptixAccelerationStructure and Matrix4x4
    // and instances.len() == transforms.len()
    unsafe {
        optix::makeInstanceAccelerationStructure(
            ctx,
            instances.as_ptr(),
            transforms.as_ptr(),
            sbt_offsets.as_ptr(),
            instances.len()
        )
    }
}

pub(crate) fn prepare_optix_acceleration_structures(
    ctx: optix::OptixDeviceContext,
    scene: &Scene,
    sbt_visitor: &mut dyn SbtVisitor,
    primitive_to_sbt: &mut HashMap<BasicPrimitiveIndex, u32>
) -> OptixAccelerationStructure { 
    fn recursive_helper(
        ctx: optix::OptixDeviceContext,
        scene: &Scene,
        sbt_visitor: &mut dyn SbtVisitor, 
        primitive_to_sbt: &mut HashMap<BasicPrimitiveIndex, u32>,
        aggregate_primitive_index: AggregatePrimitiveIndex
    ) -> OptixAccelerationStructure {

        let descendants = scene.descendants_iter(aggregate_primitive_index);
        let mut descendant_acceleration_structures: Vec<OptixAccelerationStructure> = Vec::with_capacity(descendants.len());
        let mut descendant_transforms: Vec<optix::Matrix4x4> = Vec::with_capacity(descendants.len());
        let mut descendant_sbt_offsets: Vec<u32> = Vec::with_capacity(descendants.len());

        for (primitive_index, child_transform) in descendants {
            descendant_transforms.push(child_transform.forward.into());
            
            match scene.get_primitive(primitive_index) {
                Primitive::Basic(basic) => {
                    let leaf_geometry_data = geometry_data_from_shape(&basic.shape);
                    let leaf_gas = make_leaf_geometry_as(ctx, &basic.shape, leaf_geometry_data);
                    let leaf_gas_material = &scene.materials[basic.material as usize];
                    let leaf_gas_sbt_offset = sbt_visitor.visit_geometry_as(leaf_geometry_data, leaf_gas_material, basic.area_light);
                    descendant_acceleration_structures.push(leaf_gas);
                    descendant_sbt_offsets.push(leaf_gas_sbt_offset);
                    primitive_to_sbt.insert(primitive_index.try_into().unwrap(), leaf_gas_sbt_offset);
                }
                Primitive::Aggregate(_) => {
                    let aggregate_index = primitive_index.try_into().expect("this should be an aggregate index");
                    let child_ias = recursive_helper(
                        ctx,
                        scene,
                        sbt_visitor,
                        primitive_to_sbt,
                        aggregate_index
                    );
                    let _child_ias_sbt_offset = sbt_visitor.visit_instance_as();
                    descendant_acceleration_structures.push(child_ias);
                }
                Primitive::Transform(_) => unreachable!("DescendantsIter should flatten transforms")
            }
        }

        make_instance_as(
            ctx, 
            &descendant_acceleration_structures, 
            &descendant_transforms,
            &descendant_sbt_offsets
        )
    }

    recursive_helper(
        ctx,
        scene,
        sbt_visitor,
        primitive_to_sbt,
        scene.root_index()
    )
}

pub(crate) fn prepare_optix_textures(
    scene: &Scene
) -> Vec<optix::Texture> {
    let mut cuda_arrays: Vec<CudaArray> = Vec::new();

    for image in &scene.images {
        let image_buffer = &image.buffer;
        if image_buffer.color_space() != Cicp::SRGB_LINEAR {
            warn!("image doesn't contain linear data");
        }
        
        // CUDA doesn't support 3 channel textures, so pad w/ alpha
        if image_buffer.color().channel_count() == 3 {
            warn!("padding image with alpha channel; ideally this is done when image is decoded from disk");
        }

        let image_buffer = match image_buffer.color() {
            image::ColorType::Rgb8 => &image::DynamicImage::ImageRgba8(image_buffer.to_rgba8()),
            image::ColorType::Rgb16 => &image::DynamicImage::ImageRgba16(image_buffer.to_rgba16()),
            image::ColorType::Rgb32F => &image::DynamicImage::ImageRgba32F(image_buffer.to_rgba32f()),
            _ => image_buffer,
        };

        let (data_ptr, layout, bytes_per_sample) = match image_buffer.color() {
            image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8 => {
                let flat = image_buffer.as_flat_samples_u8().unwrap();
                if !flat.is_normal(image::flat::NormalForm::RowMajorPacked) {
                    unreachable!("ImageBuffer should always be row-major packed data");
                }

                (flat.samples.as_ptr() as *const c_void, flat.layout, size_of::<u8>())
            },
            image::ColorType::L16
            | image::ColorType::La16
            | image::ColorType::Rgb16
            | image::ColorType::Rgba16 => {
                let flat = image_buffer.as_flat_samples_u16().unwrap();
                if !flat.is_normal(image::flat::NormalForm::RowMajorPacked) {
                    unreachable!("ImageBuffer should always be row-major packed data");
                }

                (flat.samples.as_ptr() as *const c_void, flat.layout, size_of::<u16>())
            },
            image::ColorType::Rgb32F
            | image::ColorType::Rgba32F => {
                let flat = image_buffer.as_flat_samples_f32().unwrap();
                if !flat.is_normal(image::flat::NormalForm::RowMajorPacked) {
                    unreachable!("ImageBuffer should always be row-major packed data");
                }
                
                (flat.samples.as_ptr() as *const c_void, flat.layout, size_of::<f32>())
            },
            
            _ => unimplemented!("unsupported dynamic image format for optix texture upload"),
        };

        let fmt = match image_buffer.color() {
            image::ColorType::L8 => optix::TextureFormat::R8,
            image::ColorType::La8 => optix::TextureFormat::RG8,
            image::ColorType::Rgba8 => optix::TextureFormat::RGBA8,
            image::ColorType::L16 => optix::TextureFormat::R16,
            image::ColorType::La16 => optix::TextureFormat::RG16,
            image::ColorType::Rgba16 => optix::TextureFormat::RGBA16,
            image::ColorType::Rgba32F => optix::TextureFormat::RGBA32F,
            _ => unimplemented!("unsupported dynamic image format for optix texture upload"),
        };

        let pitch = layout.height_stride * bytes_per_sample;
        let cuda_array = unsafe {
            makeCudaArray(
                data_ptr, 
                pitch, 
                layout.width as usize, 
                layout.height as usize, 
                fmt
            )
        };

        cuda_arrays.push(cuda_array);
    }

    let mut cuda_textures = Vec::new();
    let mut optix_textures = Vec::new();

    for texture in &scene.textures {
        let optix_texture = match texture {
            raytracing::materials::Texture::ImageTexture { image, sampler } => {
                let backing_array = cuda_arrays[image.0 as usize];
                let texture_object = unsafe { makeCudaTexture(backing_array, (*sampler).into()) };
                cuda_textures.push(texture_object);
                
                Texture {
                    kind: optix::Texture_TextureKind::ImageTexture,
                    variant: Texture_TextureVariant {
                        image_texture: Texture_TextureVariant_ImageTexture { texture_object: texture_object.handle }
                    },
                }
            },
            raytracing::materials::Texture::ConstantTexture { value } => {
                Texture { 
                    kind: optix::Texture_TextureKind::ConstantTexture, 
                    variant: Texture_TextureVariant {
                        constant_texture: Texture_TextureVariant_ConstantTexture { value: (*value).into() }
                    } 
                }
                
            },
            raytracing::materials::Texture::CheckerTexture { color1, color2 } => {
                Texture {
                    kind: optix::Texture_TextureKind::CheckerTexture,
                    variant: Texture_TextureVariant {
                        checker_texture: optix::Texture_TextureVariant_CheckerTexture {
                            color1: (*color1).into(),
                            color2: (*color2).into(),
                        }
                    }
                }
            },
            raytracing::materials::Texture::ScaleTexture { a, b } => {
                Texture {
                    kind: optix::Texture_TextureKind::ScaleTexture,
                    variant: Texture_TextureVariant {
                        scale_texture: optix::Texture_TextureVariant_ScaleTexture {
                            a: a.0,
                            b: b.0,
                        }
                    }
                }
            },
            raytracing::materials::Texture::MixTexture { a, b, c } => {
                Texture {
                    kind: optix::Texture_TextureKind::MixTexture,
                    variant: Texture_TextureVariant {
                        mix_texture: optix::Texture_TextureVariant_MixTexture {
                            a: a.0,
                            b: b.0,
                            c: c.0,
                        }
                    }
                }
            },
        };

        optix_textures.push(optix_texture);
    }

    optix_textures
}

fn prepare_optix_lights(
    scene: &Scene,
    primitive_to_sbt: &HashMap<BasicPrimitiveIndex, u32>
) -> Vec<optix::Light> {
    let mut optix_lights = Vec::with_capacity(scene.lights.len());
    for light in &scene.lights {
        let optix_light = match light {
            Light::PointLight { position, intensity } => {
                optix::Light { 
                    kind: optix::Light_LightKind::PointLight, 
                    variant: optix::Light_LightVariant { 
                        point_light: optix::Light_LightVariant_PointLight { position: (*position).into(), intensity: (*intensity).into() } 
                    } 
                }
            },
            Light::DirectionLight { direction, radiance } => {
                optix::Light {
                    kind: optix::Light_LightKind::DirectionLight,
                    variant: optix::Light_LightVariant {
                        direction_light: optix::Light_LightVariant_DirectionLight {
                            direction: (*direction).into(),
                            radiance: (*radiance).into(),
                        }
                    }
                }
            },
            Light::DiffuseAreaLight { prim_id, radiance, light_to_world } => {
                optix::Light {
                    kind: optix::Light_LightKind::DiffuseAreaLight,
                    variant: optix::Light_LightVariant {
                        area_light: optix::Light_LightVariant_DiffuseAreaLight {
                            prim_id: primitive_to_sbt[prim_id],
                            radiance: (*radiance).into(),
                            light_to_world: (*light_to_world).into(),
                        }
                    }
                }
            },
        };

        optix_lights.push(optix_light);
    }

    optix_lights
}

pub(crate) struct OptixSceneData {
    pub(crate) optix_camera: optix::Camera,
    pub(crate) optix_lights: Vec<optix::Light>,
    pub(crate) optix_textures: Vec<optix::Texture>,
}

pub(crate) fn prepare_optix_scene_data(
    scene: &Scene,
    optix_textures: Vec<optix::Texture>,
    primitive_to_sbt: &HashMap<BasicPrimitiveIndex, u32>,
) -> OptixSceneData {
    OptixSceneData { 
        optix_camera: scene.camera.clone().into(), 
        optix_lights: prepare_optix_lights(scene, primitive_to_sbt), 
        optix_textures 
    }
}

pub(crate) struct OptixScene<'scene> {
    pub(crate) ffi: optix::Scene,
    _data: PhantomData<&'scene OptixSceneData>
}

impl<'scene> OptixScene<'scene> {
    pub(crate) fn new(scene_data: &'scene OptixSceneData) -> Self {
        let ffi_scene = optix::Scene {
            camera: &scene_data.optix_camera,
            num_lights: scene_data.optix_lights.len(),
            lights: scene_data.optix_lights.as_ptr(),
            num_textures: scene_data.optix_textures.len(),
            textures: scene_data.optix_textures.as_ptr(),
        };

        OptixScene {
            ffi: ffi_scene,
            _data: PhantomData,
        }
    }
}