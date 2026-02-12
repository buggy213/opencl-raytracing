//! RAII types around SBT

use std::marker::PhantomData;

use raytracing::{geometry::Shape, materials::Material, scene::Scene};

use crate::{optix::{self, AovPipelineWrapper, AovSbtWrapper, GeometryData, GeometryKind, Material_MaterialVariant, Material_MaterialVariant_Diffuse, PathtracerPipelineWrapper, PathtracerSbtWrapper, Vec2SliceWrap, Vec3SliceWrap, Vec3uSliceWrap, addHitRecordAovSbt, addHitRecordPathtracerSbt, finalizeAovSbt, finalizePathtracerSbt, makeAovSbt, makePathtracerSbt, releaseAovSbt, releasePathtracerSbt}, scene::SbtVisitor};

// PhantomData tells borrowck that AovSbtWrapper acts as though it references scene data (it does)
// Once finalized, the AovSbt doesn't reference scene data anymore, and lifetime annotation is unneeded
pub(crate) struct AovSbtBuilder<'scene> {
    ptr: AovSbtWrapper,
    current_sbt_offset: u32,
    _data: PhantomData<&'scene Scene>
}

pub(crate) struct AovSbt {
    pub(crate) ptr: AovSbtWrapper
}

fn geometry_data_from_shape(shape: &Shape) -> GeometryData {
    match shape {
        Shape::TriangleMesh(mesh) => {
            let tris: Vec3uSliceWrap = mesh.tris.as_slice().into();
            let (tris, num_tris) = (tris.as_ptr(), tris.len());
            
            let num_vertices = mesh.vertices.len();
            let normals = if mesh.normals.len() == mesh.vertices.len() {
                let normals: Vec3SliceWrap = mesh.normals.as_slice().into();
                normals.as_ptr()
            }
            else {
                std::ptr::null()
            };

            let uvs = if mesh.uvs.len() == mesh.vertices.len() {
                let uvs: Vec2SliceWrap = mesh.uvs.as_slice().into();
                uvs.as_ptr()
            }
            else {
                std::ptr::null()
            };

            GeometryData {
                kind: GeometryKind::TRIANGLE,
                num_tris,
                tris,
                num_vertices,
                normals,
                uvs,
            }
        },
        Shape::Sphere { .. } => {
            GeometryData {
                kind: GeometryKind::SPHERE,
                num_tris: 0,
                tris: std::ptr::null(),
                num_vertices: 0,
                normals: std::ptr::null(),
                uvs: std::ptr::null(),
            }
        },
    }
}

fn optix_material_from_material(material: &Material) -> optix::Material {
    match material {
        Material::Diffuse { albedo } => {
            optix::Material {
                kind: optix::Material_MaterialKind::Diffuse,
                variant: Material_MaterialVariant { 
                    diffuse: Material_MaterialVariant_Diffuse { albedo: albedo.0 } 
                },
            }
        },
        
        // TODO: other materials
        Material::CoatedDiffuse { diffuse_albedo, .. } => {
            optix::Material {
                kind: optix::Material_MaterialKind::Diffuse,
                variant: Material_MaterialVariant { 
                    diffuse: Material_MaterialVariant_Diffuse { albedo: diffuse_albedo.0 } 
                },
            }
        }
        
        _ => {
            optix::Material {
                kind: optix::Material_MaterialKind::Diffuse,
                variant: Material_MaterialVariant { 
                    diffuse: Material_MaterialVariant_Diffuse { albedo: 0 } 
                },
            }
        }
    }
}

impl<'scene> AovSbtBuilder<'scene> {
    pub(crate) fn new(_scene: &'scene Scene) -> AovSbtBuilder<'scene> {
        // SAFETY: no preconditions
        Self {
            ptr: unsafe { makeAovSbt() },
            current_sbt_offset: 0,
            _data: PhantomData
        }
    }

    fn add_hitgroup_record(&mut self, shape: &Shape) {
        let geometry_data = geometry_data_from_shape(shape);

        let sbt_entries = unsafe { addHitRecordAovSbt(self.ptr, geometry_data) };

        self.current_sbt_offset += sbt_entries as u32;
    }

    pub(crate) fn finalize(self, pipeline_wrapper: AovPipelineWrapper) -> AovSbt {
        unsafe { finalizeAovSbt(self.ptr, pipeline_wrapper); }

        AovSbt { ptr: self.ptr }
    }
}

impl Drop for AovSbt {
    fn drop(&mut self) {
        unsafe { releaseAovSbt(self.ptr); }
    }
}

impl SbtVisitor for AovSbtBuilder<'_> {
    fn visit_geometry_as(&mut self, shape: &Shape, _material: &Material, _area_light: Option<u32>) -> u32 {
        let old_sbt_offset = self.current_sbt_offset;
        self.add_hitgroup_record(shape);

        old_sbt_offset
    }

    fn visit_instance_as(&mut self) -> u32 {
        0
    }
}

// see note above for `PhantomData` usage
pub(crate) struct PathtracerSbtBuilder<'scene> {
    ptr: PathtracerSbtWrapper,
    current_sbt_offset: u32,
    _data: PhantomData<&'scene Scene>
}

pub(crate) struct PathtracerSbt {
    pub(crate) ptr: PathtracerSbtWrapper
}

impl<'scene> PathtracerSbtBuilder<'scene> {
    pub(crate) fn new(_scene: &'scene Scene) -> PathtracerSbtBuilder<'scene> {
        // SAFETY: no preconditions
        Self {
            ptr: unsafe { makePathtracerSbt() },
            current_sbt_offset: 0,
            _data: PhantomData
        }
    }

    fn add_hitgroup_record(&mut self, shape: &Shape, material: &Material, area_light: Option<u32>) {
        let geometry_data = geometry_data_from_shape(shape);
        let optix_material = optix_material_from_material(material);
        let area_light = if let Some(idx) = area_light {
            idx as i32
        } else {
            -1
        };

        let sbt_entries = unsafe { 
            addHitRecordPathtracerSbt(self.ptr, geometry_data, optix_material, area_light) 
        };

        self.current_sbt_offset += sbt_entries as u32;
    }

    pub(crate) fn finalize(self, pipeline_wrapper: PathtracerPipelineWrapper) -> PathtracerSbt {
        unsafe { finalizePathtracerSbt(self.ptr, pipeline_wrapper); }

        PathtracerSbt { ptr: self.ptr }
    }
}

impl Drop for PathtracerSbt {
    fn drop(&mut self) {
        unsafe { releasePathtracerSbt(self.ptr); }
    }
}

impl SbtVisitor for PathtracerSbtBuilder<'_> {
    fn visit_geometry_as(&mut self, shape: &Shape, material: &Material, area_light: Option<u32>) -> u32 {
        let old_sbt_offset = self.current_sbt_offset;
        self.add_hitgroup_record(shape, material, area_light);

        old_sbt_offset
    }

    fn visit_instance_as(&mut self) -> u32 {
        0
    }
}

