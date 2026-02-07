//! RAII types around SBT

use std::marker::PhantomData;

use raytracing::{geometry::Shape, scene::Scene};

use crate::optix::{AovPipelineWrapper, AovSbtWrapper, GeometryData, GeometryData_GeometryKind, Vec2SliceWrap, Vec3SliceWrap, Vec3uSliceWrap, addHitRecordAovSbt, finalizeAovSbt, makeAovSbt, releaseAovSbt};

// PhantomData tells borrowck that AovSbtWrapper acts as though it references scene data (it does)
// Once finalized, the AovSbt doesn't reference scene data anymore, and lifetime annotation is unneeded
struct AovSbtBuilder<'scene> {
    ptr: AovSbtWrapper,
    _data: PhantomData<&'scene Scene>
}

struct AovSbt {
    ptr: AovSbtWrapper
}

impl<'scene> AovSbtBuilder<'scene> {
    pub(crate) fn new(_scene: &'scene Scene) -> AovSbtBuilder<'scene> {
        // SAFETY: no preconditions
        Self {
            ptr: unsafe { makeAovSbt() },
            _data: PhantomData
        }
    }

    pub(crate) fn add_hitgroup_record(&mut self, shape: &Shape) {
        let geometry_data = match shape {
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
                    kind: GeometryData_GeometryKind::TRIANGLE,
                    num_tris,
                    tris,
                    num_vertices,
                    normals,
                    uvs,
                }
            },
            Shape::Sphere { .. } => {
                GeometryData {
                    kind: GeometryData_GeometryKind::SPHERE,
                    num_tris: 0,
                    tris: std::ptr::null(),
                    num_vertices: 0,
                    normals: std::ptr::null(),
                    uvs: std::ptr::null(),
                }
            },
        };

        unsafe { addHitRecordAovSbt(self.ptr, geometry_data); }
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