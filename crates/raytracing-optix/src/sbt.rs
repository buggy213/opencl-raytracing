//! RAII types around SBT

use std::marker::PhantomData;

use raytracing::{materials::Material, scene::Scene};

use crate::{optix::{self, AovPipelineWrapper, AovSbtWrapper, Material_MaterialVariant, Material_MaterialVariant_Diffuse, Material_MaterialVariant_SmoothDielectric, Material_MaterialVariant_SmoothConductor, Material_MaterialVariant_RoughDielectric, Material_MaterialVariant_RoughConductor, PathtracerPipelineWrapper, PathtracerSbtWrapper, addHitRecordAovSbt, addHitRecordPathtracerSbt, finalizeAovSbt, finalizePathtracerSbt, makeAovSbt, makePathtracerSbt, releaseAovSbt, releasePathtracerSbt}, scene::SbtVisitor};

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
        Material::SmoothDielectric { eta } => {
            optix::Material {
                kind: optix::Material_MaterialKind::SmoothDielectric,
                variant: Material_MaterialVariant {
                    smooth_dielectric: Material_MaterialVariant_SmoothDielectric { eta: eta.0 }
                },
            }
        },
        Material::SmoothConductor { eta, kappa } => {
            optix::Material {
                kind: optix::Material_MaterialKind::SmoothConductor,
                variant: Material_MaterialVariant {
                    smooth_conductor: Material_MaterialVariant_SmoothConductor { eta: eta.0, kappa: kappa.0 }
                },
            }
        },
        Material::RoughDielectric { eta, roughness, remap_roughness } => {
            optix::Material {
                kind: optix::Material_MaterialKind::RoughDielectric,
                variant: Material_MaterialVariant {
                    rough_dielectric: Material_MaterialVariant_RoughDielectric {
                        eta: eta.0,
                        remap_roughness: *remap_roughness,
                        roughness: roughness.0,
                    }
                },
            }
        },
        Material::RoughConductor { eta, kappa, roughness, remap_roughness } => {
            optix::Material {
                kind: optix::Material_MaterialKind::RoughConductor,
                variant: Material_MaterialVariant {
                    rough_conductor: Material_MaterialVariant_RoughConductor {
                        eta: eta.0,
                        kappa: kappa.0,
                        remap_roughness: *remap_roughness,
                        roughness: roughness.0,
                    }
                },
            }
        },
        // CoatedDiffuse falls back to diffuse for now
        Material::CoatedDiffuse { diffuse_albedo, .. } => {
            optix::Material {
                kind: optix::Material_MaterialKind::Diffuse,
                variant: Material_MaterialVariant {
                    diffuse: Material_MaterialVariant_Diffuse { albedo: diffuse_albedo.0 }
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

    fn add_hitgroup_record(&mut self, geometry_data: optix::DeviceGeometryData) {

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
    fn visit_geometry_as(&mut self, geometry_data: optix::DeviceGeometryData, _material: &Material, _area_light: Option<u32>) -> u32 {
        let old_sbt_offset = self.current_sbt_offset;
        self.add_hitgroup_record(geometry_data);

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

    fn add_hitgroup_record(&mut self, geometry_data: optix::DeviceGeometryData, material: &Material, area_light: Option<u32>) {
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
    fn visit_geometry_as(&mut self, geometry_data: optix::DeviceGeometryData, material: &Material, area_light: Option<u32>) -> u32 {
        let old_sbt_offset = self.current_sbt_offset;
        self.add_hitgroup_record(geometry_data, material, area_light);

        old_sbt_offset
    }

    fn visit_instance_as(&mut self) -> u32 {
        0
    }
}

