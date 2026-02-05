#pragma once

/*
 * Types shared between Rust/C++ to manage OptiX runtime state and configuration
 */

#include <optix_types.h>

struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixBuildInputType primitive_type;
    OptixTraversableHandle handle;
};

struct AovPipelineWrapper {
    OptixPipeline pipeline;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgram;
    OptixProgramGroup sphereHitProgramGroup;
    OptixProgramGroup triHitProgramGroup;
};

struct PathtracerPipelineWrapper {
    OptixPipeline pipeline;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgramRadiance;
    OptixProgramGroup missProgramShadow;

    OptixProgramGroup hitProgramGroupShadow;
    OptixProgramGroup diffuseHitProgramGroupRadiance;
    // OptixProgramGroup smoothDielectricHitProgramGroupRadiance
    // ...
};