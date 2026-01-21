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

struct OptixPipelineWrapper {
    OptixPipeline pipeline;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgram;
    OptixProgramGroup sphereHitProgramGroup;
    OptixProgramGroup triHitProgramGroup;
};