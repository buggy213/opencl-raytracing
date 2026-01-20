#pragma once

/*
 * Types shared between Rust/C++ to manage OptiX runtime state and configuration
 */

#include <optix_types.h>

struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixTraversableHandle handle;
};

struct OptixPipelineWrapper {
    OptixPipeline pipeline;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgram;
    OptixProgramGroup hitProgramGroup;
};