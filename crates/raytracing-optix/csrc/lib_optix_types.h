#pragma once

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