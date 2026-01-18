// Ray-generation w/ pinhole camera; only performs primary visibility and only calculates geometric normals
#include <optix_device.h>
#include "params.cuh"

// populates buffer w/ debug values for now
extern "C" __global__ void __raygen__debug() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();
    uint2 *debug = pipeline_params.debug;

    debug[tid.y * dim.x + tid.x] = uint2 { tid.x, tid.y };
}