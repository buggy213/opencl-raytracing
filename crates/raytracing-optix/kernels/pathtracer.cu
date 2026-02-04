#define USE_PATHTRACER_PIPELINE_PARAMS
#include "params.h"

#include <optix_device.h>

#include "lib_types.h"
#include "kernel_types.h"
#include "kernel_math.h"

#include "camera.h"
// #include "sample.h"

// primary entry point
extern "C" __global__ void __raygen__main() {
    // todo
}

// -- Radiance Rays
// The radiance ray type uses OPTIX_PAYLOAD_TYPE_ID_0,
// which... (TODO: what goes in the ray payload?)

// one miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
    // todo
}

// one closest-hit program per material
extern "C" __global__ void __closesthit__radiance_diffuse() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
    // todo
}

// -- Shadow Rays
// The shadow ray type uses OPTIX_PAYLOAD_TYPE_ID_1,
// which defines a single payload value corresponding to whether there was a hit or not

// one miss program for shadow rays
extern "C" __global__ void __miss__shadow() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);
    optixSetPayload_0(0);
}

// one closest-hit program for shadow rays
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1);
    optixSetPayload_0(1);
}