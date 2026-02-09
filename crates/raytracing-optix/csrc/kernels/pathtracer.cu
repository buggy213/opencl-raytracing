#define USE_PATHTRACER_PIPELINE_PARAMS
#include "kernel_params.h"

#include <optix_device.h>

#include "types.h"
#include "kernel_types.h"
#include "kernel_math.h"

#include "camera.h"
#include "materials.h"
#include "sample.h"

enum RayType
{
    RADIANCE_RAY,
    SHADOW_RAY,
    RAY_TYPE_COUNT
};

// primary entry point
// uses radiance ray type payload only to read out reported radiance from primary (camera) ray
extern "C" __global__ void __raygen__main() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    const Scene& scene = pipeline_params.scene;
    Ray ray = generate_ray(*scene.camera, tid.x, tid.y);

    uint x, y, z;
    optixTrace(
        OPTIX_PAYLOAD_TYPE_ID_0,
        pipeline_params.root_handle,
        ray.origin,
        ray.direction,
        scene.camera->near_clip,
        scene.camera->far_clip,
        0.0f,
        (OptixVisibilityMask)(-1),
        OPTIX_RAY_FLAG_NONE,
        RADIANCE_RAY,
        RAY_TYPE_COUNT,
        RADIANCE_RAY,
        x,
        y,
        z
    );

    float4& target = pipeline_params.radiance[tid.y * dim.x + tid.x];
    target = make_float4(
        __uint_as_float(x),
        __uint_as_float(y),
        __uint_as_float(z),
        1.0f
    );
}

// -- Radiance Rays
// The radiance ray type uses OPTIX_PAYLOAD_TYPE_ID_0,
// which contains the following fields
// - radiance: float3

// one miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    // todo: environment map
    optixSetPayload_0(__float_as_uint(1.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}

// one closest-hit program per material
// `optixGetHitKind` / `optixGetPrimitiveType` is used to distinguish between different primitives
extern "C" __global__ void __closesthit__radiance_diffuse() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);
    // todo

    float t = optixGetRayTmax();
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(__float_as_uint(1.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
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