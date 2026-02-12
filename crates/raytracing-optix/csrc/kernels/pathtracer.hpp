#pragma once

#include "kernel_types.hpp"
#include "sample.hpp"

#ifdef USE_AOV_PIPELINE_PARAMS
#error mixing aov and pathtracer code not allowed
#endif

#define USE_PATHTRACER_PIPELINE_PARAMS
#include "kernel_params.hpp"
#
extern "C" __constant__ PathtracerPipelineParams pipeline_params;

/* TODO: in the case of fully opaque geometry + no volumes, can possibly do better by removing SHADOW_RAY, associated payload, and optixTrace call - just do optixTraverse */
enum RayType
{
    RADIANCE_RAY,
    SHADOW_RAY,
    RAY_TYPE_COUNT
};

inline __device__ OptixPayloadTypeID payloadTypeFromRayType(RayType ray_type)
{
    switch (ray_type)
    {
    case RADIANCE_RAY:
        return OPTIX_PAYLOAD_TYPE_ID_0;
    case SHADOW_RAY:
        return OPTIX_PAYLOAD_TYPE_ID_1;
    default:
        printf("bad ray type %d in payloadTypeFromRayType\n", ray_type);
        return OPTIX_PAYLOAD_TYPE_ID_0;
    }
}

// global memory per ray
// TODO: this is wasteful, every ray has a copy of the whole sampler configuration - they only need sampler state, and that can go in payload possibly
struct PerRayData {
    sample::OptixSampler sampler;
};

static_assert(sizeof(PerRayData) == sizeof(PathtracerPerRayData), "pathtracer per-ray data size mismatch");
static_assert(alignof(PerRayData) == alignof(PathtracerPerRayData), "pathtracer per-ray data alignment mismatch");

// payload passed between rays through registers (optixGetPayload_*, optixSetPayload_*)

/*
 * Radiance Rays
 * The radiance ray type uses OPTIX_PAYLOAD_TYPE_ID_0,
 * which contains the following fields
 * - radiance: float3. written in closest-hit / miss, read out after trace
 * - path weight: float3. written before trace, read out in closest-hit / miss, updated by closest-hit
 * - specular bounce: bool. written before trace, read out in closest-hit / miss, updated by closest-hit
 * - done: bool. written by closest-hit / miss, read out after trace. indicates that this ray is finished
 *   (either because it missed the scene, or russian roulette, or an invalid bsdf sample was drawn, or whatever)
 * - origin_w: float3. written by closest-hit, read out after trace; only valid if done is true
 * - origin_d: float3. written by closest-hit, read out after trace; only valid if done is true
 * - depth: uint. written by caller of trace, read out in closest-hit to inform shading decision
 */
struct RadianceRayPayloadTraceWrite
{
    float3 path_weight;
    bool specular_bounce;
    u32 depth;
};

struct RadianceRayPayloadTraceRead
{
    float3 radiance;
    float3 path_weight;
    bool specular_bounce;
    bool done;
    float3 origin_w;
    float3 dir_w;
};

struct RadianceRayPayloadCHWrite
{
    float3 radiance;
    float3 path_weight;
    bool specular_bounce;
    bool done;
    float3 origin_w;
    float3 dir_w;
};

struct RadianceRayPayloadCHRead
{
    float3 path_weight;
    bool specular_bounce;
    u32 depth;
};

struct RadianceRayPayloadMSWrite
{
    float3 radiance;
    bool done;
};

struct RadianceRayPayloadMSRead
{
    float3 path_weight;
    bool specular_bounce;
};

inline __device__ RadianceRayPayloadCHRead readRadiancePayloadCH()
{
    return RadianceRayPayloadCHRead {
        .path_weight = make_float3(
            __uint_as_float(optixGetPayload_3()),
            __uint_as_float(optixGetPayload_4()),
            __uint_as_float(optixGetPayload_5())
        ),
        .specular_bounce = static_cast<bool>(optixGetPayload_6()),
        .depth = optixGetPayload_14()
    };
}

inline __device__ void writeRadiancePayloadCH(const RadianceRayPayloadCHWrite& payload)
{
    auto [radiance_r, radiance_g, radiance_b] = payload.radiance;
    optixSetPayload_0(__float_as_uint(radiance_r));
    optixSetPayload_1(__float_as_uint(radiance_g));
    optixSetPayload_2(__float_as_uint(radiance_b));

    auto [path_weight_r, path_weight_g, path_weight_b] = payload.path_weight;
    optixSetPayload_3(__float_as_uint(path_weight_r));
    optixSetPayload_4(__float_as_uint(path_weight_g));
    optixSetPayload_5(__float_as_uint(path_weight_b));

    optixSetPayload_6(payload.specular_bounce);

    optixSetPayload_7(payload.done);

    auto [origin_x, origin_y, origin_z] = payload.origin_w;
    optixSetPayload_8(__float_as_uint(origin_x));
    optixSetPayload_9(__float_as_uint(origin_y));
    optixSetPayload_10(__float_as_uint(origin_z));

    auto [dir_x, dir_y, dir_z] = payload.dir_w;
    optixSetPayload_11(__float_as_uint(dir_x));
    optixSetPayload_12(__float_as_uint(dir_y));
    optixSetPayload_13(__float_as_uint(dir_z));
}

inline __device__ RadianceRayPayloadMSRead readRadiancePayloadMS()
{
    return RadianceRayPayloadMSRead {
        .path_weight = make_float3(
            __uint_as_float(optixGetPayload_3()),
            __uint_as_float(optixGetPayload_4()),
            __uint_as_float(optixGetPayload_5())
        ),
        .specular_bounce = static_cast<bool>(optixGetPayload_6()),
    };
}

inline __device__ RadianceRayPayloadTraceRead traceRadianceRay(
    float3 ray_o,
    float3 ray_d,
    float t_min, float t_max,
    const RadianceRayPayloadTraceWrite& payload_in
) {
    u32 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14;

    p3 = __float_as_uint(payload_in.path_weight.x);
    p4 = __float_as_uint(payload_in.path_weight.y);
    p5 = __float_as_uint(payload_in.path_weight.z);
    p6 = payload_in.specular_bounce;
    p14 = payload_in.depth;

    optixTrace(
        payloadTypeFromRayType(RayType::RADIANCE_RAY),
        pipeline_params.root_handle,
        ray_o,
        ray_d,
        t_min,
        t_max,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        RayType::RADIANCE_RAY,
        RayType::RAY_TYPE_COUNT,
        RayType::RADIANCE_RAY,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14
    );

    float3 out_radiance = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );

    float3 out_path_weight = float3_zero;
    bool out_specular_bounce = false;
    float3 out_origin_w = float3_zero;
    float3 out_dir_w = float3_zero;
    bool out_done = static_cast<bool>(p7);
    if (!out_done)
    {
        out_path_weight = make_float3(
            __uint_as_float(p3),
            __uint_as_float(p4),
            __uint_as_float(p5)
        );

        out_specular_bounce = static_cast<bool>(p6);
        out_origin_w = make_float3(
            __uint_as_float(p8),
            __uint_as_float(p9),
            __uint_as_float(p10)
        );

        out_dir_w = make_float3(
            __uint_as_float(p11),
            __uint_as_float(p12),
            __uint_as_float(p13)
        );
    }

    return RadianceRayPayloadTraceRead {
        out_radiance,
        out_path_weight,
        out_specular_bounce,
        out_done,
        out_origin_w,
        out_dir_w
    };
}

// Shadow Rays
// The shadow ray type uses OPTIX_PAYLOAD_TYPE_ID_1,
// which defines a single payload value corresponding to whether there was a hit or not
// trace caller reads, CH / MS write

struct ShadowRayPayload
{
    bool hit;
};

inline __device__ void writeShadowPayload(const ShadowRayPayload& payload)
{
    optixSetPayload_0(payload.hit);
}

inline __device__ ShadowRayPayload traceShadowRay(
    float3 ray_o,
    float3 ray_d,
    float t_min,
    float t_max
) {
    u32 p0;
    optixTrace(
        payloadTypeFromRayType(RayType::SHADOW_RAY),
        pipeline_params.root_handle,
        ray_o,
        ray_d,
        t_min,
        t_max,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RayType::SHADOW_RAY,
        RayType::RAY_TYPE_COUNT,
        RayType::SHADOW_RAY,
        p0
    );

    bool out_hit = static_cast<bool>(p0);
    return ShadowRayPayload { out_hit };
}

inline __device__ PerRayData& get_ray_data() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    return (reinterpret_cast<PerRayData*>(pipeline_params.ray_datas))[tid.y * dim.x + tid.x];
}
