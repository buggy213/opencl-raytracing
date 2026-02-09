#include "accel.h"
#include "kernel_params.h"
#include "lights.h"
extern "C" __constant__ PathtracerPipelineParams pipeline_params;

#include <optix_device.h>

#include "types.h"
#include "kernel_types.h"
#include "kernel_math.h"

#include "camera.h"
#include "materials.h"
#include "geometry.h"
#include "sample.h"
#include "sbt.h"

enum RayType
{
    RADIANCE_RAY,
    SHADOW_RAY,
    RAY_TYPE_COUNT
};

struct PerRayData {
    sample::OptixSampler sampler;
};

static_assert(sizeof(PerRayData) == sizeof(PathtracerPerRayData), "pathtracer per-ray data size mismatch");
static_assert(alignof(PerRayData) == alignof(PathtracerPerRayData), "pathtracer per-ray data alignment mismatch");

inline __device__ PerRayData& get_ray_data() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    return (reinterpret_cast<PerRayData*>(pipeline_params.ray_datas))[tid.y * dim.x + tid.x];
}

// returns ray in world-space, intended to be called from closest-hit program
inline __device__ Ray get_ray()
{
    return Ray {
        .origin = optixGetWorldRayOrigin(),
        .direction = optixGetWorldRayDirection()
    };
}

// primary entry point
// uses radiance ray type payload only to read out reported radiance from primary (camera) ray
extern "C" __global__ void __raygen__main() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    const Scene& scene = pipeline_params.scene;

    // initialize per-ray data with sampler
    PerRayData& prd = get_ray_data();
    prd.sampler = sample::OptixSampler::from_sampler(pipeline_params.settings.sampler, pipeline_params.settings.seed);

    Ray ray = generate_ray(*scene.camera, tid.x, tid.y);

    uint r, g, b;
    uint r_weight, g_weight, b_weight;
    r_weight = g_weight = b_weight = __float_as_uint(1.0f);
    uint specular_bounce = 1;
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
        r,
        g,
        b,
        r_weight,
        g_weight,
        b_weight,
        specular_bounce
    );

    float4& target = pipeline_params.radiance[tid.y * dim.x + tid.x];
    target = make_float4(
        __uint_as_float(r),
        __uint_as_float(g),
        __uint_as_float(b),
        1.0f
    );
}

// -- Radiance Rays
// The radiance ray type uses OPTIX_PAYLOAD_TYPE_ID_0,
// which contains the following fields
// - radiance: float3. written in closest-hit / miss, read out after trace
// - path weight: float3. written before trace, read out in closest-hit / miss
// - specular bounce: bool. written before trace, read out in closest-hit / miss

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

// should only be called from closest-hit programs
inline __device__ const Material& get_material_data() {
    auto* material_ptr = reinterpret_cast<const Material*>(optixGetSbtDataPointer() + sizeof(HitgroupRecord::MeshData));
    return *material_ptr;
}

extern "C" __global__ void __closesthit__radiance_diffuse() {
    optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0);

    const Scene& scene = pipeline_params.scene;
    const OptixRaytracerSettings& settings = pipeline_params.settings;

    Ray ray = get_ray();
    HitInfo hit {};
    unsigned int hit_kind = optixGetHitKind();
    OptixPrimitiveType geometry = optixGetPrimitiveType(hit_kind);

    switch (geometry) {
        case OPTIX_PRIMITIVE_TYPE_SPHERE:
            hit = get_hit_info_sphere();
            break;
        case OPTIX_PRIMITIVE_TYPE_TRIANGLE:
            hit = get_hit_info_tri();
            break;
        default:
            // TODO: handle more geometry types
            optixSetPayload_0(__float_as_uint(0.0f));
            optixSetPayload_1(__float_as_uint(0.0f));
            optixSetPayload_2(__float_as_uint(0.0f));
            return;
    }

    unsigned int specular_bounce = optixGetPayload_6();
    float3 radiance = float3_zero;
    float3 path_weight = make_float3(
        __uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5())
    );

    // TODO: direct lighting from intersection with light

    const Material& material = get_material_data();
    float3 o2w_x, o2w_y;
    cuda::std::tie(o2w_x, o2w_y) = geometry::make_orthonormal_basis(hit.normal);

    // so hopefully OptiX can optimize this lol. it seems especially problematic for both matrices
    // to be live across direct lighting / indirect lighting optixTrace.....
    // TODO: is this a perf issue?
    Matrix4x4 o2w = {
        o2w_x.x, o2w_y.x, hit.normal.x, 0.0f,
        o2w_x.y, o2w_y.y, hit.normal.y, 0.0f,
        o2w_x.z, o2w_y.z, hit.normal.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    Matrix4x4 w2o = {
        o2w_x.x, o2w_x.y, o2w_x.z, 0.0f,
        o2w_y.x, o2w_y.y, o2w_y.z, 0.0f,
        hit.normal.x, hit.normal.y, hit.normal.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    float3 wo = matrix4x4_apply_vector(w2o, -ray.direction);
    if (optixGetRemainingTraceDepth() == 0)
    {
        optixSetPayload_0(__float_as_uint(radiance.x));
        optixSetPayload_1(__float_as_uint(radiance.y));
        optixSetPayload_2(__float_as_uint(radiance.z));
        return;
    }

    materials::OptixBsdfDiffuse bsdf = materials::get_diffuse_bsdf(material, hit.uv);

    bool delta_bsdf = materials::OptixBsdfDiffuse::is_delta_bsdf();
    bool add_direct_illumination = settings.accumulate_bounces || optixGetRemainingTraceDepth() == 1;
    if (!delta_bsdf && add_direct_illumination)
    {
        float3 direct_illumination = float3_zero;
        for (int light_idx = 0; light_idx < scene.num_lights; light_idx += 1)
        {
            float3 light_contribution = float3_zero;
            unsigned int light_samples = lights::is_delta_light(scene.lights[light_idx]) ? 1 : settings.light_sample_count;
            for (int sample_idx = 0; sample_idx < light_samples; sample_idx += 1)
            {
                unsigned int occluded;
                optixTrace(
                    OPTIX_PAYLOAD_TYPE_ID_1,
                    pipeline_params.root_handle,
                    float3_zero,
                    float3_zero,
                    )
                if (!occluded)
                {

                }
            }

            light_contribution /= light_samples;
            direct_illumination += light_contribution;
        }

        radiance += path_weight * direct_illumination;
    }


    // optixSetPayload_0(__float_as_uint(t));
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