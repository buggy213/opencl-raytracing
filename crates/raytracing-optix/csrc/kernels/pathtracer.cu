#include "pathtracer.hpp"

#include <optix_device.h>

#include "types.h"
#include "kernel_types.hpp"
#include "kernel_math.hpp"

#include "camera.hpp"
#include "materials.hpp"
#include "geometry.hpp"
#include "sample.hpp"
#include "sbt.hpp"
#include "accel.hpp"
#include "lights.hpp"

// returns ray in world-space, intended to be called from closest-hit program
inline __device__ Ray get_ray()
{
    return Ray {
        .origin = optixGetWorldRayOrigin(),
        .direction = optixGetWorldRayDirection()
    };
}

// @raytracing_cpu::ray_radiance
inline __device__ float3 ray_radiance(Ray camera_ray)
{
    const Scene& scene = pipeline_params.scene;

    Ray ray = camera_ray;
    u32 depth = 0;

    bool specular_bounce = true;
    float3 radiance = float3_zero;
    float3 path_weight = make_float3(1.0f, 1.0f, 1.0f);

    while (true)
    {
        // TODO: try use +infty instead? not sure if it makes a difference
        float t_min = depth == 0 ? scene.camera->near_clip : 0.0001f;
        float t_max = depth == 0 ? scene.camera->far_clip : pipeline_params.scene_diameter;

        RadianceRayPayloadTraceWrite payload_in = {
            path_weight,
            specular_bounce,
            depth
        };

        RadianceRayPayloadTraceRead payload_out = traceRadianceRay(
            ray.origin,
            ray.direction,
            t_min,
            t_max,
            payload_in
        );

        depth += 1;
        radiance += payload_out.radiance;

        // this can cause divergence
        // 1. miss shader causes writes done = true, causes some threads in warp to finish earlier than others.
        //    in theory, the compiler already has enough information to deal with this case
        // 2. it's the last bounce, and so closest-hit will write done = true. i don't think the compiler can reason
        //    about this one, though if it never reorders anyways, then this won't be a problem since depth is uniform
        //    across the warp
        // 3. closest-hit writes done = true because an invalid sample was drawn or the path weight falls to zero.
        //    this case is hardest to deal with, since it's hard to know a priori before invoking CH whether
        //    or not the ray will finish. if we hoist the russian-roulette logic out of CH, maybe it's easier?

        // TODO: try out SER to fix this? SER can maybe help with
        //  1. divergence caused by hit vs miss shader
        //  2. divergence caused by switch over geometry in closest-hit shaders
        //  3. divergence from last bounce
        //  4. divergence from russian-roulette / other ways path is terminated early
        //     - question: if some threads already exited loop, do they participate in reorder? how does this actually work under the hood??
        //  5. spatial divergence in hit shaders
        //  however, it seems unlikely that SER can resolve divergence caused by lights being occluded and the bsdf evaluation
        //  that occurs as a result without major restructuring of the code, and the overhead is probably too high anyways
        if (payload_out.done)
        {
            break;
        }

        path_weight = payload_out.path_weight;
        specular_bounce = payload_out.specular_bounce;
        ray.origin = payload_out.origin_w;
        ray.direction = payload_out.dir_w;
    }

    return radiance;
}

// primary entry point, roughly corresponds to inner loop for one pixel of
// @raytracing_cpu::render_tile
extern "C" __global__ void __raygen__main() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    const Scene& scene = pipeline_params.scene;
    const OptixRaytracerSettings& settings = pipeline_params.settings;

    float3 radiance = float3_zero;

    // initialize per-ray data with sampler
    PerRayData& prd = get_ray_data();
    prd.sampler = sample::OptixSampler::from_sampler(settings.sampler, settings.seed);

    for (int sample_index = 0; sample_index < settings.samples_per_pixel; sample_index += 1)
    {
        prd.sampler.start_sample(make_uint2(tid.x, tid.y), sample_index);

        Ray camera_ray = generate_ray(*scene.camera, tid.x, tid.y, &prd.sampler);

        radiance += ray_radiance(camera_ray);
    }

    radiance /= settings.samples_per_pixel;

    float4& target = pipeline_params.radiance[tid.y * dim.x + tid.x];
    target = make_float4(
        radiance.x,
        radiance.y,
        radiance.z,
        1.0f
    );
}


// one miss program for radiance rays
extern "C" __global__ void __miss__radiance() {
    optixSetPayloadTypes(RADIANCE_RAY_PAYLOAD_TYPE);

    // todo: environment map
    RadianceRayPayloadMSWrite payload_out = {};
    payload_out.done = true;
    payload_out.radiance = make_float3(0.0f, 0.0f, 0.0f);
    writeRadiancePayloadMS(payload_out);
}

// one closest-hit program per material
// `optixGetHitKind` / `optixGetPrimitiveType` is used to distinguish between different primitives

// should only be called from closest-hit programs
inline __device__ const Material& get_material_data() {
    auto* material_ptr = reinterpret_cast<const Material*>(optixGetSbtDataPointer() + sizeof(HitgroupRecord::MeshData));
    return *material_ptr;
}

extern "C" __global__ void __closesthit__radiance_diffuse() {
    optixSetPayloadTypes(RADIANCE_RAY_PAYLOAD_TYPE);

    const Scene& scene = pipeline_params.scene;
    const OptixRaytracerSettings& settings = pipeline_params.settings;

    Ray ray = get_ray();
    PerRayData& prd = get_ray_data();
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
            RadianceRayPayloadCHWrite payload_out = {};
            payload_out.done = true;
            payload_out.radiance = float3_zero;
            writeRadiancePayloadCH(payload_out);
            return;
    }

    RadianceRayPayloadCHRead payload_in = readRadiancePayloadCH();
    auto [path_weight, specular_bounce, depth] = payload_in;
    float3 radiance = float3_zero;


    bool add_zero_bounce = settings.accumulate_bounces || settings.max_ray_depth == depth;
    if (specular_bounce && add_zero_bounce && hit.area_light)
    {
        const Light& light = scene.lights[*hit.area_light];
        radiance += path_weight * lights::light_radiance(light);
    }

    const Material& material = get_material_data();
    float3 o2w_x, o2w_y;
    cuda::std::tie(o2w_x, o2w_y) = geometry::make_orthonormal_basis(hit.normal);

    // so hopefully OptiX can optimize this lol. it seems especially problematic for both matrices
    // to be live across direct lighting optixTrace... rematerializing might be more performant
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
    // some divergence, see note in raygen program
    depth += 1;
    if (depth > settings.max_ray_depth)
    {
        RadianceRayPayloadCHWrite payload_out = {};
        payload_out.radiance = radiance;
        payload_out.done = true;
        writeRadiancePayloadCH(payload_out);
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
            const Light& light = scene.lights[light_idx];
            float3 light_contribution = float3_zero;
            unsigned int light_samples = lights::is_delta_light(light) ? 1 : settings.light_sample_count;
            for (int sample_idx = 0; sample_idx < light_samples; sample_idx += 1)
            {
                lights::LightSample light_sample = lights::sample_light(light, hit.point, prd.sampler);
                // traces a shadow ray
                bool occluded = lights::occluded(light_sample);

                if (!occluded)
                {
                    float3 wi = matrix4x4_apply_vector(w2o, -light_sample.shadow_ray.direction);
                    float3 bsdf_value = materials::evaluate_bsdf(bsdf, wo, wi);

                    float cos_theta = wi.z;
                    light_contribution += bsdf_value * light_sample.radiance * fmax(0.0f, cos_theta) / light_sample.pdf;
                }
            }

            light_contribution /= light_samples;
            direct_illumination += light_contribution;
        }

        radiance += path_weight * direct_illumination;
    }

    materials::BsdfSample bsdf_sample = materials::sample_bsdf(bsdf, wo, materials::BsdfComponentFlags::ALL(), prd.sampler);

    // some divergence: see note in raygen program
    if (!bsdf_sample.valid || bsdf_sample.bsdf == float3_zero || bsdf_sample.pdf == 0.0f)
    {
        RadianceRayPayloadCHWrite payload_out = {};
        payload_out.radiance = radiance;
        payload_out.done = true;
        writeRadiancePayloadCH(payload_out);
        return;
    }

    float cos_theta = fabs(bsdf_sample.wi.z);
    path_weight *= bsdf_sample.bsdf * cos_theta / bsdf_sample.pdf;
    specular_bounce = bsdf_sample.component.is_specular();

    float3 world_dir = matrix4x4_apply_vector(o2w, bsdf_sample.wi);
    auto new_ray = Ray {
        .origin = hit.point,
        .direction = world_dir
    };

    RadianceRayPayloadCHWrite payload_out = {
        radiance,
        path_weight,
        specular_bounce,
        .done = false,
        new_ray.origin,
        new_ray.direction,
    };

    writeRadiancePayloadCH(payload_out);
}

// one miss program for shadow rays
extern "C" __global__ void __miss__shadow() {
    optixSetPayloadTypes(SHADOW_RAY_PAYLOAD_TYPE);
    ShadowRayPayload payload_out = {
        .hit = false
    };

    writeShadowPayload(payload_out);
}

// one closest-hit program for shadow rays
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayloadTypes(SHADOW_RAY_PAYLOAD_TYPE);
    ShadowRayPayload payload_out = {
        .hit = true
    };

    writeShadowPayload(payload_out);
}