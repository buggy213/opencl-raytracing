#ifndef RT_GEOMETRY
#define RT_GEOMETRY
typedef float16 matrix4x4_t;

typedef struct {
    matrix4x4_t m;
    matrix4x4_t inverse;
} transform;

typedef struct {
    float3 origin;
    float3 direction;
} ray_t;

matrix4x4_t transpose(matrix4x4_t m) {
    return m.s048c159d26ae37bf;
}

matrix4x4_t matmul(matrix4x4_t a, matrix4x4_t b) {
    float4 a1t = a.s0123;
    float4 a2t = a.s4567;
    float4 a3t = a.s89ab;
    float4 a4t = a.scdef;
    float4 b1 = b.s048c;
    float4 b2 = b.s159d;
    float4 b3 = b.s26ae;
    float4 b4 = b.s37bf;
    return (float16) (
        dot(a1t, b1), dot(a1t, b2), dot(a1t, b3), dot(a1t, b4),
        dot(a2t, b1), dot(a2t, b2), dot(a2t, b3), dot(a2t, b4),
        dot(a3t, b1), dot(a3t, b2), dot(a3t, b3), dot(a3t, b4),
        dot(a4t, b1), dot(a4t, b2), dot(a4t, b3), dot(a4t, b4)
    );
}

float3 apply_transform_vector(transform t, float3 v) {
    return (float3) (
        dot(t.m.s012, v),
        dot(t.m.s456, v),
        dot(t.m.s89a, v)
    );
}

float3 apply_transform_point(transform t, float3 p) {
    float4 tmp_p = (float4) (p.x, p.y, p.z, 1.0f);
    float3 tmp = (float3) (
        dot(t.m.s0123, tmp_p),
        dot(t.m.s4567, tmp_p),
        dot(t.m.s89ab, tmp_p)
    );

    float w = dot(t.m.scdef, tmp_p);
    if (w == 1.0f) {
        return tmp;
    }
    else {
        return tmp / w;
    }
}

// reflects a vector about a normal vector
inline float3 reflect(float3 vec, float3 normal) {
    return vec - 2 * dot(vec, normal) * normal;
}


bool ray_triangle_intersect(float3 v0, float3 v1, float3 v2, ray_t ray, float t_min, float t_max, float3* tuv) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    
    float3 P = cross(ray.direction, e2);
    float denom = dot(P, e1);

    if (denom > -FLT_EPSILON && denom < FLT_EPSILON) {
        return false;
    }

    float3 T = ray.origin - v0;
    float u = dot(P, T) / denom;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 Q = cross(T, e1);
    float v = dot(Q, ray.direction) / denom;

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = dot(Q, e2) / denom;
    if (t < t_min || t > t_max) {
        return false;
    }

    *tuv = (float3) (t, u, v);
    return true;
}

float3 ray_at(ray_t r, float t) {
    return r.origin + t * r.direction;
}

typedef struct {
    bool hit;
    float3 tuv;
    float3 point;
    float3 tangent;
    float3 normal;
    uint tri_idx;
} hit_info_t;

typedef struct {
    float2 uv;
    float3 p;
    // 3 orthonormal vectors define local coordinate system, with normal as 3rd basis vector
    // world to local is vstack(tangent, bitangent, normal)
    // local to world is transpose of the above
    float3 normal;
    float3 tangent;
    float3 bitangent;
} surface_interaction_t;

surface_interaction_t surface_interaction_from_hit(hit_info_t hit) {
    surface_interaction_t interaction = {
        .uv = (float2) (0.0f, 0.0f),
        .p = hit.point,
        .normal = hit.normal,
        .tangent = hit.tangent,
        .bitangent = cross(normal, tangent)
    };

    return interaction;
}

#endif