#pragma once

struct Ray {
    float3 origin;
    float3 direction;
};

inline __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float a, float4 b) {
    return b * a;
}

inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator*(float a, float3 b) {
    return b * a;
}

inline __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 normalize(float3 a) {
    float inv_len = rsqrtf(dot(a, a));
    return a * inv_len;
}

// left-multiply by matrix
inline __device__ float3 operator*(const Matrix4x4& transform, float3 vector) {
    float4 v = make_float4(vector.x, vector.y, vector.z, 1.0f);
    float4 a = make_float4(transform.m[0], transform.m[1], transform.m[2], transform.m[3]);
    float4 b = make_float4(transform.m[4], transform.m[5], transform.m[6], transform.m[7]);
    float4 c = make_float4(transform.m[8], transform.m[9], transform.m[10], transform.m[11]);
    float4 d = make_float4(transform.m[12], transform.m[13], transform.m[14], transform.m[15]);

    float va = dot(a, v);
    float vb = dot(b, v);
    float vc = dot(c, v);
    float vd = dot(d, v);

    return make_float3(va / vd, vb / vd, vc / vd);
}