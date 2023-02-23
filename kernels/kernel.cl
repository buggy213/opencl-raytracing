typedef struct {
    float3 origin;
    float3 direction;
} ray_t;

typedef float16 matrix4x4;

typedef struct {
    matrix4x4 m;
    matrix4x4 inverse;
} transform;

matrix4x4 transpose(matrix4x4 m) {
    return m.s048c159d26ae37bf;
}

// only works for matrices of the form
// [ux vx wx tx]
// [uy vy wy ty]
// [uz vz wz tz]
// [ 0  0  0  1]
matrix4x4 invert_transform_matrix(matrix4x4 m) {
    // [ux uy uz -dot(u,t)]
    // [vx vy vz -dot(v,t)]
    // [wx wy wz -dot(w,t)]
    // [ 0  0  0     1    ]
    matrix4x4 inverse;
    inverse.s012 = m.s048;
    inverse.s456 = m.s159;
    inverse.s89a = m.s26a;
    inverse.scdef = (float4) (0.0f, 0.0f, 0.0f, 1.0f);
    inverse.s3 = dot(m.s048, m.s37b);
    inverse.s7 = dot(m.s159, m.s37b);
    inverse.sb = dot(m.s26a, m.s37b);
    return inverse;
}

matrix4x4 matmul(matrix4x4 a, matrix4x4 b) {
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

transform look_at(float3 position, float3 look, float3 up) {
    float3 direction = normalize(look - position);
    float3 right = normalize(cross(normalize(up), direction));
    float3 new_up = cross(direction, right);

    matrix4x4 m;
    m.s37b = position;
    m.scdef = (float4) (0.0f, 0.0f, 0.0f, 1.0f);
    m.s048 = right;
    m.s159 = new_up;
    m.s26a = direction;
    transform t = {
        .m = m,
        .inverse = invert_transform_matrix(m)
    };
}

void generate_ray() {

}

typedef struct {
    transform world_to_object;
    float radius;
} sphere;

void __kernel render(__global float* frame_buffer, int frame_width, int frame_height, 
    __constant transform* raster_to_world, __constant sphere* spheres, int num_spheres) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= frame_width || j >= frame_height) {
        return;
    }

    int pixel_index = (j * frame_width + i) * 3;
    float u = (float) i / frame_width;
    float v = (float) j / frame_height;

    frame_buffer[pixel_index] = u;
    frame_buffer[pixel_index + 1] = v;
    frame_buffer[pixel_index + 2] = 0.2f;
}