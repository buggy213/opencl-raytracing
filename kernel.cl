/*typedef struct {
    float3 origin;
    float3 direction;
} ray_t;*/
/*
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

    return t;
}

float3 apply_transform_vector(transform t, float3 v) {
    return (float3) (
        dot(t.m.s012, v),
        dot(t.m.s456, v),
        dot(t.m.s89a, v)
    );
}
*/
/*
// copied from https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
} // how does this work ???

ray_t generate_ray(uint2 *rng_state, int x, int y, int frame_width, int frame_height,
                   float view_width, float view_height, float focal_length) {
    float x_disp = MWC64X(rng_state);
    float y_disp = MWC64X(rng_state);
    float3 direction = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);
    direction.x = ((direction.x / frame_width) - 0.5f) * view_width;
    direction.y = ((direction.y / frame_height) - 0.5f) * view_height;
    direction.z = -focal_length;

    ray_t r = {
        .origin = (float3) (0.0f, 0.0f, 0.0f),
        .direction = direction
    };

    return r;
}
*/
/*
float3 ray_color(ray_t ray) {
    float3 normalized_direction = normalize(ray.direction);
    float t = 0.5f * (normalized_direction.y + 1.0f);
    return (1.0f - t) * (float3) (1.0f, 1.0f, 1.0f) + t * (float3) (0.5f, 0.7f, 1.0f);
}
*/
/*
typedef struct {
    transform world_to_object;
    float radius;
} sphere;
*/

void __kernel render(
    __global float* frame_buffer, int frame_width, int frame_height
    // __constant transform* raster_to_world, 
    // __constant sphere* spheres, int num_spheres, 
    // int num_samples, __global uint* seeds_buffer
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= frame_width || j >= frame_height) {
        return;
    }

    frame_buffer[0] = 1.0f;

    int pixel_index = (j * frame_width + i) * 3;
    float u = (float) i / frame_width;
    float v = (float) j / frame_height;

    frame_buffer[pixel_index] = u;
    frame_buffer[pixel_index + 1] = v;
    frame_buffer[pixel_index + 2] = 0.2f;

    // uint2 seed = (uint2) (seeds_buffer[(j * frame_width + i) * 2], seeds_buffer[(j * frame_width + i) * 2 + 1]);
    // ray_t ray = generate_ray(&seed, i, j, frame_width, frame_height, 3.55f, 2.0f, 1.0f);
    // float3 color = ray_color(ray);
}