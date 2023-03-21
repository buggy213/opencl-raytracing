/* #region geometry */
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

/* #endregion */

// #region utilities
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
// #endregion

typedef struct {
    float3 origin;
    float3 direction;
} ray_t;

ray_t generate_ray(uint2 *rng_state, int x, int y, transform raster_to_camera) {
    float x_disp = (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float y_disp = (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float3 raster_loc = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);

    float3 camera_loc = apply_transform_point(raster_to_camera, raster_loc);

    // int i = get_global_id(0);
    // int j = get_global_id(1);
    // if (i == 0 && j == 0) {
    //     printf("%v3f\n", camera_loc);
    // }

    float3 camera_dir = camera_loc / length(camera_loc);

    ray_t r = {
        .origin = (float3) (0.0f, 0.0f, 0.0f),
        .direction = camera_dir
    };

    return r;
}

bool ray_triangle_intersect(float3 v0, float3 v1, float3 v2, ray_t ray, float t_min, float t_max, float3* tuv) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    
    float3 P = cross(ray.direction, e2);
    float denom = dot(P, e1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (denom > -FLT_EPSILON && denom < FLT_EPSILON) {
        if (i == 0 && j == 0) {
            printf("parallel");
        }
        return false;
    }

    float3 T = ray.origin - v0;
    float u = dot(P, T) / denom;
    if (u < 0.0f || u > 1.0f) {
        if (i == 0 && j == 0) {
            printf("oob (u)");
        }
        return false;
    }

    float3 Q = cross(T, e1);
    float v = dot(Q, ray.direction) / denom;

    if (v < 0.0f || u + v > 1.0f) {
        if (i == 0 && j == 0) {
            printf("oob (v)");
        }
        return false;
    }

    float t = dot(Q, e2) / denom;
    if (t < t_min || t > t_max) {
        if (i == 0 && j == 0) {
            printf("invalid t %d", t);
        }
        return false;
    }

    *tuv = (float3) (t, u, v);
    return true;
}

float3 ray_color(ray_t ray) {
    // float3 v0 = (float3) (-0.4969f, 0.8404f, 4.033f);
    // float3 v1 = (float3) (-0.2648f, -0.0854f, 4.777f);
    // float3 v2 = (float3) (0.7556f, -0.3176f, 4.169f);

    // float3 v0 = (float3) (-0.2648f, -0.3903f, 3.438f);
    // float3 v1 = (float3) (0.7756f, -0.6231f, 4.046f);
    // float3 v2 = (float3) (-0.4969f, 0.5349f, 4.182f);

    float3 v0 = (float3) (-0.5f, -0.5f, 10.0f);
    float3 v1 = (float3) (0.5f, -0.5f, 4.0f);
    float3 v2 = (float3) (-0.5f, 0.5f, 4.0f);
    float3 tuv;
    if (ray_triangle_intersect(v0, v1, v2, ray, 0.01, INFINITY, &tuv)) {
        return tuv;
    }

    float3 normalized_direction = normalize(ray.direction);
    float t = 0.5f * (normalized_direction.y + 1.0f);
    return (1.0f - t) * (float3) (1.0f, 1.0f, 1.0f) + t * (float3) (0.5f, 0.7f, 1.0f);
}

void __kernel render(
    __global float* frame_buffer, 
    int frame_width,
     int frame_height,
    int num_samples, 
    __global uint* seeds_buffer,
    __global float* raster_to_camera_buf
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= frame_width || j >= frame_height) {
        return;
    }

    int pixel_index = (j * frame_width + i) * 3;
    float u = (float) i / frame_width;
    float v = (float) j / frame_height;

    uint2 seed = (uint2) (seeds_buffer[(j * frame_width + i) * 2], seeds_buffer[(j * frame_width + i) * 2 + 1]);

    __local transform raster_to_camera_transform;
    int k = get_local_id(0);
    int l = get_local_id(0);
    if (k == 0 && l == 0) {
        raster_to_camera_transform = (transform) {
            .m = vload16(0, raster_to_camera_buf + 16),
            .inverse = vload16(0, raster_to_camera_buf)
        };
        // if (i == 0 && j == 0) {
        //     for (int i = 0; i < 32; i += 1) {
        //         printf("%f\n", raster_to_camera_buf[i]);
        //     }

        //     printf("%v16f\n", raster_to_camera_transform.m);
        //     printf("%v16f\n", raster_to_camera_transform.inverse);
        // }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    ray_t ray = generate_ray(&seed, i, j, raster_to_camera_transform);
    if ((i == 0 && j == 0) || (i == 511 && j == 511)) {
        printf("%v3f\n", ray.origin);
        printf("%v3f\n", ray.direction);
    }

    float3 color = ray_color(ray);

    frame_buffer[pixel_index] = color.r;
    frame_buffer[pixel_index + 1] = color.g;
    frame_buffer[pixel_index + 2] = color.b;    
}