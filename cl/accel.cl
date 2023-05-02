#include "cl/geometry.cl"

#ifndef RT_ACCEL
#define RT_ACCEL
// if triCount is 0, this is index of left BVH child node. index of right BVH child node is leftIndex + 1
// otherwise, if triCount != 0, then this is the index of the first triangle in triangle array
typedef struct {
    float min_x, min_y, min_z;
    uint leftFirst; 
    float max_x, max_y, max_z;
    uint triCount;
} bvh_node_t;

typedef struct {
    __global bvh_node_t* bvh_tree; 
    __global uint* triangles;
    __global float* vertices;
} bvh_data_t;

// returns t at which ray intersections an axis-aligned bounding box, or inf if it does not intersect it
float bvh_intersect(bvh_node_t* node, ray_t ray) {
    // x slab
    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
        // printf("min=%f %f %f\n", node->min_x, node->min_y, node->min_z);
        // printf("max=%f %f %f\n", node->max_x, node->max_y, node->max_z);
        // printf("ray dir=%v3f\n", ray.direction);
    }
    float a = (node->min_x - ray.origin.x) / ray.direction.x;
    float b = (node->max_x - ray.origin.x) / ray.direction.x;
    float t0 = min(a, b);
    float t1 = max(a, b);
    // y slab
    a = (node->min_y - ray.origin.y) / ray.direction.y;
    b = (node->max_y - ray.origin.y) / ray.direction.y;
    t0 = max(t0, min(a, b));
    t1 = min(t1, max(a, b));
    // z slab
    a = (node->min_z - ray.origin.z) / ray.direction.z;
    b = (node->max_z - ray.origin.z) / ray.direction.z;
    t0 = max(t0, min(a, b));
    t1 = min(t1, max(a, b));
    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
        // printf("t0=%f t1=%f\n", t0, t1);
    }
    if (t0 <= t1 && t1 >= 0) {
        return t0;
    }
    else {
        return INFINITY;
    }
}

// todo: consider passing in as SoA instead for better performance
void traverse_bvh(
    ray_t ray,
    float t_min,
    float t_max,
    hit_info_t* hit_info,
    bvh_data_t bvh,
    bool early_exit // exit on first intersected triangle (useful for visibility testing where you don't care which particular one you hit) 
) {
    __global bvh_node_t* bvh_tree = bvh.bvh_tree; 
    __global uint* triangles = bvh.triangles; 
    __global float* vertices = bvh.vertices;

    bvh_node_t* node;
    bvh_node_t* stack[32];
    
    node = &bvh_tree[0];
    uint stack_ptr = 0;
    
    while (true) {
        if (node->triCount > 0) {
            if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                // printf("leaf node, first tri=%d, tri count=%d\n", node->leftFirst, node->triCount);
            }
            // leaf node
            for (uint i = 0; i < node->triCount; i += 1) {
                uint tri_idx = node->leftFirst + i;
                if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                    // printf("intersection test with tri %d\n", tri_idx);
                }
                uint3 tri = (uint3) (triangles[tri_idx * 3], triangles[tri_idx * 3 + 1], triangles[tri_idx * 3 + 2]);
                float3 p0 = (float3) (vertices[tri.s0 * 3], vertices[tri.s0 * 3 + 1], vertices[tri.s0 * 3 + 2]);
                float3 p1 = (float3) (vertices[tri.s1 * 3], vertices[tri.s1 * 3 + 1], vertices[tri.s1 * 3 + 2]);
                float3 p2 = (float3) (vertices[tri.s2 * 3], vertices[tri.s2 * 3 + 1], vertices[tri.s2 * 3 + 2]);
                if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                    // printf("indices= %v3d\n", tri);
                    // printf("p0= %v3f\n", p0);
                    // printf("p1= %v3f\n", p1);
                    // printf("p2= %v3f\n", p2);
                }

                bool result = ray_triangle_intersect(p0, p1, p2, ray, t_min, t_max, &hit_info->tuv);
                if (result) {
                    hit_info->hit = true;
                    hit_info->tri_idx = tri_idx;
                    t_max = hit_info->tuv.s0; // update t_max
                    if (early_exit) {
                        return;
                    }
                    
                    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                        // printf("intersected with tri %d\n", tri_idx);
                    }
                }

                if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                    // printf("intersection test with tri %d complete\n", tri_idx);
                }
            }
            if (stack_ptr == 0) {
                break;
            }
            else {
                if (get_global_id(0) == 255 && get_global_id(1) == 255) {
                    // printf("backtracking from %d\n", stack_ptr);
                }
                // printf("backtracking from %d\n", stack_ptr);
                node = stack[--stack_ptr];
                // printf("finished backtracking\n");
                volatile int x = node->triCount;
                // printf("node deref\n");
            }
        }
        else {
            // inner node
            if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                // printf("inner node\n");
                // printf("%d %d\n", node->leftFirst, node->leftFirst + 1);
            }
            bvh_node_t* left_child = &bvh_tree[node->leftFirst];
            bvh_node_t* right_child = &bvh_tree[node->leftFirst + 1];
            float left = bvh_intersect(left_child, ray);
            float right = bvh_intersect(right_child, ray);
            if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                // printf("%f %f\n", left, right);
            }
            bool swapped = (left > right);
            if (left > right) {
                float tmp = left;
                left = right;
                right = tmp;
                bvh_node_t* tmp_node = left_child;
                left_child = right_child;
                right_child = tmp_node;
            }
            if (isinf(left)) {
                if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                    // printf("no intersects\n");
                }
                if (stack_ptr == 0) {
                    break; // finished traversal
                }
                else {
                    node = stack[--stack_ptr];
                }
            }
            else {
                if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                    // printf("intersected left\n");
                }
                int right_child_idx = node->leftFirst + 1;
                node = left_child;
                if (!isinf(right)) {
                    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
                        // printf("intersected right, placing %d on stack\n", swapped ? right_child_idx - 1 : right_child_idx);
                    }
                    stack[stack_ptr++] = right_child;
                }
            }
        }
    }

    if (hit_info->hit) {
        int tri_idx = hit_info->tri_idx;
        uint3 tri = (uint3) (triangles[tri_idx * 3], triangles[tri_idx * 3 + 1], triangles[tri_idx * 3 + 2]);
        float3 p0 = (float3) (vertices[tri.s0 * 3], vertices[tri.s0 * 3 + 1], vertices[tri.s0 * 3 + 2]);
        float3 p1 = (float3) (vertices[tri.s1 * 3], vertices[tri.s1 * 3 + 1], vertices[tri.s1 * 3 + 2]);
        float3 p2 = (float3) (vertices[tri.s2 * 3], vertices[tri.s2 * 3 + 1], vertices[tri.s2 * 3 + 2]);
        hit_info->point = ray_at(ray, t_max);
        hit_info->normal = normalize(cross(p1 - p0, p2 - p0));
    }
}
#endif