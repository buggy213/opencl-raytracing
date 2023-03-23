#include "cl/geometry.cl"

#ifndef RT_ACCEL
#define RT_ACCEL
// if triCount is 0, this is index of left BVH child node. index of right BVH child node is leftIndex + 1
// otherwise, if triCount != 0, then this is the index of the first triangle in triangle array
typedef struct {
    float3 aabbMin;
    float3 aabbMax;
    uint leftFirst; 
    uint triCount;
} bvh_node_t;

#endif