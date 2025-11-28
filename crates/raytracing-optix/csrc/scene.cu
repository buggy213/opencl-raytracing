#include <cuda.h>
#include <optix_types.h>

__host__ void prepareOptixAccelerationStructures(
    OptixDeviceContext optixContext,
    void* scene
) {
    // depth-first traverse the scene graph, creating a GAS for all triangle mesh / all spheres

}
