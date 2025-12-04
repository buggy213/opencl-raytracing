#include "lib.h"

#include "shared_lib.h"
#include <optix_stubs.h>

__host__ RT_API OptixDeviceContext initOptix() {
    CUDA_CHECK(cudaSetDevice(0));

    OPTIX_CHECK(optixInit());

    OptixDeviceContext optix_ctx;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &optix_ctx));

    return optix_ctx;
}
