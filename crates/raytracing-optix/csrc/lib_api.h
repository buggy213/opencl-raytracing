#pragma once

#include "shared_lib.h"
#include <optix_types.h>

RT_API OptixDeviceContext initOptix();

RT_API void prepareOptixAccelerationStructures(
    OptixDeviceContext context,
    void* scene
);

RT_API void render();