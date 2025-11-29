#pragma once

#include <optix_types.h>

#include "lib.h"

RT_API void prepareOptixAccelerationStructures(
    OptixDeviceContext context,
    void* scene
);

RT_API void render();

