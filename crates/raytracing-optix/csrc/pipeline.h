#pragma once

#include <cstdint>
#include <optix_types.h>

#include "lib_optix_types.h"

OptixPipelineWrapper makeBasicPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize);
void launchBasicPipelineImpl(OptixPipelineWrapper pipeline);