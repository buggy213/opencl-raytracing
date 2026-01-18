#pragma once

#include <cstdint>
#include <optix_types.h>

OptixPipeline makeBasicPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize);