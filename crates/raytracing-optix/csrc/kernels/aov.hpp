#pragma once

#ifdef USE_PATHTRACER_PIPELINE_PARAMS
#error mixing aov and pathtracer code not allowed
#endif

#define USE_AOV_PIPELINE_PARAMS
#include "kernel_params.hpp"
extern "C" __constant__ AovPipelineParams pipeline_params;