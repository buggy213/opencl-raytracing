#pragma once

#include "sbt.hpp"

/// Accessors for SBT data within kernel; handles the annoying pointer arithmetic
namespace sbt
{
// callable from IS / AH / CH
inline __device__ const HitgroupRecord& get_hitgroup_record()
{
    CUdeviceptr base = optixGetSbtDataPointer() - sizeof(HitgroupRecord::header);
    return *reinterpret_cast<HitgroupRecord*>(base);
}
}