include(FetchContent)

message(STATUS "Downloading MathDx 25.12 (cuRANDDx)...")

FetchContent_Declare(
  mathdx
  URL https://developer.nvidia.com/downloads/compute/cuRANDDx/redist/cuRANDDx/cuda13/nvidia-mathdx-25.12.1-cuda13.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(mathdx)

set(MATHDX2512_INCLUDE_DIR "${mathdx_SOURCE_DIR}/nvidia/mathdx/25.12/include" CACHE PATH "MathDx 25.12 include directory" FORCE)
