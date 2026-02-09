include(FetchContent)

message(STATUS "Downloading OptiX 9.1 headers from GitHub...")

FetchContent_Declare(
  optix91
  GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
  GIT_TAG v9.1.0
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(optix91)

set(OPTIX91_INCLUDE_DIR "${optix91_SOURCE_DIR}/include" CACHE PATH "OptiX 9.1 include directory" FORCE)
