ifeq ($(origin OPTIX90_PATH),undefined)
$(error OPTIX90_PATH must be set)
endif

KERNELS := \
	normals.cu

OPTIXIR := $(KERNELS:.cu=.optixir)

NVCC := nvcc
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include

.PHONY: all clean

all: $(OPTIXIR)

%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(OPTIXIR)

