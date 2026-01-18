KERNELS := \
	normals.cu

OPTIXIR := $(KERNELS:.cu=.optixir)

NVCC := nvcc
NVCCFLAGS := --optix-ir --use_fast_math

.PHONY: all clean

all: $(OPTIXIR)

%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(OPTIXIR)

