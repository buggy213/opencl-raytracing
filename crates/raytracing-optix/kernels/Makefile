ifeq ($(origin OPTIX90_PATH),undefined)
$(error OPTIX90_PATH must be set)
endif

ifeq ($(origin OUT_DIR),undefined)
$(error OUT_DIR should be set by cargo)
endif

KERNELS := \
	normals.cu

OPTIXIR := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.optixir))

NVCC := nvcc
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include

.PHONY: all clean

all: $(OPTIXIR)

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(OPTIXIR)

