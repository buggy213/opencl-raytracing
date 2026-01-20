ifeq ($(origin OPTIX90_PATH),undefined)
$(error OPTIX90_PATH must be set)
endif

KERNELS := \
	normals.cu

NVCC := nvcc

ifeq ($(origin OUT_DIR),undefined)
$(warning OUT_DIR should be set by cargo; assuming that this is a run for IDE)
OPTIXIR := $(KERNELS:.cu=.optixir)
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include
else
OPTIXIR := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.optixir))
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include -I$(OUT_DIR)
endif

.PHONY: all clean

all: $(OPTIXIR)

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(OPTIXIR)

