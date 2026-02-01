ifeq ($(origin OPTIX90_PATH),undefined)
$(error OPTIX90_PATH must be set)
endif

KERNELS := \
	normals.cu

NVCC := nvcc

ifeq ($(origin OUT_DIR),undefined)
# the .optixir pattern rule here isn't actually meant to be used, it's just to assist with CLion's builtin Makefile support
# hopefully this works for other IDEs too
$(warning OUT_DIR should be set by cargo; assuming that this is a run for IDE)

OPTIXIR := $(KERNELS:.cu=.optixir)
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include

%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
else
OPTIXIR := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.optixir))
NVCCFLAGS := --optix-ir --use_fast_math -arch=compute_86 -I$(OPTIX90_PATH)/include -I$(OUT_DIR)

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
endif

.PHONY: all clean

all: $(OPTIXIR)



clean:
	rm -f $(OPTIXIR)

