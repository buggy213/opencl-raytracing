ifeq ($(origin OPTIX90_PATH),undefined)
$(error OPTIX90_PATH must be set)
endif

ifeq ($(origin MATHDX_25_12_PATH),undefined)
$(error MATHDX_25_12_PATH must be set)
endif

KERNELS := \
	normals.cu \
	pathtracer.cu
SHARED_HEADERS := $(realpath ../common)
NVCC := nvcc
NVCCFLAGS := \
	--optix-ir \
	--use_fast_math \
	-arch=compute_86 \
	-I$(SHARED_HEADERS) \
	-I$(OPTIX90_PATH)/include \
	-I$(MATHDX_25_12_PATH)/nvidia/mathdx/25.12/include


ifeq ($(origin OUT_DIR),undefined)
# the .optixir pattern rule here isn't actually meant to be used, it's just to assist with CLion's builtin Makefile support
# hopefully this works for other IDEs too
$(warning OUT_DIR should be set by cargo; assuming that this is a run for IDE)

OPTIXIR := $(KERNELS:.cu=.optixir)
OPTIXIR_DEPS :=

%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
else
OPTIXIR := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.optixir))
NVCCFLAGS += -I$(OUT_DIR) -MMD -MP
OPTIXIR_DEPS := $(OPTIXIR:.optixir=.d)

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

-include $(OPTIXIR_DEPS)

endif

.PHONY: all clean

all: $(OPTIXIR)

clean:
	rm -f $(OPTIXIR) $(OPTIXIR_DEPS)

