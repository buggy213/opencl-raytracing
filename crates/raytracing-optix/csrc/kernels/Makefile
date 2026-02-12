ifndef OPTIX_INCLUDE_DIR
  ifdef OPTIX91_PATH
    OPTIX_INCLUDE_DIR := $(OPTIX91_PATH)/include
  else
    $(error Set OPTIX91_PATH for standalone builds; OPTIX_INCLUDE_DIR is set automatically by cargo)
  endif
endif

ifndef MATHDX_INCLUDE_DIR
  ifdef MATHDX_25_12_PATH
    MATHDX_INCLUDE_DIR := $(MATHDX_25_12_PATH)/nvidia/mathdx/25.12/include
  else
    $(error Set MATHDX_25_12_PATH for standalone builds; MATHDX_INCLUDE_DIR is set automatically by cargo)
  endif
endif

KERNELS := \
	aov.cu \
	pathtracer.cu
SHARED_HEADERS := $(realpath ../common)
NVCC := nvcc
NVCCFLAGS := \
	--use_fast_math \
	-arch=compute_86 \
	-I$(SHARED_HEADERS) \
	-I$(OPTIX_INCLUDE_DIR) \
	-I$(MATHDX_INCLUDE_DIR) \
	-Werror all-warnings \
	--diag-suppress 20044 # extern "C" used for global variable is blessed by OptiX programming guide


ifeq ($(origin OUT_DIR),undefined)
# the .optixir pattern rule here isn't actually meant to be used, it's just to assist with CLion's builtin Makefile
# support to get good intellisense. hopefully this works for other IDEs too
$(warning OUT_DIR should be set by cargo; assuming that this is a run for IDE)

OPTIXIR := $(KERNELS:.cu=.optixir)

%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) --optix-ir $< -o $@

.PHONY: all clean

all: $(OPTIXIR)

else
OPTIXIR := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.optixir))
OPTIXIR_DEBUG := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.debug.optixir))
PTX := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.ptx))
PTX_DEBUG := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.debug.ptx))

NVCCFLAGS += -MMD -MP
OPTIXIR_DEPS := $(OPTIXIR:%.optixir=%.optixir.d)
OPTIXIR_DEBUG_DEPS := $(OPTIXIR_DEBUG:%.debug.optixir=%.debug.optixir.d)
PTX_DEPS := $(PTX:%.ptx=%.ptx.d)
PTX_DEBUG_DEPS := $(PTX_DEBUG:%.debug.ptx=%.debug.ptx.d)

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) -MF $@.d --optix-ir $< -o $@

$(OUT_DIR)/%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -MF $@.d --ptx $< -o $@

$(OUT_DIR)/%.debug.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) -MF $@.d --optix-ir -G --generate-line-info $< -o $@

$(OUT_DIR)/%.debug.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -MF $@.d --ptx -G --generate-line-info $< -o $@

-include $(OPTIXIR_DEPS)
-include $(PTX_DEPS)
-include $(OPTIXIR_DEBUG_DEPS)
-include $(PTX_DEBUG_DEPS)

.PHONY: all clean optix-ir ptx

optix-ir: $(OPTIXIR)
optix-ir-debug: $(OPTIXIR_DEBUG)
ptx: $(PTX)
ptx-debug: $(PTX_DEBUG)
all: optix-ir optix-ir-debug ptx ptx-debug

clean:
	rm -f $(OPTIXIR) $(OPTIXIR_DEPS)

endif
