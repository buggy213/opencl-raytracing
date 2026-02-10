ifndef OPTIX_INCLUDE_DIR
  ifdef OPTIX91_PATH
    OPTIX_INCLUDE_DIR := $(OPTIX91_PATH)/include
  else
    $(error Set OPTIX91_PATH for standalone builds; OPTIX_INCLUDE_DIR is set automatically by cargo)
  endif
endif

ifeq ($(origin MATHDX_25_12_PATH),undefined)
$(error MATHDX_25_12_PATH must be set)
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
	-I$(MATHDX_25_12_PATH)/nvidia/mathdx/25.12/include


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
OPTIXIR_DEBUG := $(addsuffix .debug,$(OPTIXIR))
PTX := $(addprefix $(OUT_DIR)/,$(KERNELS:.cu=.ptx))
PTX_DEBUG := $(addsuffix .debug,$(PTX))

NVCCFLAGS += -MMD -MP
OPTIXIR_DEPS := $(addsuffix .d,$(OPTIXIR))
OPTIXIR_DEBUG_DEPS := $(addsuffix .d,$(OPTIXIR_DEBUG))
PTX_DEPS := $(addsuffix .d,$(PTX))
PTX_DEBUG_DEPS := $(addsuffix .d,$(PTX_DEPS))

$(OUT_DIR)/%.optixir: %.cu
	$(NVCC) $(NVCCFLAGS) --optix-ir $< -o $@

$(OUT_DIR)/%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $@

$(OUT_DIR)/%.optixir.debug: %.cu
	$(NVCC) $(NVCCFLAGS) --optix-ir -G --generate-line-info $< -o $@

$(OUT_DIR)/%.ptx.debug: %.cu
	$(NVCC) $(NVCCFLAGS) --ptx -G --generate-line-info $< -o $@

-include $(OPTIXIR_DEPS)
-include $(PTX_DEPS)
-include $(OPTIXIR_DEBUG_DEPS)
-include $(PTX_DEBUG_DEPS)

.PHONY: all clean optix-ir ptx

optix-ir: $(OPTIXIR)
ptx: $(PTX)
all: optix-ir ptx

clean:
	rm -f $(OPTIXIR) $(OPTIXIR_DEPS)

endif
