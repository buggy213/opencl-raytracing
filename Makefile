XLEN ?= 32

LLVM_PREFIX ?= /opt/llvm-riscv
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
POCL_CC_PATH ?= /opt/pocl/compiler
POCL_RT_PATH ?= /opt/pocl/runtime

OPTS ?= -n64

VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

K_LLCFLAGS += "-O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32 -mattr=+m,+f -mattr=+vortex -float-abi=hard -code-model=small"
K_CFLAGS   += "-v -O3 --sysroot=$(SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -march=rv32imf -mabi=ilp32f -Xclang -target-feature -Xclang +vortex -I$(VORTEX_RT_PATH)/include -fno-rtti -fno-exceptions -ffreestanding -nostartfiles -fdata-sections -ffunction-sections"
K_LDFLAGS  += "-Wl,-Bstatic,-T$(VORTEX_RT_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a -lm"

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter -Wno-narrowing

CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_DRV_PATH)/stub -lOpenCL -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

all: kernel.pocl

kernel.pocl: cl/kernel.cl
	LLVM_PREFIX=$(LLVM_PREFIX) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_PREFIX)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -LLCFLAGS $(K_LLCFLAGS) -CFLAGS $(K_CFLAGS) -LDFLAGS $(K_LDFLAGS) -o cl/kernel.pocl cl/kernel.cl