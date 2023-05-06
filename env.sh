# setup environment variables
export LLVM_PREFIX=/scratch/jyou12/build/vortex-toolchain-prebuilt/llvm-riscv/
export POCL_CC_PATH=/scratch/jyou12/build/vortex-toolchain-prebuilt/pocl/compiler
export POCL_RT_PATH=/scratch/jyou12/build/vortex-toolchain-prebuilt/pocl/runtime
export VERILATOR_ROOT=/scratch/jyou12/build/vortex-toolchain-prebuilt/verilator
export RISCV_TOOLCHAIN_PATH=/scratch/jyou12/build/vortex-toolchain-prebuilt/riscv-gnu-toolchain
export PATH="/scratch/jyou12/build/vortex-toolchain-prebuilt/verilator/bin:$PATH"
export VORTEX_DRV_PATH="/home/eecs/jyou12/vortex/driver"
export VORTEX_RT_PATH="/home/eecs/jyou12/vortex/runtime"
export VORTEX_DRV_STUB_PATH="/home/eecs/jyou12/vortex/driver/stub"

export LD_LIBRARY_PATH="$POCL_RT_PATH/lib:$VORTEX_DRV_PATH/simx:$LD_LIBRARY_PATH"