# Looks for the environment variable:
# MATHDX_25_12_PATH

# Sets the variables :
# MATHDX2512_INCLUDE_DIR

# MathDx2512_FOUND

set(MATHDX_25_12_PATH $ENV{MATHDX_25_12_PATH})

find_path(MATHDX2512_INCLUDE_DIR curanddx.hpp ${MATHDX_25_12_PATH}/nvidia/mathdx/25.12/include)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MathDx2512 DEFAULT_MSG MATHDX2512_INCLUDE_DIR)

mark_as_advanced(MATHDX2512_INCLUDE_DIR)
