# Looks for the environment variable:
# OPTIX91_PATH

# Sets the variables :
# OPTIX91_INCLUDE_DIR

# OptiX91_FOUND

set(OPTIX91_PATH $ENV{OPTIX91_PATH})

find_path(OPTIX91_INCLUDE_DIR optix_host.h ${OPTIX91_PATH}/include)

# message("OPTIX91_INCLUDE_DIR = " "${OPTIX91_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX91 DEFAULT_MSG OPTIX91_INCLUDE_DIR)

mark_as_advanced(OPTIX91_INCLUDE_DIR)

# message("OptiX91_FOUND = " "${OptiX91_FOUND}")