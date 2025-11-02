# add subdirectories
add_subdirectory(base_model)
add_subdirectory(utils)
add_subdirectory(ensemble)
add_subdirectory(tensor)
add_subdirectory(neural_network)

# Find MPI
# This will set MPI_FOUND and create the imported target MPI::MPI_CXX
# if MPI C++ components are found. It will fail if not found due to 'REQUIRED'.
find_package(MPI REQUIRED COMPONENTS CXX)
if(MPI_FOUND)
    message(STATUS "MPI found! Enabling MPI support.")
else()
    # This block will only be reached if MPI is not found and 'REQUIRED' was not used,
    # or if you want a custom error message. With 'REQUIRED', CMake will stop earlier.
    message(FATAL_ERROR "MPI C++ components not found. Please ensure MPI is installed and configured correctly.")
endif()

# add executables
add_executable(ml main.cpp)

# include directories
target_include_directories(ml PRIVATE "${PROJECT_SOURCE_DIR}")

# compile options
target_compile_options(ml PRIVATE -Wall -g -O3)

# link libraries
target_link_libraries(ml PRIVATE neural_network utils tensor ensemble base_model)

# Enable OpenMP if found
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling OpenMP support.")
    target_link_libraries(ml PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link MPI
target_link_libraries(ml PRIVATE MPI::MPI_CXX)
