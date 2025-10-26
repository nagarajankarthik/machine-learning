# add subdirectories
add_subdirectory(base_model)
add_subdirectory(utils)
add_subdirectory(ensemble)
add_subdirectory(tensor)
add_subdirectory(neural_network)

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
