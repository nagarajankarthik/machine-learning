cmake_minimum_required(VERSION 3.20)

# set the project name and version
project(MachineLearning VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Fetch googletest dependency
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        "release"
)

FetchContent_MakeAvailable(googletest)

# add subdirectories
add_subdirectory(base_model)
add_subdirectory(utils)
add_subdirectory(ensemble)
add_subdirectory(computational_graph)

# add executables
add_executable(ml main.cpp)

# include directories
target_include_directories(ml PRIVATE "${PROJECT_SOURCE_DIR}")

# link libraries
target_link_libraries(ml PRIVATE utils computational_graph ensemble base_model)

# Enable OpenMP if found
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling OpenMP support.")
    target_link_libraries(ml PRIVATE OpenMP::OpenMP_CXX)
endif()
