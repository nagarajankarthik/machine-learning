cmake_minimum_required(VERSION 3.20)

# set the project name and version
project(MachineLearning VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add subdirectories
add_subdirectory(utils)
add_subdirectory(ensemble)

# add executables
add_executable(ml main.cpp)

# include directories
target_include_directories(ml PRIVATE "${PROJECT_SOURCE_DIR}")

# link libraries
target_link_libraries(ml PRIVATE utils ensemble)