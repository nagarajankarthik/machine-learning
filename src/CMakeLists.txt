# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Machine Learning VERSION 1.0)

add_subdirectory(utils)
add_executable(main main.cpp)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
