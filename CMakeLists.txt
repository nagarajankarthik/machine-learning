cmake_minimum_required(VERSION 3.20)

# set the project name and version
project(MachineLearning VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add subdirectories
add_subdirectory(src)
add_subdirectory(docs)

# add executables

# install(TARGETS ml DESTINATION bin)