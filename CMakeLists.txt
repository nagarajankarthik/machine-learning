cmake_minimum_required(VERSION 3.20)

# set the project name and version
project(MachineLearning VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# add subdirectories
add_subdirectory(src)
add_subdirectory(docs)

# add executables

# install(TARGETS ml DESTINATION bin)