# CMakeLists.txt
cmake_policy(SET CMP0028 NEW)
add_library(utils STATIC
    utils.cpp
    utils.h
    logging.cpp
    logging.h
)