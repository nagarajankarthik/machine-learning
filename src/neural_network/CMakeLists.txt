cmake_policy(SET CMP0028 NEW)
add_library(neural_network STATIC
    layer.h
    optimizer.h
    neural_network.h
    neural_network.cpp
)

