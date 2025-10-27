cmake_policy(SET CMP0028 NEW)
add_library(neural_network STATIC
    layer.h
    optimizer.h
    neural_network.h
    neural_network.cpp
)

add_executable(neural_network_test neural_network_test.cpp)
target_link_libraries(neural_network_test PRIVATE neural_network tensor utils base_model PUBLIC GTest::gtest GTest::gtest_main)
add_test(NAME NeuralNetworkTest COMMAND neural_network_test)
