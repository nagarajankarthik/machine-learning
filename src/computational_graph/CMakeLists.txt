cmake_policy(SET CMP0028 NEW)
add_library(tensor STATIC
    tensor.cpp
    tensor.h
    tensor_operations.cpp
)

add_executable(tensor_ops_test tensor_operations_test.cpp)
target_link_libraries(tensor_ops_test PRIVATE ml_test tensor PUBLIC glog::glog gtest::gmock_main)
add_test(NAME TensorOpsTest COMMAND tensor_ops_test)
