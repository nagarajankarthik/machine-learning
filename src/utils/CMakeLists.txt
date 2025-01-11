# CMakeLists.txt
cmake_policy(SET CMP0028 NEW)
target_sources(ml PRIVATE "json.hpp" "read_input.cpp" "logging.cpp")
