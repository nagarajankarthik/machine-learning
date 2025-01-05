# CMakeLists.txt
find_package(nlohmann_json 3.2.0 REQUIRED)
add_library(utils read_input.cpp logging.cpp)
target_link_libraries(utils PRIVATE nlohmann_json::nlohmann_json)
