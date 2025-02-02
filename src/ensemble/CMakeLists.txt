# CMakeLists.txt
cmake_policy(SET CMP0028 NEW)
add_library(ensemble STATIC
    decision_tree_classifier.cpp
    decision_tree_classifier.h
    random_forest_classifier.cpp
    random_forest_classifier.h
)