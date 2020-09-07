cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dotnetlibtorch)

find_package(Torch REQUIRED)

add_library(dotnetlibtorch SHARED dotnetlibtorch.cpp)
target_link_libraries(dotnetlibtorch "${TORCH_LIBRARIES}")
set_property(TARGET dotnetlibtorch PROPERTY CXX_STANDARD 14)
