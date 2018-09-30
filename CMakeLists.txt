cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project (GradientClustering LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include_directories(include)
add_executable(Algorithm src/algorithm.cpp)