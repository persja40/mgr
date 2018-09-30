cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project (GradientClustering LANGUAGES CXX CUDA)
find_package(CUDA QUIET REQUIRED)

CUDA_ADD_LIBRARY(mylib SHARED
    cuda/kernel.cpp
    OPTIONS  -arch sm_20
)

TARGET_LINK_LIBRARIES(mylib ${CUDA_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# COMPILE AND LINK
cuda_add_executable(algorithm ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm.cpp)