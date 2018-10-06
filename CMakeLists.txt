cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project (GradientClustering LANGUAGES CXX CUDA)
find_package(CUDA QUIET REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/cuda_include)

#set_source_files_properties( algorithm.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
#set_source_files_properties( kernel.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

add_library(cluster STATIC
    cuda_include/kernel.cu
)
target_compile_features(cluster PUBLIC cxx_std_14)
set_target_properties(cluster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# COMPILE AND LINK
#cuda_add_executable(algorithm ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithm.cpp)

set_source_files_properties( src/algorithm.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
add_executable(algorithm src/algorithm.cpp)

set_property(TARGET algorithm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(algorithm PRIVATE cluster)