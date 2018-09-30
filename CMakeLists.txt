cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 17)
project (GradientClustering LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
IF(CMAKE_BUILD_TYPE MATCHES CPU)
    message("CPU build")
ELSEIF(CMAKE_BUILD_TYPE MATCHES GPU)
    message("GPU build")
ELSE()
    message(FATAL_ERROR "No build type specified")
ENDIF()