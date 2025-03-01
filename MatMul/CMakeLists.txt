cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(MATMUL LANGUAGES CXX)

message(STATUS "===========================================")
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "===========================================")

# 增加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++14")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/core)

file(GLOB MATMUL_SRCS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/core/*.cpp
)

set_property(SOURCE ${MATMUL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16 ")

add_executable(matmul ${CMAKE_SOURCE_DIR}/src/main.cpp ${MATMUL_SRCS} )