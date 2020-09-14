cmake_minimum_required(VERSION 3.12.0)
set(RENEU_VERSION 0.2.0)
project(libreneu VERSION ${RENEU_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
set(RENEU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PYBIND11_CPP_STANDARD  "-std=c++17")

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Numpy REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Numpy)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

pybind11_add_module(libreneu main.cpp)
# pybind11 do not recognize our NUMPY_INCLUDE_DIR variable
# we need to add it explicitly
target_include_directories(libreneu PUBLIC ${NUMPY_INCLUDE_DIR})

# options
option(BUILD_TESTS "build test suite" OFF)

# Unit tests
if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# packaging
#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
