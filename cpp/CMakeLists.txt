cmake_minimum_required(VERSION 3.12.0)
set(RENEU_VERSION 0.2.0)
project(libreneu VERSION ${RENEU_VERSION})

# this only works with absolute path!
set(CONDA_PREFIX /Users/jwu/opt/anaconda3/envs/reneu)
# set(CONDA_PREFIX ${CONDA_PREFIX_1}/envs/reneu)
set(pybind11_DIR ${CONDA_PREFIX}/share/cmake/pybind11)
set(xtensor_DIR ${CONDA_PREFIX}/lib/cmake/xtensor)
set(xtensor-python_DIR ${CONDA_PREFIX}/lib/cmake/xtensor-python)
set(xtl_DIR ${CONDA_PREFIX}/lib/cmake/xtl)
set(Boost_INCLUDE_DIR ${CONDA_PREFIX}/include)
set(BOOST_ROOT ${CONDA_PREFIX})
set(RENEU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR} ${CONDA_PREFIX}/include)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PYBIND11_CPP_STANDARD  "-std=c++17")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../python/reneu/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../python/reneu/lib)


find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Numpy REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Numpy)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)


pybind11_add_module(skeleton ${CMAKE_CURRENT_SOURCE_DIR}/skeleton.cpp)
pybind11_add_module(segmentation ${CMAKE_CURRENT_SOURCE_DIR}/segmentation.cpp)

# pybind11 do not recognize our NUMPY_INCLUDE_DIR variable
# we need to add it explicitly
target_include_directories(skeleton PUBLIC ${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR})
target_include_directories(segmentation PUBLIC ${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR})

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
