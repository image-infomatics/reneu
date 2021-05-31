cmake_minimum_required(VERSION 3.12.0)
set(RENEU_VERSION 0.2.0)
project(libreneu VERSION ${RENEU_VERSION} LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE on)
# this only works with absolute path!
# set(CONDA_PREFIX ${HOME}/opt/anaconda3/envs/cf)
set(CONDA_PREFIX /Users/jwu/opt/anaconda3/envs/cf)
# set(CONDA_PREFIX ${CONDA_PREFIX_1}/envs/reneu)
set(pybind11_DIR ${CONDA_PREFIX}/share/cmake/pybind11)
set(xtensor_DIR ${CONDA_PREFIX}/lib/cmake/xtensor)
set(xtensor-python_DIR ${CONDA_PREFIX}/lib/cmake/xtensor-python)
set(xtl_DIR ${CONDA_PREFIX}/lib/cmake/xtl)

add_subdirectory(third-party/robin-map)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED OFF)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_NO_SYSTEM_PATH FALSE)

if(Boost_NO_SYSTEM_PATH)
    set(BOOST_ROOT /usr/local/Cellar/boost/1.74.0)
    set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include")
    set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
    set(CMAKE_INCLUDE_PATH ${BOOST_INCLUDEDIR} ${CMAKE_INCLUDE_PATH} )
    set(CMAKE_LIBRARY_PATH ${BOOST_LIBRARYDIR} ${CMAKE_LIBRARY_PATH})
    message(STATUS "found Boost libraries in ${BOOST_LIBRARYDIR}")
endif(Boost_NO_SYSTEM_PATH)
find_package(Boost REQUIRED serialization)

set(RENEU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR} 
                        ${CONDA_PREFIX}/include ${BOOST_INCLUDEDIR})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PYBIND11_CPP_STANDARD  "-std=c++20")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../python/reneu/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../python/reneu/lib)


find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Numpy REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Numpy)
find_package(pybind11 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(xtensor-blas REQUIRED)
# find_package(tsl REQUIRED)
# find_package(TBB REQUIRED)

pybind11_add_module(skeleton ${CMAKE_CURRENT_SOURCE_DIR}/skeleton.cpp)
pybind11_add_module(segmentation ${CMAKE_CURRENT_SOURCE_DIR}/segmentation.cpp)

# pybind11 do not recognize our NUMPY_INCLUDE_DIR variable
# we need to add it explicitly
target_include_directories(skeleton PUBLIC ${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR})
target_link_libraries(skeleton)

message("boost libarries: ${Boost_LIBRARIES}")
target_include_directories(segmentation PUBLIC ${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR})
target_link_libraries(segmentation PRIVATE ${Boost_LIBRARIES} tsl::robin_map)

# target_link_libraries(segmentation ${Boost_LIBRARIES} )
# target_link_libraries(segmentation "${Boost_LIBRARIES}/libboost_serialization.dylib" )

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
