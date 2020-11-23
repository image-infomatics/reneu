cmake_minimum_required(VERSION 3.12.0)
set(RENEU_TEST_VERSION 0.2.0)
project(reneu-test VERSION ${RENEU_VERSION})

# this only works with absolute path!
set(CONDA_PREFIX /Users/jwu/opt/anaconda3/envs/reneu)
# set(CONDA_PREFIX ${CONDA_PREFIX_1}/envs/reneu)
set(pybind11_DIR ${CONDA_PREFIX}/share/cmake/pybind11)
set(xtensor_DIR ${CONDA_PREFIX}/lib/cmake/xtensor)
set(xtensor-python_DIR ${CONDA_PREFIX}/lib/cmake/xtensor-python)
set(xtl_DIR ${CONDA_PREFIX}/lib/cmake/xtl)
#set(Boost_INCLUDE_DIR ${CONDA_PREFIX}/include)
set(Boost_INCLUDE_DIR /Users/jwu/opt/boost_1_74_0)
set(BOOST_ROOT ${CONDA_PREFIX})
set(RENEU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)
include_directories(${NUMPY_INCLUDE_DIR} ${RENEU_INCLUDE_DIR} ${CONDA_PREFIX}/include)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/cmake/Modules)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PYBIND11_CPP_STANDARD  "-std=c++17")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib)


find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Numpy REQUIRED)
#find_package(Python3 REQUIRED COMPONENTS Numpy)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

# googletest package setup
find_package(GTest CONFIG REQUIRED)

target_include_directories(test_segmentation PUBLIC ${RENEU_INCLUDE_DIR})

add_executable(test_segmentation test_segmentation.cpp)
# add_executable(test_main main.cpp)
# target_link_libraries(test_main GTest::GTest)
target_link_libraries(test_segmentation GTest::GTest)

#gtest_add_tests(TARGET test_main TEST_PREFIX old:)
gtest_discover_tests(test_main TEST_PREFIX new:)
#add_test(NAME monolithic COMMAND tests)