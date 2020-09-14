include(GoogleTest)

add_executable(test tests.cpp)
target_link_libraries(test GTest::GTest)

gtest_add_tests(TARGET tests TEST_PREFIX old:)
gtest_discover_tests(tests TEST_PREFIX new:)
#add_test(NAME monolithic COMMAND tests)