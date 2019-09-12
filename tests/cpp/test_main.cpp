
#include <gtest/gtest.h>
//#include "xiuli/xiuli.hpp"
#include "neuron/nblast_test.hpp"

int main(int argc, char ** argv){
    testing::InitGoogleTest(&argc, argv);
    //testing::FLAGS_gtest_death_test_style = "threadsafe";    

    return RUN_ALL_TESTS();
}