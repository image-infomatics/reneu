#pragma once

#include <gtest/gtest.h>
#include "xiuli/neuron/nblast.hpp"

using namespace xiuli::neuron::nblast;

/**
 * \brief Score table construction and indexing
 */
TEST(SCORE_TABLE, NBLAST){
    ScoreTable scoreTable = ScoreTable();

    ASSERT_FLOAT_EQ( scoreTable.get_index(0., 0.), 9.50009681841246 );

}