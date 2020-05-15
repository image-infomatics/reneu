#pragma once

#include <iostream>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

namespace reneu::utils {

/**
 * \brief compute the principle component, only return the first component to
 * save some computation.
 */
auto pca_first_component(xt::xtensor<float, 2> sample);
inline auto py_pca_first_component(xt::pytensor<float, 2> pysample);

}  // namespace reneu::utils
