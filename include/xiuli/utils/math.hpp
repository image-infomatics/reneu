#pragma once

#include <iostream>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-python/pytensor.hpp"


namespace xiuli::utils{

/**
 * \brief compute the principle component, only return the first component to save some computation.
 */
inline auto pca_first_component(xt::xtensor<float, 2> sample){
    assert(sample.shape(1) == 3);
    auto nodeNum = sample.shape(0);
    sample -= xt::mean(sample, {0});
    sample /= std::sqrt( nodeNum - 1 );
    auto [S, V, D] = xt::linalg::svd(sample);
    std::size_t maxIdx = xt::argmax( V )(0);
    xt::xtensor<float, 1> ret = xt::view(D, maxIdx, xt::all());
    return ret;
    //return xt::view(D, maxIdx, xt::all());
}

inline auto py_pca_first_component(xt::pytensor<float, 2> pysample){
    return pca_first_component( pysample );
}

} // namespace xiuli::utils