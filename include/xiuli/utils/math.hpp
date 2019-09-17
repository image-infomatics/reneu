#pragma once

#include <iostream>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-python/pytensor.hpp"


namespace xiuli::utils{

/**
 * \brief compute the principle component, only return the first component to save some computation.
 */
auto pca_first_component(xt::xtensor<float, 2> sample){
    auto nodeNum = sample.shape(0);
    sample -= xt::mean(sample, {0});
    sample /= std::sqrt( nodeNum - 1 );
    auto [S, V, D] = xt::linalg::svd(sample);
    
    //std::size_t maxIdx = xt::argmax( V )(0);
    // this is a manual implementation to replace above code
    // manual implementation should be faster since it avoid to create a temporal vector 
    std::size_t maxIdx = 0;
    float maxValue = std::numeric_limits<float>::min();
    for (std::size_t i = 0; i<nodeNum; i++){
        if (V(i) > maxValue){
            maxValue = V(i);
            maxIdx = i;
        }
    }
    
    //return xt::view(D, maxIdx, xt::all());
    xt::xtensor<float, 1> ret = xt::view(D, maxIdx, xt::all());
    return ret;
}

inline auto py_pca_first_component(xt::pytensor<float, 2> pysample){
    xt::xtensor<float, 2> sample = pysample;
    auto pca = pca_first_component( sample );
    // we have to transform to a real xtensor to return as numpy array
    //xt::xtensor<float, 1> ret = pca;
    return pca;
}

} // namespace xiuli::utils