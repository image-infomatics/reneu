#pragma once

#include <iostream>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-python/pytensor.hpp"


namespace reneu::utils{

/**
 * \brief compute the principle component, only return the first component to save some computation.
 */
auto pca_first_component(xt::xtensor<float, 2> sample){
    auto nodeNum = sample.shape(0);
    sample -= xt::mean(sample, {0});
    sample /= std::sqrt( nodeNum - 1 );
    auto [S, V, D] = xt::linalg::svd(sample);

    // std::cout<< "S, V, D: " << S << V << D << std::endl; 
    auto maxIdx = xt::argmax( V )(0);
    // std::cout<< "max index of " << maxIdx << "in V: " << V << std::endl;
    // std::cout<< "D in C++: "<< D << std::endl;

    //return xt::view(D, maxIdx, xt::all());
    xt::xtensor<float, 1> ret = xt::view(D, maxIdx, xt::all());
    // std::cout<< "component in c++: " << ret << std::endl;
    return ret;
}

inline auto py_pca_first_component(xt::pytensor<float, 2> pysample){
    xt::xtensor<float, 2> sample = pysample;
    return pca_first_component( sample );
}

} // namespace reneu::utils