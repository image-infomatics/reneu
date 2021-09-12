#pragma once

#include <vector>
#include "../type_aliase.hpp"
#include <xtensor/xbuilder.hpp>

namespace reneu{


/**
 * @brief detect points based on connected components and distance field
 * 
 * @param com connected component analysis result. Each component is an object. The segmentation ID should be numerically ordered without gap. Meaning the segmentation IDs should be 1,2,3,... 
 * @param df distance field. The result of distance transformation. The value indicates the distance to the nearest object boundary.
 * @return auto 
 */
auto detect_points(const Segmentation& com, const ProbabilityMap& df){
    assert(com.shape(0) == df.shape(0));
    assert(com.shape(1) == df.shape(1));
    assert(com.shape(2) == df.shape(2));

    auto object_num = xt::reduce(
        [](std::uint64_t a, std::uint64_t b) {return std::max(a, b);},
        com)();
    
    // map segmentation ID to distance from current voxel to 
    // nearest boundary
    xt::xtensor<float, 1>::shape_type sh = {object_num};
    xt::xtensor<float, 1> sid2dis = xt::zeros<float>(sh);

    xt::xtensor<std::size_t, 2>::shape_type sh1 = {object_num, 3};
    xt::xtensor<std::size_t, 2> sid2pos = xt::zeros<std::size_t>(sh1);

    for(std::size_t z=1; z<com.shape(0)-1; z++){
        for(std::size_t y=1; y<com.shape(1)-1; y++){
            for(std::size_t x=1; x<com.shape(2)-1; x++){
                // if this is a local maxima, collect it
                const auto& sid = com(z,y,x);
                if(sid > 0){
                    const auto& d = df(z,y,x);
                    if( d>sid2dis(sid-1) ){
                        sid2dis(sid-1) = d;
                        sid2pos(sid-1, 0) = z;
                        sid2pos(sid-1, 1) = y;
                        sid2pos(sid-1, 2) = x;
                    }
                }
            }
        }
    }
    return sid2pos;
}


auto py_detect_points(
        const PySegmentation& pyCom, 
        const PyProbabilityMap& pyDF){
    return detect_points(std::move(pyCom), std::move(pyDF));
}

} // end of name space reneu