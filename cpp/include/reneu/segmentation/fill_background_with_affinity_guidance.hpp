#pragma once

#include <deque>

#include "../type_aliase.hpp"
#include <xtensor/xview.hpp>

namespace reneu{


/* 
 * @brief Fill background by finding the maximum affinity connected segment 
 */
template<class E1, class E2>
void fill_background_with_affinity_guidance2d(E1& seg2d, const E2& affx2d, const E2& affy2d){
    std::ptrdiff_t sy = seg2d.shape(0);
    std::ptrdiff_t sx = seg2d.shape(1);

    // current position: x,y; max affinity position: x,y
    std::deque<std::tuple<std::ptrdiff_t, std::ptrdiff_t>> positions = {};
    for(std::ptrdiff_t y = 0; y < sy; y++){
        for(std::ptrdiff_t x = 0; x < sx; x++){
            if(seg2d(y,x)==0)
                positions.push_back(std::make_tuple(y,x));
        }
    }

    std::ptrdiff_t x1, y1;
    const aff_edge_t low = -1.; 
    while(!positions.empty()){
        auto [y,x] = positions.back();
        positions.pop_back();
        
        auto negx = (x>0)       && seg2d(y,x-1)     ? affx2d(y,x)    : low;
        auto negy = (y>0)       && seg2d(y-1,x)     ? affy2d(y,x)    : low;
        auto posx = (x<sx-1)    && seg2d(y, x+1)    ? affx2d(y, x+1) : low;
        auto posy = (y<sy-1)    && seg2d(y+1, x)    ? affy2d(y+1, x) : low;

        auto maxAff = std::max({negy, negx, posy, posx});
        
        // keep edges with maximum affinity
        // only consider all the edges connecting objects
        if(maxAff == low){
            // surrounded by all zeros
            positions.push_front(std::make_tuple(y,x));
            continue;
        }

        if(maxAff==negx)
            seg2d(y,x) = seg2d(y, x-1);
        else if( maxAff==negy )
            seg2d(y,x) = seg2d(y-1, x);
        else if( maxAff==posx )
            seg2d(y,x) = seg2d(y, x+1);
        else if( maxAff==posy )
            seg2d(y,x) = seg2d(y+1, x);
        else{
            // did not find any nonzero neighbor
            positions.push_front(std::make_tuple(y,x));
        }
    }
}

template<class E1, class E2>
void fill_background_with_affinity_guidance(E1& seg, const E2& affs){
    for(size_t z = 0; z<seg.shape(0); z++){
        auto seg2d = xt::view(seg, z, xt::all(), xt::all());
        auto affx2d = xt::view(affs, 0, z, xt::all(), xt::all());
        auto affy2d = xt::view(affs, 1, z, xt::all(), xt::all());
        fill_background_with_affinity_guidance2d(seg2d, affx2d, affy2d);
    }
}

void py_fill_background_with_affinity_guidance(PySegmentation& pySeg, const PyAffinityMap& pyAffs){
    fill_background_with_affinity_guidance(pySeg, pyAffs);
}

} // namespace reneu