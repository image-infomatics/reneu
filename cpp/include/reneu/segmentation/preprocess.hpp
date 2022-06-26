#pragma once

#include <deque>

#include "../types.hpp"
#include <xtensor/xview.hpp>
#include "reneu/utils/print.hpp"

namespace reneu{

// using namespace xt::placeholders;  // required for `_` in range view to work


auto remove_contact(PySegmentation& seg){
    std::ptrdiff_t sz = seg.shape(0);
    std::ptrdiff_t sy = seg.shape(1);
    std::ptrdiff_t sx = seg.shape(2);
    
    // direction, +z,+y,+x,-z,-y,-x
    const std::array<std::ptrdiff_t, 6> dir = {sx*sy, sx, 1, -sx*sy, -sx, -1};

    assert(seg.size()==sz*sy*sx);
    std::vector<std::ptrdiff_t> contact_indices = {};
    for(std::ptrdiff_t idx=0; idx<seg.size(); idx++){
        const auto& sid0 = seg[idx];
        if(sid0 > 0){
            for(std::ptrdiff_t d=0; d<6; d++){
                auto idx1 = idx+dir[d];
                if(idx1>=0 && idx1<seg.size()){
                    const auto& sid1 = seg[idx1];
                    if(sid1>0 && sid0!=sid1){
                        contact_indices.push_back(idx);
                        break;
                    }
                }
            }
        }
    }

    std::cout<<"number of voxels to be black out: "<< contact_indices.size()<<std::endl;
    for(const auto& idx: contact_indices){
        seg[idx] = 0;
    }
    return seg;
}

/* 
 * @brief Fill background by finding the maximum affinity connected segment 
 */
template<class SEG, class AFF, class AFF_EDGE>
void fill_background_with_affinity_guidance2d(
        SEG& seg2d, const AFF& affx2d, const AFF& affy2d, 
        const AFF_EDGE& threshold){
    std::ptrdiff_t sy = seg2d.shape(0);
    std::ptrdiff_t sx = seg2d.shape(1);

    // collect all the zero positions contacting objects
    std::deque<std::tuple<std::ptrdiff_t, std::ptrdiff_t>> positions = {};
    for(std::ptrdiff_t y = 0; y < sy; y++){
        for(std::ptrdiff_t x = 0; x < sx; x++){
            if(seg2d(y,x)==0)
                positions.push_back(std::make_tuple(y,x));
        }
    }

    std::ptrdiff_t x1, y1;
    const AFF_EDGE low = -1.; 
    while(!positions.empty()){
        auto [y,x] = positions.back();
        positions.pop_back();
        
        auto negx = (x>0)       && seg2d(y,x-1)     ? affx2d(y,x)    : low;
        auto negy = (y>0)       && seg2d(y-1,x)     ? affy2d(y,x)    : low;
        auto posx = (x<sx-1)    && seg2d(y, x+1)    ? affx2d(y, x+1) : low;
        auto posy = (y<sy-1)    && seg2d(y+1, x)    ? affy2d(y+1, x) : low;

        auto maxAff = std::max({negy, negx, posy, posx});

        // skip the points surrounded by membrane
        if(maxAff > low && maxAff <= threshold){
            continue;
        }

        // keep edges with maximum affinity
        // only consider all the edges connecting objects
        if(maxAff == low){
            // surrounded by all zeros
            positions.push_front(std::make_tuple(y,x));
            continue;
        }

        // expand the object by one pixel
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
            std::cout<<"did not find any nonzero neighbor, this should never happen!"<<std::endl;
            positions.push_front(std::make_tuple(y,x));
        }
    }
}

void fill_background_with_affinity_guidance(
        PySegmentation& seg, const PyAffinityMap& affs, 
        const aff_edge_t& threshold){
    for(size_t z = 0; z<seg.shape(0); z++){
        auto seg2d = xt::view(seg, z, xt::all(), xt::all());
        auto affx2d = xt::view(affs, 0, z, xt::all(), xt::all());
        auto affy2d = xt::view(affs, 1, z, xt::all(), xt::all());
        fill_background_with_affinity_guidance2d(seg2d, affx2d, affy2d, threshold);
    }
}


} // namespace reneu