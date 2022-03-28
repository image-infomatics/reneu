#pragma once

#include <deque>

#include "reneu/type_aliase.hpp"
#include <xtensor/xview.hpp>
#include "reneu/utils/print.hpp"

namespace reneu{

// using namespace xt::placeholders;  // required for `_` in range view to work

template<class SEG1D>
void remove_contact_1d(SEG1D seg1d){
    for(std::size_t x = 1; x < seg1d.shape(0); x++){
        if(seg1d(x)>0 && seg1d(x-1)>0 && seg1d(x)!=seg1d(x-1)){
            seg1d(x) = 0;
            seg1d(x-1) = 0;
            // std::cout<<x<<", ";
        }
    }
    // std::cout<<std::endl;
}

/**
 * @brief remove the object boundary voxel if it contacted another object
 * 
 * @param seg the input plain segmentation
 */
auto remove_contact(PySegmentation& seg){
    // z direction
    for(std::size_t y=0; y<seg.shape(1); y++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            remove_contact_1d(xt::view(seg, xt::all(), y, x));
        }
    }

    // y direction
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            remove_contact_1d(xt::view(seg, z, xt::all(), x));
        }
    }
    
    // x direction
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t y=0; y<seg.shape(1); y++){
            remove_contact_1d(xt::view(seg, z, y, xt::all()));
        }
    }
    // reneu::utils::print_array(seg);
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