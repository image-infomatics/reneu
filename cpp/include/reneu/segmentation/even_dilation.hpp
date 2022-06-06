#pragma once

#include <iostream>
#include <queue>

#include "../types.hpp"

#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>

namespace reneu{

/**
 * @brief dilate the objects evenly to fill up all the background space
 *
 * @tparam SEG 
 * @param seg1d 
 * @return auto 
 */
template<class E>
auto even_dilation_1d(xt::xexpression<E>& seg1d_){
    E& seg1d = seg1d_.derived_cast();
    
   

    // we should take turns to bias to start or stop
    bool bias_to_start;
    if(rand()>0.5)
        bias_to_start = true;
    else
        bias_to_start = false;

    // std::cout<<"\n\nprocess a row: "<<std::endl;
    std::size_t new_start;
    std::size_t start=0, stop = seg1d.shape(0);
    segid_t obj0=0, obj1=0; 
    
    while(start<seg1d.size()){
        if(seg1d(start) > 0)
            start++;
        else{
            if(start==0)
                // this column start with backgound 
                obj0 = 0;
            else
                obj0 = seg1d(start-1);

            // find the stop of background region
            for(std::size_t j=start; j<seg1d.size(); j++){
                if(seg1d(j)>0){
                    stop = j - 1;
                    obj1 = seg1d(j);
                    break;
                }
                if(j==seg1d.size()-1){
                    // the end of column is still background
                    stop = j;
                    obj1 = obj0;
                }
            }
            // stop+1 will be positive or out of the bounds
            new_start = stop+2;

            if(start == 0){
                // we have to filled it up with the first object id
                if(obj1>0)
                    obj0 = obj1;
                else
                    // the whole column is all 0!
                    return;
            }

            // fill the region with nearest object
            while(start < stop){
                // std::cout<<"filled a start voxel "<<seg1d(start)<<" at "<< start << " with "<< obj0<<"."<<std::endl;
                assert(seg1d(start)==0);
                seg1d(start) = obj0;
                start++;
                // std::cout<<"filled a stop voxel "<<seg1d(stop)<<" at "<< stop <<" with "<< obj1<<"."<<std::endl;
                assert(seg1d(stop)==0);
                seg1d(stop) = obj1;
                stop--;
            }
            if(start == stop){
                // choose a random one
                if(bias_to_start){
                    seg1d(start) = obj0;
                }else{
                    seg1d(stop) = obj1;
                }
                bias_to_start = ~bias_to_start;
            }
            start = new_start;
        }
    }
}

template<class E>
auto even_dilation_2d_v1(xt::xexpression<E>& seg2d_){
    E& seg2d = seg2d_.derived_cast();

    for(std::size_t y=0; y<seg2d.shape(0); y++){
        // std::cout<<"y: "<<y<<std::endl;
        auto row = xt::view(seg2d, y, xt::all());
        even_dilation_1d(row);
    }
}

template<class E>
auto even_dilation_2d(xt::xexpression<E>& seg2d_){
    E& seg2d = seg2d_.derived_cast();

    for(std::size_t y=0; y<seg2d.shape(0); y++){
        // std::cout<<"y: "<<y<<std::endl;
        auto row = xt::view(seg2d, y, xt::all());
        even_dilation_1d(row);
    }
}

auto even_dilation(PySegmentation& seg){
    for(std::size_t z=0; z<seg.shape(0); z++){
        // std::cout<<"z: "<<z<<std::endl;
        auto seg2d = xt::view(seg, z, xt::all(), xt::all());
        even_dilation_2d(seg2d);
    }
    // in-place dilation is not working, we have to return this!
    return seg;
}

} // end of namespace reneu