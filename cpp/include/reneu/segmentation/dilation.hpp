#pragma once

#include <deque>

#include "reneu/type_aliase.hpp"
#include "xtensor/xview.hpp"

namespace reneu{

// To-do: copied the seg2d, do in inplace would be better
template < class Seg >
void dilate2d(Seg& seg){
    std::ptrdiff_t sz = seg.shape(0);
    std::ptrdiff_t sy = seg.shape(1);
    std::ptrdiff_t sx = seg.shape(2);
    const std::array<std::ptrdiff_t, 4> dir2d = {-1, -sx, 1, sx};

    std::deque<std::ptrdiff_t> backgroundIndices = {};
    for(std::ptrdiff_t idx = 0; idx < seg.size(); idx++){
        if(seg[idx] == 0) {
            backgroundIndices.push_back(idx);
        }
    }

    while(!backgroundIndices.empty()){
        auto idx = backgroundIndices.back();
        backgroundIndices.pop_back();

        for(const std::ptrdiff_t& d : dir2d){
            std::ptrdiff_t him = idx + d;
            if(him>0 && him<seg.size() && seg[him]!=0){
                seg[idx] = seg[him];
                break;
            }
        }

        if(seg[idx] == 0){
            backgroundIndices.push_front(idx);
        }
    }
}

void py_dilate2d(PySegmentation& py_seg){
    dilate2d(py_seg);
}

} // namespace reneu