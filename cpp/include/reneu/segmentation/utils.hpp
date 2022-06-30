#pragma once
#include <initializer_list>
#include <set>
#include <limits.h>

#include <tsl/robin_map.h>
#include "disjoint_sets.hpp"
#include "../types.hpp"

namespace reneu{

using Segid2VoxelNum = tsl::robin_map<segid_t, size_t>;

auto get_nonzero_bounding_box(const PySegmentation& seg){
    std::array<std::size_t, 3> start = {
        seg.shape(0)-1, 
        seg.shape(1)-1, 
        seg.shape(2)-1
    };
    std::array<std::size_t, 3> stop = {0, 0, 0};

    // find start of x
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t y=0; y<seg.shape(1); y++){
            for(std::size_t x=0; x<start[2]; x++){
                if(seg(z,y,x)>0 && x<start[2]){
                    start[2] = x;
                    break;
                }
            }
        }
    }
    
    // find start of y
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            for(std::size_t y=0; y<start[1]; y++){
                if(seg(z,y,x)>0 && y<start[1]){
                    start[1] = x;
                    break;
                }
            }
        }
    }
    
    // find start of z
    for(std::size_t y=0; y<seg.shape(1); y++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            for(std::size_t z=0; z<start[0]; z++){
                if(seg(z,y,x)>0 && z<start[0]){
                    start[0] = z;
                    break;
                }
            }
        }
    }

    // find stop of x
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t y=0; y<seg.shape(1); y++){
            for(std::size_t x=seg.shape(2)-1; x>stop[2]; x--){
                if(seg(z,y,x)>0 && x>stop[2]){
                    stop[2] = x;
                    break;
                }
            }
        }
    }
    
    // find stop of y
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            for(std::size_t y=seg.shape(1)-1; y>stop[1]; y--){
                if(seg(z,y,x)>0 && y>stop[1]){
                    stop[1] = y;
                    break;
                }
            }
        }
    }
 
    // find stop of z
    for(std::size_t y=0; y<seg.shape(1); y++){
        for(std::size_t x=0; x<seg.shape(2); x++){
            for(std::size_t z=seg.shape(0)-1; z>stop[0]; z--){
                if(seg(z,y,x)>0 && z>stop[0]){
                    stop[0] = z;
                    break;
                }
            }
        }
    }
    
    // stop should not be inclusive
    for(std::size_t i=0; i<3; i++)
        stop[i] += 1;

    assert(start[0]<stop[0]);
    assert(start[1]<stop[1]);
    assert(start[2]<stop[2]);
    return std::make_pair(start, stop);
}

inline auto get_segid_to_voxel_num(const Segmentation& seg){
    Segid2VoxelNum id2count;

    for(const auto& segid : seg){
        if(segid>0)
            ++id2count[segid];
    }
    return id2count;
}

inline auto get_nonzero_segids(const PySegmentation& seg){
    
    std::set<segid_t> segids = {};
    for(const auto& segid : seg){
        if(segid>0)
            segids.insert(segid);
    }

    return segids;
}

auto get_label_map(const PySegmentation& frag, const PySegmentation& seg){
    std::set<std::pair<segid_t, segid_t>> pairs = {};

    if(seg(0)>0){
        pairs.emplace(frag(0), seg(0));
    }

    for(std::size_t idx = 1; idx < seg.size(); idx ++){
        // skip most of repetitive voxels and speed it up
        if(seg(idx)!=seg(idx-1) && frag(idx)!=seg(idx) && seg(idx)>0){
            pairs.emplace(frag(idx), seg(idx));
        }
    }

    std::cout<<"build array from the set..."<<std::endl;
    xt::xtensor<segid_t, 2>::shape_type sh = {pairs.size(), 2};
    auto labelMapArray = xt::empty<segid_t>(sh);
    std::size_t idx = 0;
    for(const auto& [sid0, root]: pairs){
        labelMapArray(idx, 0) = sid0;
        labelMapArray(idx, 1) = root;
        idx ++;
    }

    return labelMapArray;
}


auto get_label_map_v1(const PySegmentation& frag, const PySegmentation& seg){
    tsl::robin_map<segid_t, segid_t> labelMap = {};

    for(std::size_t idx = 0; idx < frag.size(); idx ++){
        const auto& sid0 = frag(idx);
        const auto& search = labelMap.find(sid0);
        if(search == labelMap.end()){
            const auto& sid1 = seg(idx);
            if(sid0!=sid1 && sid0>0 && sid1>0){
                labelMap[sid0] = sid1;
            }
        }
    }

    std::cout<<"build array from the map..."<<std::endl;
    xt::xtensor<segid_t, 2>::shape_type sh = {labelMap.size(), 2};
    auto labelMapArray = xt::empty<segid_t>(sh);
    std::size_t idx = 0;
    for(const auto& [sid0, root]: labelMap){
        labelMapArray(idx, 0) = sid0;
        labelMapArray(idx, 1) = root;
        idx ++;
    }

    return labelMapArray;
}

// using SegPairs = std::vector<std::pair<segid_t, segid_t>>;
// auto seg_pairs_to_array(SegPairs& pairs){
//     // std::cout<<"convert to array."<<std::endl;
//     const auto& pairNum = pairs.size();
//     xt::xtensor<segid_t, 2>::shape_type sh = {pairNum, 2};
//     auto arr = xt::empty<segid_t>(sh);
//     for(std::size_t idx=0; idx<pairNum; idx++){
//         const auto& [segid0, root] = pairs[idx];
//         arr(idx, 0) = segid0;
//         arr(idx, 1) = root;
//         // std::cout<<"merge "<<segid0<<", "<<root<<std::endl;
//     }
//     return arr;
// }




} // namespace reneu