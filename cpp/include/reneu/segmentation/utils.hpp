#pragma once
#include <initializer_list>
#include <set>
#include <tsl/robin_map.h>
#include "disjoint_sets.hpp"
#include "../types.hpp"

namespace reneu{

using Segid2VoxelNum = tsl::robin_map<segid_t, size_t>;

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
    xt::pytensor<segid_t, 2>::shape_type sh = {pairs.size(), 2};
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
    xt::pytensor<segid_t, 2>::shape_type sh = {labelMap.size(), 2};
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