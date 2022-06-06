#pragma once
#include <initializer_list>
#include <set>
#include <tsl/robin_map.h>

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

inline auto get_nonzero_segids(const Segmentation& seg){
    
    std::set<segid_t> segids = {};
    for(const auto& segid : seg){
        if(segid>0)
            segids.insert(segid);
    }

    return segids;
}

auto get_label_map(const PySegmentation& frag, const PySegmentation& seg){
    tsl::robin_map<segid_t, segid_t> labelMap = {};

    for(std::size_t idx = 0; idx < frag.size(); idx ++){
        const auto& sid0 = frag(idx);
        const auto& sid1 = seg(idx);
        if(sid0!=sid1 && sid0>0){
            labelMap[sid0] = sid1;
        }
    }

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


} // namespace reneu