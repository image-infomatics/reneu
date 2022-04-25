#pragma once
#include <initializer_list>
#include <set>
#include <tsl/robin_map.h>

#include "../type_aliase.hpp"

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


} // namespace reneu