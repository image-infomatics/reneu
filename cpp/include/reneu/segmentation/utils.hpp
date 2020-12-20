#pragma once
#include <initializer_list>
#include <map>

#include <boost/pending/disjoint_sets.hpp>

#include "../type_aliase.hpp"

namespace reneu{

inline auto get_id2count(const Segmentation& seg){
    std::map<segid_t, size_t> id2count = {};

    for(const auto& segid : seg){
        id2count[segid]++;
    }
    return id2count;
}

inline auto get_nonzero_segids(const Segmentation& seg){
    auto id2count = get_id2count(seg);
    std::vector<segid_t> segids;
    segids.reserve(id2count.size());
    for(const auto& [segid, count] : id2count){
        if(segid>0) segids.push_back(segid);
    }
    return segids;
}
} // namespace reneu