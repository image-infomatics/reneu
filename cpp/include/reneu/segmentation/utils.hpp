#pragma once
#include <initializer_list>
#include <map>

#include <boost/pending/disjoint_sets.hpp>

#include "../type_aliase.hpp"

namespace reneu{

inline auto get_nonzero_segids(const Segmentation& seg){

    std::map<segid_t, std::size_t> id2count;

    // std::cout<< "accumulate segment ids..." << std::endl;
    for(const auto& segid : seg){
        // std::cout<< segid << ", ";
        if(segid>0)
            ++id2count[segid];
    }

    // std::cout<< "size of map: "<< id2count.size() << std::endl;
    std::vector<segid_t> segids = {};
    segids.reserve(id2count.size());
    for(const auto& [segid, count] : id2count){
        segids.push_back(segid);
    }
    return segids;
}
} // namespace reneu