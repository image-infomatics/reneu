#pragma once

#include <boost/pending/disjoint_sets.hpp>
#include "reneu/type_aliase.hpp"
#include "reneu/segmentation/utils.hpp"


namespace reneu{

class DisjointSets{

private:
using Rank_t = std::map<segid_t, size_t>;
using Parent_t = std::map<segid_t, segid_t>;
using PropMapRank_t = boost::associative_property_map<Rank_t>;
using PropMapParent_t = boost::associative_property_map<Parent_t>;
using BoostDisjointSets = boost::disjoint_sets<PropMapRank_t, PropMapParent_t>; 

Rank_t _mapRank;
Parent_t _mapParent;
PropMapRank_t _propMapRank;
PropMapParent_t _propMapParent;
BoostDisjointSets _dsets;

public:
DisjointSets(const Segmentation& seg):
        _propMapRank(_mapRank), _propMapParent(_mapParent),
        _dsets(_propMapRank, _propMapParent){
    
    auto segids = get_nonzero_segids(seg);
    for(const auto& segid : segids){
        _dsets.make_set(segid);
    }
}


inline void union_set(segid_t s0, segid_t s1){
    _dsets.union_set(s0, s1);
}

auto find_set(segid_t sid){
    return _dsets.find_set(sid);
}

void relabel(Segmentation& seg){
    auto segids = get_nonzero_segids(seg);
    // Flatten the parents tree so that the parent of every element is its representative.
    _dsets.compress_sets(segids.begin(), segids.end());
    std::cout<< "get "<< 
                _dsets.count_sets(segids.begin(), segids.end()) << 
                " final objects."<< std::endl;

    std::cout<< "relabel the fragments to a flat segmentation." << std::endl;
    std::transform(seg.begin(), seg.end(), seg.begin(), 
        [this](segid_t segid)->segid_t{return this->find_set(segid);}
    );
    return;
}

};

} // namespace reneu