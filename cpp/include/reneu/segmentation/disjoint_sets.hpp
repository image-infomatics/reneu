#pragma once

#include <assert.h>
#include <boost/pending/disjoint_sets.hpp>
#include "../type_aliase.hpp"
#include "./utils.hpp"

// #include <boost/serialization/serialization.hpp>
// #include <boost/serialization/map.hpp>


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
DisjointSets():
    _propMapRank(_mapRank), _propMapParent(_mapParent),
    _dsets(_propMapRank, _propMapParent){}


DisjointSets(const Segmentation& seg):
        _propMapRank(_mapRank), _propMapParent(_mapParent),
        _dsets(_propMapRank, _propMapParent){
    
    auto segids = get_nonzero_segids(seg);
    for(const auto& segid : segids){
        _dsets.make_set(segid);
    }
}



// friend class boost::serialization::access;
// template<class Archive>
// void serialize(Archive ar, const unsigned int version){
//     ar & _mapRank;
//     ar & _mapParent;
//     ar & _propMapRank;
//     ar & _propMapParent;
//     ar & _dsets;
// }

void make_set(const segid_t& segid ){
    _dsets.make_set(segid);
}

void union_set(segid_t s0, segid_t s1){
    _dsets.union_set(s0, s1);
}

segid_t find_set(segid_t sid){
    const auto& root = _dsets.find_set(sid);
    if(root == 0)
        return sid;
    else
        return root;
}

auto merge_array(const xt::xtensor<segid_t, 2>& arr){
    std::set<std::pair<segid_t, segid_t>> pairs = {};
    assert(arr.shape(0) == 2);

    // in case there exist a lot of duplicates in this array
    // we make a small set first to make it more efficient
    for(std::size_t idx=0; idx<arr.shape(1); idx++){
        const auto& segid0 = arr(0, idx);
        const auto& segid1 = arr(1, idx);
        // const auto& pair = std::make_tuple(segid0, segid1);
        pairs.emplace(segid0, segid1);
    }

    for(const auto& [segid0, segid1] : pairs){
        make_set(segid0);
        make_set(segid1);
        union_set(segid0, segid1);
    }
}

inline auto py_merge_array(xt::pytensor<segid_t, 2>& pyarr){
    return merge_array(std::move(pyarr));
}

auto to_array(){
    std::vector<std::pair<segid_t, segid_t>> pairs = {};
    for(const auto& [segid0, parent]: _mapParent){
        const auto& root = find_set(segid0);
        if(root != segid0)
            pairs.emplace_back(segid0, root);
    }

    const auto& pairNum = pairs.size();
    xt::xtensor<segid_t, 2>::shape_type sh = {2, pairNum};
    auto arr = xt::empty<segid_t>(sh);
    for(std::size_t idx=0; idx<pairNum; idx++){
        const auto& [segid0, root] = pairs[idx];
        arr(0, idx) = segid0;
        arr(1, idx) = root;
    }
    return arr;
}

auto relabel(Segmentation&& seg){
    auto segids = get_nonzero_segids(seg);
    // Flatten the parents tree so that the parent of every element is its representative.
    _dsets.compress_sets(segids.begin(), segids.end());
    std::cout<< "get "<< 
                _dsets.count_sets(segids.begin(), segids.end()) << 
                " final objects."<< std::endl;

    std::cout<< "relabel the fragments to a flat segmentation." << std::endl;
    const auto& [sz, sy, sx] = seg.shape();
    for(std::size_t z=0; z<sz; z++){
        for(std::size_t y=0; y<sy; y++){
            for(std::size_t x=0; x<sx; x++){
                const auto& sid = seg(z,y,x);
                if(sid > 0){
                    const auto& rootID = find_set(sid);
                    if(sid!=rootID){
                        assert(rootID > 0);
                        seg(z,y,x) = rootID;
                    }
                }
            }
        }
    }

    // this implementation will mask out all the objects that is not in the set!
    // We should not do it in the global materialization stage.
    // std::transform(seg.begin(), seg.end(), seg.begin(), 
    //    [this](segid_t segid)->segid_t{return this->find_set(segid);}
    // );
    return seg;
}

inline auto py_relabel(PySegmentation& pyseg){
    return relabel(std::move(pyseg));
}

};

} // namespace reneu