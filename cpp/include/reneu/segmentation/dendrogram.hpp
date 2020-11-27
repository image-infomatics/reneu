#pragma once

#include <vector>

#include <xtensor/xtensor.hpp>
#include "disjoint_sets.hpp"
#include "reneu/type_aliase.hpp"


namespace reneu{


struct DendEdge{
segid_t segid0, segid1;
aff_edge_t affinity;
};

class Dendrogram{
private:
// segid0, segid1, affinity
std::vector<DendEdge> _dend;

// the lowest threshold we can go in this dendrogram.
// dendrogram is created by iterative merging and we stop the merging in a threshold
// Thus, it is an error if we merge supervoxels lower than this threshold
aff_edge_t _minThreshold;

public:
Dendrogram(aff_edge_t minThreshold): _dend({}), _minThreshold(minThreshold){}

auto as_array(){
    xt::xtensor<aff_edge_t, 2>::shape_type sh = {_dend.size(),3};
    auto arr = xt::empty<aff_edge_t>(sh);
    for(std::size_t i=0; i<_dend.size(); i++){
        const auto& dendEdge = _dend[i];
        arr(i, 0) = dendEdge.segid0;
        arr(i, 1) = dendEdge.segid1;
        arr(i, 2) = dendEdge.affinity;
    }
    return arr;
}

// Note that the edges should be pushed in descending order
// this is an implicity assumption, otherwise the materialize function will not work correctly!
inline auto push_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& affinity){
    assert(affinity >= _minThreshold);
    _dend.emplace_back(DendEdge({segid0, segid1, affinity}));
}

auto materialize(Segmentation&& seg, const aff_edge_t& threshold){
    assert(threshold >= _minThreshold);

    std::cout<< "build disjoint set..." << std::endl;
    auto dsets = DisjointSets(seg); 

    for(const auto& dendEdge: _dend){
        const auto& segid0 = dendEdge.segid0;
        const auto& segid1 = dendEdge.segid1;
        const auto& affinity = dendEdge.affinity;

        // Union the two sets that contain elements x and y. 
        // This is equivalent to link(find_set(x),find_set(y)).
        dsets.union_set(segid0, segid1);
    }
    dsets.relabel(seg);
    return seg;
}

inline auto py_materialize(PySegmentation& pySeg, const aff_edge_t& threshold){
    return materialize(std::move(pySeg), threshold);
}

}; // class of Dendrogram

}// namespace of reneu