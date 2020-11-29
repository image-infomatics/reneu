#pragma once

#include <vector>
#include <algorithm>
#include <execution>

#include <xtensor/xtensor.hpp>
#include "disjoint_sets.hpp"
#include "reneu/type_aliase.hpp"


namespace reneu{


struct DendEdge{
segid_t segid0, segid1;
aff_edge_t affinity;
};

bool compare_edgeList_edge(const DendEdge& de1, const DendEdge& de2) {
    return (de1.affinity > de2.affinity);
}

class Dendrogram{
private:
// segid0, segid1, affinity
std::vector<DendEdge> _edgeList;

// the lowest threshold we can go in this dendrogram.
// dendrogram is created by iterative merging and we stop the merging in a threshold
// Thus, it is an error if we merge supervoxels lower than this threshold
aff_edge_t _minThreshold;

public:
Dendrogram(aff_edge_t minThreshold): _edgeList({}), _minThreshold(minThreshold){}

void print() const {
    std::cout<<"dendrogram minimum threshold: "<< _minThreshold<< std::endl;
    for(const auto& edge: _edgeList){
        std::cout<< edge.segid0 << "--"<< edge.segid1<<":"<<edge.affinity<<", ";
    }
    std::cout<<std::endl;
}

auto as_array() const {
    xt::xtensor<aff_edge_t, 2>::shape_type sh = {_edgeList.size(),3};
    auto arr = xt::empty<aff_edge_t>(sh);
    for(std::size_t i=0; i<_edgeList.size(); i++){
        const auto& dendEdge = _edgeList[i];
        arr(i, 0) = dendEdge.segid0;
        arr(i, 1) = dendEdge.segid1;
        arr(i, 2) = dendEdge.affinity;
    }
    return arr;
}

auto get_min_threshold() const {
    return _minThreshold;
}

// Note that the edges should be pushed in descending order
// this is an implicity assumption, otherwise the materialize function will not work correctly!
inline auto push_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& affinity){
    assert(affinity >= _minThreshold);
    _edgeList.emplace_back(DendEdge({segid0, segid1, affinity}));
}

auto merge(Dendrogram other){
    _minThreshold = std::min(_minThreshold, other._minThreshold);
    for(const auto& dendEdge: other._edgeList){
        _edgeList.push_back(dendEdge);
    }
    // sort the dendEdge?
    std::sort(std::execution::par_unseq, _edgeList.begin(), _edgeList.end(), compare_edgeList_edge);
}

auto materialize(Segmentation&& seg, const aff_edge_t& threshold) const {
    assert(threshold >= _minThreshold);

    std::cout<< "build disjoint set..." << std::endl;
    auto dsets = DisjointSets(seg); 

    for(const auto& dendEdge: _edgeList){
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

inline auto py_materialize(PySegmentation& pySeg, const aff_edge_t& threshold) const {
    return materialize(std::move(pySeg), threshold);
}

}; // class of Dendrogram

}// namespace of reneu