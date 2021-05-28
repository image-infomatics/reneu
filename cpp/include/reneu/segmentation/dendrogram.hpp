#pragma once

#include <vector>
#include <algorithm>
// #include <execution>
#include <string>
#include <sstream>
#include <set>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <xtensor/xtensor.hpp>
#include "disjoint_sets.hpp"
#include "../type_aliase.hpp"


namespace reneu{

class Dendrogram;

class DendEdge{
friend class Dendrogram;


public:
segid_t segid0, segid1;
aff_edge_t affinity;

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version){
    ar & segid0;
    ar & segid1;
    ar & affinity;
}


DendEdge(): segid0(0), segid1(0), affinity(0){}
DendEdge(segid_t _segid0, segid_t _segid1, aff_edge_t _affinity): 
            segid0(_segid0), segid1(_segid1), affinity(_affinity){}

bool operator<(const DendEdge& other){
    return affinity < other.affinity;
}

};


class Dendrogram{

private:
// segid0, segid1, affinity
std::vector<DendEdge> _edgeList;

// the lowest threshold we can go in this dendrogram.
// dendrogram is created by iterative merging and we stop the merging in a threshold
// Thus, it is an error if we merge supervoxels lower than this threshold
aff_edge_t _minThreshold;

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version){
    ar & _minThreshold;
    ar & _edgeList;
}

public:

Dendrogram(): _edgeList({}), _minThreshold(0){};
Dendrogram(aff_edge_t minThreshold): _edgeList({}), _minThreshold(minThreshold){}


std::string as_string() const {
    // to-do: use std::format in C++20
    std::ostringstream stringStream;
    stringStream<< "minimum threshold: "<< _minThreshold<< "\n";
    stringStream<< "edges: ";
    for(const auto& edge: _edgeList){
        stringStream<< edge.segid0 << "--"<< edge.segid1<<":"<<edge.affinity<<", ";
    }
    stringStream<<"\n";
    return stringStream.str();
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

auto get_edge_num() const {
    return _edgeList.size();
}

// Note that the edges should be pushed in descending order
// this is an implicity assumption, otherwise the materialize function will not work correctly!
inline auto push_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& affinity){
    assert(affinity >= _minThreshold);
    _edgeList.emplace_back(DendEdge(segid0, segid1, affinity));
}

void py_keep_only_contacting_edges(PySegmentation& seg, std::tuple<std::size_t, std::size_t, std::size_t> margin_sizes){
    keep_only_contacting_edges(std::move(seg), margin_sizes);
}

/*  
This function was used for global agglomeration in distributed computation.
For contacting segmentation chunks with unique object IDs, it extract the edges
connecting the neighboring segmentation chunk. The extracted dendrogram could be 
aggregated as a glocal one and used for global agglomeration of objects.

This implementation might not be correct, we might need to extract the contacting edges
by comparing the segmentation. This is a future plan to improve it.
*/
void keep_only_contacting_edges(Segmentation&& seg, std::tuple<std::size_t, std::size_t, std::size_t> margin_sizes){
    const auto& [mz, my, mx] = margin_sizes;
    assert( mz > 0 );
    assert( my > 0 );
    assert( mx > 0 );

    const auto& [sz, sy, sx] = seg.shape();
    assert(sz>mz);
    assert(sy>my);
    assert(sx>mx);
    
    // the edges across internal chunk boundary
    // only lower part edges are included
    // The upper edges will be handled in the upper chunk processing
    // this is like the voxel wise affinity edges.
    using EdgeType = std::pair<segid_t, segid_t>;
    std::set<EdgeType> edgeSet({});

    for(std::size_t y = my; y<sy-my; y++){
        for(std::size_t x = mx; x<sx-mx; x++){
            const segid_t& id0 = seg(mz-1, y, x);
            const segid_t& id1 = seg(mz, y, x);
            if(id0>0 && id1>0){
                assert(id0!=id1);
                edgeSet.emplace(std::minmax(id0, id1));
            }
        }
    }

    for(std::size_t z = mz; z<sz-mz; z++){
        for(std::size_t x=mx; x<sx-mx; x++){
            const segid_t& id0 = seg(z, my-1, x);
            const segid_t& id1 = seg(z, my, x);
            if(id0>0 && id1>0){
                assert(id0!=id1);
                edgeSet.emplace(std::minmax(id0, id1));
            }
        }
    }

    for(std::size_t z = mz; z<sz-mz; z++){
        for(std::size_t y=my; y<sy-my; y++){
            const segid_t& id0 = seg(z, y, mx-1);
            const segid_t& id1 = seg(z, y, mx);
            if(id0>0 && id1>0){
                assert(id0!=id1);
                edgeSet.emplace(std::minmax(id0, id1));
            }
        }
    }

    // To-Do: c++20 have erase_if function
    _edgeList.erase(std::remove_if(_edgeList.begin(), _edgeList.end(),
        [edgeSet](const auto& edge)->bool{
            const auto& e = std::minmax(edge.segid0, edge.segid1);
            const auto& search = edgeSet.find(e);
            return search == edgeSet.end();
        }
    ));
    return;
}

auto merge(Dendrogram other){
    _minThreshold = std::min(_minThreshold, other._minThreshold);
    for(const auto& dendEdge: other._edgeList){
        _edgeList.push_back(dendEdge);
    }
}

auto to_disjoint_sets(const aff_edge_t& threshold) const {
    std::cout<< "build disjoint set..." << std::endl;
    auto dsets = DisjointSets();

    for(const auto& edge : _edgeList){
        if(edge.affinity >= threshold){
            dsets.make_set(edge.segid0);
            dsets.make_set(edge.segid1);
        }
    }

    // the make_set and union_set should be done separately!
    for(const auto& edge : _edgeList){
        if(edge.affinity >= threshold){
            dsets.union_set(edge.segid0, edge.segid1);
        }
    }
    return dsets;
}

/* 
There are some big mergers after agglomeration.
We would like to split them up using a higher threshold.
We first get the disjoint sets and find out all the connecting edges.
The we can delete the edges.
*/
auto split_objects(
            const aff_edge_t& agglomerationThreshold, 
            const std::set<segid_t>& segids, 
            const aff_edge_t& splitThreshold) {
    assert(agglomerationThreshold < splitThreshold);
    auto dsets = to_disjoint_sets(agglomerationThreshold);

    for(auto it = _edgeList.begin(); it != _edgeList.end(); ){
        if (it->affinity < splitThreshold){
            if( segids.count( dsets.find_set(it->segid0) ) )
                it = _edgeList.erase(it);
            else if ( segids.count( dsets.find_set(it->segid1) ) )
                it = _edgeList.erase(it);
            else 
                ++it;
        } else {
            ++it;
        }
    }
}

auto materialize(Segmentation&& seg, const aff_edge_t& threshold) const {
    assert(threshold >= _minThreshold);
    auto dsets = to_disjoint_sets(threshold);
    
    auto seg2 = dsets.relabel(std::move(seg));
    return seg2;
}

inline auto py_materialize(PySegmentation& pySeg, const aff_edge_t& threshold) const {
    return materialize(std::move(pySeg), threshold);
}

}; // class of Dendrogram

}// namespace of reneu