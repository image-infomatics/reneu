#pragma once
#include <limits>
#include <map>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>

#include "region_graph.hpp"

namespace reneu{

class RegionGraphChunk: public RegionGraph{
private:

// segmentation id --> chunk surface count
std::map<segid_t, std::uint8_t> _frozenSegIDs;

// 9223372036854775808
static const segid_t FREEZING_BIT = ~(std::numeric_limits<segid_t>::max() >> 1);

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version){
    ar & boost::serialization::base_object<RegionGraph>(*this);
    ar & _frozonSegIDs; 
}


void freeze_boundary_segment_ids(const Segmentation& seg){

    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t y=0; y<seg.shape(1); y++){
            for(std::size_t x=0; x<seg.shape(2); x++){
                if(z==0 || z==seg.shape(0)-1 || y==0 || y==seg.shape(1)-1 || x==0 || x==seg.shape(2)-1){
                    const auto& segid = seg(z,y,x);
                    if(segid > 0){
                        _frozenSegIDs[segid]++;
                    }
                }
            }
        }
    }

    return;
}

inline bool is_frozen(const segid_t& sid){
    // To-Do: replace it using contains in C++20
    auto search = _frozenSegIDs.find(sid);
    return search != _frozenSegIDs.end();
}

public:

RegionGraphChunk(): RegionGraph(), _frozenSegIDs({}){}

RegionGraphChunk(const AffinityMap& affs, const Segmentation& seg): RegionGraph(affs, seg), _frozenSegIDs({}){
    freeze_boundary_segment_ids(seg); 
}


auto greedy_merge(const Segmentation& seg, const aff_edge_t& threshold){
    
}

inline auto py_greedy_merge(const Segmentation& seg, const aff_edge_t& threshold){
    return greedy_merge(seg, threshold);
}

}; // class of RegionGraphChunk

} // namespace of reneu