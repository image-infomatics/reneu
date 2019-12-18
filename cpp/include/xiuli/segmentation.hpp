#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "type_aliase.hpp"

namespace xiuli{

class RegionEdge{
public:
    // segid0 should always be smaller than segid1 for fast indexing
    const segid_t segid0;
    const segid_t segid1;
    uint32_t count;
    aff_edge_t weight_sum;

RegionEdge(const segid_t &segid0_, const segid_t &segid1_, 
                uint32_t count_, aff_edge_t weight_sum_): segid0(segid0_), segid1(segid1_), 
                count(count_), weight_sum(weight_sum_){
    assert(segid0 < segid1);
}

inline aff_edge_t get_mean(){
    return weight_sum / count;
}

friend bool operator<(const RegionEdge &left, const RegionEdge &right);

}; // class of edge

inline bool operator<(RegionEdge &left, RegionEdge &right){
    return left.get_mean() < right.get_mean();
}


class RegionGraph{
private:
    /* segment id1, segment id2, number of affinity edges, sum of affinity edges
     we use float as the data type of edge count to make it consistent with weight
     we'll do division of total weight and count, same data type will avoid data type
     conversion overhead. 
    */
    vector<RegionEdge> rg;
    
public:
RegionGraph(const AffinityMap &aff, const Segmentation &seg){
    // only contains x,y,z affinity edges
    assert(aff.shape[0] == 3);
    assert(aff.shape[1]==seg.shape[0]);
    assert(aff.shape[2]==seg.shape[1]);
    assert(aff.shape[3]==seg.shape[2]);

    unordered_map<pair<segid_t, segid_t>, pair<uint32_t, aff_edge_t>> rg_map = {};
    for(size_t z=0; z<seg.shape[0]; z++){
        for(size_t y=0; y<seg.shape[1]; y++){
            for(size_t x=0; x<seg.shape[2]; x++){
                
                auto seg1 = seg[z,y,x]; 

                if (z>0 && seg1!=seg[z-1,y,x]){
                    auto key = minmax(seg1, seg[z-1,y,x]);
                    rg_map[key].first += 1;
                    rg_map[key].second += aff[2,z,y,x];
                }
                
                if (y>0 && seg1!=seg[z,y-1,x]){
                    auto key = minmax(seg1, seg[z,y-1,x]);
                    rg_map[key].first += 1;
                    rg_map[key].second += aff[1,z,y,x];
                }
                
                if (x>0 && seg1!=seg[z,y,x-1]){
                    auto key = minmax(seg1, seg[z,y,x-1]);
                    rg_map[key].first += 1;
                    rg_map[key].second += aff[0,z,y,x];
                }

            }
        }
    }

    // make it more solid as a vector
    for(const auto & it: rg_map){
        const auto edge = RegionEdge(it.first.first,  it.first.second, 
                               it.second.first, it.second.second);
        rg.push_back(edge);
    }
}
}; // class of RegionGraph


class MinimumSpanningTree{
private:
vector<tuple<segid_t, segid_t, aff_edge_t>> mst;

public:
MinimumSpanningTree( const RegionGraph &rg, const aff_edge_t &threshold ){
    // fibonacci heap might be more efficient
    // use std data structure to avoid denpendency for now
    priority_queue<RegionEdge> region_heap;

}

}; // class of minimum spanning tree

} // namespace