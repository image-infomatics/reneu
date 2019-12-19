#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <boost/pending/disjoint_sets.hpp>

#include "xtensor/xsort.hpp"
#include "xtensor/xbuilder.hpp"

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

RegionEdge(const pair<segid_t, segid_t> &ids, 
            pair<uint32_t, aff_edge_t> &weights): segid0(ids.first), segid1(ids.second), 
            count(weights.first), weight_sum(weights.second){}

RegionEdge( pair<pair<segid_t, segid_t>, 
            pair<uint32_t, aff_edge_t>> & kv): segid0(kv.first.first), segid1(kv.first.second), count(kv.second.first), weight_sum(kv.second.second){}


inline aff_edge_t get_mean(){
    return weight_sum / count;
}

friend bool operator<(const RegionEdge &left, const RegionEdge &right);

}; // class of edge

inline bool operator<(RegionEdge &left, RegionEdge &right){
    return left.get_mean() < right.get_mean();
}

class MinimumSpanningTree{
private:

using MST = vector<tuple<segid_t, segid_t, aff_edge_t>>;
MST mst;
const Segmentation fragments;

auto emplace_back(const segid_t &segid0, const segid_t &segid1, const aff_edge_t &weight){
    assert( segid0 < segid1 );
    mst.emplace_back(segid0, segid1, weight);
}

auto build_region_graph(const AffinityMap &affs){
    unordered_map<pair<segid_t, segid_t>, pair<uint32_t, aff_edge_t>> rg = {};
    // only contains x,y,z affinity rg
    assert(affs.shape[0] == 3);
    assert(affs.shape[1]==fragments.shape[0]);
    assert(affs.shape[2]==fragments.shape[1]);
    assert(affs.shape[3]==fragments.shape[2]);

    for(size_t z=0; z<fragments.shape[0]; z++){
        for(size_t y=0; y<fragments.shape[1]; y++){
            for(size_t x=0; x<fragments.shape[2]; x++){
                
                auto segid = fragments[z,y,x];
                if(segid==0){ continue; } 

                if (z>0 && fragments[z-1,y,x]>0 && segid!=fragments[z-1,y,x]){
                    auto key = minmax(segid, fragments[z-1,y,x]);
                    rg[key].first += 1;
                    rg[key].second += affs[2,z,y,x];
                }
                
                if (y>0 && fragments[z,y-1,x]>0 && segid!=fragments[z,y-1,x]){
                    auto key = minmax(segid, fragments[z,y-1,x]);
                    rg[key].first += 1;
                    rg[key].second += affs[1,z,y,x];
                }
                
                if (x>0 && fragments[z,y,x-1]>0 && segid!=fragments[z,y,x-1]){
                    auto key = minmax(segid, fragments[z,y,x-1]);
                    rg[key].first += 1;
                    rg[key].second += affs[0,z,y,x];
                }
            }
        }
    }
    return rg;
}

auto greedy_mean_affinity_agglomeration( const AffinityMap &affs, 
                                        const aff_edge_t &minThreshold){
    auto rg = build_region_graph(affs);

    unordered_map<segid_t, unordered_set<segid_t>> segid2neighbors;
    for(const auto &edge: rg){
        auto [segid0, segid1] = edge.first;
        segid2neighbors[ segid0 ].insert( segid1 );
        segid2neighbors[ segid1 ].insert( segid0 );
    }

    // fibonacci heap might be more efficient
    // use std data structure to avoid denpendency for now
    priority_queue<RegionEdge> region_heap;
    for(const auto &edge : rg){
        region_heap.emplace(edge);
    }

    // merge regions from the highest affinity edge
    while(!region_heap.empty()){
        auto edge = region_heap.top();
        if( edge.get_mean() < minThreshold ){
            cout<< "reached mean affinity lower than threshold: "<< edge.get_mean()<< endl;
            break;
        }
        region_heap.pop();

        const auto &segid0 = edge.segid0;
        const auto &segid1 = edge.segid1; 
        // add this edge to mst 
        mst.emplace_back(segid0, segid1, rg[make_pair(segid0, segid1)]);
        
        // merge segid1 to segid0 and update the weights
        // always merge to lower id of segments
        for(const auto &neighbor_segid : segid2neighbors[segid1]){
            if(neighbor_segid != segid0){
                // merge the affinity edge count and weight sum
                auto old_segid_pair = minmax( segid1, neighbor_segid);
                auto new_segid_pair = minmax( segid0, neighbor_segid);
                rg[new_segid_pair].first += rg[old_segid_pair].first;
                rg[new_segid_pair].second += rg[old_segid_pair].second;
            }
        }
    }
    return mst;
}

public:
MinimumSpanningTree(const PyAffinityMap &affs, const PySegmentation &fragments_, 
                    const aff_edge_t &minThreshold): fragments(fragments_){
    greedy_mean_affinity_agglomeration(affs, minThreshold);
}

auto segment( const aff_edge_t &threshold ){
    auto index2segid = xt::unique(fragments);
    auto segNum = index2segid.size();

    unordered_map<segid_t, segid_t> segid2index;
    for(size_t index=0; index<segNum; index++){
        auto segid = index2segid[ index ];
        segid2index[ segid ] = index;
    }

    xt::xtensor<segid_t, 1> rank = xt::ones<segid_t>(segNum);
    xt::xtensor<segid_t, 1> parent = xt::arange<segid_t>(0, segNum);
    boost::disjoint_sets dsets(rank.data(), parent.data());

    for(const auto &[segid0, segid1, aff] : mst){
        if(aff>=threshold){
            // union these two sets
            auto index0 = segid2index[ segid0 ];
            auto index1 = segid2index[ segid1 ];
            // Union the two sets that contain elements x and y. 
            // This is equivalent to link(find_set(x),find_set(y)).
            dsets.union_set(index0, index1);
        }
    }

    // Flatten the parents tree so that the parent of every element is its representative.
    dsets.compress_sets(parent.begin(), parent.end());

    cout<< "get "<< dsets.count_sets(parent.begin(), parent.end()) << " segments after merging."<< endl;

    cout<< "relabel the fragments to a flat segmentation." << endl;
    // copy fragments  
    Segmentation segmentation = fragments;
    for(auto it=segmentation.begin(); it!=segmentation.end(); it++){
        auto childSegid = *it;
        auto childIndex = segid2index[ childSegid ];
        auto parentIndex = parent[ childIndex ];
        auto parentSegid = index2segid[ parentIndex ];
        *it = parentSegid;
    }
    return segmentation;
}


}; // class of minimum spanning tree



} // namespace