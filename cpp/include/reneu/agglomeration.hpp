#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>
#include <boost/pending/disjoint_sets.hpp>

#include "xtensor/xsort.hpp"
#include "xtensor/xbuilder.hpp"

#include "type_aliase.hpp"

namespace reneu{

class SupervoxelDendrogram{
private:

using SegPair = pair<segid_t, segid_t>;
using EdgeProps = pair<aff_edge_t, aff_edge_t>;
using Edge = pair<SegPair, EdgeProps>;
using Dendrogram = vector<tuple<segid_t, segid_t, aff_edge_t>>;
Dendrogram dendrogram;
const Segmentation fragments;

auto emplace_back(const segid_t &segid0, const segid_t &segid1, const aff_edge_t &weight){
    assert( segid0 < segid1 );
    dendrogram.emplace_back(segid0, segid1, weight);
}

auto build_region_graph(const AffinityMap &affs){
    unordered_map<SegPair, EdgeProps, boost::hash<SegPair>> rg = {};
    // only contains x,y,z affinity rg
    assert(affs.shape(0)== 3);
    assert(affs.shape(1)==fragments.shape(0));
    assert(affs.shape(2)==fragments.shape(1));
    assert(affs.shape(3)==fragments.shape(2));

    for(std::ptrdiff_t z=0; z<fragments.shape(0); z++){
        for(std::ptrdiff_t y=0; y<fragments.shape(1); y++){
            for(std::ptrdiff_t x=0; x<fragments.shape(2); x++){
                
                segid_t segid = fragments(z,y,x);
                // skip background voxels
                if(segid==0){ continue; } 

                if (z>0 && fragments(z-1,y,x)>0 && segid!=fragments(z-1,y,x)){
                    auto key = minmax(segid, fragments(z-1,y,x));
                    rg[key].first++;
                    rg[key].second += affs(2,z,y,x);
                    //std::cout<< "region graph, first: "<< rg[key].first << ", second: "<< rg[key].second << std::endl; 
                }
                
                if (y>0 && fragments(z,y-1,x)>0 && segid!=fragments(z,y-1,x)){
                    auto key = minmax(segid, fragments(z,y-1,x));
                    rg[key].first++;
                    rg[key].second += affs(1,z,y,x);
                }
                
                if (x>0 && fragments(z,y,x-1)>0 && segid!=fragments(z,y,x-1)){
                    auto key = minmax(segid, fragments(z,y,x-1));
                    rg[key].first++;
                    rg[key].second += affs(0,z,y,x);
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
    auto cmp = [](Edge left, Edge right){return 
                left.second.second  / left.second.first < 
                right.second.second / right.second.first;};
    // TO-DO: replace with fibonacci heap
    priority_queue<Edge, vector<Edge>, decltype(cmp)> heap(cmp);
    for(auto &edge : rg){
        heap.emplace(edge);
    }

    // merge regions from the highest affinity edge
    while(!heap.empty()){
        const auto [segPair, edgeProps] = heap.top();
        // remove the edge in heap
        heap.pop();

        const auto [segid0, segid1] = segPair;
        const aff_edge_t meanWeight = edgeProps.second / edgeProps.first;
        //std::cout<< "mean weight: "<< meanWeight << std::endl;
        if( meanWeight < minThreshold ){
            std::cout<< "reached mean affinity lower than threshold: "<< meanWeight << std::endl;
            break;
        }

        // add this edge to dendrogram 
        dendrogram.emplace_back(segid0, segid1, meanWeight);
        
        // merge segid1 to segid0 and update the weights
        // always merge to lower id of segments
        for(const segid_t &neighbor_segid : segid2neighbors[segid1]){
            if(neighbor_segid != segid0){
                // merge the affinity edge count and weight sum
                auto old_segid_pair = minmax( segid1, neighbor_segid);
                auto new_segid_pair = minmax( segid0, neighbor_segid);
                rg[new_segid_pair].first += rg[old_segid_pair].first;
                rg[new_segid_pair].second += rg[old_segid_pair].second;
                rg.erase(old_segid_pair);
            }
        }
    }
    return dendrogram;
}

public:
SupervoxelDendrogram(const PyAffinityMap &affs, const PySegmentation &fragments_, 
                    const aff_edge_t &minThreshold): fragments(fragments_){
    greedy_mean_affinity_agglomeration(affs, minThreshold);
}

auto segment( const aff_edge_t &threshold, const size_t& sizeThreshold){
    using Rank_t = std::map<segid_t, size_t>;
    using Parent_t = std::map<segid_t, segid_t>;
    using PropMapRank_t = boost::associative_property_map<Rank_t>;
    using PropMapParent_t = boost::associative_property_map<Parent_t>;

    Rank_t mapRank;
    Parent_t mapParent;
    PropMapRank_t propMapRank(mapRank);
    PropMapParent_t propMapParent(mapParent);
    boost::disjoint_sets<PropMapRank_t, PropMapParent_t> dsets( propMapRank, propMapParent);

    auto segids = xt::unique(fragments);
    for(const segid_t& segid : segids){
        dsets.make_set(segid);
    }
    
    // count voxel number
    std::map<segid_t, size_t> mapVoxelNum = {};
    segids = xt::unique(fragments);
    for(const auto& segid : segids){
        mapVoxelNum[segid] = 0;
    }
    for(const auto& segid : fragments){
        mapVoxelNum[segid]++;
    }
    
    size_t mergeNum = 0;
    // greedy mean affinity agglomeration
    for(const auto &[segid0, segid1, aff] : dendrogram){
        
        //std::cout<< "aff: "<< aff << ", voxel count 0: "<< mapVoxelNum[segid0] << 
         //               ",      voxel count 1: "<< mapVoxelNum[segid1]<< std::endl;
        if( (aff>=threshold) || 
                (mapVoxelNum[segid0] <= sizeThreshold) || 
                (mapVoxelNum[segid1] <= sizeThreshold) ){
            // Union the two sets that contain elements x and y. 
            // This is equivalent to link(find_set(x),find_set(y)).
            dsets.union_set(segid0, segid1);
            mergeNum++;

            //std::cout<< "aff: "<< aff << ", voxel count 0: "<< mapVoxelNum[segid0] << 
            //            ",      voxel count 1: "<< mapVoxelNum[segid1]<< std::endl;

            size_t mergedVoxelNum = mapVoxelNum[segid0] + mapVoxelNum[segid1];
            mapVoxelNum[segid0] = mergedVoxelNum;
            mapVoxelNum[segid1] = mergedVoxelNum;
        }
    }
    
    // Flatten the parents tree so that the parent of every element is its representative.
    dsets.compress_sets(segids.begin(), segids.end());

    std::cout<< "merged "<< mergeNum << " times to get "<< 
                dsets.count_sets(segids.begin(), segids.end()) << 
                " final objects."<< std::endl;

    std::cout<< "relabel the fragments to a flat segmentation." << std::endl;
    // copy fragments  
    Segmentation segmentation = fragments;
    std::transform(fragments.begin(), fragments.end(), segmentation.begin(), 
        [&dsets](segid_t segid)->segid_t{return dsets.find_set(segid);} 
    );

    return segmentation;
}


}; // class of minimum spanning tree



} // namespace
