#pragma once

#include <vector>
#include <initializer_list>

#include <xtensor/xsort.hpp>

#include <boost/pending/disjoint_sets.hpp>

#include "reneu/type_aliase.hpp"
#include "utils.hpp"


namespace reneu{

class RegionEdge{
public:
// we use the same type for both value for type stability in the average division. 
aff_edge_t count;
aff_edge_t sum;
// version is used to tell whether an edge is outdated or not in priority queue
size_t version;

RegionEdge(): count(0.), sum(0.), version(1){}
RegionEdge(const aff_edge_t& aff): count(1), sum(aff), version(1){}

friend std::ostream& operator<<(std::ostream& os, const RegionEdge& re);

inline aff_edge_t get_mean(){
    return sum / count;
}

inline void accumulate(const aff_edge_t& aff){
    count++;
    sum += aff;
}

inline void absorb(RegionEdge& re2){
    count += re2.count;
    sum += re2.sum;
    version++;
    re2.cleanup();
}

inline void cleanup(){
    count = 0;
    sum = 0;
    // other edges in the priority queue will always be outdated!
    version = std::numeric_limits<size_t>::max();
}

}; // class of RegionEdge

std::ostream& operator<<(std::ostream& os, const RegionEdge& re){
    os << "count: " << re.count << ", sum: "<< re.sum << ", version: "<< re.version << ". ";
    return os;
}


class RegionGraph{
private:
    using Neighbors = std::map<segid_t, size_t>;
    std::map<segid_t, Neighbors> _rg;
    std::vector<RegionEdge> _edgeList;

inline bool has_connection(const segid_t& sid0, const segid_t& sid1){
    auto& neighbors = _rg[sid0];
    return (neighbors.find(sid1) != neighbors.end());
}

inline void accumulate_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff){
    // we assume that segid0 is greater than 0 !
    if(segid1>0 && segid0 != segid1){
        if(has_connection(segid0, segid1)){
            auto& edgeIndex = _rg[segid0][segid1];
            _edgeList[edgeIndex].accumulate(aff);
        } else {
            // create a new edge
            _edgeList.emplace_back(RegionEdge(aff));
            size_t edgeIndex = _edgeList.size()-1;
            _rg[segid0][segid1] = edgeIndex;
            _rg[segid1][segid0] = edgeIndex;
        }
    }
    return;
}


public:
/**
 * @brief build region graph. 
 * 
 */
RegionGraph(const AffinityMap& affs, const Segmentation& fragments){
    // only contains x,y,z affinity 
    // Note that our format of affinity channels are ordered by x,y,z
    // although we are using C order with z,y,x in indexing!
    // This is reasonable, because the last axis x is always changing fastest in memory
    // when we tranvers the memory, the x axis changes first, so this is sort of 
    // consistent with the order of channels. 

    std::cout<< "accumulate the affinity edges..." << std::endl;
    for(std::ptrdiff_t z=0; z<fragments.shape(0); z++){
        for(std::ptrdiff_t y=0; y<fragments.shape(1); y++){
            for(std::ptrdiff_t x=0; x<fragments.shape(2); x++){
                const auto segid = fragments(z,y,x);
                // skip background voxels
                if(segid>0){ 
                    if (z>0)
                        accumulate_edge(segid, fragments(z-1,y,x), affs(2,z,y,x));
                    
                    if (y>0)
                        accumulate_edge(segid, fragments(z,y-1,x), affs(1,z,y,x));
                    
                    if (x>0)
                        accumulate_edge(segid, fragments(z,y,x-1), affs(0,z,y,x));
                }
            }
        }
    }
}

auto greedy_merge_until(Segmentation&& fragments, const aff_edge_t& threshold){

    std::cout<< "build priority queue..." << std::endl;
    struct EdgeInQueue{
        segid_t segid0;
        segid_t segid1;
        aff_edge_t aff;
        size_t version;
    };
    // fibonacci heap might be more efficient
    // use std data structure to avoid denpendency for now
    auto cmp = [](const EdgeInQueue& left, const EdgeInQueue& right){
        return left.aff < right.aff;
    };
    // TO-DO: replace with fibonacci heap
    std::priority_queue<EdgeInQueue, vector<EdgeInQueue>, decltype(cmp)> heap(cmp);
    for(auto& [segid0, neighbors0] : _rg){
        for(auto& [segid1, edgeIndex] : neighbors0){
            if(segid0 < segid1){
                auto meanAff = _edgeList[edgeIndex].get_mean();
                if(meanAff > threshold){
                    // initial version is set to 1
                    heap.emplace(EdgeInQueue({segid0, segid1, meanAff, 1}));
                }
            }
        }
    }

    std::cout<< "build disjoint set..." << std::endl;
    using Rank_t = std::map<segid_t, size_t>;
    using Parent_t = std::map<segid_t, segid_t>;
    using PropMapRank_t = boost::associative_property_map<Rank_t>;
    using PropMapParent_t = boost::associative_property_map<Parent_t>;
    Rank_t mapRank;
    Parent_t mapParent;
    PropMapRank_t propMapRank(mapRank);
    PropMapParent_t propMapParent(mapParent);
    boost::disjoint_sets<PropMapRank_t, PropMapParent_t> dsets( propMapRank, propMapParent);

    auto segids = get_nonzero_segids(fragments);
    for(const auto& segid : segids){
        dsets.make_set(segid);
    }

    std::cout<< "iterative greedy merging..." << std::endl; 
    size_t mergeNum = 0;
    while(!heap.empty()){
        const auto& edgeInQueue = heap.top();
        segid_t segid0 = edgeInQueue.segid0;
        segid_t segid1 = edgeInQueue.segid1;
        heap.pop();
        
        const auto& edgeIndex = _rg[segid1][segid0];
        const auto& edge = _edgeList[edgeIndex];
        if((edge.count==0) || (edge.version > edgeInQueue.version)){
            // skip outdated region edge
            //std::cout<< "skip outdated edge: "<< segid0 << " -- "<< segid1 << 
            //                                " = " << edgeInQueue.aff<< std::endl;
            continue;
        }

        std::cout<< "merge edge: "<< edge<< std::endl;
        // merge segid1 and segid0
        mergeNum++;
        // Union the two sets that contain elements x and y. 
        // This is equivalent to link(find_set(x),find_set(y)).
        dsets.union_set(segid0, segid1);
        
        // make segid1 bigger than segid0
        // always merge object with less neighbors to more neighbors
        if(_rg[segid0].size() > _rg[segid1].size()){
            std::swap(segid0, segid1);
        }
        auto& neighbors0 = _rg[segid0];
        auto& neighbors1 = _rg[segid1];

        // merge all the edges to segid1
        for(const auto& [nid0, edgeIndex] : neighbors0){
            auto& edge = _edgeList[edgeIndex];
            // skip the bad edges
            // we should not have bad edges here since have already erased them in the 
            // region graph! There is a bug here!
            if(edge.count == 0) continue;

            if(nid0 != segid1){
                if (has_connection(segid1, nid0)){
                    // combine two region edges
                    const auto& newEdgeIndex = neighbors1[nid0];
                    auto& newEdge = _edgeList[newEdgeIndex];
                    newEdge.absorb(edge);
                    const auto& meanAff = newEdge.get_mean();
                    if(meanAff > threshold){
                        heap.emplace(
                            EdgeInQueue({segid1, nid0, meanAff, newEdge.version})
                        );
                    }
                } else {
                    // directly assign nid0-segid0 to nid0-segid1
                    neighbors1[nid0] = edgeIndex;
                    _rg[nid0][segid1] = edgeIndex;
                    // make the original edge in priority queue outdated
                    std::cout<< "edge before assignment: "<< edge << std::endl;
                    edge.version++;
                    std::cout<< "edge after  assignment: " << edge << std::endl;
                    const auto& meanAff = edge.get_mean();
                    if(meanAff > threshold){
                        heap.emplace(EdgeInQueue({nid0, segid1, meanAff, edge.version}                        ));
                    }
                }
            }
            _rg[nid0].erase(segid0);
        }
        _rg.erase(segid0);
    }

    // Flatten the parents tree so that the parent of every element is its representative.
    dsets.compress_sets(segids.begin(), segids.end());
    std::cout<< "merged "<< mergeNum << " times to get "<< 
                dsets.count_sets(segids.begin(), segids.end()) << 
                " final objects."<< std::endl;

    std::cout<< "relabel the fragments to a flat segmentation." << std::endl;
    std::transform(fragments.begin(), fragments.end(), fragments.begin(), 
        [&dsets](segid_t segid)->segid_t{return dsets.find_set(segid);} 
    );

    return fragments;
}

inline auto py_greedy_merge_until(PySegmentation& pyseg, const aff_edge_t& threshold){
    return greedy_merge_until(std::move(pyseg), threshold);
}

};

} // namespace reneu