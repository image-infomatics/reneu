#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <initializer_list>

#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include "../type_aliase.hpp"
#include "utils.hpp"
#include "disjoint_sets.hpp"
#include "dendrogram.hpp"
#include "priority_queue.hpp"


namespace reneu{

class RegionGraph;

class RegionEdge{
public:
// we use the same type for both value for type stability in the average division. 
aff_edge_t count;
aff_edge_t sum;
segid_t segid0;
segid_t segid1;
// version is used to tell whether an edge is outdated or not in priority queue
size_t version;

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version){
    ar & segid0;
    ar & segid1;
    ar & count;
    ar & sum;
}


public:
friend std::ostream& operator<<(std::ostream& os, const RegionEdge& re);
// friend class RegionGraph;

RegionEdge(): count(0), sum(0), segid0(0), segid1(0) {}

RegionEdge(const segid_t& _segid0, const segid_t& _segid1, const aff_edge_t& aff): 
            count(1), sum(aff), segid0(_segid0), segid1(_segid1), version(1){}


inline aff_edge_t get_mean() const {
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
    segid0=0;
    segid1=0;
    count = 0;
    sum = 0;
    // other edges in the priority queue will always be outdated!
    version = std::numeric_limits<size_t>::max();
}

// inline bool operator<(const RegionEdge& other) const {
//     return get_mean() < other.get_mean();
// }

}; // class of RegionEdge

std::ostream& operator<<(std::ostream& os, const RegionEdge& re){
    os << re.segid0 << "--" << re.segid1 << "| count: " << re.count << ", sum: "<< re.sum << 
        " mean aff: "<< re.sum/re.count <<", version: "<< re.version << ". ";
    return os;
}

using RegionEdgeList = std::vector<RegionEdge>;
using Neighbors = std::map<segid_t, size_t>;
using Segid2Neighbor = std::map<segid_t, Neighbors>;

class RegionGraph{
protected:
    // flat_map is much slower than std::map 
    // using RegionMap = boost::container::flat_map<segid_t, Neighbors>;

    Segid2Neighbor _segid2neighbor;
    RegionEdgeList _edgeList;

    
friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version){
    ar & _segid2neighbor;
    ar & _edgeList;
}

inline auto _get_edge_index(const segid_t& sid0, const segid_t& sid1) const {
    return _segid2neighbor.at(sid0).at(sid1);
}

inline bool has_connection (const segid_t& sid0, const segid_t& sid1) const {
    // return _segid2neighbor[sid0].count(sid1);
    return (_segid2neighbor.count(sid0)) && (_segid2neighbor.at(sid0).count(sid1));
    // return (_segid2neighbor.find(sid0) != _segid2neighbor.end()) && (_segid2neighbor[sid0].find(sid1)!= _segid2neighbor[sid0].end());
}

inline void accumulate_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff){
    // we assume that segid0 is greater than 0 !
    if( (segid1>0) && (segid0 != segid1)){
        if(has_connection(segid0, segid1)){
            const auto& edgeIndex = _segid2neighbor.at(segid0).at(segid1);
            _edgeList[edgeIndex].accumulate(aff);
        } else {
            // create a new edge
            _edgeList.emplace_back(segid0, segid1, aff);
            const size_t& edgeIndex = _edgeList.size() - 1;
            _segid2neighbor[segid0][segid1] = edgeIndex; 
            _segid2neighbor[segid1][segid0] = edgeIndex;
        }
    }
    return;
}


auto _build_priority_queue (const aff_edge_t& threshold) const {
    PriorityQueue heap;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            // the connection is bidirectional, 
            // so only half of the pairs need to be handled
            if(segid0 < segid1){
                const auto& meanAff = _edgeList[edgeIndex].get_mean();
                if(meanAff > threshold){
                    // initial version is set to 1
                    heap.emplace_back(segid0, segid1, meanAff, 1);
                }
            }
        }
    }

    heap.make_heap();

    std::cout<< "initial heap size: "<< heap.size() << std::endl;
    return heap;
}

auto _merge_segments(segid_t& segid0, segid_t& segid1, const RegionEdge& edge,
            Dendrogram& dend, PriorityQueue& heap, 
            const aff_edge_t& threshold){
    
    dend.push_edge(segid0, segid1, edge.get_mean());

    // always merge object with less neighbors to more neighbors
    if(_segid2neighbor.at(segid0).size() > _segid2neighbor.at(segid1).size()){
        std::swap(segid0, segid1);
    }
    auto& neighbors0 = _segid2neighbor.at(segid0);
    auto& neighbors1 = _segid2neighbor.at(segid1);
    neighbors0.erase(segid1);
    neighbors1.erase(segid0);

    // merge all the edges to segid1
    for(auto& [nid0, neighborEdgeIndex] : neighbors0){
        _segid2neighbor.at(nid0).erase(segid0);
        
        auto& neighborEdge = _edgeList[neighborEdgeIndex];

        if (has_connection(segid1, nid0)){
            // combine two region edges
            const auto& newEdgeIndex = neighbors1[nid0];
            auto& newEdge = _edgeList[newEdgeIndex];
                            
            auto meanAff0 = neighborEdge.get_mean();
            auto meanAff1 = newEdge.get_mean();
            newEdge.absorb(neighborEdge);
            const auto& meanAff = newEdge.get_mean();
            
            if(meanAff > threshold){
                heap.emplace_push(nid0, segid1, meanAff, newEdge.version);
            }
        } else {
            // directly assign nid0-segid0 to nid0-segid1
            _segid2neighbor.at(segid1)[nid0] = neighborEdgeIndex;
            _segid2neighbor.at(nid0)[segid1] = neighborEdgeIndex;
            // make the original edge in priority queue outdated
            // this is a new edge, so the version should be 1
            neighborEdge.version = 1;
            neighborEdge.segid0 = nid0;
            neighborEdge.segid1 = segid1;

            const auto& meanAff = neighborEdge.get_mean();
            if(meanAff > threshold){
                heap.emplace_push(nid0, segid1, meanAff, neighborEdge.version);
            }
        }
    }
    _segid2neighbor.erase(segid0);

}
public:

RegionGraph(): _segid2neighbor({}), _edgeList({}){}
RegionGraph(const Segid2Neighbor& segid2neighbor, const RegionEdgeList& edgeList): 
    _segid2neighbor(segid2neighbor), _edgeList(edgeList) {}

/**
 * @brief Construct a new Region Graph object
 * 
 * @param affs 
 * @param fragments 
 */
RegionGraph(const AffinityMap& affs, const Segmentation& fragments) {
    // only contains x,y,z affinity 
    // Note that our format of affinity channels are ordered by x,y,z
    // although we are using C order with z,y,x in indexing!
    // This is reasonable, because the last axis x is always changing fastest in memory
    // when we tranvers the memory, the x axis changes first, so this is sort of 
    // consistent with the order of channels. 
    assert(affs.shape(1) == fragments.shape(0));
    assert(affs.shape(1) == fragments.shape(0));
    assert(affs.shape(1) == fragments.shape(0));

    std::cout<< "accumulate the affinity edges..." << std::endl;
    for(std::size_t z=0; z<fragments.shape(0); z++){
        for(std::size_t y=0; y<fragments.shape(1); y++){
            for(std::size_t x=0; x<fragments.shape(2); x++){
                const auto& segid = fragments(z,y,x);
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
    std::cout<<"total edge number: "<< _edgeList.size() <<std::endl;
}

auto get_edge_num() const {
    std::size_t edgeNum = 0;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        edgeNum += neighbors0.size();
    }
    return edgeNum / 2;
}

std::string as_string(){
    std::ostringstream stringStream;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        stringStream << segid0 << ": ";
        for(const auto& [segid1, edgeIndex] : neighbors0){
            const auto& meanAff = _edgeList[edgeIndex].get_mean();
            stringStream << segid1 << "--" << meanAff << ", ";
        }
        stringStream<< "\n";
    }
    stringStream<<"\n";
    return stringStream.str(); 
}

auto as_array() const {
    const auto& edgeNum = get_edge_num();

    xt::xtensor<aff_edge_t, 2>::shape_type sh = {edgeNum, 3};
    auto arr = xt::empty<aff_edge_t>(sh);

    std::size_t n = 0;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            if(segid0 < segid1){
                arr(n, 0) = segid0;
                arr(n, 1) = segid1;
                arr(n, 2) = _edgeList[edgeIndex].get_mean();
                n++;
            }
        }
    }
    return arr;
}



auto greedy_merge(const Segmentation& seg, const aff_edge_t& threshold){

    std::cout<< "build priority queue..." << std::endl;
    auto heap = _build_priority_queue(threshold);

    Dendrogram dend(threshold);

    std::cout<< "iterative greedy merging..." << std::endl; 
    size_t mergeNum = 0;
    while(!heap.empty()){
        const auto& edgeInQueue = heap.pop();
        
        auto segid0 = edgeInQueue.segid0;
        auto segid1 = edgeInQueue.segid1;

        if(!has_connection(segid0, segid1)){
            continue;
        }
        
        const auto& edgeIndex = _get_edge_index(segid1, segid0);
        const auto& edge = _edgeList[edgeIndex];
        if(edge.version > edgeInQueue.version){
            // found an outdated edge
            continue;
        }
        
        // merge segid1 and segid0
        mergeNum++;
        _merge_segments(segid0, segid1, edge, dend, heap, threshold);
        
    }
    
    std::cout<< "merged "<< mergeNum << " times." << std::endl;
    return dend;
}

inline auto py_greedy_merge(const PySegmentation& pyseg, const aff_edge_t& threshold){
    return greedy_merge(pyseg, threshold);
}

}; // class of RegionGraph
} // namespace reneu