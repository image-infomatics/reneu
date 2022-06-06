#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <initializer_list>

// #include <absl/container/node_hash_map.h>
#include <tsl/robin_map.h>

#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "../types.hpp"
#include "utils.hpp"
#include "disjoint_sets.hpp"
#include "dendrogram.hpp"
#include "priority_queue.hpp"

// BOOST_SERIALIZATION_SPLIT_MEMBER()

namespace reneu{

class RegionGraph;

struct EdgeInHeap{
    segid_t segid0;
    segid_t segid1;
    aff_edge_t aff;
    size_t version;

    // constructor for emplace operation
    EdgeInHeap(
        const segid_t& segid0_, const segid_t& segid1_, 
        const aff_edge_t& aff_, const std::size_t& version_):
        segid0(segid0_), segid1(segid1_), aff(aff_), version(version_){}

    bool operator<(const EdgeInHeap& other) const {
        return aff < other.aff;
    }
};
// struct LessThanByAff{
//     bool operator()(const EdgeInQueue& lhs, const EdgeInQueue& rhs) const {
//         return lhs.aff < rhs.aff;
//     }
// };
// using PriorityQueue = std::priority_queue<EdgeInQueue, std::vector<EdgeInQueue>, std::less<EdgeInQueue>>;


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
    ar & this->version;
}


public:
friend std::ostream& operator<<(std::ostream& os, const RegionEdge& re);
// friend class RegionGraph;

RegionEdge(): count(0), sum(0), segid0(0), segid1(0) {}

RegionEdge(const segid_t& _segid0, const segid_t& _segid1, 
                const aff_edge_t& _sum, aff_edge_t _count = 1.): 
            count(_count), sum(_sum), segid0(_segid0), segid1(_segid1), version(1){}


inline aff_edge_t get_mean() const {
    return sum / count;
}

inline void accumulate(const aff_edge_t& newEdgeSum, std::size_t newEdgeNum = 1){
    count += newEdgeNum;
    sum += newEdgeSum;
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
using Neighbors = std::unordered_map<segid_t, size_t>;
// using Segid2Neighbor = absl::node_hash_map<segid_t, Neighbors>;
using Segid2Neighbor = tsl::robin_map<segid_t, Neighbors>;

class RegionGraph{
protected:
    // flat_map is much slower than std::map 
    // using RegionMap = boost::container::flat_map<segid_t, Neighbors>;

    Segid2VoxelNum _segid2voxelNum;
    Segid2Neighbor _segid2neighbor;
    RegionEdgeList _edgeList;

    
friend class boost::serialization::access;
BOOST_SERIALIZATION_SPLIT_MEMBER()
template<class Archive>
void save(Archive& ar, const unsigned int version) const {
    // invoke serialization of the base class 
    // ar << boost::serialization::base_object<const base_class_of_T>(*this);
    ar & _edgeList;
    
    // ar & _segid2neighbor;
    auto serializer = [&ar](const auto& v) { ar & v; };
    _segid2neighbor.serialize(serializer);

}

template<class Archive>
void load(Archive& ar, const unsigned int version){
    // invoke serialization of the base class 
    // ar >> boost::serialization::base_object<base_class_of_T>(*this);
    ar & _edgeList;
    // _segid2neighbor.deserialize(ar);

    auto deserializer = [&ar]<typename U>() { U u; ar & u; return u; };
    _segid2neighbor = Segid2Neighbor::deserialize(deserializer);

}


inline auto _get_edge(const segid_t& sid0, const segid_t& sid1) const {
    const auto& edgeIndex = _segid2neighbor.at(sid0).at(sid1);
    return _edgeList[edgeIndex];
}

inline bool _has_connection (const segid_t& sid0, const segid_t& sid1) const {
    // return _segid2neighbor[sid0].count(sid1);
    // return (_segid2neighbor.find(sid0) != _segid2neighbor.end()) && (_segid2neighbor[sid0].find(sid1)!= _segid2neighbor[sid0].end());
    return (_segid2neighbor.count(sid0)) && (_segid2neighbor.at(sid0).count(sid1));
}

inline void _accumulate_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& newEdgeSum, aff_edge_t newEdgeNum= 1){
    // we assume that segid0 is greater than 0 !
    if( (segid1>0) && (segid0 != segid1)){
        if(_has_connection(segid0, segid1)){
            const auto& edgeIndex = _segid2neighbor.at(segid0).at(segid1);
            _edgeList[edgeIndex].accumulate(newEdgeSum, newEdgeNum);
        } else {
            // create a new edge
            _edgeList.emplace_back(segid0, segid1, newEdgeSum, newEdgeNum);
            const size_t& edgeIndex = _edgeList.size() - 1;
            _segid2neighbor[segid0][segid1] = edgeIndex; 
            _segid2neighbor[segid1][segid0] = edgeIndex;
        }
    }
    return;
}


auto _build_priority_queue (const aff_edge_t& threshold) const {
    PriorityQueue<EdgeInHeap> heap;
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
            PriorityQueue<EdgeInHeap>& heap, 
            const aff_edge_t& affinityThreshold){
    
    // always merge object with less neighbors to more neighbors
    // after swaping, we'll merge segid0 to segid1
    if(_segid2neighbor.at(segid0).size() > _segid2neighbor.at(segid1).size()){
        std::swap(segid0, segid1);
    }

    _segid2voxelNum[segid1] += _segid2voxelNum[segid0];
    _segid2voxelNum.erase(segid0);

    auto& neighbors0 = _segid2neighbor.at(segid0);
    auto& neighbors1 = _segid2neighbor.at(segid1);
    neighbors0.erase(segid1);
    neighbors1.erase(segid0);

    // merge all the edges to segid1
    for(auto& [nid0, neighborEdgeIndex] : neighbors0){
        _segid2neighbor.at(nid0).erase(segid0);
        
        auto& neighborEdge = _edgeList[neighborEdgeIndex];

        if (_has_connection(segid1, nid0)){
            // combine two region edges
            const auto& newEdgeIndex = neighbors1[nid0];
            auto& newEdge = _edgeList[newEdgeIndex];
                            
            auto meanAff0 = neighborEdge.get_mean();
            auto meanAff1 = newEdge.get_mean();
            newEdge.absorb(neighborEdge);
            const auto& meanAff = newEdge.get_mean();
            
            if(meanAff > affinityThreshold){
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
            if(meanAff > affinityThreshold){
                heap.emplace_push(nid0, segid1, meanAff, neighborEdge.version);
            }
        }
    }
    _segid2neighbor.erase(segid0);

}
public:

RegionGraph(): _segid2neighbor({}), _edgeList({}), _segid2voxelNum({}){}
RegionGraph(const Segid2VoxelNum& segid2voxelNum, const Segid2Neighbor& segid2neighbor, const RegionEdgeList& edgeList): 
    _segid2voxelNum(segid2voxelNum), _segid2neighbor(segid2neighbor), _edgeList(edgeList) {}

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

    _segid2voxelNum = get_segid_to_voxel_num(fragments);

    std::cout<< "accumulate the affinity edges..." << std::endl;
    for(std::size_t z=0; z<fragments.shape(0); z++){
        for(std::size_t y=0; y<fragments.shape(1); y++){
            for(std::size_t x=0; x<fragments.shape(2); x++){
                const auto& segid = fragments(z,y,x);
                // skip background voxels
                if(segid>0){ 
                    if (z>0)
                        _accumulate_edge(segid, fragments(z-1,y,x), affs(2,z,y,x));
                    
                    if (y>0)
                        _accumulate_edge(segid, fragments(z,y-1,x), affs(1,z,y,x));
                    
                    if (x>0)
                        _accumulate_edge(segid, fragments(z,y,x-1), affs(0,z,y,x));
                }
            }
        }
    }
    std::cout<<"total edge number: "<< _edgeList.size() <<std::endl;
}

std::size_t get_edge_num() const {
    std::size_t edgeNum = 0;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        edgeNum += neighbors0.size();
    }
    return edgeNum / 2;
}

std::string as_string() const {
    std::ostringstream stringStream;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        stringStream << segid0 << ": ";
        for(const auto& [segid1, edgeIndex] : neighbors0){
            if(segid0 < segid1){
                const auto& meanAff = _edgeList[edgeIndex].get_mean();
                stringStream << segid1 << "--" << meanAff << ", ";
            }
        }
        stringStream<< "\n";
    }
    stringStream<<"\n";
    return stringStream.str(); 
}

auto to_arrays() const {
    const auto& edgeNum = get_edge_num();

    xt::xtensor<segid_t, 2>::shape_type sh_arr = {edgeNum, 3};
    auto arr = xt::empty<segid_t>(sh_arr);
    
    xt::xtensor<aff_edge_t, 1>::shape_type sh_sums = {edgeNum};
    auto sums = xt::empty<aff_edge_t>(sh_sums);


    std::size_t n = 0;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            if(segid0 < segid1){
                const auto& edge = _edgeList[edgeIndex];
                arr(n, 0) = segid0;
                arr(n, 1) = segid1;
                arr(n, 2) = edge.count;
                sums(n) = edge.sum;
                n++;
            }
        }
    }
    return std::make_tuple(arr, sums);
}

auto merge_arrays(const xt::pytensor<segid_t, 2>& arr, 
        const xt::pytensor<aff_edge_t, 1>& sums){
    for(std::size_t i=0; i<sums.shape(0); i++){
        _accumulate_edge(arr(i, 0), arr(i, 1), sums(i), aff_edge_t(arr(i, 2)));
    }
}

/**
 * @brief greedy mean affinity agglomeration 
 * 
 * @param seg 
 * @param affinityThreshold 
 * @param voxelNumThreshold 
 * @return dendrogram 
 */
auto greedy_mean_affinity_agglomeration(const PySegmentation& seg, const aff_edge_t& affinityThreshold=0., 
        const size_t& voxelNumThreshold=std::numeric_limits<size_t>::max()){

    std::cout<< "build priority queue..." << std::endl;
    auto heap = _build_priority_queue(affinityThreshold);

    Dendrogram dend(affinityThreshold);

    std::cout<< "iterative greedy merging..." << std::endl; 
    size_t mergeNum = 0;
    while(!heap.empty()){
        const auto& edgeInQueue = heap.pop();
        
        auto segid0 = edgeInQueue.segid0;
        auto segid1 = edgeInQueue.segid1;

        if(!_has_connection(segid0, segid1)){
            continue;
        }
        
        const auto& edge = _get_edge(segid1, segid0);
        if(edge.version > edgeInQueue.version){
            // found an outdated edge
            continue;
        }

        const auto& voxelNum0 = _segid2voxelNum[segid0];
        const auto& voxelNum1 = _segid2voxelNum[segid1];
        if(voxelNum0 > voxelNumThreshold && voxelNum1 > voxelNumThreshold){
            // both objects are too big
            continue;
        }
        
        // merge segid1 and segid0
        mergeNum++;
        dend.push_edge(segid0, segid1, edge.get_mean());
        _merge_segments(segid0, segid1, edge, heap, affinityThreshold);
    }
    
    std::cout<< "merged "<< mergeNum << " times." << std::endl;
    return dend;
}


}; // class of RegionGraph
} // namespace reneu