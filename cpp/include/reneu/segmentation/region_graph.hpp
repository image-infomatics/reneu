#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <initializer_list>

#include <xtensor/xsort.hpp>

#include "reneu/type_aliase.hpp"
#include "utils.hpp"
#include "disjoint_sets.hpp"
#include "dendrogram.hpp"


namespace reneu{

class RegionGraph;

class RegionEdge{
private:
// we use the same type for both value for type stability in the average division. 
aff_edge_t count;
aff_edge_t sum;
segid_t segid0;
segid_t segid1;
// version is used to tell whether an edge is outdated or not in priority queue
size_t version;

public:
friend std::ostream& operator<<(std::ostream& os, const RegionEdge& re);
friend class RegionGraph;


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

}; // class of RegionEdge

std::ostream& operator<<(std::ostream& os, const RegionEdge& re){
    os << re.segid0 << "--" << re.segid1 << "| count: " << re.count << ", sum: "<< re.sum << 
        " mean aff: "<< re.sum/re.count <<", version: "<< re.version << ". ";
    return os;
}

using Neighbors = std::map<segid_t, size_t>;
// using RegionMap = boost::container::flat_map<segid_t, Neighbors>;
using RegionMap = std::map<segid_t, Neighbors>;

class RegionGraph{
private:
    RegionMap _rm;
    std::vector<RegionEdge> _edgeList;

    size_t _edgeNum;

    struct EdgeInQueue{
        segid_t segid0;
        segid_t segid1;
        aff_edge_t aff;
        size_t version;
    };


inline bool has_connection (const segid_t& sid0, const segid_t& sid1) const {
    // return _rm[sid0].count(sid1);
    return (_rm.count(sid0)) && (_rm.at(sid0).count(sid1));
    // return (_rm.find(sid0) != _rm.end()) && (_rm[sid0].find(sid1)!= _rm[sid0].end());
}

inline void accumulate_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff){
    // we assume that segid0 is greater than 0 !
    if( (segid1>0) && (segid0 != segid1)){
        if(has_connection(segid0, segid1)){
            const auto& edgeIndex = _rm.at(segid0).at(segid1);
            _edgeList[edgeIndex].accumulate(aff);
        } else {
            // create a new edge
            _edgeList.emplace_back(RegionEdge(segid0, segid1, aff));
            const size_t& edgeIndex = _edgeList.size() - 1;
            _rm[segid0][segid1] = edgeIndex; 
            _rm[segid1][segid0] = edgeIndex;
            _edgeNum++;
        }
    }
    return;
}


auto build_priority_queue (const aff_edge_t& threshold) const {
    std::vector<EdgeInQueue> heap({});
    heap.reserve(_rm.size() * 4);
    for(const auto& [segid0, neighbors0] : _rm){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            // the connection is bidirectional, 
            // so only half of the pairs need to be handled
            if(segid0 < segid1){
                const auto& meanAff = _edgeList[edgeIndex].get_mean();
                if(meanAff > threshold){
                    // initial version is set to 1
                    heap.emplace_back(EdgeInQueue({segid0, segid1, meanAff, 1}));
                }
            }
        }
    }

    std::cout<< "initial heap size: "<< heap.size() << std::endl;
    return heap;
}


public:

auto get_edge_num() const {
    std::size_t edgeNum = 0;
    for(const auto& [segid0, neighbors0] : _rm){
        edgeNum += neighbors0.size();
    }
    return edgeNum / 2;
}

void print(){
    std::cout<<std::endl << "region graph: " <<std::endl;
    for(const auto& [segid0, neighbors0] : _rm){
        std::cout << segid0 << ": ";
        for(const auto& [segid1, edgeIndex] : neighbors0){
            const auto& meanAff = _edgeList[edgeIndex].get_mean();
            std::cout << segid1 << "--" << meanAff << ", ";
        }
        std::cout << std::endl;
    }
    std::cout<< std::endl;
    return; 
}

auto as_array() const {
    auto edgeNum = get_edge_num();
    xt::xtensor<aff_edge_t, 2>::shape_type sh = {edgeNum, 3};
    auto arr = xt::empty<aff_edge_t>(sh);

    std::size_t n = 0;
    for(const auto& [segid0, neighbors0] : _rm){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            arr(n, 0) = segid0;
            arr(n, 1) = segid1;
            arr(n, 2) = _edgeList[edgeIndex].get_mean();
        }
    }
    return arr;
}

/**
 * @brief build region graph. 
 * 
 */
RegionGraph(const AffinityMap& affs, const Segmentation& fragments): _edgeNum(0){
    // only contains x,y,z affinity 
    // Note that our format of affinity channels are ordered by x,y,z
    // although we are using C order with z,y,x in indexing!
    // This is reasonable, because the last axis x is always changing fastest in memory
    // when we tranvers the memory, the x axis changes first, so this is sort of 
    // consistent with the order of channels. 

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
    std::cout<<"total edge number: "<< _edgeNum<<std::endl;
}

auto greedy_merge_until(Segmentation&& seg, const aff_edge_t& threshold){

    size_t edgeNum = 0;
    for(const auto& edge : _edgeList){
        if(edge.get_mean() > threshold){
            edgeNum++;
        }
    }
    std::cout<< "effective edge number: "<< edgeNum << std::endl;

    size_t totalEdgeNumInRegionGraph = 0;
    size_t edgeNumInRegionGraph0 = 0;
    size_t edgeNumInRegionGraph1 = 0;
    for(const auto& [segid0, neighbors0] : _rm){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            auto& edge = _edgeList[edgeIndex];
            if(std::minmax(segid0, segid1) != std::minmax(edge.segid0, edge.segid1)){
                std::cout<< "the edge pointer is pointing to a wrong place!"<< std::endl;
            }
            if(segid0 > segid1){
                totalEdgeNumInRegionGraph++;
                if(edge.get_mean() > threshold ){
                    edgeNumInRegionGraph0++;
                }
            } else if(segid0 == segid1){
                std::cout<< "equivalent edge: "<< segid0<< std::endl;
            } else {
                if(edge.get_mean()> threshold){
                    edgeNumInRegionGraph1++;
                }
            }
        }
    }
    std::cout << "total edge number in region graph: "<< totalEdgeNumInRegionGraph << std::endl;
    std::cout << "effective edge number in region graph: "<< edgeNumInRegionGraph0 << std::endl;
    std::cout << "effective edge number in region graph: "<< edgeNumInRegionGraph1 << std::endl;

    std::cout<< "build priority queue..." << std::endl;
    auto heap = build_priority_queue(threshold);
    auto cmp = [](const EdgeInQueue& left, const EdgeInQueue& right){
        return left.aff < right.aff;
    };
    std::make_heap(heap.begin(), heap.end(), cmp);


    Dendrogram dend(threshold);

    std::cout<< "iterative greedy merging..." << std::endl; 
    size_t mergeNum = 0;
    while(!heap.empty()){
        std::pop_heap(heap.begin(), heap.end(), cmp);
        const auto& edgeInQueue = heap.back();
        heap.pop_back();
        
        auto segid0 = edgeInQueue.segid0;
        auto segid1 = edgeInQueue.segid1;

        if(!has_connection(segid0, segid1)){
            continue;
        }
        
        const auto& edgeIndex = _rm.at(segid1).at(segid0);
        const auto& edge = _edgeList[edgeIndex];
        if(edge.version > edgeInQueue.version){
            // found an outdated edge
            // recompute its mean affinity and push to heap
            continue;
        }
        
        // print out heap
        // std::cout<< "heap: " << std::endl;
        // for(const auto& edge : heap){
        //     std::cout<< edge.segid0 << "--" << edge.segid1 << ": " 
        //                 << edge.aff << "," << edge.version << "; "; 
        // }
        // print();
        
        // merge segid1 and segid0
        mergeNum++;
        
        dend.push_edge(segid0, segid1, edge.get_mean());

        // make segid1 bigger than segid0
        // always merge object with less neighbors to more neighbors
        if(_rm.at(segid0).size() > _rm.at(segid1).size()){
            std::swap(segid0, segid1);
        }
        auto& neighbors0 = _rm.at(segid0);
        auto& neighbors1 = _rm.at(segid1);
        neighbors0.erase(segid1);
        neighbors1.erase(segid0);

        // merge all the edges to segid1
        for(auto& [nid0, neighborEdgeIndex] : neighbors0){
            _rm.at(nid0).erase(segid0);
            
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
                    heap.emplace_back(
                        EdgeInQueue(
                            {nid0, segid1, meanAff, newEdge.version}
                        )
                    );
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            } else {
                // directly assign nid0-segid0 to nid0-segid1
                _rm.at(segid1)[nid0] = neighborEdgeIndex;
                _rm.at(nid0)[segid1] = neighborEdgeIndex;
                // make the original edge in priority queue outdated
                // this is a new edge, so the version should be 1
                neighborEdge.version = 1;
                neighborEdge.segid0 = nid0;
                neighborEdge.segid1 = segid1;

                const auto& meanAff = neighborEdge.get_mean();
                if(meanAff > threshold){
                    heap.emplace_back(
                        EdgeInQueue(
                            {nid0, segid1, meanAff, neighborEdge.version}
                        )
                    );
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }
        _rm.erase(segid0);
    }
    
    std::cout<< "merged "<< mergeNum << " times." << std::endl;
    return dend;
}

inline auto py_greedy_merge_until(PySegmentation& pyseg, const aff_edge_t& threshold){
    return greedy_merge_until(std::move(pyseg), threshold);
}

};




} // namespace reneu