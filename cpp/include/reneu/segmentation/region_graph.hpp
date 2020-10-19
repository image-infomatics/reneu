#pragma once

#include <vector>
#include <initializer_list>

#include <xtensor/xsort.hpp>

#include "reneu/type_aliase.hpp"
#include "utils.hpp"
#include "disjoint_sets.hpp"


namespace reneu{

class RegionEdge;
using RegionEdgePtr = std::shared_ptr<RegionEdge>;


class RegionEdge{
public:
// we use the same type for both value for type stability in the average division. 
aff_edge_t count;
aff_edge_t sum;
segid_t segid0;
segid_t segid1;
// version is used to tell whether an edge is outdated or not in priority queue
size_t version;

RegionEdge(const segid_t& _segid0, const segid_t& _segid1, const aff_edge_t& aff): 
            count(1), sum(aff), segid0(_segid0), segid1(_segid1), version(1){}

friend std::ostream& operator<<(std::ostream& os, const RegionEdge& re);

inline aff_edge_t get_mean() const {
    return sum / count;
}

inline void accumulate(const aff_edge_t& aff){
    count++;
    sum += aff;
}

inline void absorb(RegionEdgePtr& re2Ptr){
    count += re2Ptr->count;
    sum += re2Ptr->sum;
    version++;
    re2Ptr->cleanup();
    re2Ptr = nullptr;
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


class RegionGraph{
private:
    using Neighbors = std::map<segid_t, std::shared_ptr<RegionEdge>>;
    using RegionMap = std::map<segid_t, Neighbors>;
    RegionMap _rg;
    std::vector<RegionEdge> _edgeList;

    size_t _edgeNum;

    struct EdgeInQueue{
        segid_t segid0;
        segid_t segid1;
        aff_edge_t aff;
        size_t version;
    };


inline bool has_connection (const segid_t& sid0, const segid_t& sid1) const {
    // return _rg[sid0].count(sid1);
    return (_rg.count(sid0)) && (_rg.at(sid0).count(sid1));
    // return (_rg.find(sid0) != _rg.end()) && (_rg[sid0].find(sid1)!= _rg[sid0].end());
}

inline void accumulate_edge(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff){
    // we assume that segid0 is greater than 0 !
    if( (segid1>0) && (segid0 != segid1)){
        if(has_connection(segid0, segid1)){
            const auto& edgePtr = _rg.at(segid0).at(segid1);
            edgePtr->accumulate(aff);
        } else {
            if(has_connection(segid1, segid0) != has_connection(segid0, segid1)){
                std::cout<< "how can the connectivity not symmetric?"<< std::endl;
            }
            // create a new edge
            _edgeList.emplace_back(RegionEdge(segid0, segid1, aff));
            const auto& edgePtr = std::make_shared<RegionEdge>(_edgeList.back());
            _rg[segid0][segid1] = edgePtr; 
            _rg[segid1][segid0] = edgePtr;
            _edgeNum++;
        }
    }
    return;
}


auto build_priority_queue (const aff_edge_t& threshold) const {
    // fibonacci heap might be more efficient
    // use std data structure to avoid denpendency for now
    auto cmp = [](const EdgeInQueue& left, const EdgeInQueue& right){
        return left.aff < right.aff;
    };
    // TO-DO: replace with fibonacci heap
    std::priority_queue<EdgeInQueue, vector<EdgeInQueue>, decltype(cmp)> heap(cmp);
    for(const auto& [segid0, neighbors0] : _rg){
        for(const auto& [segid1, edgePtr] : neighbors0){
            if(std::minmax(segid0, segid1) != std::minmax(edgePtr->segid0, edgePtr->segid1)){
                std::cout<< "the edge pointer is pointing to a wrong place!"<< std::endl;
            }
            if(segid0 == segid1){
                std::cout<< "equivalant segmentation id: "<< segid0 << std::endl;
            }
            // the connection is bidirectional, 
            // so only half of the pairs need to be handled
            if(segid0 < segid1){
                const auto& meanAff = edgePtr->get_mean();
                if(meanAff > threshold){
                    // initial version is set to 1
                    heap.emplace(EdgeInQueue({segid0, segid1, meanAff, 1}));
                    // const auto& newEdgeInQueue = EdgeInQueue({segid0, segid1, meanAff, 1});
                    // heap.push(newEdgeInQueue);
                }
            }
        }
    }
    std::cout<< "initial heap size: "<< heap.size() << std::endl;
    return heap;
}


public:
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

    size_t edgeNumInRegionGraph0 = 0;
    size_t edgeNumInRegionGraph1 = 0;
    for(const auto& [segid0, neighbors0] : _rg){
        for(const auto& [segid1, edgePtr] : neighbors0){
            if(segid0 > segid1){
                if(edgePtr->get_mean() > threshold ){
                    edgeNumInRegionGraph0++;
                }
            } else if(segid0 == segid1){
                std::cout<< "equivalent edge: "<< segid0<< std::endl;
            } else {
                if(edgePtr->get_mean()> threshold){
                    edgeNumInRegionGraph1++;
                }
            }
        }
    }
    std::cout << "effective edge number in region graph: "<< edgeNumInRegionGraph0 << std::endl;
    std::cout << "effective edge number in region graph: "<< edgeNumInRegionGraph1 << std::endl;

    std::cout<< "build priority queue..." << std::endl;
    auto heap = build_priority_queue(threshold);

    std::cout<< "build disjoint set..." << std::endl;
    auto dsets = DisjointSets(seg); 

    std::cout<< "iterative greedy merging..." << std::endl; 
    size_t mergeNum = 0;
    while(!heap.empty()){
        const auto& edgeInQueue = heap.top();
        auto segid0 = edgeInQueue.segid0;
        auto segid1 = edgeInQueue.segid1;
        // std::cout<< "edge: " << segid0 << "--" << segid1 << "=" << edgeInQueue.aff<<std::endl;
        heap.pop();

        if(!has_connection(segid0, segid1)){
            // std::cout<< segid0 << "--" << segid1 << " do not exist anymore!" << std::endl;
            continue;
        }
        const auto& edgePtr = _rg.at(segid1).at(segid0);
        // std::cout<< "checking edge: "<< *edgePtr << 
            // " with mean aff: "<< edgePtr->get_mean() << std::endl;
        if(edgePtr->version > edgeInQueue.version){
            // found an outdated edge
            // recompute its mean affinity and push to heap
            // const auto& meanAff = edgePtr->get_mean();
            // if(meanAff>threshold){
            //     heap.emplace(EdgeInQueue(
            //         {segid0, segid1, edgePtr->get_mean(), edgePtr->version}
            //     ));
            // // }
            // std::cout<< "skip outdated edge: "<< segid0 << " -- "<< segid1 << 
            //                             " = " << edgeInQueue.aff<< std::endl;
            // std::cout<< "latest edge: "<< *edgePtr << std::endl;
            // std::cout<< "current version: "<< edgeInQueue.version 
            //             << ", latest version: "<< edgePtr->version<<std::endl; 
            continue;
        }

        //std::cout<< "merge edge: "<< e << std::endl;
        // merge segid1 and segid0
        mergeNum++;
        // Union the two sets that contain elements x and y. 
        // This is equivalent to link(find_set(x),find_set(y)).
        dsets.union_set(segid0, segid1);
        
        // make segid1 bigger than segid0
        // always merge object with less neighbors to more neighbors
        if(_rg.at(segid0).size() > _rg.at(segid1).size()){
            std::swap(segid0, segid1);
        }
        auto& neighbors0 = _rg.at(segid0);
        auto& neighbors1 = _rg.at(segid1);
        neighbors0.erase(segid1);
        neighbors1.erase(segid0);

        // merge all the edges to segid1
        for(auto& [nid0, neighborEdgePtr] : neighbors0){
            //std::cout<< "find neighbor edge: " << *neighborEdgePtr << std::endl; 
            
            if(!has_connection(nid0, segid0)){
                std::cout<< "we are still iterating erased edges? "<< 
                                    nid0 << "--" << segid0 << std::endl;
                continue;
            }
            
            // it seems that erase disrupted the iteration!
            //std::cout << "before erase neighbor segid0 " << std::endl;
            _rg.at(nid0).erase(segid0);

            if(neighborEdgePtr->count == 0){
                std::cout<< "0 count edge: " << *neighborEdgePtr << std::endl;
                continue;
            }

            if (has_connection(segid1, nid0)){
                // combine two region edges
                const auto& newEdgePtr = neighbors1[nid0];
                // if(newEdgePtr->get_mean() > edgeInQueue.aff){
                //     std::cout<< "we should iterate this new edge first!: "<< *newEdgePtr << std::endl;
                // }
                auto meanAff0 = neighborEdgePtr->get_mean();
                auto meanAff1 = newEdgePtr->get_mean();
                newEdgePtr->absorb(neighborEdgePtr);
                // std::cout<< "after combine two region edges: "<< *newEdgePtr << std::endl;
                const auto& meanAff = newEdgePtr->get_mean();
                
                // if(meanAff > edgeInQueue.aff){
                    // std::cout<< meanAff0 <<" + " <<meanAff1 << " = "<< meanAff<< std::endl; 
                    // std::cout<< "combined new edge: "<< *newEdgePtr<< ", mean: "<<meanAff<<std::endl;
                // }
                if(meanAff > threshold){
                    heap.emplace(
                        EdgeInQueue(
                            {nid0, segid1, meanAff, newEdgePtr->version}
                        )
                    );
                }
            } else {
                // directly assign nid0-segid0 to nid0-segid1
                // std::cout<< "edge before assignment: " << *neighborEdgePtr << std::endl;
                _rg[segid1][nid0] = neighborEdgePtr;
                _rg[nid0][segid1] = neighborEdgePtr;
                // make the original edge in priority queue outdated
                // std::cout<< "edge before assignment: "<< *neighborEdgePtr << std::endl;
                // this is a new edge, so the version should be 1
                neighborEdgePtr->version = 1;
                neighborEdgePtr->segid0 = nid0;
                neighborEdgePtr->segid1 = segid1;

                const auto& meanAff = neighborEdgePtr->get_mean();
                if(meanAff > threshold){
                    // if(meanAff > edgeInQueue.aff){
                    //     std::cout<< "assigned new edge: "<< *neighborEdgePtr<<std::endl;
                    // }

                    heap.emplace(
                        EdgeInQueue(
                            {nid0, segid1, meanAff, neighborEdgePtr->version}
                        )
                    );
                }
            }
        }
        // std::cout<< "before erase segid0 map" << std::endl;
        _rg.erase(segid0);
        // std::cout<< "after  erase segid0 map" << std::endl;
        //_rg[segid0].clear();
    }
    
    std::cout<< "merged "<< mergeNum << " times." << std::endl;
    dsets.relabel(seg);
    return seg;
}

inline auto py_greedy_merge_until(PySegmentation& pyseg, const aff_edge_t& threshold){
    return greedy_merge_until(std::move(pyseg), threshold);
}

};

} // namespace reneu