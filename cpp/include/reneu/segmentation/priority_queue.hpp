#pragma once

#include <queue>
#include "../type_aliase.hpp"


namespace reneu{

struct EdgeInQueue{
    segid_t segid0;
    segid_t segid1;
    aff_edge_t aff;
    size_t version;

    // constructor for emplace operation
    EdgeInQueue(const segid_t& segid0_, const segid_t& segid1_, const aff_edge_t& aff_, const std::size_t& version_):
        segid0(segid0_), segid1(segid1_), aff(aff_), version(version_){}

    bool operator<(const EdgeInQueue& other) const {
        return aff < other.aff;
    }
};
// struct LessThanByAff{
//     bool operator()(const EdgeInQueue& lhs, const EdgeInQueue& rhs) const {
//         return lhs.aff < rhs.aff;
//     }
// };
// using PriorityQueue = std::priority_queue<EdgeInQueue, std::vector<EdgeInQueue>, std::less<EdgeInQueue>>;

class PriorityQueue{
private:
    std::vector<EdgeInQueue> _edges;

public:
    PriorityQueue(): _edges({}) { }

    bool empty(){
        return _edges.empty();
    }

    auto size(){ return _edges.size(); }

    /**
     * @brief simply add new element
     * 
     * @param segid0 
     * @param segid1 
     * @param aff 
     * @param version 
     * @return auto 
     */
    auto emplace_back(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff, const std::size_t& version){
        _edges.emplace_back(segid0, segid1, aff, version);
    }

    void make_heap(){
        std::make_heap(_edges.begin(), _edges.end());
    }
    
    /**
     * @brief add new element in-place and update the heap
     * 
     * @param segid0 
     * @param segid1 
     * @param aff 
     * @param version 
     * @return auto 
     */
    auto emplace_push(const segid_t& segid0, const segid_t& segid1, const aff_edge_t& aff, const std::size_t& version){
        _edges.emplace_back(segid0, segid1, aff, version);
        std::push_heap(_edges.begin(), _edges.end());
    }

    auto pop(){
        // move largest one to the back
        std::pop_heap(_edges.begin(), _edges.end());
        // get the largest one
        const auto& edge = _edges.back();
        // pop out the largest one
        _edges.pop_back();
        return edge;
    }
}; // class of PriorityQueue

}// namespace reneu