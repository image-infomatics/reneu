#pragma once

#include <queue>
#include "../types.hpp"


namespace reneu{


template<class E>
class PriorityQueue{
private:
    std::vector<E> _edges;

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