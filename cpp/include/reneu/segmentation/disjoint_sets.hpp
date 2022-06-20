#pragma once

// #include <assert.h>
// #include <iostream>
#include "../types.hpp"
#include "./utils.hpp"

#include <set>
#include <tsl/robin_map.h>
#include <boost/serialization/vector.hpp>
#include "../utils/serialization.hpp"


namespace reneu{

template<class T>
struct Ele{
    T id;
    std::size_t parentIndex;
    std::size_t size;

    Ele(){};
    Ele(const T& id_, const std::size_t& parentIndex_, const std::size_t& size_): id(id_), parentIndex(parentIndex_), size(size_){};

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & id;
        ar & parentIndex;
        ar & size;
    }
};

template<class T>
class DisjointSets{

protected:
tsl::robin_map<T, std::size_t> _id2index;
std::vector<Ele<T>> _elements;

public:

DisjointSets(): _id2index({}), _elements({}){}
DisjointSets(std::size_t n): _id2index({}), _elements({}){
    _elements.reserve(n);
}
DisjointSets(std::set<T> ids): _id2index({}), _elements({}){
    _elements.reserve(ids.size());

    std::size_t idx=0;
    for(const auto& id : ids){
        _elements.emplace_back(id, idx, 1);
        _id2index[id] = idx;
        idx++;
    }
}

DisjointSets(const PySegmentation& seg){
    auto ids = get_nonzero_segids(seg);

    // this code is a duplicate of DisjointSets(std::set<T>)
    // To-Do: remove this duplicate
    _elements.reserve(ids.size());

    std::size_t idx=0;
    for(const auto& id : ids){
        _elements.emplace_back(id, idx, 1);
        _id2index[id] = idx;
        idx++;
    }
}

friend class boost::serialization::access;
BOOST_SERIALIZATION_SPLIT_MEMBER()

template<class Archive>
void save(Archive& ar, const unsigned int /*version*/) const {
    ar & _id2index;
    ar & _elements;
}

template<class Archive>
void load(Archive& ar, const unsigned int /*version*/) {
    ar & _id2index;
    ar & _elements;
}

inline auto size() const {
    return _elements.size();
}

bool contains(const T& id) const {
    const auto& search = _id2index.find(id);
    return search != _id2index.end();
}

inline auto find_root_element_index(const T& id){
    assert(id > 0);
    auto idx = _id2index[id];
    auto ele = _elements[idx];

    // Path halving
    while(ele.parentIndex != idx){
        auto& parentIndex = ele.parentIndex;
        auto& parent = _elements[parentIndex];
        auto& grandParentIndex = parent.parentIndex;
        auto& grandParent = _elements[grandParentIndex];
        ele.parentIndex = grandParentIndex;
        
        // go to parent node
        idx = parentIndex;
        ele = _elements[idx];
    }
    // return std::make_pair(ele, idx);
    return idx;
}

T find_set(const T& id){
    if(!this->contains(id)){
        return id;
    } else {
        const auto& idx = find_root_element_index(id);
        const auto& ele = _elements[idx];
        return ele.id;
    }
}

void make_set(const T& id){
    if(!this->contains(id)){
        const auto& newIndex = _elements.size();
        _id2index[id] = newIndex;
        // add a new element, the parent is itself
        _elements.emplace_back(id, newIndex, 1); 
    }
}

void union_set(const T& id0, const T& id1,
        bool bySize = true){
    const auto& rootIndex0 = find_root_element_index(id0);
    const auto& rootIndex1 = find_root_element_index(id1);

    auto& rootEle0 = _elements[rootIndex0];
    auto& rootEle1 = _elements[rootIndex1];

    if(rootIndex0 == rootIndex1){
        // they have already in the same set
        return;
    } else {
        if(bySize && rootEle0.size < rootEle1.size){
            // merge rootEle0 to rootEle1
            rootEle0.parentIndex = rootIndex1;
            rootEle1.size += rootEle0.size;
        } else {
            // merge rootEle1 to rootEle0
            rootEle1.parentIndex = rootIndex0;
            rootEle0.size += rootEle1.size;
        }
    }
}

void make_and_union_set(const T& id0, const T& id1,
        bool bySize=true){
    if(id0 != id1){
        make_set(id0);
        make_set(id1);
        union_set(id0, id1, bySize=bySize);
    }
}

auto merge_array(const xt::pytensor<T, 2>& arr,
        bool hasRoot=false){

    if(!hasRoot){
        std::set<T> ids = {};
        for(std::size_t i=0; i<arr.shape(0); i++){
            ids.insert(arr(i, 0));
            if(arr(i,0)!=arr(i,1)) ids.insert(arr(i, 1));
        }

        // make all the set
        for(const auto& id: ids) 
            make_set(id);

        for(std::size_t i=0; i<arr.shape(0); i++){
            const auto& id0 = arr(i, 0);
            const auto& id1 = arr(i, 1);
            union_set(id0, id1);
        } 
    } else {
        // make all the set
        for(std::size_t i=0; i<arr.shape(0); i++){
            const auto& id0 = arr(i, 0);
            const auto& id1 = arr(i, 1);
            if(id0 == id1) make_set(id0);
        }

        for(std::size_t i=0; i<arr.shape(0); i++){
            const auto& id0 = arr(i, 0);
            const auto& id1 = arr(i, 1);
            // we assume that all the identical set is made.
            // otherwise, we'll get a wrong result.
            // make_and_union_set will fix it but slower.
            if(id0!=id1) union_set(id0, id1);
        }
    }
}

auto to_array(){
    xt::pytensor<segid_t, 2>::shape_type sh = {
        static_cast<long>(this->size()), 2};
    auto arr = xt::empty<segid_t>(sh);

    for(std::size_t i=0; i<this->size(); i++){
        auto& ele = _elements[i];
        arr(i,0) = ele.id;
        arr(i,1) = find_set(ele.id);
    }
    return arr;
}

void compress_sets(){
    for(std::size_t i=0; i<this->size(); i++){
        auto& ele = _elements[i];
        auto& root = find_set(ele.id);
        auto& rootIndex = _id2index[root.id]; 
        ele.parentIndex = rootIndex;
    }
}

auto relabel(Segmentation&& seg){
    auto segids = get_nonzero_segids(seg);

    std::cout<< "relabel the fragments to a flat segmentation." << std::endl;
    const auto& [sz, sy, sx] = seg.shape();
    for(std::size_t z=0; z<sz; z++){
        for(std::size_t y=0; y<sy; y++){
            for(std::size_t x=0; x<sx; x++){
                const auto& sid = seg(z,y,x);
                if(sid > 0){
                    const auto& rootID = find_set(sid);
                    if(sid!=rootID){
                        assert(rootID > 0);
                        seg(z,y,x) = rootID;
                    }
                }
            }
        }
    }

    return seg;
}

inline auto py_relabel(PySegmentation& pyseg){
    return relabel(std::move(pyseg));
}

}; // class of DisjointSets

} // namespace reneu