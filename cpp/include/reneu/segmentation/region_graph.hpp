#pragma once

#include <vector>
#include <initializer_list>

#include <xtensor/xsort.hpp>

#include <boost/pending/disjoint_sets.hpp>

#include "reneu/type_aliase.hpp"

namespace reneu{

class RegionEdge{
public:
// we use the same type for both value for type stability in the average division. 
aff_edge_t count;
aff_edge_t sum;

RegionEdge(): count(0.), sum(0.){}

aff_edge_t get_mean(){
    return sum / count;
}

void absorb(RegionEdge& re2){
    count += re2.count;
    sum += re2.sum;
    re2.count = 0;
    re2.sum = 0;
}
}; // class of RegionEdge

class RegionProps{
public:
segid_t segid;
size_t voxelNum;
std::map<segid_t, RegionEdge> neighbors;

private:
void _cleanup(){
    voxelNum = 0;
    neighbors = {};
}

public:
RegionProps(segid_t _segid):segid(_segid), voxelNum(0), neighbors({}){}


void absorb(RegionProps& smallerRegionProps){
    voxelNum += smallerRegionProps.voxelNum;
    for(auto& [segid, regionEdge] : smallerRegionProps.neighbors){
        neighbors[segid].absorb( regionEdge );
    }

    smallerRegionProps._cleanup();
    return;
}

}; // class of RegionProps

class RegionGraph{
private:
    std::vector<RegionProps> _rg;

/**
 * @brief we assume that the lower_bound is the index!
 */
inline auto find_index(const segid_t& segid){
    auto compare = [](RegionProps rp1, RegionProps rp2){
        return rp1.voxelNum < rp2.voxelNum;
    };
    auto lower = std::lower_bound(_rg.begin(), _rg.end(), segid, compare);
    auto idx = std::distance(_rg.begin(), lower);
    // if we really want speed, we can delete this assersion after extensive tests.
    assert(_rg[idx].segid == segid);
    return idx;
}

inline void accumulate_edge(const segid_t& segid1, const segid_t& segid2, const aff_edge_t& aff){
    // we assume that segid1 is greater than 0 !
    if(segid2>0 && segid1 != segid2){
        auto [s1, s2] = std::minmax(segid1, segid2);
        auto idx1 = find_index(s1);
        _rg[idx1].neighbors[s2].count++;
        _rg[idx1].neighbors[s2].sum += aff;
    }
    return;
}

/** 
 * @brief always merge small segment to large one
 */
inline void merge(const segid_t& segid1, const segid_t& segid2){
    auto idx1 = find_index(segid1);
    auto idx2 = find_index(segid2);
    auto& voxelNum1 = _rg[idx1].voxelNum;
    auto& voxelNum2 = _rg[idx2].voxelNum;
    
    // make i1 the smaller segid index
    size_t i1, i2;
    if(voxelNum1 < voxelNum2){
        i1 = idx1;
        i2 = idx2;
    } else {
        i1 = idx2;
        i2 = idx1;
    }

    _rg[i2].absorb( _rg[i1] );
    // To-Do: try really erase the element to see whether it will speed it up or not
    // the erase operation is expensive, but it shrinks the region graph gradually.
    // if we erase the element, we might not be able to find the segment id correctly 
    // in the find_index function.
    return;
}

public:
/**
 * @brief build region graph. 
 * 
 */
RegionGraph(const AffinityMap& affs, const Segmentation& fragments){
    // only contains x,y,z affinity 
    assert(affs.shape(0)== 3);
    // Note that our format of affinity channels are ordered by x,y,z
    // although we are using C order with z,y,x in indexing!
    // This is reasonable, because the last axis x is always changing fastest in memory
    // when we tranvers the memory, the x axis changes first, so this is sort of 
    // consistent with the order of channels. 
    assert(affs.shape(1)==fragments.shape(0));
    assert(affs.shape(2)==fragments.shape(1));
    assert(affs.shape(3)==fragments.shape(2));

    auto segids = xt::unique(fragments);
    std::map<segid_t, size_t> id2count = {};
    for(auto segid : fragments){
        id2count[segid]++;
    }

    _rg.reserve(segids.size());
    for(auto segid : segids){
        if(segid > 0){
            _rg.emplace_back( RegionProps(segid) );
        }
    }

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

void greedy_merge_until(Segmentation& fragments, const aff_edge_t& threshold){
    // fibonacci heap might be more efficient
    // use std data structure to avoid denpendency for now
    using Edge = std::tuple<segid_t, segid_t, aff_edge_t>;
    auto cmp = [](const Edge& left, const Edge& right){
        return std::get<2>(left) < std::get<2>(right);
    };
    // TO-DO: replace with fibonacci heap
    priority_queue<Edge, vector<Edge>, decltype(cmp)> heap(cmp);
    for(auto& rg1 : _rg){
        for(auto& [segid2, props] : rg1.neighbors){
            heap.emplace(std::make_tuple(rg1.segid, segid2, props.get_mean()));
        }
    }

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

    size_t mergeNum = 0;
    while(!heap.empty()){
        auto [segid0, segid1, meanAffinity] = heap.top();
        if(meanAffinity < threshold) break;
        heap.pop();

        mergeNum++;
        merge(segid0, segid1);
        // Union the two sets that contain elements x and y. 
        // This is equivalent to link(find_set(x),find_set(y)).
        dsets.union_set(segid0, segid1);
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

    return;
}

inline void py_greedy_merge_until(PySegmentation& pyseg, const aff_edge_t& threshold){
    greedy_merge_until(std::move(pyseg), threshold);
    return;
}

};

} // namespace reneu