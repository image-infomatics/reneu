#pragma once
#include <map>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>

#include "region_graph.hpp"

namespace reneu{

class RegionGraphChunk: public RegionGraph{
private:

// segmentation id --> chunk surface frozen status
// the 8 bits: negative z,y,x, positive z,y,x, 0, existance
// for the last bit, if it is 1, this segment exist
// otherwise, this segment do not belong to this region graph.
// since the map will have a default value of 0 for all the keys
using SegID2Frozen = std::map<segid_t, std::uint8_t>;
SegID2Frozen _segid2frozen;

// bit flag of chunk surface
// if the bit is 1, the segment is frozen by the corresponding surface
// it should be melted in that bit if the corresponding surface got merged
// if the bit is 0, the segment is not frozen by the corresponding surface 


// bit order: NEG_Z, NEG_Y, NEG_X, 0, POS_Z, POS_Y, POS_X, 0 
static const std::uint8_t NEG_Z = 0x80;
static const std::uint8_t NEG_Y = 0x40;
static const std::uint8_t NEG_X = 0x20;
static const std::uint8_t POS_Z = 0x08;
static const std::uint8_t POS_Y = 0x04;
static const std::uint8_t POS_X = 0x02;
static const std::array<std::uint8_t, 6> constexpr SURFACE_BITS = {NEG_Z, NEG_Y, NEG_X, POS_Z, POS_Y, POS_X};

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version) {
    // To-Do: clean up merged segments?
    // do we need to propagate the freezing face?
    ar & boost::serialization::base_object<RegionGraph>(*this);
    ar & _segid2frozen; 
}


inline bool _is_frozen(const segid_t& sid) const {
    const auto& search = _segid2frozen.find(sid);
    return search != _segid2frozen.end();
}

inline auto _freeze_both(const segid_t& sid0, const segid_t& sid1){
    _segid2frozen[sid0] |= _segid2frozen.at(sid1);
    _segid2frozen[sid1] |= _segid2frozen.at(sid0);
}

auto _build_priority_queue (const aff_edge_t& threshold) const {
    PriorityQueue heap;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            // the connection is bidirectional, 
            // so only half of the pairs need to be handled
            if((segid0 < segid1) && !_is_frozen(segid0) && !_is_frozen(segid1)){
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

auto _greedy_merge(const aff_edge_t& threshold){
    std::cout<< "build heap/priority queue..." << std::endl;
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

        if(_is_frozen(segid0) || _is_frozen(segid1)){
            // mark both segment as frozen from both chunk faces
            // these two could be merged in the 
            _freeze_both(segid0, segid1);
            continue;
        } 

        // merge segid1 and segid0
        mergeNum++;
        _merge_segments(segid0, segid1, edge, dend, heap, threshold); 
    }
    
    std::cout<< "merged "<< mergeNum << " times." << std::endl;

    // clean up the edge list 
    auto dsets = dend.to_disjoint_sets(threshold);

    auto residualSegid2Neighbor = Segid2Neighbor({});
    auto residualEdgeList = RegionEdgeList({});
    auto residualSegid2Frozen = SegID2Frozen({});

    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            if(segid0 < segid1){
                if(_is_frozen(segid0) && _is_frozen(segid1)){
                    // keep frozen edges to be proccessed in future
                    auto root0 = dsets.find_set(segid0);
                    auto root1 = dsets.find_set(segid1);
                    if(root0 == root1){
                        std::cout<< "segment "<< segid0 << " and " << segid1 << " has same root " << root0 << std::endl;
                    }

                    residualSegid2Neighbor[root0][root1] = residualEdgeList.size();
                    residualEdgeList.push_back(_edgeList[edgeIndex]);
                    residualSegid2Frozen[root0] = _segid2frozen[segid0] | _segid2frozen[root0];
                    residualSegid2Frozen[root1] = _segid2frozen[segid1] | _segid2frozen[root1];
                } 
            }
        }
    }

    // update internally
    _segid2neighbor = residualSegid2Neighbor;
    _edgeList = residualEdgeList;
    _segid2frozen = residualSegid2Frozen;
    return dend;
}

public:

RegionGraphChunk(): RegionGraph(), _segid2frozen({}){}
RegionGraphChunk(const RegionGraph& rg, const SegID2Frozen& segid2frozen): 
    RegionGraph(rg), _segid2frozen(segid2frozen) {}

/**
 * @brief Construct a new Region Graph Chunk object
 * 
 * @param affs affinity map. the starting offset should be (1,1,1) compared with segmentation
 * @param seg segmentation. The size should be larger than affinity map by (1,1,1).
 * @param volumeBoundaryFlags 
 */
RegionGraphChunk(const AffinityMap& affs, const Segmentation& seg, const std::array<bool, 6> &volumeBoundaryFlags): 
        RegionGraph(affs, seg), _segid2frozen({}){
    assert(affs.shape(1) == seg.shape(0) + 1);
    assert(affs.shape(2) == seg.shape(1) + 1);
    assert(affs.shape(3) == seg.shape(2) + 1);
   
    // freeze the segment ids touching chunk boundary 
    // the ones touching volume boundary are excluded
    for(std::size_t z=1; z<seg.shape(0); z++){
        for(std::size_t y=1; y<seg.shape(1); y++){
            for(std::size_t x=1; x<seg.shape(2); x++){
                const auto& segid = seg(z,y,x);
                if(segid == 0) continue;

                if(z==1 && !volumeBoundaryFlags[0]){
                    _segid2frozen[segid] |= NEG_Z;
                    const auto& contactingSegid = seg(z-1, y, x);
                    if(contactingSegid > 0){
                        _segid2frozen[contactingSegid] |= NEG_Z;
                    }
                }else if(z==seg.shape(0)-1 && !volumeBoundaryFlags[3])
                    _segid2frozen[segid] |= POS_Z;

                if(y==1 && !volumeBoundaryFlags[1]){
                    _segid2frozen[segid] |= NEG_Y;
                    const auto& contactingSegid = seg(z, y-1, x);
                    if(contactingSegid > 0){
                        _segid2frozen[contactingSegid] |= NEG_Y;
                    }
                }else if(y==seg.shape(1)-1 && !volumeBoundaryFlags[4])
                    _segid2frozen[segid] |= POS_Y;
                
                if(x==1 && !volumeBoundaryFlags[2]){
                    _segid2frozen[segid] |= NEG_X;
                    const auto& contactingSegid = seg(z, y, x-1);
                    if(contactingSegid > 0){
                        _segid2frozen[contactingSegid] |= NEG_X;
                    }
                }else if(x==seg.shape(2)-1 && !volumeBoundaryFlags[5])
                    _segid2frozen[segid] |= POS_X;

            }
        }
    }

    std::cout<< "accumulate the affinity edges..." << std::endl;
    // start from 1 since we included the contacting neighbor chunk segmentation
    for(std::size_t z=1; z<seg.shape(0); z++){
        for(std::size_t y=1; y<seg.shape(1); y++){
            for(std::size_t x=1; x<seg.shape(2); x++){
                const auto& segid = seg(z,y,x);
                // skip background voxels
                if(segid>0){ 
                    if (z>0)
                        accumulate_edge(segid, seg(z-1,y,x), affs(2,z-1,y,x));
                    
                    if (y>0)
                        accumulate_edge(segid, seg(z,y-1,x), affs(1,z,y-1,x));
                    
                    if (x>0)
                        accumulate_edge(segid, seg(z,y,x-1), affs(0,z,y,x-1));
                }
            }
        }
    }
}

/**
 * @brief merge nodes in leaf chunk
 * 
 * @param threshold 
 * @return auto 
 */
auto merge_in_leaf_chunk(const aff_edge_t& threshold){
    return _greedy_merge(threshold);
}

/** Merge the other node in the binary bounding box tree. 
 * 
 * The nodes freezed by the contacting face should be melted.
 * 
 * @param rgc2 The other region graph chunk with some frozen nodes.
 */
auto merge_upper_chunk(const RegionGraphChunk& upperRegionGraphChunk, 
                                const std::size_t& dim, const aff_edge_t& threshold){
    assert(dim>=0 && dim<3);
    const auto& LOWER_SURFACE_BIT = SURFACE_BITS[dim];
    const auto& UPPER_SURFACE_BIT = SURFACE_BITS[3+dim];

    // merge the frozen set
    // the contacting face should be melted
    for(auto& [segid, frozen] : _segid2frozen){
        frozen &= (~LOWER_SURFACE_BIT);
        if(frozen == 0){
            _segid2frozen.erase(segid);
        }
    }
    for(auto& [segid, frozen] : upperRegionGraphChunk._segid2frozen){
        const auto& newFrozen = frozen & (~UPPER_SURFACE_BIT);
        if(newFrozen > 0 ){
            // this segment is still frozen by other faces
            _segid2frozen[segid] |= newFrozen;
        }
    }

    // merge the region graphs
    for(const auto& [segid0, neighbor] : upperRegionGraphChunk._segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbor){
            if(segid0 < segid1){
                const auto& upperEdge = upperRegionGraphChunk._edgeList[edgeIndex];
                if(_segid2neighbor[segid0][segid1] == 0){
                    _segid2neighbor[segid0][segid1] = _edgeList.size();
                    _segid2neighbor[segid1][segid0] = _edgeList.size();
                    _edgeList.push_back(upperEdge);
                } else {
                    // both have the same edge, merge them
                    // theoretically, this should not happen?
                    std::cout<<"both region graph chunk have the same edge: "<<std::endl;
                    const auto& lowerEdgeIndex = _segid2neighbor.at(segid0).at(segid1);
                    _edgeList[lowerEdgeIndex].absorb(upperEdge);
                }
            }
        }
    }

    // greedy iterative agglomeration
    return _greedy_merge(threshold); 
}

}; // class of RegionGraphChunk

} // namespace of reneu