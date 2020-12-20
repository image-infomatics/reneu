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
std::map<segid_t, std::uint8_t> _segid2frozen;

// bit flag of chunk surface
// if the bit is 1, the segment is frozen by the corresponding surface
// it should be melted in that bit if the corresponding surface got merged
// if the bit is 0, the segment is not frozen by the corresponding surface 
static const std::uint8_t NEG_Z = ~(std::numeric_limits<std::uint8_t>::max() >> 1);
static const std::uint8_t NEG_Y = NEG_Z >> 1;
static const std::uint8_t NEG_X = NEG_Y >> 1;
static const std::uint8_t POS_Z = NEG_X >> 1;
static const std::uint8_t POS_Y = POS_Z >> 1;
static const std::uint8_t POS_X = POS_Y >> 1;

friend class boost::serialization::access;
template<class Archive>
void serialize(Archive& ar, const unsigned int version) {
    // To-Do: clean up merged segments?
    // do we need to propagate the freezing face?
    ar & boost::serialization::base_object<RegionGraph>(*this);
    ar & _segid2frozen; 
}


inline bool _is_frozen(const segid_t& sid) const {
    return (_segid2frozen.at(sid) >> 1) > 0;
}

inline auto _freeze_both(const segid_t& sid0, const segid_t& sid1){
    _segid2frozen[sid0] |= _segid2frozen.at(sid1);
    _segid2frozen[sid1] |= _segid2frozen.at(sid0);
}

public:

RegionGraphChunk(): RegionGraph(), _segid2frozen({}){}

RegionGraphChunk(const AffinityMap& affs, const Segmentation& seg, const std::array<bool, 6> &volumeBoundaryFlags): 
        RegionGraph(affs, seg), _segid2frozen({}){
    
    // freeze the segment ids touching chunk boundary 
    // the ones touching volume boundary are excluded
    for(std::size_t z=0; z<seg.shape(0); z++){
        for(std::size_t y=0; y<seg.shape(1); y++){
            for(std::size_t x=0; x<seg.shape(2); x++){
                const auto& segid = seg(z,y,x);
                if(segid == 0) continue;

                // mark that this segment id exist
                // std::map will return 0 for nonexist segmentation id
                if(_segid2frozen[segid] == 0)
                    _segid2frozen[segid] = 1;
                
                if(z==0 && !volumeBoundaryFlags[0])
                    _segid2frozen[segid] |= NEG_Z;
                else if(z==seg.shape(0)-1 && !volumeBoundaryFlags[3])
                    _segid2frozen[segid] |= POS_Z;

                if(y==0 && !volumeBoundaryFlags[1])
                    _segid2frozen[segid] |= NEG_Y;
                else if(y==seg.shape(1)-1 && !volumeBoundaryFlags[4])
                    _segid2frozen[segid] |= POS_Y;
                
                if(x==0 && !volumeBoundaryFlags[2])
                    _segid2frozen[segid] |= NEG_X;
                else if(x==seg.shape(2)-1 && !volumeBoundaryFlags[5])
                    _segid2frozen[segid] |= POS_X;

            }
        }
    }
    return;
}

/** Merge fragments inside a leaf node.
 * 
 * @param seg The segmentation volume containing many small fragments.
 * @param threshold The stopping threshold of the agglomeration.
 */
auto merge_in_leaf(const Segmentation& seg, const aff_edge_t& threshold) {

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
    return dend;
}

inline auto py_merge_in_leaf(const Segmentation& seg, const aff_edge_t& threshold) {
    return merge_in_leaf(seg, threshold);
}

/** Merge the other node in the binary bounding box tree. 
 * 
 * The nodes freezed by the contacting face should be melted.
 * 
 * @param rgc2 The other region graph chunk with some frozen nodes.
 */
auto merge_another_node(const RegionGraphChunk& rgc2){
    return;
}

}; // class of RegionGraphChunk

} // namespace of reneu