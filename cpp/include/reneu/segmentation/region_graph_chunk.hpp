#pragma once

#include <map>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include "reneu/types.hpp"
#include "region_graph.hpp"
#include "utils.hpp"

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
    // const auto& search = _segid2frozen.find(sid);
    // return search != _segid2frozen.end();
    // to-do: use contains in C++20
    return _segid2frozen.count(sid) > 0;
}

inline auto _frozen_neighbor_flag(const segid_t& segid0, const aff_edge_t& threshold) const {
    std::uint8_t flag = 0;
    for(const auto& [segid1, edgeIndex]: _segid2neighbor.at(segid0)){
        const auto& edge = _edgeList[edgeIndex];
        if( _is_frozen(segid1) /* && edge.get_mean()> threshold */) 
            flag |= _segid2frozen.at(segid1);
    }
    return flag;
}


auto _build_priority_queue (const aff_edge_t& threshold) const {
    PriorityQueue<EdgeInHeap> heap;
    for(const auto& [segid0, neighbors0] : _segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbors0){
            // the connection is bidirectional, 
            // so only half of the pairs need to be handled
            // exclude frozen segmentations with same flag
            if((segid0 < segid1) && 
                    !(_is_frozen(segid0) && _is_frozen(segid1) && 
                    _segid2frozen.at(segid0) == _segid2frozen.at(segid1))){

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

auto _greedy_mean_affinity_agglomeration(const aff_edge_t& threshold){
    std::cout<< "build heap/priority queue..." << std::endl;
    auto heap = _build_priority_queue(threshold);

    Dendrogram dend(threshold);

    std::cout<< "iterative greedy merging..." << std::endl;
    // std::cout<< "this region graph chunk: "<< as_string(); 
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

        if(_is_frozen(segid0) || _is_frozen(segid1)){
            // mark both segment as frozen from both chunk faces
            // these two could be merged in the higher order agglomeration
            _segid2frozen[segid0] |= _segid2frozen[segid1];
            _segid2frozen[segid1] |= _segid2frozen.at(segid0);
            continue;
        } 

        // if there is any segment has frozen neighbor, they should also be frozen.
        const auto& frozen_neighbor_flag0 = _frozen_neighbor_flag(segid0, threshold);
        const auto& frozen_neighbor_flag1 = _frozen_neighbor_flag(segid1, threshold);
        if(frozen_neighbor_flag0 || frozen_neighbor_flag1){
            _segid2frozen[segid0] = frozen_neighbor_flag0 | frozen_neighbor_flag1; 
            _segid2frozen[segid1] = frozen_neighbor_flag0 | frozen_neighbor_flag1; 
            continue;
        }

        // merge segid1 and segid0
        mergeNum++;
        dend.push_edge(segid0, segid1, edge.get_mean());
        _merge_segments(segid0, segid1, edge, heap, threshold); 
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
                    const auto& root0 = dsets.find_set(segid0);
                    const auto& root1 = dsets.find_set(segid1);
                    if(root0 == root1){
                        std::cout<< "segment "<< segid0 << " and " << segid1 << " has same root " << root0 << std::endl;
                        continue;
                    }

                    residualSegid2Neighbor[root0][root1] = residualEdgeList.size();
                    residualSegid2Neighbor[root1][root0] = residualEdgeList.size();
                    auto& edge = _edgeList[edgeIndex];
                    edge.segid0 = root0;
                    edge.segid1 = root1;
                    edge.version = 1;
                    residualEdgeList.push_back(edge);
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
 * @param seg segmentation. The size should be larger than affinity map by (1,1,1). the expanded part is in the negative directions.
 */
RegionGraphChunk(const PyAffinityMap& affs, const PySegmentation& frag): 
        RegionGraph(), _segid2frozen({}){
    
    assert(affs.shape(1)==frag.shape(0)-1);
    assert(affs.shape(2)==frag.shape(1)-1);
    assert(affs.shape(3)==frag.shape(2)-1);

    std::cout<< "accumulate the affinity edges..." << std::endl;
    // start from 1 since we included the contacting neighbor chunk segmentation
    for(std::size_t z=1; z<frag.shape(0); z++){
        for(std::size_t y=1; y<frag.shape(1); y++){
            for(std::size_t x=1; x<frag.shape(2); x++){
                const auto& segid = frag(z,y,x);
                // skip background voxels
                if(segid>0){ 
                    if (z>1)
                        _accumulate_edge(segid, frag(z-1,y,x), affs(2,z,y,x));
                    
                    if (y>1)
                        _accumulate_edge(segid, frag(z,y-1,x), affs(1,z,y,x));
                    
                    if (x>1)
                        _accumulate_edge(segid, frag(z,y,x-1), affs(0,z,y,x));
                }
            }
        }
    }

    std::cout<< "deal with boundary faces..." << std::endl;
    // negative z 
    // it has a contacting chunk face
    // we have already included the contacting face here
    // freeze both contacting face
    const auto& contactingFacesZ = xt::view(frag, 
        xt::range(0,2), xt::range(1, _), xt::range(1, _));
    const auto& contactingFaceIDsZ = xt::unique(contactingFacesZ);
    for(const auto& segid: contactingFaceIDsZ){
        if(segid) _segid2frozen[segid] |= NEG_Z; 
    }
    // accumulate edges
    for(std::size_t y=1; y<frag.shape(1); y++){
        for(std::size_t x=1; x<frag.shape(2); x++){
            const auto& segid = frag(1,y,x);
            if(segid)
                // the three affinity channels are ordered as xyz!
                _accumulate_edge(segid, frag(0,y,x), affs(2, 0, y, x));
        }
    }

    // negative y
    const auto& contactingFacesY = xt::view(frag, 
        xt::range(1, _), xt::range(0,2), xt::range(1, _));
    const auto& contactingFaceIDsY = xt::unique(contactingFacesY);
    for(const auto& segid: contactingFaceIDsY){
        if(segid) _segid2frozen[segid] |= NEG_Y;
    }
    // accumulate edges
    for(std::size_t z=1; z<frag.shape(0); z++){
        for(std::size_t x=1; x<frag.shape(2); x++){
            const auto& segid = frag(z,1,x);
            if(segid)
                // the three affinity channels are ordered as xyz!
                _accumulate_edge(segid, frag(z, 0, x), affs(1, z, 0, x));
        }
    }

    // negative x 
    const auto& contactingFacesX = xt::view(frag, 
        xt::range(1, _), xt::range(1, _), xt::range(0,2));
    const auto& contactingFaceIDsX = xt::unique(contactingFacesX);
    for(const auto& segid: contactingFaceIDsX){
        if(segid) _segid2frozen[segid] = NEG_X; 
    }

    // accumulate edges
    for(std::size_t z=1; z<frag.shape(0); z++){
        for(std::size_t y=1; y<frag.shape(1); y++){
            const auto& segid = frag(z, y, 1);
            if(segid)
                // the three affinity channels are ordered as xyz!
                _accumulate_edge(segid, frag(z, y, 0), affs(0, z, y, 0));
        }
    }

}




std::string as_string() const {
    std::ostringstream stringStream;
    stringStream<< "frozen set: ";
    for(const auto& [segid, frozen] : _segid2frozen){
        // directly print out uint8_t could be regarded as ASCII code and be empty!
        // https://stackoverflow.com/questions/19562103/uint8-t-cant-be-printed-with-cout
        stringStream<< segid << "--" << unsigned(frozen) << ", ";
    }
    stringStream<<"\n";
    stringStream << RegionGraph::as_string();
    return stringStream.str();
}

/**
 * @brief merge nodes in leaf chunk
 * 
 * @param threshold 
 * 
 * @return reneu::Dendrogram
 */
auto merge_in_leaf_chunk(const aff_edge_t& threshold){
    return _greedy_mean_affinity_agglomeration(threshold);
}

/** 
 * @brief the other node in the binary bounding box tree. 
 * 
 * The nodes freezed by the contacting face should be melted.
 * 
 * @param upperRegionGraghChunk The other region graph chunk with some frozen nodes.
 * @param dim dimension or axis to merge across.
 * @param threshold affinity threshold during agglomeration.
 * 
 * @return reneu::Dendrogram
 */
auto merge_upper_chunk(const RegionGraphChunk& upperRegionGraphChunk, 
                                const std::size_t& dim, const aff_edge_t& threshold){
    assert(dim>=0 && dim<3);
    const auto& NEG_SURFACE_BIT = SURFACE_BITS[dim];
    const auto& POS_SURFACE_BIT = SURFACE_BITS[3+dim];

    // merge the frozen set
    // the contacting face should be melted
    for(auto it = _segid2frozen.begin(); it != _segid2frozen.end();){
        auto& frozen = it->second;
        frozen &= (~POS_SURFACE_BIT);
        if(frozen == 0)
            it = _segid2frozen.erase(it); 
        else
            ++it;
    }
    
    for(const auto& [segid, frozen] : upperRegionGraphChunk._segid2frozen){
        // the contacting face of upper chunk is in negative direction!
        const auto& meltedFlag = frozen & (~NEG_SURFACE_BIT);
        if(meltedFlag){
            // this segment is still frozen by other faces
            _segid2frozen[segid] |= meltedFlag;
        }
    }
    std::cout<<"number of remaining frozen segments: "<< _segid2frozen.size() << std::endl;


    // merge the region graphs
    for(const auto& [segid0, neighbor] : upperRegionGraphChunk._segid2neighbor){
        for(const auto& [segid1, edgeIndex] : neighbor){
            if(segid0 < segid1){
                const auto& upperEdge = upperRegionGraphChunk._edgeList[edgeIndex];
                _segid2neighbor[segid0][segid1] = _edgeList.size();
                _segid2neighbor[segid1][segid0] = _edgeList.size();
                _edgeList.push_back(upperEdge);
            }
        }
    }

    // std::cout<< "region graph after merging: "<< as_string() << std::endl;
    // greedy iterative agglomeration
    return _greedy_mean_affinity_agglomeration(threshold); 
}

}; // class of RegionGraphChunk


} // namespace of reneu