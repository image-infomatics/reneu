#pragma once

#include <queue>
// #include "reneu/type_aliase.hpp"
#include "../type_aliase.hpp"
// #include "reneu/utils/priority_queue.hpp"


namespace reneu{


struct AffEdge{
    aff_edge_t aff;
    std::size_t channel;
    std::ptrdiff_t z;
    std::ptrdiff_t y;
    std::ptrdiff_t x;
    
    // constructor for emplace operation
    AffEdge(const aff_edge_t& aff_, const std::size_t& channel_, 
        const std::ptrdiff_t& z_, const std::ptrdiff_t& y_,
        const std::ptrdiff_t& x_): aff(aff_), channel(channel_),
        z(z_), y(y_), x(x_){}

    // comparitor for sorting in heap
    // sort by decremental rather than default increamental
    bool operator<(const AffEdge& other) const {
        return aff < other.aff;
    }
};

template<class SEG, class AFFS>
auto _build_priority_queue(const SEG& seg, const AFFS& affs, 
        const aff_edge_t& threshold){
    
    std::priority_queue<AffEdge> heap = {};
    // the first dimension of affs is channel
    assert(affs.shape(1)==seg.shape(0));
    assert(affs.shape(2)==seg.shape(1));
    assert(affs.shape(3)==seg.shape(2));

    // z direction
    // the order in channels is xyz due to historical definition!
    std::size_t channel = 2;
    for(std::ptrdiff_t z=1; z<seg.shape(0); z++){
        for(std::ptrdiff_t y=0; y<seg.shape(1); y++){
            for(std::ptrdiff_t x=0; x<seg.shape(2); x++){
                const auto& sid0 = seg(z,   y, x);
                const auto& sid1 = seg(z-1, y, x);
                if((sid0==0 && sid1>0) || (sid1==0 && sid0>0)){
                    // only one of them is background
                    const auto& aff = affs(channel, z,y,x);
                    if(aff > threshold)
                        heap.emplace(aff, channel, z, y, x);
                } 
            }
        }
    }

    // y direction
    channel = 1;
    for(std::ptrdiff_t z=0; z<seg.shape(0); z++){
        for(std::ptrdiff_t y=1; y<seg.shape(1); y++){
            for(std::ptrdiff_t x=0; x<seg.shape(2); x++){
                const auto& sid0 = seg(z, y, x);
                const auto& sid1 = seg(z, y-1, x);
                if((sid0==0 && sid1>0) || (sid1==0 && sid0>0)){
                    // only one of them is background
                    const auto& aff = affs(channel, z,y,x);
                    if(aff > threshold)
                        heap.emplace(aff, channel, z, y, x);
                } 
            }
        }
    }
    
    // x direction
    channel = 0;
    for(std::ptrdiff_t z=0; z<seg.shape(0); z++){
        for(std::ptrdiff_t y=0; y<seg.shape(1); y++){
            for(std::ptrdiff_t x=1; x<seg.shape(2); x++){
                const auto& sid0 = seg(z, y, x);
                const auto& sid1 = seg(z, y, x-1);
                if((sid0==0 && sid1>0) || (sid1==0 && sid0>0)){
                    // only one of them is background
                    const auto& aff = affs(channel, z,y,x);
                    if(aff > threshold)
                        heap.emplace(aff, channel, z, y, x);
                } 
            }
        }
    }
    return heap;
}

template<class HEAP, class SEG, class AFFS>
void _update_priority_queue(HEAP& heap, const SEG& seg, 
        const AFFS& affs, const aff_edge_t& threshold, 
        const std::ptrdiff_t& z, const std::ptrdiff_t& y, 
        const std::ptrdiff_t& x){
    
    assert(seg(z,y,x)>0);
    // search all the six surrounding directions
    // we are sure that voxel in z,y,x is non-zero since 
    // this is the voxel we just expanded
    if(z>0 && seg(z-1, y, x)==0 && affs(2, z, y, x)>threshold){
        const auto& sid0 = seg(z, y, x);
        const auto& sid1 = seg(z-1, y, x);
        assert(seg(z-1, y, x)==0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(2, z, y, x), 2, z, y, x);
    } 
    if(y>0 && seg(z, y-1, x)==0 && affs(1, z, y, x)>threshold){
        const auto& sid0 = seg(z, y, x);
        const auto& sid1 = seg(z, y-1, x);
        assert(seg(z, y-1, x)==0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(1, z, y, x), 1, z, y, x);
    }
    if(x>0 && seg(z, y, x-1)==0 && affs(0, z, y, x)>threshold){
        const auto& sid0 = seg(z, y, x);
        const auto& sid1 = seg(z, y, x-1);
        assert(seg(z, y, x-1)==0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(0, z, y, x), 0, z, y, x);
    }

    if(z<seg.shape(0)-1 && seg(z+1, y, x)==0 && affs(2, z+1, y, x)>threshold){
        const auto& sid0 = seg(z+1, y, x);
        const auto& sid1 = seg(z, y, x);
        assert(seg(z+1, y, x)==0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(2, z+1, y, x), 2, z+1, y, x);
    }
    if(y<seg.shape(1)-1 && seg(z, y+1, x)==0 && affs(1, z, y+1, x)>threshold){
        const auto& sid0 = seg(z, y+1, x);
        const auto& sid1 = seg(z, y, x);
        assert(seg(z, y+1, x)==0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(1, z, y+1, x), 1, z, y+1, x);
    }
    if(x<seg.shape(2)-1 && seg(z, y, x+1)==0 && affs(0, z, y, x+1)>threshold){
        const auto& sid0 = seg(z, y, x+1);
        const auto& sid1 = seg(z, y, x);
        assert(seg(z, y, x+1)==0);
        assert(seg(z, y, x)>0);
        assert((sid0==0 && sid1>0) || (sid1==0 && sid0>0));
        heap.emplace(affs(0, z, y, x+1), 0, z, y, x+1);
    }

}

void seeded_watershed(PySegmentation& seg, const PyAffinityMap& affs,   
        const aff_edge_t& threshold){
    
    std::cout<< "start building priority queue..."<<std::endl;
    auto heap = _build_priority_queue(seg, affs, threshold);
    std::cout<< "Length of priority queue: "<< heap.size() << std::endl;

    while(!heap.empty()){
        const auto& affEdge = heap.top();
        heap.pop();
        std::cout<< affEdge.aff << ", ";

        const auto& channel = affEdge.channel;
        const auto& z = affEdge.z;
        const auto& y = affEdge.y;
        const auto& x = affEdge.x;

        switch(channel){
            case 2:
                if(seg(z, y, x) == 0){
                    assert(seg(z-1, y, x)>0);
                    seg(z,y,x) = seg(z-1, y, x);
                    _update_priority_queue(heap, seg, affs, threshold, z, y, x);
                }else if(seg(z-1, y, x) ==0){
                    assert(seg(z, y, x)>0);
                    seg(z-1, y, x) = seg(z, y, x);
                    _update_priority_queue(heap, seg, affs, threshold, z-1, y, x);
                }
            case 1:
                if(seg(z, y, x) == 0){
                    assert(seg(z, y-1, x)>0);
                    seg(z, y, x) = seg(z, y-1, x);
                    _update_priority_queue(heap, seg, affs, threshold, z, y, x);
                }else if(seg(z, y-1, x)==0){
                    assert(seg(z, y, x)>0);
                    seg(z, y-1, x) = seg(z, y, x);
                    _update_priority_queue(heap, seg, affs, threshold, z, y-1, x);
                }
            case 0:
                if(seg(z, y, x) == 0){
                    if(seg(z,y,x-1)==0)
                        std::cout<<"\nposition: "<<z<<", "<<y<<", "<<x<<std::endl;
                    assert(seg(z, y, x-1)>0);
                    seg(z, y, x) = seg(z, y, x-1);
                    _update_priority_queue(heap, seg, affs, threshold, z, y, x);
                }else if(seg(z, y, x-1) == 0){
                    assert(seg(z, y, x)>0);
                    seg(z, y, x-1) = seg(z, y, x);
                    _update_priority_queue(heap, seg, affs, threshold, z, y, x-1);
                } 
        }


    }
}

} // namespace of reneu