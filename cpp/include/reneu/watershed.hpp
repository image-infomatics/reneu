#pragma once

#include <iostream>
#include <algorithm>
#include <tuple>

#include "type_aliase.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xfixed.hpp"

namespace reneu{

using SAG = Segmentation;

template < typename T > struct watershed_traits;

template <> struct watershed_traits<uint32_t>
{
    static const uint32_t high_bit      = 0x80000000;
    static const uint32_t mask          = 0x7FFFFFFF;
    static const uint32_t sag_visited   = 0x00000040;
};

template <> struct watershed_traits<uint64_t>
{
    static const uint64_t high_bit      = 0x8000000000000000LL;
    static const uint64_t mask          = 0x7FFFFFFFFFFFFFFFLL;
    static const uint64_t sag_visited   = 0x00000000000000040LL;
};

using traits = watershed_traits<segid_t>;

// direction mask
const std::array<std::uint32_t, 6> dirmask  = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
// inverse direction mask
const std::array<std::uint32_t, 6> idirmask = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};

auto steepest_ascent(const AffinityMap &affs, aff_edge_t low, aff_edge_t high ){
    // initialize the steepest ascent graph
    assert(affs.shape(0) == 3);
    auto sz = affs.shape(1);
    auto sy = affs.shape(2);
    auto sx = affs.shape(3);
    
    SAG::shape_type sag_shape = {sz, sy, sx};
    SAG sag = xt::zeros<segid_t>(sag_shape);

    for(auto z=0; z<sz; z++){
        for(auto y=0; y<sy; y++){
            for(auto x=0; x<sx; x++){
                // weights of all six edges incident to (z,y,x)
                // the affinity map channels are ordered in (x,y,z)
                auto negx = (x==0) ? low : affs(0, z,y,x);
                auto negy = (y==0) ? low : affs(1, z,y,x);
                auto negz = (z==0) ? low : affs(2, z,y,x);
                auto posx = (x>=sx-1) ? low : affs(0, z, y, x+1);
                auto posy = (y>=sy-1) ? low : affs(1, z, y+1, x);
                auto posz = (z>=sz-1) ? low : affs(2, z+1, y, x);

                auto m = std::max({negz, negy, negx, posz, posy, posx});

                // keep edges with maximum affinity
                if(m > low){ // no edges at all if m<=low
                    if(m==negx || negx >= high) sag(z,y,x) |= 0x01;
                    if(m==negy || negy >= high) sag(z,y,x) |= 0x02;
                    if(m==negz || negz >= high) sag(z,y,x) |= 0x04;
                    if(m==posx || posx >= high) sag(z,y,x) |= 0x08;
                    if(m==posy || posy >= high) sag(z,y,x) |= 0x10;
                    if(m==posz || posz >= high) sag(z,y,x) |= 0x20;
                }
            }
        }
    }
    return sag;
}

auto divide_plateaus(SAG& sag){
    auto sz = sag.shape(0);
    auto sy = sag.shape(1);
    auto sx = sag.shape(2);
    
    // direction
    const std::array<std::ptrdiff_t, 6> dir = {-1, -sx, -sx*sy, 1, sx, sx*sy};
    
    // queue all vertices for which a purely outgoing edge exists
    std::cout<<"queue all vertices for which a purely outgoing edge exists"<<std::endl;
    std::vector<std::ptrdiff_t> bfs;
    //bfs.reserve(sag.size());
    
    for(std::ptrdiff_t idx = 0; idx < sag.size(); idx++){
        for(size_t d=0; d<6; d++){
            if((sag[idx] & dirmask[d] != 0) && 
                        idx+dir[d]>=0 && 
                        idx+dir[d]<sag.size() && 
                        ((sag[idx+dir[d]] & idirmask[d]) == 0)){
                    // outgoing edge exists, no incoming edge
                    sag[idx] |= traits::sag_visited;
                    bfs.push_back(idx);
                    break;
            }
        }
    }

    // divide plateaus
    std::cout<<"divide plateaus for the sag"<< std::endl;
    size_t bfs_index = 0;
    while(bfs_index < bfs.size()){
        std::ptrdiff_t idx = bfs[bfs_index];
        segid_t to_set = 0;
        for(size_t d=0; d<6; d++){
            if(((sag[idx] & dirmask[d]) != 0) && 
                        idx+dir[d]>=0 && 
                        idx+dir[d]<sag.size()){
                // outgoing edge exists
                if(sag[idx+dir[d]] & idirmask[d]){
                    // incoming edge
                    if(!(sag[idx+dir[d]] & traits::sag_visited)){
                        bfs.push_back(idx+dir[d]);
                        sag[idx+dir[d]] |= traits::sag_visited;
                    }
                } else {
                    // purely outgoing edge
                    to_set = dirmask[d];
                }
            }
        }
        // picks unique outgoing edge, unsets visited bit
        sag[idx] = to_set;
        bfs_index++;
    }
    return sag;
}


auto find_basins(Segmentation& seg){
    // seg is initially the steepest ascent graph
    // and will be transformed in-place to yield the segmentation into basins
    
    auto sz = seg.shape(0);
    auto sy = seg.shape(1);
    auto sx = seg.shape(2);
    
    // direction
    const std::array<std::ptrdiff_t, 6> dir = {-1, -sx, -sx*sy, 1, sx, sx*sy};

    // voxel counts for each basin
    std::vector<size_t> counts = {0};
    // breadth first search
    std::vector<std::ptrdiff_t> bfs = {};

    segid_t next_id = 1;
    for(std::ptrdiff_t idx=0; idx<seg.size(); idx++){
        if(seg[idx]==0){
            // background voxel (no edges at all)
            // mark as assigned
            seg[idx] |= traits::high_bit;
            counts[0]++; 
        } else if((seg[idx] & traits::high_bit) == 0){
            // not yet assigned, enqueue
            bfs.push_back(idx);
            // mark as visited
            seg[idx] |= traits::sag_visited;

            // follow trajectory starting from idx
            size_t bfs_index = 0;
            while( bfs_index < bfs.size()){
                std::ptrdiff_t me = bfs[bfs_index];
                for(size_t d=0; d<6; d++){
                    if( seg[me] & dirmask[d] ){
                        // this is an outgoing edge
                        // target of edge
                        auto him = me + dir[d];
                        if( him>=0 && him<seg.size() ){
                            // target is inside this volume
                            if( seg[him] & traits::high_bit ){
                                // already assigned
                                for(auto& it : bfs){
                                    // assign entire queue to same ID including high bit
                                    seg[it] = seg[him];
                                }
                                counts[seg[him] & traits::mask] += bfs.size();
                                bfs.clear();
                                break;
                            } else if( !(seg[him] & traits::sag_visited) ){
                                // not visited
                                // make as visited
                                seg[him] |= traits::sag_visited;
                                bfs.push_back(him);
                            } // else ignore since visited (try next direction)
                        }
                    }
                }
                // go to next vertex in queue
                bfs_index++;
            }

            if(!bfs.empty()){
                // create a new basin 
                for(auto& it : bfs){
                    seg[it] = traits::high_bit | next_id;
                }
                counts.push_back(bfs.size());
                next_id++;
                bfs.clear();
            }
        }
    }
    std::cout<< "found: "<< next_id - 1 << " components" << std::endl;

    for(size_t idx = 0; idx < seg.size(); idx++){
        // clear high bit visited label
        seg[idx] &= traits::mask;
    }
    return std::make_tuple(seg, counts);
}

auto watershed(const AffinityMap& affs, const aff_edge_t& low, const aff_edge_t& high){
    std::cout<< "start steepest ascent..." << std::endl;
    auto sag = steepest_ascent(affs, low, high);
    std::cout<< "start divide plateaus..." << std::endl;
    divide_plateaus(sag);
    std::cout<< "start find basins ..." << std::endl;
    auto [seg, counts] = find_basins(sag);
    return seg;
}

auto py_watershed(const PyAffinityMap& affs, const aff_edge_t& low, const aff_edge_t& high){
    return watershed(affs, low, high);  
}

} // namespace reneu