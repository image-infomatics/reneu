#pragma once

#include <fstream>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <filesystem>
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include "xtensor/xcsv.hpp"
#include "xiuli/utils/math.hpp"
#include "nanoflann.hpp"

// use the c++17 nested namespace
namespace xiuli::neuron::nblast{

using NodesType = xt::xtensor<float, 2>;
using TableType = xt::xtensor_fixed<float, xt::xshape<21, 10>>; 
using DistThresholdsType = xt::xtensor_fixed<float, xt::xshape<22>>; 
using ADPThresholdsType = xt::xtensor_fixed<float, xt::xshape<11>>; 

namespace nf = nanoflann;


class ScoreTable{

private:
    TableType table;
    const DistThresholdsType distThresholds = {0., 0.75, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 
                                        12, 14, 16, 20, 25, 30, 40, std::numeric_limits<float>::max()};
    const ADPThresholdsType adpThresholds = {0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    template<std::size_t N>
    auto binary_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, const float &value ){
        std::size_t start = 0;
        // Note that the last one index is N-1 rather than N_
        // This is following python indexing style, the last index is uninclusive
        std::size_t stop = N;
        while( stop - start > 1 ){
            auto middle = std::div(stop - start, 2).quot + start;
            if ( value > thresholds[ middle ] ){
                // Move forward
                start = middle;
            }else{
                // Move backward
                stop = middle;
            }
        }
        // the start is the matching index
        return start;
    }

public:
    ScoreTable(const xt::pytensor<float, 2> &table_): table(table_){}

    ScoreTable( const std::string fileName = std::filesystem::current_path() / "../../../data/smat_fcwb.csv"){
        std::ifstream in(fileName); 
        table = xt::load_csv<float>( in );
    }

    inline auto get_pytable(){
        // xtensor_fixed can not be converted to pytensor directly
        return xt::xtensor<float, 2>(table);
    }

    /**
     * \brief 
     * \param dist: physical distance
     * \param dp: absolute dot product of vectors
     */
    inline auto operator()(const float &dist, const float &adp){
        std::size_t distIdx = binary_search( distThresholds, dist );
        std::size_t adpIdx = binary_search( adpThresholds, adp );
        return table( distIdx, adpIdx );
    }

    inline auto operator()(const std::tuple<float, float> &slice){
        return this->operator()(std::get<0>(slice), std::get<1>(slice));
    }
}; // ScoreTable class 

/**
 * \brief adapt nodes to nanoflann datastructure without copy
 */
struct NodesAdaptor{
    const NodesType &nodes; //!< A const ref to the data set origin

    // the constructor that sets the data set source
    NodesAdaptor(const NodesType &nodes_) : nodes(nodes_) {}

    // CRTP helper method
    inline const NodesType& derived() const { return nodes; }

    // Must return the number of data points
    inline std::size_t kdtree_get_point_count() const { return derived().shape(0); }

    float kdtree_distance(const float *p, const std::size_t idx_q, std::size_t size) const{
        //assert(size == 3);
        return std::sqrt(std::pow(p[0] - nodes(idx_q, 0), 2) +  
                         std::pow(p[1] - nodes(idx_q, 1), 2) +
                         std::pow(p[2] - nodes(idx_q, 2), 2));   
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value,
    // the "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const{
        if (dim==0) return derived()(idx, 0);
        else if (dim==1) return derived()(idx, 1);
        else if (dim==2) return derived()(idx, 2);
        else std::unexpected();
    } 

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
}; // end of NodesAdaptor

typedef NodesAdaptor Nodes2KD;

class VectorCloud{

private:
    typedef nf::KDTreeSingleIndexAdaptor<
            nf::L2_Simple_Adaptor<float, Nodes2KD>,
            Nodes2KD, 3 /* dim */> my_kd_tree_t;

    NodesType nodes;
    xt::xtensor<float, 2> vectors;
    // the adaptor
    Nodes2KD nodes2kd;
    my_kd_tree_t kdtree;

public:
    // our nodes array contains radius direction, but we do not need it.
    VectorCloud( NodesType &nodes_, const std::size_t nearestNodeNum = 20 )
        : nodes(nodes_),
          nodes2kd(nodes),
          kdtree(3 /*dim*/, nodes2kd, nf::KDTreeSingleIndexAdaptorParams(10 /*max leaf*/))
    {
        auto nodeNum = nodes.shape(0);

        kdtree.buildIndex();
        
        // find the nearest k nodes and compute the first principle component as the main direction
        xt::xtensor<float, 1>::shape_type shape1D = {nearestNodeNum};
        xt::xtensor<std::size_t, 1> nearestNodeIndexes = xt::empty<std::size_t>(shape1D);
        xt::xtensor<float, 1> distances = xt::empty<std::size_t>(shape1D);
        
        xt::xtensor<float, 2>::shape_type shape = {nearestNodeNum, 3};
        NodesType nearestNodes = xt::empty<float>(shape);
        nf::KNNResultSet<float> resultSet( nearestNodeNum );
        resultSet.init(nearestNodeIndexes.data(), distances.data());

        xt::xtensor<float, 2>::shape_type vshape = {nodeNum, 3};
        vectors = xt::empty<float>( vshape );

        for (std::size_t nodeIdx = 0; nodeIdx < nodeNum; nodeIdx++){
            auto node = xt::view(nodes, nodeIdx, xt::range(0, 3));
            kdtree.findNeighbors( resultSet, node.data(), nf::SearchParams(nearestNodeNum) );    

            for (std::size_t i = 0; i < nearestNodeNum; i++){
                auto nearestNodeIdx = nearestNodeIndexes(i);
                //nearestNodes(i, xt::all()) = nodes(nearestNodeIdx, xt::range(0,3));
                nearestNodes(i, 0) = nodes(nearestNodeIdx, 0);
                nearestNodes(i, 1) = nodes(nearestNodeIdx, 1);
                nearestNodes(i, 2) = nodes(nearestNodeIdx, 2);
            }
            // use the first principle component as the main direction
            //vectors(nodeIdx, xt::all()) = xiuli::utils::pca_first_component( nearestNodes ); 
            auto direction = xiuli::utils::pca_first_component( nearestNodes );
            vectors(nodeIdx, 0) = direction(0);
            vectors(nodeIdx, 1) = direction(1);
            vectors(nodeIdx, 2) = direction(2);
        }
    }
    
    VectorCloud( const xt::pytensor<float, 2> &nodes_, const std::size_t &nearestNodeNum = 20 )
        : nodes(nodes_),
          nodes2kd(nodes),
          kdtree(3 /*dim*/, nodes2kd, nf::KDTreeSingleIndexAdaptorParams(10 /*max leaf*/))
    {
        VectorCloud( nodes, nearestNodeNum );
    }

    inline auto get_vectors(){
        return vectors;
    }
    //auto query_by(xt::xtensor<float, 1> &node){
    //}

}; // VectorCloud class

} // end of namespace xiuli::neuron::nblast