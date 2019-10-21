#pragma once

#include <fstream>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <filesystem>
#include <memory>
#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include "xtensor/xcsv.hpp"
#include "xiuli/utils/math.hpp"
#include "xiuli/utils/kd_tree.hpp"

// use the c++17 nested namespace
namespace xiuli::neuron::nblast{

using NodesType = xt::xtensor<float, 2>;

namespace py = pybind11;

class ScoreTable{

private:
    xt::xtensor_fixed<float, xt::xshape<21, 10>> table; 
    //const DistThresholdsType distThresholds = {0., 0.75, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 
    //                                    12, 14, 16, 20, 25, 30, 40, std::numeric_limits<float>::max()};
    // this is using nanometer rather than micron
    const xt::xtensor_fixed<float, xt::xshape<22>> distThresholds = {
            1000., 750, 1500, 2000, 2500, 3000, 3500, 4000, 
            5000, 6000, 7000, 8000, 9000, 10000, 
            12000, 14000, 16000, 20000, 25000, 30000, 40000, 
            std::numeric_limits<float>::max()};

    const xt::xtensor_fixed<float, xt::xshape<11>> adpThresholds = {
            0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    template<std::size_t N>
    auto binary_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, const float &value ) const {
        std::size_t start = 0;
        // Note that the last one index is N-1 rather than N_
        // This is following python indexing style, the last index is not inclusive
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

    ScoreTable( const std::string fileName ){
        std::ifstream in(fileName); 
        table = xt::load_csv<float>( in );
    }

    ScoreTable(){
        std::string fileName = std::filesystem::current_path() / "../../../data/smat_fcwb.csv";
        std::ifstream in(fileName); 
        table = xt::load_csv<float>( in );
    }

    inline auto get_pytable() const {
        // xtensor_fixed can not be converted to pytensor directly
        return xt::xtensor<float, 2>(table);
    }

    /**
     * \brief 
     * \param dist: physical distance
     * \param dp: absolute dot product of vectors
     */
    inline auto operator()(const float &dist, const float &adp) const {
        std::size_t distIdx = binary_search( distThresholds, dist );
        std::size_t adpIdx = binary_search( adpThresholds, adp );
        return table( distIdx, adpIdx );
    }

    inline auto operator()(const std::tuple<float, float> &slice){
        return this->operator()(std::get<0>(slice), std::get<1>(slice));
    }

}; // ScoreTable class 

class VectorCloud{

private:
    const NodesType nodes;
    xt::xtensor<float, 2> vectors;
    xiuli::utils::ThreeDTree kdTree;
    auto construct_vectors(const std::size_t &nearestNodeNum=20){
        auto nodeNum = nodes.shape(0);
        
        // find the nearest k nodes and compute the first principle component as the main direction
        xt::xtensor<float, 2>::shape_type shape = {nearestNodeNum, 3};
        NodesType nearestNodes = xt::empty<float>(shape);

        xt::xtensor<float, 2>::shape_type vshape = {nodeNum, 3};
        vectors = xt::empty<float>( vshape );

        for (std::size_t nodeIdx = 0; nodeIdx < nodeNum; nodeIdx++){
            xt::xtensor<float, 1> queryNode = xt::view(nodes, nodeIdx, xt::range(0, 3));
            auto nearestNodeIndices = kdTree.find_nearest_k_node_indices(
                                                        queryNode, nearestNodeNum);
            std::cout<< "nearest node indices: " << nearestNodeIndices << std::endl;
            for (std::size_t i=0; i<nearestNodeIndices.size(); i++){
                auto nearestNodeIndex = nearestNodeIndices(i);
                nearestNodes(i, 0) = nodes( nearestNodeIndex, 0 );
                nearestNodes(i, 1) = nodes( nearestNodeIndex, 1 );
                nearestNodes(i, 2) = nodes( nearestNodeIndex, 2 );
            }
            // use the first principle component as the main direction
            //vectors(nodeIdx, xt::all()) = xiuli::utils::pca_first_component( nearestNodes ); 
            std::cout<< "nearest nodes: " << nearestNodes << std::endl;
            auto direction = xiuli::utils::pca_first_component( nearestNodes );
            vectors(nodeIdx, 0) = direction(0);
            vectors(nodeIdx, 1) = direction(1);
            vectors(nodeIdx, 2) = direction(2);
            std::cout<< "vector: " << direction << std::endl;
        }
    }

public:
    inline auto size() const {
        return nodes.shape(0);
    }

    inline auto get_nodes() const {
        return nodes;
    }

    inline auto get_vectors() const {
        return vectors;
    }

    // our nodes array contains radius direction, but we do not need it.
    VectorCloud( const NodesType &nodes_, const std::size_t &nearestNodeNum = 20 )
        : nodes(nodes_), kdTree(xiuli::utils::ThreeDTree(nodes, nearestNodeNum)){
        // kdTree = xiuli::utils::ThreeDTree(nodes, nearestNodeNum);
        construct_vectors( nearestNodeNum );
    }
    
    VectorCloud( const xt::pytensor<float, 2> &nodes_, const std::size_t &nearestNodeNum = 20 )
        : nodes(nodes_), kdTree(xiuli::utils::ThreeDTree(nodes, nearestNodeNum) ) {
        // kdTree = xiuli::utils::ThreeDTree(nodes, nearestNodeNum);
        construct_vectors( nearestNodeNum );
    }

    auto query_by(VectorCloud &query, const ScoreTable &scoreTable) const {
        // raw NBLAST is accumulated by query nodes
        float rawScore = 0;

        float distance = 0;
        float absoluteDotProduct = 0;
        std::size_t nearestNodeIdx = 0; 

        NodesType queryNodes = query.get_nodes();
        for (std::size_t queryNodeIdx = 0; queryNodeIdx<query.size(); queryNodeIdx++){
            xt::xtensor<float, 1> queryNode = xt::view(queryNodes, queryNodeIdx, xt::range(0, 3));
            // std::cout<< "\nquery node: " << queryNode <<std::endl;
            // find the best match node in target and get physical distance
            auto nearestNodeIndex = kdTree.find_nearest_k_node_indices( queryNode, 1 )(0);
            // std::cout<< "nearest node index: "<< nearestNodeIndex << std::endl;

            auto nearestNode = xt::view(nodes, nearestNodeIndex, xt::range(0,3));
            distance = xt::norm_l2( nearestNode - queryNode )(0);
            // std::cout<< "distance: " << distance << std::endl;

            // compute the absolute dot product between the principle vectors
            auto queryVector = xt::view(query.get_vectors(), queryNodeIdx, xt::all());
            auto targetVector = xt::view(vectors, nearestNodeIdx, xt::all());
            //auto dot = xt::linalg::dot(queryVector, targetVector);
            //assert( dot.size() == 1 );
            //absoluteDotProduct = std::abs(dot(0));
            // std::cout<< "vectors: "<<queryVector << "   "<< targetVector << std::endl;
            absoluteDotProduct = std::abs(xt::linalg::dot( queryVector, targetVector )(0));
            // std::cout<< "absolute dot product: " << absoluteDotProduct << std::endl;
            // lookup the score table and accumulate the score
            rawScore += scoreTable( distance,  absoluteDotProduct );
            // std::cout<< "score: "<< scoreTable(distance, absoluteDotProduct) << std::endl; 
            // std::cout<< "accumulated score: " << rawScore <<std::endl; 
        }
        return rawScore; 
    }

}; // VectorCloud class

class NBLASTScoreMatrix{
private:
// the rows are targets, the columns are queries
xt::xtensor<float, 2> rawScoreMatrix;

public:
    NBLASTScoreMatrix(  const std::vector<VectorCloud> &vectorClouds, 
                        const ScoreTable &scoreTable){
        std::size_t vcNum = vectorClouds.size();
        xt::xtensor<float, 2>::shape_type shape = {vcNum, vcNum};
        rawScoreMatrix = xt::empty<float>( shape );

        for (std::size_t targetIdx = 0; targetIdx<vcNum; targetIdx++){
            VectorCloud target = vectorClouds[ targetIdx ];
            for (std::size_t queryIdx = targetIdx; queryIdx<vcNum; queryIdx++){
                auto query = vectorClouds[ queryIdx ];
                rawScoreMatrix( targetIdx, queryIdx ) = target.query_by( query, scoreTable ); 
            }
        }
    }

    //NBLASTScoreMatrix( const py::list &vectorClouds, const ScoreTable scoreTable ){
    //    
    //}

    //NBLASTScoreMatrix(  const std::vector<xiuli::neuron::Skeleton> &skeletonList, 
    //                    const ScoreTable &scoreTable){
    //    
    //}

    inline auto get_neuron_number() const {
        return rawScoreMatrix.shape(0);
    }

    inline auto get_raw_score_matrix() const {
        return rawScoreMatrix;
    }

    /*
     * \brief normalized by the self score of query
     */
    inline auto get_normalized_score_matrix() const {
        xt::xtensor<float, 2> normalizedScoreMatrix = xt::zeros_like(rawScoreMatrix);
        for (std::size_t queryIdx = 0; queryIdx<get_neuron_number(); queryIdx++){
            auto selfQueryScore = rawScoreMatrix( queryIdx, queryIdx );
            for (std::size_t targetIdx = 0; targetIdx<get_neuron_number(); targetIdx++){
                normalizedScoreMatrix(targetIdx, queryIdx) = rawScoreMatrix(targetIdx, queryIdx) / selfQueryScore;
            }
        }
        return normalizedScoreMatrix; 
    }

    inline auto get_mean_score_matrix() const {
        auto normalizedScoreMatrix = get_normalized_score_matrix();
        auto meanScoreMatrix = xt::ones_like( normalizedScoreMatrix );
        for (std::size_t targetIdx = 0; targetIdx<get_neuron_number(); targetIdx++){
            for (std::size_t queryIdx = targetIdx+1; queryIdx<get_neuron_number(); queryIdx++){
                meanScoreMatrix(targetIdx, queryIdx) = (normalizedScoreMatrix(targetIdx, queryIdx) + 
                                                        normalizedScoreMatrix(queryIdx, targetIdx)) / 2;
                meanScoreMatrix(queryIdx, targetIdx) = meanScoreMatrix(targetIdx, queryIdx);
            }
        }
        return meanScoreMatrix;
    }
};

} // end of namespace xiuli::neuron::nblast
