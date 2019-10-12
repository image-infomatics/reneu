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

// use the c++17 nested namespace
namespace xiuli::neuron::nblast{

using NodesType = xt::xtensor<float, 2>;
using TableType = xt::xtensor_fixed<float, xt::xshape<21, 10>>; 
using DistThresholdsType = xt::xtensor_fixed<float, xt::xshape<22>>; 
using ADPThresholdsType = xt::xtensor_fixed<float, xt::xshape<11>>; 

namespace py = pybind11;

class ScoreTable{

private:
    TableType table;
    //const DistThresholdsType distThresholds = {0., 0.75, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 
    //                                    12, 14, 16, 20, 25, 30, 40, std::numeric_limits<float>::max()};
    // this is using nanometer rather than micron
    const DistThresholdsType distThresholds = {1000., 750, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 
                                        12000, 14000, 16000, 20000, 25000, 30000, 40000, std::numeric_limits<float>::max()};

    const ADPThresholdsType adpThresholds = {0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    template<std::size_t N>
    auto binary_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, const float &value ) const {
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

// ThreeDTree
class AbstractThreeDNode{
public:
    // we need to make base class polymorphic for dynamic cast
    virtual ~AbstractThreeDNode() = default;
    virtual std::size_t size() const;
    virtual xt::xtensor<std::size_t, 1> get_node_indices() const;
    virtual void fill_node_indices(xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                    std::size_t &filledNum) const;
};

using ThreeDNodePtr = std::shared_ptr<AbstractThreeDNode>;

class ThreeDLeafNode: public AbstractThreeDNode{
public:
    ~ThreeDLeafNode() = default;
    xt::xtensor<std::size_t, 1> nodeIndices;

    ThreeDLeafNode(const xt::xtensor<std::size_t, 1> &nodeIndices_) : 
                                            nodeIndices(nodeIndices_){}

    std::size_t size() const {
        return nodeIndices.size();
    }

    xt::xtensor<std::size_t, 1> get_node_indices() const {
        return nodeIndices;
    }

    void fill_node_indices(xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                                        std::size_t &filledNum){
        for (std::size_t i=0; i<nodeIndices.size(); i++){
            nodeIndicesBuffer( i + filledNum ) = nodeIndices(i);
        }
    }
};
using ThreeDLeafNodePtr = std::shared_ptr<ThreeDLeafNode>;

class ThreeDInsideNode: public AbstractThreeDNode{
public: 
    ~ThreeDInsideNode() = default;
    std::size_t middleNodeIndex;
    ThreeDNodePtr leftNodePtr;
    ThreeDNodePtr rightNodePtr;
    std::size_t nodeNum;

    ThreeDInsideNode(const std::size_t &middleNodeIndex_, 
            ThreeDNodePtr leftNodePtr_, ThreeDNodePtr rightNodePtr_, std::size_t nodeNum_):
            middleNodeIndex(middleNodeIndex_), leftNodePtr(leftNodePtr_), 
            rightNodePtr(rightNodePtr_), nodeNum(nodeNum_){};

    std::size_t size() const {
        // return leftNodePtr->size() + rightNodePtr->size() + 1;
        return nodeNum;
    }
    
    void fill_node_indices(
                xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, std::size_t &filledNum) const {
        leftNodePtr->fill_node_indices(nodeIndicesBuffer, filledNum);
        filledNum += leftNodePtr->size();
        nodeIndicesBuffer(filledNum) = middleNodeIndex;
        filledNum += 1;
        rightNodePtr->fill_node_indices( nodeIndicesBuffer, filledNum );
        filledNum += rightNodePtr->size();
    }

    xt::xtensor<std::size_t, 1> get_node_indices(){
        xt::xtensor<std::size_t, 1>::shape_type sh = {nodeNum};
        xt::xtensor<std::size_t, 1> nodeIndicesBuffer =  xt::empty<std::size_t>(sh);

        std::size_t filledNum = 0;
        fill_node_indices( nodeIndicesBuffer, filledNum );
        return nodeIndicesBuffer;
    }

}; // ThreeDNode class

using ThreeDInsideNodePtr = std::shared_ptr<ThreeDInsideNode>;

class ThreeDTree{
private:
    ThreeDNodePtr root;
    const NodesType nodes;
    const std::size_t leafSize;
    auto next_dim(std::size_t &dim) const {
        if(dim==3)
            dim = 0;
        else
            dim += 1;
        return dim;
    }

    ThreeDNodePtr make_ThreeDtree_node(const xt::xtensor<float, 1> &coords, std::size_t &dim){
        // find the median value index
        const auto medianIndex = coords.size() / 2;
        auto sortedIndices = xt::argpartition(coords, medianIndex);
        auto middleNodeIndex = sortedIndices(medianIndex);

        ThreeDNodePtr leftNodePtr, rightNodePtr;

        // recursively loop the dimension in 3D
        dim = next_dim(dim);       
        if (medianIndex > leafSize){
            auto coords = xt::view(nodes, xt::all(), dim);
            auto leftIndices = xt::view(sortedIndices, xt::range(0, medianIndex));
            auto rightIndices = xt::view(sortedIndices, xt::range(medianIndex+1, _));
            auto leftCoords = xt::index_view(coords, leftIndices);
            auto rightCoords = xt::index_view(coords, rightIndices);
            ThreeDNodePtr leftNodePtr = make_ThreeDtree_node( leftCoords, dim );
            ThreeDNodePtr rightNodePtr = make_ThreeDtree_node( rightCoords, dim );
        } else {
            // include all the nodes as a leaf
            auto leftIndices = xt::view(sortedIndices, xt::range(0, medianIndex));
            auto rightIndices = xt::view(sortedIndices, xt::range(medianIndex+1, _));
            ThreeDNodePtr leftNodePtr = std::make_shared<ThreeDLeafNode>( leftIndices );
            ThreeDNodePtr rightNodePtr = std::make_shared<ThreeDLeafNode>( rightIndices );
        }

        return std::make_shared<AbstractThreeDNode>(
            ThreeDInsideNode(middleNodeIndex, leftNodePtr, rightNodePtr, coords.size()));
    }

    auto is_inside_node(const ThreeDNodePtr nodePtr){
        return std::dynamic_pointer_cast<ThreeDInsideNode>( nodePtr );
    }

    auto is_leaf_node(const ThreeDNodePtr nodePtr){
        return std::dynamic_pointer_cast<ThreeDLeafNode>( nodePtr );
    }

    xt::xtensor<std::size_t, 1> search_nearest_nodes_in_3dtree (
                ThreeDNodePtr nodePtr, const xt::xtensor<float, 1> &queryNode, 
                std::size_t &dim, const std::size_t &nearestNodeNum=1) const {
        
        ThreeDNodePtr closeNodePtr, farNodePtr;
        
        // use dynamic_cast to check the class
        if( ThreeDInsideNodePtr insideNodePtr = 
                            std::dynamic_pointer_cast<ThreeDInsideNode>( nodePtr ) ){
            // compare with the median value
            if (queryNode(dim) < nodes(insideNodePtr->middleNodeIndex, dim)){
                // continue checking left nodes
                closeNodePtr = insideNodePtr->leftNodePtr;
                farNodePtr = insideNodePtr->rightNodePtr;
            } else {
                closeNodePtr = insideNodePtr->rightNodePtr;
                farNodePtr = insideNodePtr->leftNodePtr;
            }

            if (closeNodePtr->size() > nearestNodeNum){
                dim = next_dim(dim);
                return search_nearest_nodes_in_3dtree(closeNodePtr, queryNode, dim);
            } else if (closeNodePtr->size() == nearestNodeNum){
                //assert( std::dynamic_pointer_cast<ThreeDLeafNode>( closeNodePtr ) );
                return closeNodePtr->get_node_indices();
            } else {
                // closeNodePtr->size() < nearestNodeNum
                // we need to add some close nodes from far node

            }
        } else if (ThreeDLeafNodePtr leafNodePtr = 
                            std::dynamic_pointer_cast<ThreeDLeafNode>( nodePtr ) ){

            // this is a leaf node, we have to compute the distance one by one
            auto nodeIndices = leafNodePtr->nodeIndices;
            std::vector< std::size_t > ret = {};
            ret.reserve( nearestNodeNum );
            if (nearestNodeNum == 1){
                // use squared distance to avoid sqrt computation
                float minSquaredDist = std::numeric_limits<float>::max();
                std::size_t nearestNodeIndex = 0;
                for (auto i : nodeIndices){
                    auto node = xt::view(nodes, i, xt::range(0, 3));
                    float squaredDist = xt::norm_sq(node - queryNode)(0);
                    if (squaredDist < minSquaredDist){
                        nearestNodeIndex = i;
                        minSquaredDist = squaredDist;
                    }
                }
                ret.push_back( nearestNodeIndex  );
                return ret;
            } else if (((nearestNodeNum - nodeIndices.size())>= -1) && 
                      ((nearestNodeNum - nodeIndices.size())<=  1)){
                for (auto nodeIndex : nodeIndices){
                    auto node = xt::view(nodes, nodeIndex, xt::range(0, 3));
                    float dist = xt::norm_l2( node - queryNode )(0);
                    ret.push_back( nodeIndex );
                }
                return ret;
            } else {
                std::cerr<< "only support nearest node number 1 or close to K"<<std::endl;
            }
        } else {
            std::cerr<< "ThreeDNodePtr is not pointing to correct type!" <<std::endl;
        }
    }

public:
    auto get_leaf_size() const {
        return leafSize;
    }

    ThreeDTree(const NodesType &nodes_, const std::size_t leafSize_=10): 
            nodes(nodes_), leafSize(leafSize_){
        // start from first dimension
        std::size_t dim = 0;
        auto coords = xt::view(nodes, xt::all(), dim);
        root = make_ThreeDtree_node( coords, dim );
    }

    auto get_nearest_node(const xt::xtensor<float, 1> &queryNode) const {
        std::size_t dim = 0;
        auto queryNodeCoord = xt::view(queryNode, xt::range(0,3));
        auto results = search_nearest_nodes_in_3dtree(root, queryNodeCoord, dim, 1);
        assert( 1 == results.size() );
        return results[0];
    }

    auto get_k_nearest_nodes(const xt::xtensor<float, 1> &queryNode, 
                                        const std::size_t &nearestNodeNum=20) const {
        std::size_t dim = 0;
        
        assert( nearestNodeNum >= 1 );
        // currently, we only support nearest node number equal or less than leafSize
        assert( nearestNodeNum <= leafSize );

        auto queryNodeCoord = xt::view(queryNode, xt::range(0, 3));
        return search_nearest_nodes_in_3dtree(root, queryNodeCoord, dim, nearestNodeNum);
    } 
}; // ThreeDTree class

class VectorCloud{

private:

    NodesType nodes;
    xt::xtensor<float, 2> vectors;
    ThreeDTree ThreeDtree;
    auto construct_vectors(const std::size_t &nearestNodeNum=20){
        auto nodeNum = nodes.shape(0);
        
        // find the nearest k nodes and compute the first principle component as the main direction
        xt::xtensor<float, 2>::shape_type shape = {nearestNodeNum, 3};
        NodesType nearestNodes = xt::empty<float>(shape);

        xt::xtensor<float, 2>::shape_type vshape = {nodeNum, 3};
        vectors = xt::empty<float>( vshape );

        for (std::size_t nodeIdx = 0; nodeIdx < nodeNum; nodeIdx++){
            auto node = xt::view(nodes, nodeIdx, xt::range(0, 3));
            auto nearestNeighbors = ThreeDtree.get_k_nearest_nodes(node, nearestNodeNum);
            for (std::size_t i = 0; i < nearestNodeNum; i++){
                auto nearestNodeIndex = nearestNeighbors[i].first;
                //nearestNodes(i, xt::all()) = nodes(nearestNodeIdx, xt::range(0,3));
                nearestNodes(i, 0) = nodes(nearestNodeIndex, 0);
                nearestNodes(i, 1) = nodes(nearestNodeIndex, 1);
                nearestNodes(i, 2) = nodes(nearestNodeIndex, 2);
            }
            // use the first principle component as the main direction
            //vectors(nodeIdx, xt::all()) = xiuli::utils::pca_first_component( nearestNodes ); 
            auto direction = xiuli::utils::pca_first_component( nearestNodes );
            vectors(nodeIdx, 0) = direction(0);
            vectors(nodeIdx, 1) = direction(1);
            vectors(nodeIdx, 2) = direction(2);
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
    VectorCloud( NodesType &nodes_, const std::size_t &nearestNodeNum = 20 )
        : nodes(nodes_), ThreeDtree(ThreeDTree(nodes_, nearestNodeNum))
    {
        construct_vectors( nearestNodeNum );
    }
    
    VectorCloud( const xt::pytensor<float, 2> &nodes_, const std::size_t &nearestNodeNum = 20 )
        : nodes(nodes_), ThreeDtree(ThreeDTree(nodes_, nearestNodeNum))
    {
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
            // find the best match node in target and get physical distance
            auto nearestNodeDist = ThreeDtree.get_nearest_node( queryNode ).second;
            
            // compute the absolute dot product between the principle vectors
            auto queryVector = xt::view(query.get_vectors(), queryNodeIdx, xt::all());
            auto targetVector = xt::view(vectors, nearestNodeIdx);
            //auto dot = xt::linalg::dot(queryVector, targetVector);
            //assert( dot.size() == 1 );
            //absoluteDotProduct = std::abs(dot(0));
            absoluteDotProduct = std::abs(xt::linalg::dot( queryVector, targetVector )(0));

            // lookup the score table and accumulate the score
            rawScore += scoreTable( nearestNodeDist,  absoluteDotProduct );
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