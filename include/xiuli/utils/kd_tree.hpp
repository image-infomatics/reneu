#pragma once

#include <limits>       // std::numeric_limits
#include <iostream>
#include <queue>
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"

// use the c++17 nested namespace
namespace xiuli::utils{

using namespace xt::placeholders;

using NodesType = xt::xtensor<float, 2>;

// the type of element in priority queue
using PriorityQueueElement = std::pair<float, std::size_t>;
// build priority queue to store the nearest neighbors
auto cmp = [](PriorityQueueElement left, PriorityQueueElement right) {
    return left.first < right.first;
};
using NearestNodePriorityQueue = std::priority_queue<
        PriorityQueueElement, std::vector<PriorityQueueElement>, decltype(cmp)>; 

inline void update_nearest_node_priority_queue(
            NearestNodePriorityQueue &nearestNodePriorityQueue, 
            const NodesType &nodes,
            const std::size_t &nodeIndex,
            const xt::xtensor<float, 1> &queryNode){
    auto node = xt::view(nodes, nodeIndex, xt::range(0, 3));

    float squaredDist = xt::norm_sq(node - queryNode)(0);
    if ( squaredDist < nearestNodePriorityQueue.top().first ){
        nearestNodePriorityQueue.pop();
        nearestNodePriorityQueue.push( std::make_pair( squaredDist, nodeIndex ) );
    }
}

inline auto next_dim(std::size_t &dim) {
    if(dim==2)
        dim = 0;
    else
        dim += 1;
    return dim;
}

// ThreeDTree
class ThreeDNode{
public:
    ThreeDNode() = default;
    // we need to make base class polymorphic for dynamic cast
    virtual std::size_t size() const = 0;
    
    virtual xt::xtensor<std::size_t, 1> get_node_indices() const = 0;
    
    virtual void fill_node_indices(xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                    std::size_t &startIndex) const = 0;
    
    virtual void find_nearest_k_node_indices(
                const xt::xtensor<float, 1> &queryNode,
                const NodesType &nodes, 
                std::size_t &dim, 
                NearestNodePriorityQueue &nearestNodePriorityQueue) const = 0; 
};
using ThreeDNodePtr = std::shared_ptr<ThreeDNode>;


class ThreeDLeafNode: public ThreeDNode{

private:
    xt::xtensor<std::size_t, 1> nodeIndices;

public:
    ThreeDLeafNode() = default;
    ThreeDLeafNode(const xt::xtensor<std::size_t, 1> &nodeIndices_) : nodeIndices(nodeIndices_){}

    std::size_t size() const {
        return nodeIndices.size();
    }

    xt::xtensor<std::size_t, 1> get_node_indices() const {
        return nodeIndices;
    }

    void fill_node_indices(xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                        std::size_t &startIndex) const {
        for (std::size_t i=0; i<nodeIndices.size(); i++){
            nodeIndicesBuffer( i + startIndex ) = nodeIndices(i);
        }
    }

    void find_nearest_k_node_indices( 
                const xt::xtensor<float, 1> &queryNode, 
                const NodesType &nodes,
                std::size_t &dim, 
                NearestNodePriorityQueue &nearestNodePriorityQueue) const {
        
        for (auto nodeIndex : nodeIndices){
            update_nearest_node_priority_queue(nearestNodePriorityQueue, nodes, nodeIndex, queryNode);
        }
    }
};
using ThreeDLeafNodePtr = std::shared_ptr<ThreeDLeafNode>;

class ThreeDInsideNode: public ThreeDNode{
public: 
    std::size_t medianNodeIndex;
    ThreeDNodePtr leftNodePtr;
    ThreeDNodePtr rightNodePtr;
    std::size_t nodeNum;

    ThreeDInsideNode() = default;

    ThreeDInsideNode(const std::size_t &medianNodeIndex_, 
            ThreeDNodePtr leftNodePtr_, ThreeDNodePtr rightNodePtr_, std::size_t nodeNum_):
            medianNodeIndex(medianNodeIndex_), leftNodePtr(leftNodePtr_), 
            rightNodePtr(rightNodePtr_), nodeNum(nodeNum_){};

    std::size_t size() const {
        // return leftNodePtr->size() + rightNodePtr->size() + 1;
        return nodeNum;
    }
     
    xt::xtensor<std::size_t, 1> get_node_indices() const {
        xt::xtensor<std::size_t, 1>::shape_type sh = {nodeNum};
        xt::xtensor<std::size_t, 1> nodeIndicesBuffer =  xt::empty<std::size_t>(sh);

        std::size_t filledNum = 0;
        fill_node_indices( nodeIndicesBuffer, filledNum );
        return nodeIndicesBuffer;
    }
    
    void fill_node_indices( xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                            std::size_t &startIndex) const {
        leftNodePtr->fill_node_indices(nodeIndicesBuffer, startIndex);
        startIndex += leftNodePtr->size();
        nodeIndicesBuffer(startIndex) = medianNodeIndex;
        startIndex += 1;
        rightNodePtr->fill_node_indices( nodeIndicesBuffer, startIndex );
        startIndex += rightNodePtr->size();
    }
   
    void find_nearest_k_node_indices( 
                const xt::xtensor<float, 1> &queryNode,
                const xt::xtensor<float, 2> &nodes, 
                std::size_t &dim, 
                NearestNodePriorityQueue &nearestNodePriorityQueue) const {

        auto nearestNodeNum = nearestNodePriorityQueue.size();
        // compare with the median value
        if (queryNode(dim) < nodes(medianNodeIndex, dim)){
            dim = next_dim(dim);
            // continue checking left nodes
            // closeNodePtr = leftNodePtr;
            leftNodePtr->find_nearest_k_node_indices(
                queryNode, nodes, dim, nearestNodePriorityQueue);
            if (nodes(medianNodeIndex, dim) - queryNode(dim) < 
                                            nearestNodePriorityQueue.top().first) {
                rightNodePtr->find_nearest_k_node_indices(
                    queryNode, nodes, dim, nearestNodePriorityQueue );
            }
        } else {
            rightNodePtr->find_nearest_k_node_indices(
                queryNode, nodes, dim, nearestNodePriorityQueue );
            if (queryNode(dim)-nodes(medianNodeIndex, dim) < 
                                    nearestNodePriorityQueue.top().first){
                leftNodePtr->find_nearest_k_node_indices(
                    queryNode, nodes, dim, nearestNodePriorityQueue);
            }
        }
    }
}; // ThreeDNode class

using ThreeDInsideNodePtr = std::shared_ptr<ThreeDInsideNode>;

class ThreeDTree{
private:
    ThreeDInsideNodePtr root;
    NodesType nodes;
    std::size_t leafSize;

    ThreeDInsideNodePtr build_node(const xt::xtensor<std::size_t, 1> &nodeIndices, std::size_t &dim) const {
        // find the median value index

        xt::xtensor<float, 1> coords = xt::index_view( 
                           xt::view(nodes, xt::all(), dim), 
                           nodeIndices);
        //xt::xtensor<float, 1>::shape_type sh = {nodeIndices.size()};
        //auto coords = xt::empty<float>(sh);
        //for (std::size_t i=0; i<coords.size(); i++){
        //    coords(i) = nodes( nodeIndices(i), dim );
        //}

        std::size_t splitIndex = nodeIndices.size() / 2;
        // although the partition can save some computation, but the order of equiverlant elements are not preserved!
        // const auto argSortIndices = xt::argpartition(coords, splitIndex);
        auto argSortIndices = xt::argsort( coords );
        xt::xtensor<std::size_t, 1> sortedNodeIndices = xt::index_view( nodeIndices, argSortIndices );
        auto middleNodeIndex = sortedNodeIndices( splitIndex );
        auto leftNodeIndices = xt::view( sortedNodeIndices, xt::range(0, splitIndex) );
        auto rightNodeIndices = xt::view( sortedNodeIndices, xt::range(splitIndex, _) );

        std::cout<< "\n\ndim: " << dim << std::endl; 
        // std::cout<< "nodes: " << nodes <<std::endl;
        // std::cout<< "coordinates in nodes: " << xt::view(nodes, xt::all(), dim) << std::endl;
        std::cout<< "node indices: " << nodeIndices << std::endl;
        std::cout<< "coordinates: " << coords << std::endl;
        std::cout<< "sorted node indices: " << sortedNodeIndices << std::endl;
        std::cout<< "median index: " << splitIndex << std::endl;
        std::cout<< "middle node index: " << middleNodeIndex << std::endl;
        std::cout << "left node indices: "<< leftNodeIndices << std::endl;
        std::cout << "right node indices: "<< rightNodeIndices << std::endl;

        ThreeDNodePtr leftNodePtr, rightNodePtr;
        // recursively loop the dimension in 3D
        dim = next_dim(dim);      

        auto leftNodeNum = leftNodeIndices.size();
        if (leftNodeNum > leafSize){
            leftNodePtr = build_node( leftNodeIndices, dim );
        } else {
            // include all the nodes as a leaf
            leftNodePtr = std::static_pointer_cast<ThreeDNode>(
                            std::make_shared<ThreeDLeafNode>( leftNodeIndices ));
        }

        auto rightNodeNum = rightNodeIndices.size();
        if (rightNodeNum > leafSize){
            rightNodePtr = build_node( rightNodeIndices, dim );
        } else {
            rightNodePtr = std::static_pointer_cast<ThreeDNode>( 
                    std::make_shared<ThreeDLeafNode>( rightNodeIndices ));
        }

        return std::make_shared<ThreeDInsideNode>(middleNodeIndex, 
                        leftNodePtr, rightNodePtr, coords.size());
    }

    auto build_root(){
        // start from first dimension
        std::size_t dim = 0;
        auto nodeIndices = xt::arange<std::size_t>(0, nodes.shape(0));
        std::cout<< "nodes: " << nodes << std::endl;
        root = build_node( nodeIndices, dim );
    }

public:
    ThreeDTree() = default;

    ThreeDTree(const NodesType &nodes_, const std::size_t leafSize_=10): 
            nodes(nodes_), leafSize(leafSize_){
        build_root();
    }

    ThreeDTree(const xt::pytensor<float, 2> &nodes_, const std::size_t leafSize_=10):
        nodes( NodesType(nodes_) ), leafSize(leafSize_){
        build_root();
    }
 
    auto get_leaf_size() const {
        return leafSize;
    }
    
    xt::xtensor<std::size_t, 1> find_nearest_k_node_indices(
                const xt::xtensor<float, 1> &queryNode, 
                const std::size_t &nearestNodeNum) const {
        assert( nearestNodeNum >= 1 );
        std::size_t dim = 0;
        auto queryNodeCoord = xt::view(queryNode, xt::range(0, 3));

        // build priority queue to store the nearest neighbors
        NearestNodePriorityQueue nearestNodePriorityQueue(cmp);
        for (auto nodeIndex : xt::arange<std::size_t>(0, nearestNodeNum)){
            auto node = xt::view(nodes, nodeIndex, xt::range(0,3));
            float squaredDist = xt::norm_sq(node - queryNodeCoord)(0);
            nearestNodePriorityQueue.push(std::make_pair(squaredDist, nodeIndex));
        }

        root->find_nearest_k_node_indices(queryNodeCoord, 
                                                 nodes, dim,nearestNodePriorityQueue);

        xt::xtensor<float,1>::shape_type sh = {nearestNodeNum};
        auto nearestNodeIndices = xt::empty<float>(sh);
        for(auto i : xt::arange<std::size_t>(0, nearestNodeNum)){
            nearestNodeIndices(i) = nearestNodePriorityQueue.top().second;
            nearestNodePriorityQueue.pop();
        }
        assert( nearestNodePriorityQueue.empty() );
        return nearestNodeIndices;
    }

    xt::xtensor<std::size_t, 1> find_nearest_k_node_indices(
            const xt::pytensor<float, 1> &queryNode, 
            const std::size_t &nearestNodeNum){
        return find_nearest_k_node_indices( xt::xtensor<float,1>(queryNode), nearestNodeNum );
    }

    auto find_nearest_k_nodes(const xt::xtensor<float, 1> &queryNode, 
                                        const std::size_t &nearestNodeNum) const {
        
        auto nearestNodeIndices = find_nearest_k_node_indices(queryNode, nearestNodeNum);
        xt::xtensor<float, 2>::shape_type shape = {nearestNodeNum, 4};

        auto nearestNodes = xt::empty<float>( shape ); 
        for(std::size_t i; i<nearestNodeNum; i++){
            auto nodeIndex = nearestNodeIndices(i);
            xt::view(nearestNodes, i, xt::all()) = xt::view(nodes, nodeIndex, xt::all());
        }
        return nearestNodes;
    } 
}; // ThreeDTree class

} // end of namespace
