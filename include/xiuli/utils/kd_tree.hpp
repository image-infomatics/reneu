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
#include "xiuli/utils/bounding_box.hpp"

// use the c++17 nested namespace
namespace xiuli::utils{

using namespace xt::placeholders;

using Nodes = xt::xtensor<float, 2>;
using Node = xt::xtensor_fixed<float, xt::xshape<3>>;
using NodeIndices = xt::xtensor<std::size_t, 1>;

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
            const Nodes &nodes,
            const std::size_t &nodeIndex,
            const Node &queryNode){
    auto node = xt::view(nodes, nodeIndex, xt::range(0, 3));

    float squaredDist = xt::norm_sq(node - queryNode)(0);
    if ( squaredDist < nearestNodePriorityQueue.top().first ){
        // std::cout<< "replacing (" << nearestNodePriorityQueue.top().first << ", " << 
                                    // nearestNodePriorityQueue.top().second << ") with ("<< squaredDist << ", " << nodeIndex << ")" << std::endl;
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
    
    virtual void find_nearest_k_node_indices(
                const Node &queryNode,
                const Nodes &nodes, 
                NearestNodePriorityQueue &nearestNodePriorityQueue) const = 0; 
};
using ThreeDNodePtr = std::shared_ptr<ThreeDNode>;


class ThreeDLeafNode: public ThreeDNode{

private:
    const NodeIndices nodeIndices;
    const BoundingBox bbox;

public:
    ThreeDLeafNode() = default;
    ThreeDLeafNode(const NodeIndices &nodeIndices_,
                    const BoundingBox &bbox_) : nodeIndices(nodeIndices_), bbox(bbox_){}

    std::size_t size() const {
        return nodeIndices.size();
    }

    void find_nearest_k_node_indices( 
                const Node &queryNode, 
                const Nodes &nodes,
                NearestNodePriorityQueue &nearestNodePriorityQueue) const {
        
        for (auto nodeIndex : nodeIndices){
            update_nearest_node_priority_queue(
                    nearestNodePriorityQueue, nodes, nodeIndex, queryNode);
        }
    }
};
using ThreeDLeafNodePtr = std::shared_ptr<ThreeDLeafNode>;

class ThreeDInsideNode: public ThreeDNode{
private: 
    const std::size_t medianNodeIndex;
    ThreeDNodePtr leftNodePtr;
    ThreeDNodePtr rightNodePtr;
    const std::size_t nodeNum;
    const std::size_t dim;
    const BoundingBox bbox; 

public:
    ThreeDInsideNode() = default;

    ThreeDInsideNode(const std::size_t medianNodeIndex_, 
            ThreeDNodePtr leftNodePtr_, ThreeDNodePtr rightNodePtr_, 
            const std::size_t nodeNum_, const std::size_t dim_, const BoundingBox &bbox_):
            medianNodeIndex(medianNodeIndex_), leftNodePtr(leftNodePtr_), 
            rightNodePtr(rightNodePtr_), nodeNum(nodeNum_), dim(dim_), bbox(bbox_){};

    std::size_t size() const {
        // return leftNodePtr->size() + rightNodePtr->size() + 1;
        return nodeNum;
    }

    auto get_bounding_box() const{
        return bbox;
    }
     
    void find_nearest_k_node_indices( 
                const Node &queryNode,
                const Nodes &nodes, 
                NearestNodePriorityQueue &nearestNodePriorityQueue) const {
        // check the bounding box first
        if (nearestNodePriorityQueue.top().first <= bbox.min_squared_distance_from(queryNode)){
            return;
        }

        auto nearestNodeNum = nearestNodePriorityQueue.size();
        
        // compare with the median value
        if (queryNode(dim) < nodes(medianNodeIndex, dim)){
            // continue checking left nodes
            // closeNodePtr = leftNodePtr;
            leftNodePtr->find_nearest_k_node_indices(
                queryNode, nodes, nearestNodePriorityQueue);
            update_nearest_node_priority_queue(nearestNodePriorityQueue, 
                                                nodes, medianNodeIndex, queryNode);
            if (nodes(medianNodeIndex, dim) - queryNode(dim) < 
                                            nearestNodePriorityQueue.top().first) {
                rightNodePtr->find_nearest_k_node_indices(
                    queryNode, nodes, nearestNodePriorityQueue );
            }
        } else {
            // right one is closer
            rightNodePtr->find_nearest_k_node_indices(
                queryNode, nodes, nearestNodePriorityQueue );
            update_nearest_node_priority_queue(nearestNodePriorityQueue, 
                                                nodes, medianNodeIndex, queryNode);
            if (queryNode(dim)-nodes(medianNodeIndex, dim) < 
                                    nearestNodePriorityQueue.top().first){
                leftNodePtr->find_nearest_k_node_indices(
                    queryNode, nodes, nearestNodePriorityQueue);
            }
        }
    }
}; // ThreeDNode class

using ThreeDInsideNodePtr = std::shared_ptr<ThreeDInsideNode>;

class ThreeDTree{
private:
    ThreeDInsideNodePtr root;
    Nodes nodes;
    std::size_t leafSize;
    
    ThreeDInsideNodePtr build_node(const NodeIndices &nodeIndices, std::size_t dim) const {
        // find the median value index

        xt::xtensor<float, 1> coords = xt::index_view( 
                           xt::view(nodes, xt::all(), dim), 
                           nodeIndices);

        const std::size_t splitIndex = nodeIndices.size() / 2;
        // partition can save some computation than full sort
        const auto argSortIndices = xt::argpartition(coords, splitIndex);
        // auto argSortIndices = xt::argsort( coords );
        const NodeIndices sortedNodeIndices = xt::index_view( nodeIndices, argSortIndices );
        const auto middleNodeIndex = sortedNodeIndices( splitIndex );
        const auto leftNodeIndices = xt::view( sortedNodeIndices, xt::range(0, splitIndex) );
        const auto rightNodeIndices = xt::view( sortedNodeIndices, xt::range(splitIndex+1, _) );

        const BoundingBox bbox( nodes, nodeIndices );
        // std::cout<< "\n\ndim: " << dim << std::endl; 
        // std::cout<< "nodes: " << nodes <<std::endl;
        // std::cout<< "coordinates in nodes: " << xt::view(nodes, xt::all(), dim) << std::endl;
        // std::cout<< "node indices: " << nodeIndices << std::endl;
        // std::cout<< "coordinates: " << coords << std::endl;
        // std::cout<< "sorted node indices: " << sortedNodeIndices << std::endl;
        // std::cout<< "split index: " << splitIndex << std::endl;
        // std::cout<< "middle node index: " << middleNodeIndex << std::endl;
        // std::cout << "left node indices: "<< leftNodeIndices << std::endl;
        // std::cout << "right node indices: "<< rightNodeIndices << std::endl;

        ThreeDNodePtr leftNodePtr, rightNodePtr;
        // recursively loop the dimension in 3D
        dim = next_dim(dim);      

        auto leftNodeNum = leftNodeIndices.size();
        if (leftNodeNum > leafSize){
            leftNodePtr = build_node( leftNodeIndices, dim );
        } else {
            BoundingBox leftBBox( nodes, leftNodeIndices );
            // include all the nodes as a leaf
            leftNodePtr = std::static_pointer_cast<ThreeDNode>(
                            std::make_shared<ThreeDLeafNode>( leftNodeIndices, leftBBox ));
        }

        auto rightNodeNum = rightNodeIndices.size();
        if (rightNodeNum > leafSize){
            rightNodePtr = build_node( rightNodeIndices, dim );
        } else {
            BoundingBox rightBBox( nodes, rightNodeIndices );
            rightNodePtr = std::static_pointer_cast<ThreeDNode>( 
                    std::make_shared<ThreeDLeafNode>( rightNodeIndices, rightBBox ));
        }

        return std::make_shared<ThreeDInsideNode>(middleNodeIndex, 
                        leftNodePtr, rightNodePtr, coords.size(), dim, bbox);
    }

    auto build_root(){
        // start from first dimension
        std::size_t dim = 0;
        auto nodeIndices = xt::arange<std::size_t>(0, nodes.shape(0));
        // std::cout<< "nodes: " << nodes << std::endl;
        root = build_node( nodeIndices, dim );
    }

public:
    ThreeDTree() = default;

    ThreeDTree(const Nodes &nodes_, const std::size_t leafSize_=10): 
            nodes(nodes_), leafSize(leafSize_){
        build_root();
    }

    ThreeDTree(const xt::pytensor<float, 2> &nodes_, const std::size_t leafSize_=10):
        nodes( Nodes(nodes_) ), leafSize(leafSize_){
        build_root();
    }
 
    auto get_leaf_size() const {
        return leafSize;
    }
    
    NodeIndices find_nearest_k_node_indices(
                const Node &queryNode, 
                const std::size_t &nearestNodeNum) const {
        assert( nearestNodeNum >= 1 );
        auto queryNodeCoord = xt::view(queryNode, xt::range(0, 3));

        // build priority queue to store the nearest neighbors
        NearestNodePriorityQueue nearestNodePriorityQueue(cmp);
        for (auto i : xt::arange<std::size_t>(0, nearestNodeNum)){
            float squaredDist = std::numeric_limits<float>::max();
            std::size_t nodeIndex = std::numeric_limits<std::size_t>::max();
            nearestNodePriorityQueue.push(std::make_pair(squaredDist, nodeIndex));
        }

        root->find_nearest_k_node_indices(queryNodeCoord, 
                                            nodes, nearestNodePriorityQueue);

        NodeIndices::shape_type sh = {nearestNodeNum};
        auto nearestNodeIndices = xt::empty<std::size_t>(sh);
        for(auto i : xt::arange<std::size_t>(0, nearestNodeNum)){
            nearestNodeIndices(i) = nearestNodePriorityQueue.top().second;
            nearestNodePriorityQueue.pop();
        }
        assert( nearestNodePriorityQueue.empty() );
        return nearestNodeIndices;
    }

    NodeIndices find_nearest_k_node_indices(
            const xt::pytensor<float, 1> &queryNode, 
            const std::size_t &nearestNodeNum){
        return find_nearest_k_node_indices( 
                Node(queryNode), nearestNodeNum );
    }

    auto find_nearest_k_nodes(const Node &queryNode, 
                                const std::size_t &nearestNodeNum) const {
        
        auto nearestNodeIndices = find_nearest_k_node_indices(
                                            queryNode, nearestNodeNum);
        Nodes::shape_type shape = {nearestNodeNum, 4};
        auto nearestNodes = xt::empty<float>( shape );

        for(std::size_t i; i<nearestNodeNum; i++){
            auto nodeIndex = nearestNodeIndices(i);
            xt::view(nearestNodes, i, xt::all()) = xt::view(
                                            nodes, nodeIndex, xt::all());
        }
        return nearestNodes;
    } 
}; // ThreeDTree class

} // end of namespace
