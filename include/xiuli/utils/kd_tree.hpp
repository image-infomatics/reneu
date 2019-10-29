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

namespace xiuli::utils{

using namespace xt::placeholders;

using Nodes = xt::xtensor<float, 2>;
using Node = xt::xtensor_fixed<float, xt::xshape<3>>;
using NodeIndices = xt::xtensor<std::size_t, 1>;

/*
 * this is a fake heap/priority queue, designed specifically for this use case.
 */
class IndexHeap{
private:
    NodeIndices nodeIndices;
    xt::xtensor<float, 1> squaredDistances;
    std::size_t maxDistIndex;

public:
    IndexHeap(const std::size_t K){
        NodeIndices::shape_type sh = {K}; 
        nodeIndices = xt::empty<std::size_t>( sh );
        nodeIndices.fill( std::numeric_limits<std::size_t>::max() );
        squaredDistances = xt::empty<float>({K});
        squaredDistances.fill( std::numeric_limits<float>::max() );
        maxDistIndex = 0;
    }

    inline auto get_node_indices() const {
        return nodeIndices;
    }

    inline auto size() const {
        return nodeIndices.size();
    }

    inline auto max_squared_dist() const {
        return squaredDistances( maxDistIndex );
    }

    void update( const std::size_t &nodeIndex, const Nodes &nodes, const Node &queryNode ){
        auto node = xt::view(nodes, nodeIndex, xt::range(0, 3));
        auto squaredDist = xt::norm_sq( node - queryNode )();
        if (squaredDist < max_squared_dist()){
            // replace the largest distance with current one
            nodeIndices( maxDistIndex ) = nodeIndex;
            squaredDistances( maxDistIndex ) = squaredDist;
            // update the max distance index
            maxDistIndex = xt::argmax( squaredDistances )();
        }
    }

    inline void update(const NodeIndices &nodeIndices, const Nodes &nodes, const Node &queryNode){
        for (auto nodeIndex : nodeIndices){
            update(nodeIndex, nodes, queryNode);
        }
    }
}; // end of class IndexHeap

// ThreeDTree
class ThreeDNode{
public:
    ThreeDNode() = default;
    // we need to make base class polymorphic for dynamic cast
    virtual std::size_t size() const = 0;
    
    virtual void find_nearest_k_node_indices(
                const Node &queryNode,
                const Nodes &nodes, 
                IndexHeap &indexHeap) const = 0; 
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
                IndexHeap &indexHeap) const {
        indexHeap.update(nodeIndices, nodes, queryNode);
    }
};
using ThreeDLeafNodePtr = std::shared_ptr<ThreeDLeafNode>;

class ThreeDInsideNode: public ThreeDNode{
private: 
    const std::size_t dim;
    const float splitValue;
    const std::size_t nodeNum;
    const BoundingBox bbox; 
    ThreeDNodePtr leftNodePtr;
    ThreeDNodePtr rightNodePtr;

public:
    ThreeDInsideNode() = default;

    ThreeDInsideNode(const std::size_t dim_, 
            const float splitValue_,
            const std::size_t nodeNum_, const BoundingBox &bbox_, 
            ThreeDNodePtr leftNodePtr_, ThreeDNodePtr rightNodePtr_): 
            dim(dim_), splitValue(splitValue_),
            nodeNum(nodeNum_), bbox(bbox_),
            leftNodePtr(leftNodePtr_), rightNodePtr(rightNodePtr_){}

    std::size_t size() const {
        return nodeNum;
    }

    auto get_bounding_box() const{
        return bbox;
    }
     
    void find_nearest_k_node_indices( 
                const Node &queryNode,
                const Nodes &nodes, 
                IndexHeap &indexHeap) const {
        // check the bounding box first
        if (indexHeap.max_squared_dist() <= bbox.min_squared_distance_from(queryNode)){
            return;
        }

        // compare with the median value
        if (queryNode(dim) < splitValue){
            // continue checking left nodes
            // closeNodePtr = leftNodePtr;
            leftNodePtr->find_nearest_k_node_indices( queryNode, nodes, indexHeap );
            if (splitValue - queryNode(dim) < indexHeap.max_squared_dist()) {
                rightNodePtr->find_nearest_k_node_indices(queryNode, nodes, indexHeap );
            }
        } else {
            // right one is closer
            rightNodePtr->find_nearest_k_node_indices( queryNode, nodes, indexHeap );
            if ((queryNode(dim)-splitValue) < indexHeap.max_squared_dist()){
                leftNodePtr->find_nearest_k_node_indices(queryNode, nodes, indexHeap);
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
    
    ThreeDInsideNodePtr build_node(const NodeIndices &nodeIndices) const {
        const BoundingBox bbox( nodes, nodeIndices );
        const std::size_t dim = bbox.get_largest_extent_dimension();      
        auto nodeNum = nodeIndices.size();

        // find the median value index
        xt::xtensor<float, 1> coords = xt::index_view( 
                           xt::view(nodes, xt::all(), dim), 
                           nodeIndices);
        const std::size_t splitIndex = nodeNum / 2;
        // partition can save some computation than full sort
        const auto argSortIndices = xt::argpartition(coords, splitIndex);
        // auto argSortIndices = xt::argsort( coords );
        const NodeIndices sortedNodeIndices = xt::index_view( nodeIndices, argSortIndices );
        const auto middleNodeIndex = sortedNodeIndices( splitIndex );
        const float splitValue = nodes( middleNodeIndex, dim );
        const auto leftNodeIndices = xt::view( sortedNodeIndices, xt::range(0, splitIndex) );
        const auto rightNodeIndices = xt::view( sortedNodeIndices, xt::range(splitIndex, _) );

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

        auto leftNodeNum = leftNodeIndices.size();
        if (leftNodeNum > leafSize){
            leftNodePtr = build_node( leftNodeIndices );
        } else {
            BoundingBox leftBBox( nodes, leftNodeIndices );
            // include all the nodes as a leaf
            leftNodePtr = std::static_pointer_cast<ThreeDNode>(
                            std::make_shared<ThreeDLeafNode>( leftNodeIndices, leftBBox ));
        }

        auto rightNodeNum = rightNodeIndices.size();
        if (rightNodeNum > leafSize){
            rightNodePtr = build_node( rightNodeIndices );
        } else {
            BoundingBox rightBBox( nodes, rightNodeIndices );
            rightNodePtr = std::static_pointer_cast<ThreeDNode>( 
                    std::make_shared<ThreeDLeafNode>( rightNodeIndices, rightBBox ));
        }

        return std::make_shared<ThreeDInsideNode>(dim, splitValue, nodeNum, bbox, 
                                                        leftNodePtr, rightNodePtr);
    }

    auto build_root(){
        // start from first dimension
        std::size_t dim = 0;
        auto nodeIndices = xt::arange<std::size_t>(0, nodes.shape(0));
        // std::cout<< "nodes: " << nodes << std::endl;
        root = build_node( nodeIndices );
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

    NodeIndices find_nearest_k_node_indices( const Node &queryNode, const std::size_t K ) const {
        auto queryNodeCoord = xt::view(queryNode, xt::range(0, 3));

        // build priority queue to store the nearest neighbors
        IndexHeap indexHeap(K);
        root->find_nearest_k_node_indices(queryNodeCoord, 
                                            nodes, indexHeap);
        return indexHeap.get_node_indices();
    }

    NodeIndices find_nearest_k_node_indices( const xt::pytensor<float, 1> &queryNode, 
                                                const std::size_t &K){
        return find_nearest_k_node_indices( Node(queryNode), K );
    }

    auto find_nearest_k_nodes( const Node &queryNode, const std::size_t K ) const {
        
        auto nearestNodeIndices = find_nearest_k_node_indices(queryNode, K);
        Nodes::shape_type shape = {K, 4};
        auto nearestNodes = xt::empty<float>( shape );

        for(std::size_t i; i<K; i++){
            auto nodeIndex = nearestNodeIndices(i);
            xt::view(nearestNodes, i, xt::all()) = xt::view(
                                            nodes, nodeIndex, xt::all());
        }
        return nearestNodes;
    } 
}; // ThreeDTree class

} // end of namespace
