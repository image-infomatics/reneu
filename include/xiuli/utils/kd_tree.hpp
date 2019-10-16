#pragma once
#include <limits>       // std::numeric_limits
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"

// use the c++17 nested namespace
namespace xiuli::utils{

using NodesType = xt::xtensor<float, 2>;

auto next_dim(std::size_t &dim) {
    if(dim==3)
        dim = 0;
    else
        dim += 1;
    return dim;
}

// ThreeDTree
class AbstractThreeDNode{
public:
    // we need to make base class polymorphic for dynamic cast
    virtual ~AbstractThreeDNode() = default;
    virtual std::size_t size() const;
    virtual xt::xtensor<std::size_t, 1> get_node_indices() const;
    virtual void fill_node_indices(xt::xtensor<std::size_t, 1> &nodeIndicesBuffer, 
                                    std::size_t &filledNum) const;
    virtual xt::xtensor<std::size_t, 1> find_nearest_k_nodes(
                const xt::xtensor<float, 1> &queryNode,
                const NodesType &nodes, 
                std::size_t &dim, const std::size_t &nearestNodeNum=1) const; 
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

    xt::xtensor<std::size_t, 1> find_nearest_k_nodes( 
                const xt::xtensor<float, 1> &queryNode, 
                const NodesType &nodes,
                std::size_t &dim, const std::size_t &nearestNodeNum=1) {
        // this is a leaf node, we have to compute the distance one by one
       
        if (nearestNodeNum == 1){
            //xt::xtensor< std::size_t, 1 > ret = xt::empty<std::size_t>( sh );
 
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
            //ret(0) = nearestNodeIndex;
            return {nearestNodeIndex};
        } else if (nearestNodeNum == nodeIndices.size()){
            return nodeIndices;
        } else {
            assert(nearestNodeNum < nodeIndices.size());
            xt::xtensor<std::size_t, 1>::shape_type sh = {nearestNodeNum};
            xt::xtensor<float, 1> dist2s = xt::empty<float>(sh); 
            for (std::size_t i=0; i<nodeIndices.size(); i++){
                auto nodeIndex = nodeIndices( i );
                auto node = xt::view(nodes, nodeIndex, xt::range(0, 3));
                dist2s(i) = xt::norm_sq( node - queryNode )(0);
            }
            auto indices = xt::argpartition( dist2s, nearestNodeNum-1 );
            auto selectedIndices = xt::view(indices, xt::range(0, nearestNodeNum));
            return xt::index_view(nodeIndices, selectedIndices );
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
     
    xt::xtensor<std::size_t, 1> get_node_indices(){
        xt::xtensor<std::size_t, 1>::shape_type sh = {nodeNum};
        xt::xtensor<std::size_t, 1> nodeIndicesBuffer =  xt::empty<std::size_t>(sh);

        std::size_t filledNum = 0;
        fill_node_indices( nodeIndicesBuffer, filledNum );
        return nodeIndicesBuffer;
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
   
    xt::xtensor<std::size_t, 1> find_nearest_k_nodes( 
                const xt::xtensor<float, 1> &queryNode,
                const xt::xtensor<float, 2> &nodes, 
                std::size_t &dim, const std::size_t &nearestNodeNum=1) {
        ThreeDNodePtr closeNodePtr, farNodePtr;
        
        // compare with the median value
        if (queryNode(dim) < nodes(middleNodeIndex, dim)){
            // continue checking left nodes
            closeNodePtr = leftNodePtr;
            farNodePtr = rightNodePtr;
        } else {
            closeNodePtr = rightNodePtr;
            farNodePtr = leftNodePtr;
        }

        if (closeNodePtr->size() >= nearestNodeNum){
            dim = next_dim(dim);
            return closeNodePtr->find_nearest_k_nodes(queryNode, nodes, dim);
        } else {
            // closeNodePtr->size() < nearestNodeNum
            // we need to add some close nodes from far node
            xt::xtensor<std::size_t, 1>::shape_type shape = {nearestNodeNum};
            xt::xtensor<std::size_t, 1> nearestNodeIndices = xt::empty<std::size_t>(shape);
            auto closeNodeIndices = closeNodePtr->get_node_indices();
            xt::view(nearestNodeIndices, xt::range(0, closeNodeIndices.size())) = closeNodeIndices;
            //for (std::size_t i = 0; i<closeNodeIndices.size(); i++){
                //nearestNodeIndices(i) = closeNodeIndices(i);
            //}
            auto remainingNodeNum = nearestNodeNum - closeNodeIndices.size();
            auto remainingNodeIndices = farNodePtr->find_nearest_k_nodes(queryNode, nodes, dim, remainingNodeNum);
            xt::view(nearestNodeIndices, xt::range(closeNodeIndices.size(), _)) = remainingNodeIndices;
            return nearestNodeIndices;
        }
    }
}; // ThreeDNode class

using ThreeDInsideNodePtr = std::shared_ptr<ThreeDInsideNode>;

class ThreeDTree{
private:
    ThreeDNodePtr root;
    const NodesType nodes;
    const std::size_t leafSize;
    
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
    
    xt::xtensor<std::size_t, 1> find_nearest_k_node_indices(
                const xt::xtensor<float, 1> &queryNode, 
                const std::size_t &nearestNodeNum=1) const {
        assert( nearestNodeNum >= 1 );
        std::size_t dim = 0;
        auto queryNodeCoord = xt::view(queryNode, xt::range(0, 3));
        return root->find_nearest_k_nodes(queryNodeCoord, nodes, dim, nearestNodeNum);
    }

    auto find_nearest_k_nodes(const xt::xtensor<float, 1> &queryNode, 
                                        const std::size_t &nearestNodeNum=20) const {
        
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