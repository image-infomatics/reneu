#pragma once

#include <limits>       // std::numeric_limits
#include <iostream>
#include <queue>
#include <variant>
#include "xiuli/type_aliase.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"

#include "xiuli/type_aliase.hpp"
#include "xiuli/utils/bounding_box.hpp"

namespace xiuli{

using namespace xt::placeholders;

using HeapElement = std::pair<float, Index>;
auto cmp = [](HeapElement left, HeapElement right) { return left.first < right.first; };

/*
 * this is a fake heap/priority queue, designed specifically for this use case.
 * the speed is similar with std::priority_queue implementation.
 * keep customized version because it makes code more clean, and could find out some speed up method later.
 */
class IndexHeap{
    // build priority queue to store the nearest neighbors
private:
    std::priority_queue<HeapElement, std::vector<HeapElement>, decltype(cmp)> pqueue;

public:
    IndexHeap(const Index K): pqueue(cmp){
        for (Index i=0; i<K; i++){
            float squaredDist = std::numeric_limits<float>::max();
            std::size_t nodeIndex = std::numeric_limits<std::size_t>::max();
            pqueue.push(std::make_pair(squaredDist, nodeIndex));
        }
    }
    
    inline auto size() const {
        return pqueue.size();
    }

    inline auto max_squared_dist() const {
        return pqueue.top().first;
    }

    auto get_point_indices() {
        auto K = pqueue.size();
        PointIndices::shape_type sh = {K};
        auto pointIndices = xt::empty<Index>(sh);
        for(Index i=0; i<K; i++ ){
            pointIndices(i) = pqueue.top().second;
            pqueue.pop();
        }
        assert( pqueue.empty() );
        return pointIndices;
    }

    void update( const Point &point, const Index &pointIndex, const Point &queryPoint ){
        auto squaredDist = xt::norm_sq( point - queryPoint )();
        if (squaredDist < max_squared_dist()){
            // replace the largest distance with current one
            pqueue.pop();
            pqueue.push( std::make_pair(squaredDist, pointIndex) );
        }
    }
}; // end of class IndexHeap



// the maximum dim is 2, so two bits is enough to encode the dimension
const std::uint32_t DIM_BIT_START = 30;

/*
* This implementation is inspired by libnabo:
* https://github.com/ethz-asl/libnabo
* we use one Point type to represent both inside node and leaf node.
* this design will avoid class inheritance and virtual functions, smart pointers...
* this is more efficient according to libnabo paper:
* Elseberg, Jan, et al. "Comparison of nearest-neighbor-search strategies and implementations for efficient shape registration." Journal of Software Engineering for Robotics 3.1 (2012): 2-12.
*/
class KDTreeNode{
private:
    // make data member public to fit the criteria of POD (Plain Old Data)
    // this variable is encoded. The first most-significant bits 
    // encodes the dimension. if the dimension is 0-2, it is a split node.
    // if the dimension is 3 (11 in binary code), this is a leaf node.
    // if this is a split node inside tree, the 30 least-significant bits 
    // encode the right child node index 
    // if this is a leaf node, the 30 least-significant bits encode 
    // the number of points in bucket.
    Index dim_child_bucketSize;
    // the starting index of points in bucket
    // the cut/split value of this node
    std::variant<Index, float> bucketIndex_cutValue;
    BoundingBox bbox;

public:
    // construct a split node
    KDTreeNode(const Index dim, const float cutValue_, const BoundingBox bbox_):
                        bucketIndex_cutValue(cutValue_), bbox(bbox_){
        // since dim is unsigned type, it is always >=0
        assert(dim<3);
        dim_child_bucketSize = dim << DIM_BIT_START;
        // we'll have to write right child node index later.
    }

    // construct a leaf node
    KDTreeNode(const Index bucketSize, const Index bucketIndex_, const BoundingBox bbox_):
                                    bucketIndex_cutValue(bucketIndex_), bbox(bbox_){
        dim_child_bucketSize = (3<<DIM_BIT_START) + bucketSize;
    }

    inline auto get_bounding_box() const {
        return bbox;
    }    

    inline auto get_bucket_index() const {
        return std::get<Index>(bucketIndex_cutValue);
    }

    inline auto get_cut_value() const {
        return std::get<float>(bucketIndex_cutValue);
    }

    inline auto get_dim() const {
        return dim_child_bucketSize >> DIM_BIT_START;
    }

    inline bool is_leaf() const {
        // whether the first two bits are 11
        return dim_child_bucketSize >= 0xC0000000;
    }

    inline auto get_right_child_node_index() const {
        // black out the two most-significant bits
        return dim_child_bucketSize & 0x3FFFFFFF;
    }

    inline auto get_bucket_size() const {
        // black out the two most-significant bits
        return dim_child_bucketSize & 0x3FFFFFFF;
    }

    inline auto write_right_child_node_index( const Index &rightChildNodeIndex ){
        dim_child_bucketSize += rightChildNodeIndex; 
    }
    
    void print() const {
        if (is_leaf()){
            std::cout<< "\nleaf node:" << std::endl;
            std::cout<< "bucket size: " << get_bucket_size() << std::endl;
            std::cout<< "bucket index: "<< get_bucket_index() << std::endl;
        } else {
            std::cout<< "\nsplit node:" << std::endl;
            std::cout<< "dim: " << get_dim() << std::endl;
            std::cout<< "right child node index: " << get_right_child_node_index() << std::endl;
            std::cout<< "cut value: " << get_cut_value() << std::endl;
        }
    }

}; // end of class KDTreeNode


class KDTree{

private:
    std::vector<KDTreeNode> kdTreeNodes;
    std::vector<Index> pointIndicesBucket;
    const Index leafSize;
    const Points points;

    Index build_kd_nodes( const PointIndices &pointIndices ){
        BoundingBox bbox( points, pointIndices );
        if (pointIndices.size() <= leafSize ){
            // build a leaf node
            Index bucketSize = pointIndices.size();
            Index bucketIndex = pointIndicesBucket.size();
            KDTreeNode node( bucketSize, bucketIndex, bbox );
            for( Index pointIndex : pointIndices ){
                pointIndicesBucket.push_back( pointIndex );
            }

            // std::cout<< "\ninserting leaf node to " << kdTreeNodes.size() << std::endl;
            // node.print();
            // std::cout<<"\nexisting nodes in kdTreeNodes: "<<std::endl;
            // for(auto n : kdTreeNodes){
                // n.print();
            // }

            kdTreeNodes.push_back( node );
            return kdTreeNodes.size() - 1;
        } else {
            // build a split node
            auto dim = bbox.get_largest_extent_dimension();
            auto pointNum = pointIndices.size();
            
            // find the median value index
            xt::xtensor<float, 1> coords = xt::index_view( 
                            xt::view(points, xt::all(), dim), 
                            pointIndices);
            const Index splitIndex = pointNum / 2;
            // partition can save some computation than full sort
            const auto argSortIndices = xt::argpartition(coords, splitIndex);
            // auto argSortIndices = xt::argsort( coords );
            const PointIndices sortedPointIndices = xt::index_view( 
                                                pointIndices, argSortIndices );
            const auto middlePointIndex = sortedPointIndices( splitIndex );
            const float cutValue = points( middlePointIndex, dim );
             
            // std::cout<< "dim: " << dim << std::endl;
            // std::cout<< "point number: " << pointNum << std::endl;
            // std::cout<< "middle point index: " << middlePointIndex << std::endl;

           
            KDTreeNode node(dim, cutValue, bbox);
            // node.print();
            Index nodeIndex = kdTreeNodes.size();
            kdTreeNodes.push_back(node);

            const auto leftPointIndices = xt::view( sortedPointIndices, 
                                                        xt::range(0, splitIndex) );
            const auto leftNodeIndex = build_kd_nodes(leftPointIndices);
            assert( leftNodeIndex == nodeIndex + 1 );

            const auto rightPointIndices = xt::view( sortedPointIndices, 
                                                        xt::range(splitIndex, _) );
            const auto rightNodeIndex = build_kd_nodes(rightPointIndices);

            kdTreeNodes[nodeIndex].write_right_child_node_index( rightNodeIndex );
            return nodeIndex;
        }
    }

    void knn_update_heap( const Point &queryPoint, IndexHeap &indexHeap, 
                                        const Index &nodeIndex) const {
        KDTreeNode node = kdTreeNodes[nodeIndex];
        
        // check the bounding box first
        BoundingBox bbox = node.get_bounding_box();
        if (indexHeap.max_squared_dist() < bbox.min_squared_distance_from( queryPoint )){
            return;
        }

        if (node.is_leaf()){
            for (Index i=node.get_bucket_index(); 
                            i<node.get_bucket_index() + node.get_bucket_size(); i++){
                auto pointIndex = pointIndicesBucket[ i ];
                auto point = xt::view( points, pointIndex, xt::range(0, 3) );
                indexHeap.update( point, pointIndex, queryPoint);
            }
        } else {
            // this is a split node
            auto cutValue = node.get_cut_value();
            auto dim = node.get_dim();
            const Index leftChildNodeIndex = nodeIndex + 1;
            const Index rightChildNodeIndex = node.get_right_child_node_index();
            if (queryPoint(dim) < cutValue){
                // left child node is closer
                knn_update_heap(queryPoint, indexHeap, leftChildNodeIndex);
                knn_update_heap(queryPoint, indexHeap, rightChildNodeIndex);
            } else {
                // right child node is closer
                knn_update_heap(queryPoint, indexHeap, rightChildNodeIndex);
                knn_update_heap(queryPoint, indexHeap, leftChildNodeIndex);
            }
        }
    }

public:
    KDTree( const Points &points_, std::size_t leafSize_ ): 
                points(points_), leafSize(leafSize_), 
                kdTreeNodes({}), pointIndicesBucket({}){
        auto pointNum = points.shape(0);
        pointIndicesBucket.reserve( pointNum );
        // std::cout<< "\nreserve node number: " << pointNum/leafSize*2 <<std::endl;
        kdTreeNodes.reserve( pointNum / leafSize * 2 );
        
        PointIndices pointIndices = xt::arange<Index>(0, pointNum);
        build_kd_nodes(pointIndices);
        assert(pointIndicesBucket.size() == pointNum);
        kdTreeNodes.shrink_to_fit();
    }

    /*
     * find the nearest k neighbors
     */
    inline auto knn(const Point &queryPoint, const Index &K) const {
        // build priority queue to store the nearest neighbors
        IndexHeap indexHeap(K);
        
        // the first one is the root node
        knn_update_heap( queryPoint, indexHeap, 0 );
        
        return indexHeap.get_point_indices();
    }

    inline auto py_knn(const PyPoint &queryPoint, const int &K) const {
        return knn(queryPoint, K);
    }
}; //end of KDTree class

} // end of namespace
