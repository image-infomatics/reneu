#pragma once

#include <limits>       // std::numeric_limits
#include <iostream>
#include <queue>
#include <variant>
#include "reneu/types.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"

#include "reneu/types.hpp"
#include "reneu/utils/bounding_box.hpp"

namespace reneu{

// namespace py=pybind11;
using namespace xt::placeholders;

using HeapElement = std::pair<float, Index>;
auto cmp = [](HeapElement left, HeapElement right) { return left.first < right.first; };

/*
 * this is a fake heap/priority queue, designed specifically for this use case.
 * the speed is similar with std::priority_queue implementation.
 * keep customized version because it makes code more clean, and could find out some speed up method later.
 */
class IndexHeap{
private:
    // build priority queue to store the nearest neighbors
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

    void update(const float &squaredDist, const Index &pointIndex){
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
    std::variant<Index, float> bucketStart_cutValue;
    BoundingBox bbox;

public:
    // construct a split node
    KDTreeNode(const Index dim, const float cutValue_, const BoundingBox bbox_):
                        bucketStart_cutValue(cutValue_), bbox(bbox_){
        // since dim is unsigned type, it is always >=0
        assert(dim<3);
        dim_child_bucketSize = dim << DIM_BIT_START;
        // we'll have to write right child node index later.
    }

    // construct a leaf node
    KDTreeNode(const Index bucketStart_, const Index bucketSize, const BoundingBox bbox_):
                                    bucketStart_cutValue(bucketStart_), bbox(bbox_){
        dim_child_bucketSize = (3<<DIM_BIT_START) + bucketSize;
    }

    inline auto get_bounding_box() const {
        return bbox;
    }    

    inline auto get_bucket_start() const {
        return std::get<Index>(bucketStart_cutValue);
    }
    
    inline auto get_bucket_size() const {
        // black out the two most-significant bits
        return dim_child_bucketSize & 0x3FFFFFFF;
    }

    inline auto get_bucket_stop() const {
        return get_bucket_start() + get_bucket_size();
    }

    inline auto get_cut_value() const {
        return std::get<float>(bucketStart_cutValue);
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

    inline auto write_right_child_node_index( const Index &rightChildNodeIndex ){
        dim_child_bucketSize += rightChildNodeIndex; 
    }
    
    void print() const {
        if (is_leaf()){
            std::cout<< "\nleaf node:" << std::endl;
            std::cout<< "bucket size: " << get_bucket_size() << std::endl;
            std::cout<< "bucket index: "<< get_bucket_start() << std::endl;
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
    using KDTreeNodes = std::vector<KDTreeNode>;
    using PointIndicesBucket = std::vector<Index>;

    KDTreeNodes kdTreeNodes;
    PointIndicesBucket pointIndicesBucket;
    Points pointsBucket;
    const Index leafSize;

    Index build_kd_nodes( const Points &points, const PointIndices &pointIndices ){
        BoundingBox bbox( points, pointIndices );
        if (pointIndices.size() <= leafSize ){
            // build a leaf node
            Index bucketSize = pointIndices.size();
            Index bucketStart = pointIndicesBucket.size();
            KDTreeNode node( bucketStart, bucketSize, bbox );

            Index bucketIndex, pointIndex;
            for (Index i=0; i<bucketSize; i++){
                pointIndex = pointIndices(i);
                pointIndicesBucket.push_back( pointIndices(i) );

                bucketIndex = bucketStart + i;
                pointsBucket( bucketIndex, 0 ) = points(pointIndex, 0);
                pointsBucket( bucketIndex, 1 ) = points(pointIndex, 1);
                pointsBucket( bucketIndex, 2 ) = points(pointIndex, 2);
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
            const auto leftNodeIndex = build_kd_nodes(points, leftPointIndices);
            assert( leftNodeIndex == nodeIndex + 1 );

            const auto rightPointIndices = xt::view( sortedPointIndices, 
                                                        xt::range(splitIndex, _) );
            const auto rightNodeIndex = build_kd_nodes(points, rightPointIndices);

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
            Index bucketStart = node.get_bucket_start(); 
            Index bucketStop = node.get_bucket_stop();
            auto leafPoints = xt::view(pointsBucket, 
                            xt::range(bucketStart, bucketStop), xt::all());
            // Points minusPoints = leafPoints - queryPoint; 
            // xt::xtensor<float, 1> squaredDistances = xt::norm_sq( leafPoints - queryPoint, 1.0, xt::evaluation_strategy::immediate)();

            // std::cout<< "leafPoints - queryPoint: " << (leafPoints - queryPoint) << std::endl;
            // std::cout<< "squared distances: " << squaredDistances << std::endl;
            // std::cout<< "bucket start: " << bucketStart << ", bucket stop: "<< bucketStop << 
                        // ", size: "<< bucketStop - bucketStart << std::endl;

            Index pointIndex;
            float squaredDist;
            for(Index i=0; i<node.get_bucket_size(); i++){
                pointIndex = pointIndicesBucket[ i + bucketStart ];
                auto point = xt::view(leafPoints, i, xt::all());
                squaredDist = xt::norm_sq( point - queryPoint )();
                indexHeap.update( squaredDist, pointIndex );
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

    auto build_kd_tree(const Points &points){
        auto pointNum = points.shape(0);
        pointIndicesBucket.reserve( pointNum );
        Points::shape_type sh = {pointNum, 3};
        pointsBucket = xt::empty<float>( sh );
        // std::cout<< "\nreserve node number: " << pointNum/leafSize*2 <<std::endl;
        kdTreeNodes.reserve( pointNum / leafSize * 2 );
        
        PointIndices pointIndices = xt::arange<Index>(0, pointNum);
        build_kd_nodes(points, pointIndices);
        assert(pointIndicesBucket.size() == pointNum);
        kdTreeNodes.shrink_to_fit();
    }

public:
    KDTree(const KDTreeNodes &kdTreeNodes_, 
            const PointIndicesBucket &pointIndicesBucket_, const PyPoints &pointsBucket_, 
            const Index &leafSize_): 
                kdTreeNodes(kdTreeNodes_), pointIndicesBucket(pointIndicesBucket_),
                pointsBucket(Points(pointsBucket_)), leafSize(leafSize_){}

    KDTree( const std::tuple<KDTreeNodes, PointIndicesBucket, PyPoint, Index> &tp ):
        kdTreeNodes(std::get<0>(tp)), pointIndicesBucket(std::get<1>(tp)), 
        pointsBucket( Points( std::get<2>(tp) ) ), leafSize(std::get<3>(tp)){}

    KDTree( const Points &points, const std::size_t &leafSize_ ): 
                leafSize(leafSize_), 
                kdTreeNodes({}), pointIndicesBucket({}){
        build_kd_tree( points );
    }

    KDTree( const PyPoints &points, const std::size_t &leafSize_ ):
                leafSize(leafSize_), kdTreeNodes({}), pointIndicesBucket({}){
        build_kd_tree( Points( points) );
    }

    auto get_kd_tree_nodes() const {
        return kdTreeNodes;
    }

    auto get_point_indices_bucket() const {
        return pointIndicesBucket;
    }

    auto get_py_points_bucket() const {
        return PyPoints(pointsBucket);
    }

    auto get_leaf_size() const {
        return leafSize;
    }

    auto get_serializable_tuple() const {
        return std::make_tuple( kdTreeNodes, pointIndicesBucket, 
                                get_py_points_bucket(), leafSize);
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
