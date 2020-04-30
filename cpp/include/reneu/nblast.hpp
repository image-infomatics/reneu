#pragma once

#include <fstream>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <filesystem>
#include <memory>
#include <cassert>
#include <pybind11/pybind11.h>
#include "xtensor/xview.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xcsv.hpp"

#include "reneu/type_aliase.hpp"
#include "reneu/utils/math.hpp"
#include "reneu/utils/kd_tree.hpp"

// use the c++17 nested namespace
namespace reneu{


namespace py = pybind11;

class ScoreTable{

private:
    xt::xtensor_fixed<float, xt::xshape<21, 10>> table; 
    //const DistThresholdsType distThresholds = {0., 0.75, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 
    //                                    12, 14, 16, 20, 25, 30, 40, std::numeric_limits<float>::max()};
    // this is using nanometer rather than micron
    const xt::xtensor_fixed<float, xt::xshape<22>> distThresholds = {
            std::numeric_limits<float>::min(), 750, 1500, 2000, 2500, 3000, 3500, 4000, 
            5000, 6000, 7000, 8000, 9000, 10000, 
            12000, 14000, 16000, 20000, 25000, 30000, 40000, 
            std::numeric_limits<float>::max()};

    // const xt::xtensor_fixed<float, xt::xshape<11>> adpThresholds = {
            // -1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2};

    template<std::size_t N>
    inline auto binary_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, 
                                                        const float &value ) const {
        Index start = 0;
        // Note that the last one index is N-1 rather than N_
        // This is following python indexing style, the last index is not inclusive
        Index stop = N;
        while( stop - start > 1 ){
            auto middle = std::div(stop - start, 2).quot + start;
            if ( value >= thresholds[ middle ] ){
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
    
    template<std::size_t N>
    inline auto sequential_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, 
                                                        const float &value ) const {
        for (Index i=0; i<N; i++){
            if(value < thresholds(i+1)){
                return i;
            }
        }
    }

public:
    ScoreTable(const xt::pytensor<float, 2> &table_): table(table_){}

    ScoreTable( const std::string fileName ){
        std::ifstream in(fileName); 
        table = xt::load_csv<float>( in );
    }

    ScoreTable(){
        const auto fileName = std::filesystem::path(__FILE__).parent_path() / 
                                                        "../../../data/smat_fcwb.csv";
        std::ifstream in(fileName);
        std::cout<< "read nblast score table file: " << fileName << std::endl; 
        table = xt::load_csv<float>( in );
        in.close();
    }

    inline auto get_pytable() const {
        // xtensor_fixed can not be converted to pytensor directly
        return xt::pytensor<float, 2>(table);
    }

    /**
     * \brief 
     * \param dist: physical distance
     * \param dp: absolute dot product of vectors
     */
    inline auto operator()(const float &dist, const float &adp) const {
        // Index distIdx = sequential_search( distThresholds, dist );
        Index distIdx = binary_search( distThresholds, dist );

        // minus a small value to make sure that adp=1, we get index 9 rather than 10
        Index adpIdx = trunc( adp * 10. - 1e-4 );
        // Index adpIdx1 = binary_search( adpThresholds, adp );
        // Index adpIdx2 = sequential_search( adpThresholds, adp );

        // if (adpIdx != adpIdx2){
            // std::cout<< adp << ": inconsistent adpidx: " << adpIdx << ", " << adpIdx1 << ", " << adpIdx2 << std::endl;
        // }
        return table( distIdx, adpIdx );
    }

    inline auto operator()(const std::tuple<float, float> &slice){
        return this->operator()(std::get<0>(slice), std::get<1>(slice));
    }

    auto self_score() const {
        return table(0, 9);
    }

}; // ScoreTable class 

class VectorCloud{

private:
    const Points points;
    xt::xtensor<float, 2> vectors;
    KDTree kdTree;
    auto construct_vectors(const Index &nearestPointNum){
        auto pointNum = points.shape(0);
        
        // find the nearest k points and compute the first principle component as the main direction
        xt::xtensor<float, 2>::shape_type shape = {nearestPointNum, 3};
        Points nearestPoints = xt::empty<float>(shape);

        xt::xtensor<float, 2>::shape_type vshape = {pointNum, 3};
        vectors = xt::empty<float>( vshape );

        for (Index pointIdx = 0; pointIdx < pointNum; pointIdx++){
            xt::xtensor<float, 1> queryPoint = xt::view(points, pointIdx, xt::range(0, 3));
            auto nearestPointIndices = kdTree.knn( queryPoint, nearestPointNum );
            // std::cout<< "nearest point indices: " << nearestPointIndices << std::endl;
            for (Index i=0; i<nearestPointIndices.size(); i++){
                auto nearestPointIndex = nearestPointIndices(i);
                nearestPoints(i, 0) = points( nearestPointIndex, 0 );
                nearestPoints(i, 1) = points( nearestPointIndex, 1 );
                nearestPoints(i, 2) = points( nearestPointIndex, 2 );
            }
            // use the first principle component as the main direction
            //vectors(pointIdx, xt::all()) = pca_first_component( nearestPoints ); 
            // std::cout<< "nearest points: " << nearestPoints << std::endl;
            auto direction = pca_first_component( nearestPoints );
            vectors(pointIdx, 0) = direction(0);
            vectors(pointIdx, 1) = direction(1);
            vectors(pointIdx, 2) = direction(2);
            // std::cout<< "vector: " << direction << std::endl;
        }
    }

public:
    VectorCloud( const Points &points_, const Points &vectors_, const KDTree &kdTree_ ):
                points(points), vectors(vectors_), kdTree(kdTree_){}
    
    VectorCloud( const PyPoints &points_, const PyPoints &vectors_, const KDTree &kdTree_ ):
                points(points), vectors(vectors_), kdTree(kdTree_){}
    
    // our points array contains radius direction, but we do not need it.
    VectorCloud( const Points &points_, const Index &leafSize, 
                    const Index &nearestPointNum ): 
                    points(points_), kdTree(points_, leafSize){
        construct_vectors( nearestPointNum );
    }
    
    VectorCloud( const xt::pytensor<float, 2> &points_, const Index &leafSize, 
                        const Index &nearestPointNum ): 
                        points(points_), kdTree(points_, leafSize) {
        construct_vectors( nearestPointNum );
    }

    
    inline auto size() const {
        return points.shape(0);
    }

    inline auto get_points() const {
        return points;
    }

    inline auto get_py_points() const {
        return PyPoints(points);
    }

    inline auto get_vectors() const {
        return vectors;
    }

    inline auto get_py_vectors() const {
        return PyPoints( vectors );
    }

    inline auto get_kd_tree() const {
        return kdTree;
    }

    inline auto get_kd_tree_serializable_tuple() const {
        return kdTree.get_serializable_tuple();
    } 

    float query_by_self(const ScoreTable &scoreTable) const {
        return size() * scoreTable.self_score();
    }

    auto query_by(const VectorCloud &query, const ScoreTable &scoreTable) const {
        // raw NBLAST is accumulated by query points
        float rawScore=0, distance, absoluteDotProduct;
        Index nearestPointIndex; 
        xt::xtensor_fixed<float, xt::xshape<3>> queryPoint, nearestPoint, queryVector, targetVector;

        const auto queryPoints = query.get_points();
        const auto queryVectors = query.get_vectors();
        for (Index queryPointIndex = 0; queryPointIndex<query.size(); queryPointIndex++){
            
            queryPoint = xt::view(queryPoints, queryPointIndex, xt::range(0, 3));
            // find the best match point in target and get physical distance
            nearestPointIndex = kdTree.knn( queryPoint, 1 )(0);
            nearestPoint = xt::view(points, nearestPointIndex, xt::range(0,3));
            distance = xt::norm_l2( nearestPoint - queryPoint )(0);
           
            // compute the absolute dot product between the principle vectors
            queryVector = xt::view(queryVectors, queryPointIndex, xt::all());
            targetVector = xt::view(vectors, nearestPointIndex, xt::all());
            auto dotProduct = xt::linalg::dot(queryVector, targetVector)(0);
            absoluteDotProduct = std::abs(dotProduct);
            // absoluteDotProduct = std::abs(xt::linalg::dot( queryVector, targetVector )(0));
            
            // lookup the score table and accumulate the score
            rawScore += scoreTable( distance,  absoluteDotProduct );
            
            // std::cout<< "\n "<< queryPointIndex << " th query point: " << queryPoint <<std::endl;
            // std::cout<< "nearest point index: "<< nearestPointIndex << " : " << nearestPoint << std::endl;
            // std::cout<< "vectors: "<<queryVector << "   "<< targetVector << std::endl;
            // std::cout<< "distance: "<< distance << ";   absolute dot product: " 
            //                                     << absoluteDotProduct << std::endl;
            // std::cout<< "accumulate score: "<< scoreTable(distance, absoluteDotProduct) << " to " << rawScore<< std::endl; 
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
        Index vcNum = vectorClouds.size();
        xt::xtensor<float, 2>::shape_type shape = {vcNum, vcNum};
        rawScoreMatrix = xt::empty<float>( shape );

        for (Index targetIdx = 0; targetIdx<vcNum; targetIdx++){
            VectorCloud target = vectorClouds[ targetIdx ];
            for (Index queryIdx = 0; queryIdx<vcNum; queryIdx++){
                if (targetIdx == queryIdx){
                    rawScoreMatrix(targetIdx, queryIdx) = target.query_by_self(scoreTable);
                } else {
                    auto query = vectorClouds[ queryIdx ];
                    rawScoreMatrix( targetIdx, queryIdx ) = target.query_by( query, scoreTable );
                }
            }
        }
    }

    //NBLASTScoreMatrix( const py::list &vectorClouds, const ScoreTable scoreTable ){
    //    
    //}

    //NBLASTScoreMatrix(  const std::vector<reneu::neuron::Skeleton> &skeletonList, 
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
        for (Index queryIdx = 0; queryIdx<get_neuron_number(); queryIdx++){
            auto selfQueryScore = rawScoreMatrix( queryIdx, queryIdx );
            for (Index targetIdx = 0; targetIdx<get_neuron_number(); targetIdx++){
                normalizedScoreMatrix(targetIdx, queryIdx) = rawScoreMatrix(targetIdx, queryIdx) / selfQueryScore;
            }
        }
        return normalizedScoreMatrix; 
    }

    inline auto get_mean_score_matrix() const {
        auto normalizedScoreMatrix = get_normalized_score_matrix();
        auto meanScoreMatrix = xt::ones_like( normalizedScoreMatrix );
        for (Index targetIdx = 0; targetIdx<get_neuron_number(); targetIdx++){
            for (Index queryIdx = targetIdx+1; queryIdx<get_neuron_number(); queryIdx++){
                meanScoreMatrix(targetIdx, queryIdx) = (normalizedScoreMatrix(targetIdx, queryIdx) + 
                                                        normalizedScoreMatrix(queryIdx, targetIdx)) / 2;
                meanScoreMatrix(queryIdx, targetIdx) = meanScoreMatrix(targetIdx, queryIdx);
            }
        }
        return meanScoreMatrix;
    }
};

} // end of namespace reneu::neuron::nblast
