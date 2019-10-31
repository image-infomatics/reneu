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
#include "xiuli/type_aliase.hpp"

// use the c++17 nested namespace
namespace xiuli{


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

    const xt::xtensor_fixed<float, xt::xshape<11>> adpThresholds = {
            -1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2};

    template<std::size_t N>
    auto binary_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, 
                                                        const float &value ) const {
        Index start = 0;
        // Note that the last one index is N-1 rather than N_
        // This is following python indexing style, the last index is not inclusive
        Index stop = N;
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
    
    template<std::size_t N>
    auto sequential_search( const xt::xtensor_fixed<float, xt::xshape<N>> &thresholds, 
                                                        const float &value ) const {
        for (Index i=0; i<N; i++){
            if(value <= thresholds(i+1)){
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
        // Index distIdx = binary_search( distThresholds, dist );
        // Index adpIdx = binary_search( adpThresholds, adp );
        
        Index distIdx = sequential_search( distThresholds, dist );
        Index adpIdx = sequential_search( adpThresholds, adp );

        return table( distIdx, adpIdx );
    }

    inline auto operator()(const std::tuple<float, float> &slice){
        return this->operator()(std::get<0>(slice), std::get<1>(slice));
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
    inline auto size() const {
        return points.shape(0);
    }

    inline auto get_points() const {
        return points;
    }

    inline auto get_vectors() const {
        return vectors;
    }

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

    auto query_by(VectorCloud &query, const ScoreTable &scoreTable) const {
        // raw NBLAST is accumulated by query points
        float rawScore = 0;

        float distance = 0;
        float absoluteDotProduct = 0;
        Index nearestPointIndex = 0; 
        xt::xtensor_fixed<float, xt::xshape<3>> queryPoint, nearestPoint, queryVector, targetVector;

        auto queryPoints = query.get_points();
        auto queryVectors = query.get_vectors();
        for (Index queryPointIdx = 0; queryPointIdx<query.size(); queryPointIdx++){
            
            queryPoint = xt::view(queryPoints, queryPointIdx, xt::range(0, 3));
            
            // find the best match point in target and get physical distance
            nearestPointIndex = kdTree.knn( queryPoint, 1 )(0);
            nearestPoint = xt::view(points, nearestPointIndex, xt::range(0,3));
            distance = xt::norm_l2( nearestPoint - queryPoint )(0);
           
            // compute the absolute dot product between the principle vectors
            queryVector = xt::view(queryVectors, queryPointIdx, xt::all());
            targetVector = xt::view(vectors, nearestPointIndex, xt::all());
            //auto dot = xt::linalg::dot(queryVector, targetVector);
            //assert( dot.size() == 1 );
            //absoluteDotProduct = std::abs(dot(0));
            absoluteDotProduct = std::abs(xt::linalg::dot( queryVector, targetVector )(0));
            
            // lookup the score table and accumulate the score
            rawScore += scoreTable( distance,  absoluteDotProduct );
            
            // std::cout<< "\nquery point: " << queryPoint <<std::endl;
            // std::cout<< "nearest point index: "<< nearestPointIndex << std::endl;
            // std::cout<< "distance: " << distance << std::endl;
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
            for (Index queryIdx = targetIdx; queryIdx<vcNum; queryIdx++){
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

} // end of namespace xiuli::neuron::nblast
