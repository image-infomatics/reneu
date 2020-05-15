#pragma once

#include <pybind11/pybind11.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <memory>

#include "reneu/type_aliase.hpp"
#include "reneu/utils/kd_tree.hpp"
#include "reneu/utils/math.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xview.hpp"

// use the c++17 nested namespace
namespace reneu {

class ScoreTable{
public:
inline auto get_pytable();
inline auto operator()(const float &dist, const float &adp);

inline auto operator()(const std::tuple<float, float> &slice);

auto self_score();
};

class VectorCloud {
  const Points points;
  xt::xtensor<float, 2> vectors;
  KDTree kdTree;
  
private:
  auto construct_vectors(const Index &nearestPointNum);
 
public:
  VectorCloud(const Points &points_, const Points &vectors_,
              const KDTree &kdTree_);

  VectorCloud(const PyPoints &points_, const PyPoints &vectors_,
              const KDTree &kdTree_)
      : points(points), vectors(vectors_), kdTree(kdTree_) {}

  // our points array contains radius direction, but we do not need it.
  VectorCloud(const Points &points_, const Index &leafSize,
              const Index &nearestPointNum)
      : points(points_), kdTree(points_, leafSize) {
    construct_vectors(nearestPointNum);
  }

  VectorCloud(const xt::pytensor<float, 2> &points_, const Index &leafSize,
              const Index &nearestPointNum)
      : points(points_), kdTree(points_, leafSize) {
    construct_vectors(nearestPointNum);
  }

  inline auto size() const { return points.shape(0); }

  inline auto get_points() const { return points; }

  inline auto get_py_points() const { return PyPoints(points); }

  inline auto get_vectors() const { return vectors; }

  inline auto get_py_vectors() const { return PyPoints(vectors); }

  inline auto get_kd_tree() const { return kdTree; }

  inline auto get_kd_tree_serializable_tuple() const {
    return kdTree.get_serializable_tuple();
  }

  float query_by_self(const ScoreTable &scoreTable) const {
    return size() * scoreTable.self_score();
  }

  auto query_by(const VectorCloud &query, const ScoreTable &scoreTable);
};  // VectorCloud class

class NBLASTScoreMatrix {
 private:
  // the rows are targets, the columns are queries
  xt::xtensor<float, 2> rawScoreMatrix;

 public:
  NBLASTScoreMatrix(const std::vector<VectorCloud> &vectorClouds,
                    const ScoreTable &scoreTable) {
    Index vcNum = vectorClouds.size();
    xt::xtensor<float, 2>::shape_type shape = {vcNum, vcNum};
    rawScoreMatrix = xt::empty<float>(shape);

    for (Index targetIdx = 0; targetIdx < vcNum; targetIdx++) {
      VectorCloud target = vectorClouds[targetIdx];
      for (Index queryIdx = 0; queryIdx < vcNum; queryIdx++) {
        if (targetIdx == queryIdx) {
          rawScoreMatrix(targetIdx, queryIdx) =
              target.query_by_self(scoreTable);
        } else {
          auto query = vectorClouds[queryIdx];
          rawScoreMatrix(targetIdx, queryIdx) =
              target.query_by(query, scoreTable);
        }
      }
    }
  }

  // NBLASTScoreMatrix( const py::list &vectorClouds, const ScoreTable
  // scoreTable ){
  //
  //}

  // NBLASTScoreMatrix(  const std::vector<reneu::neuron::Skeleton>
  // &skeletonList,
  //                    const ScoreTable &scoreTable){
  //
  //}

  inline auto get_neuron_number() const { return rawScoreMatrix.shape(0); }

  inline auto get_raw_score_matrix() const { return rawScoreMatrix; }

  /*
   * \brief normalized by the self score of query
   */
  inline auto get_normalized_score_matrix() const {
    xt::xtensor<float, 2> normalizedScoreMatrix =
        xt::zeros_like(rawScoreMatrix);
    for (Index queryIdx = 0; queryIdx < get_neuron_number(); queryIdx++) {
      auto selfQueryScore = rawScoreMatrix(queryIdx, queryIdx);
      for (Index targetIdx = 0; targetIdx < get_neuron_number(); targetIdx++) {
        normalizedScoreMatrix(targetIdx, queryIdx) =
            rawScoreMatrix(targetIdx, queryIdx) / selfQueryScore;
      }
    }
    return normalizedScoreMatrix;
  }

  inline auto get_mean_score_matrix() const {
    auto normalizedScoreMatrix = get_normalized_score_matrix();
    auto meanScoreMatrix = xt::ones_like(normalizedScoreMatrix);
    for (Index targetIdx = 0; targetIdx < get_neuron_number(); targetIdx++) {
      for (Index queryIdx = targetIdx + 1; queryIdx < get_neuron_number();
           queryIdx++) {
        meanScoreMatrix(targetIdx, queryIdx) =
            (normalizedScoreMatrix(targetIdx, queryIdx) +
             normalizedScoreMatrix(queryIdx, targetIdx)) /
            2;
        meanScoreMatrix(queryIdx, targetIdx) =
            meanScoreMatrix(targetIdx, queryIdx);
      }
    }
    return meanScoreMatrix;
  }
};



}// end of namespace reneu
