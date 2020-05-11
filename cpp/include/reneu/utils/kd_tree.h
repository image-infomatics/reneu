#pragma once

#include <iostream>
#include <limits>  // std::numeric_limits
#include <queue>
#include <variant>

#include "reneu/type_aliase.hpp"
#include "reneu/utils/bounding_box.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

namespace reneu {

// namespace py=pybind11;
using namespace xt::placeholders;

using HeapElement = std::pair<float, Index>;
auto cmp = [](HeapElement left, HeapElement right) {
  return left.first < right.first;
};

class IndexHeap{
public:
  inline auto size();
  inline auto max_squared_dist();
  auto get_point_indices();
  void update(const float &squaredDist, const Index &pointIndex); 
}; // end of class IndexHeap

} // end of namespace reneu