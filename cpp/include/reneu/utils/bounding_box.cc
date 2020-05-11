#pragma once

#include "bounding_box.h"

namespace reneu {

class BoundingBox {
 private:
  xt::xtensor_fixed<float, xt::xshape<2, 3>> corner;

 public:
  BoundingBox(const Points &points, const PointIndices &pointIndices)
      : corner(xt::zeros<float>({2, 3})) {
    for (Index i = 0; i < 3; i++) {
      auto coords =
          xt::index_view(xt::view(points, xt::all(), i), pointIndices);
      auto minmax = xt::minmax(coords)();
      corner(0, i) = minmax[0];
      corner(1, i) = minmax[1];
    }
  }

  inline auto get_min_corner() const { return xt::view(corner, 0, xt::all()); }

  inline auto get_max_corner() const { return xt::view(corner, 1, xt::all()); }

  auto get_largest_extent_dimension() const {
    auto minCorner = get_min_corner();
    auto maxCorner = get_max_corner();
    return xt::argmax(maxCorner - minCorner)(0);
  }

  /**
   * compute the distance from bounding box using a smart way
  https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
  code in JS
  function distance(rect, p) {
    var dx = Math.max(rect.min.x - p.x, 0, p.x - rect.max.x);
    var dy = Math.max(rect.min.y - p.y, 0, p.y - rect.max.y);
    return Math.sqrt(dx*dx + dy*dy);
  }
  */
  float min_squared_distance_from(const xt::xtensor<float, 1> &node) const {
    float squaredDist = 0;
    float tmp;
    for (Index i = 0; i < 3; i++) {
      tmp = 0;
      tmp = std::max(tmp, corner(0, i) - node(i));
      tmp = std::max(tmp, node(i) - corner(1, i));
      squaredDist += tmp * tmp;
    }

    // std::cout<< "min corner: "<< minCorner << " max corner: " << maxCorner <<
    // " node: " << node << " min squared distance: " << squaredDist <<
    // std::endl;
    return squaredDist;
  }

};  // end of class

}  // namespace reneu
