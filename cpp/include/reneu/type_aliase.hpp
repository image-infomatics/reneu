# pragma once

#include "xtensor/xtensor.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor-python/pytensor.hpp>

namespace reneu{

using namespace std;
using namespace xt::placeholders;  // required for `_` in range view to work

using aff_edge_t = float;
using segid_t = std::uint64_t;

using AffinityMap = xt::xtensor<aff_edge_t, 4>;
using PyAffinityMap = xt::pytensor<aff_edge_t, 4>;
using Segmentation = xt::xtensor<segid_t, 3>;
using PySegmentation = xt::pytensor<segid_t, 3>;
using Segmentation2D = xt::xtensor<segid_t, 2>;
using Segmentation1D = xt::xtensor<segid_t, 1>;

using ProbabilityMap = xt::xtensor<float, 3>;
using PyProbabilityMap = xt::pytensor<float, 3>;

using Index = std::uint32_t;
using Point = xt::xtensor_fixed<float, xt::xshape<3>>;
using PyPoint = xt::pytensor<float, 1>;
using Points = xt::xtensor<float, 2>;
using PyPoints = xt::pytensor<float, 2>;
using PointIndices = xt::xtensor<Index, 1>;

} // namespace