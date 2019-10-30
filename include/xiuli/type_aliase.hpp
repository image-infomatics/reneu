# pragma once

#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor-python/pytensor.hpp"

using Index = std::uint32_t;
using Point = xt::xtensor_fixed<float, xt::xshape<3>>;
using Points = xt::xtensor<float, 2>;
using PyPoint = xt::pytensor<float, 1>;
using PyPoints = xt::pytensor<float, 2>;
using PointIndices = xt::xtensor<Index, 1>;