# pragma once

#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor-python/pytensor.hpp"

using Index = std::size_t;
using PyNode = xt::pytensor<float, 2>;
using Node = xt::xtensor_fixed<float, xt::xshape<3>>;
using Nodes = xt::xtensor<float, 2>;
using NodeIndices = xt::xtensor<Index, 1>;