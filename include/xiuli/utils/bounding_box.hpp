#pragma once

// #include "xtensor/xio.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"


namespace xiuli::utils{

class BoundingBox{
private:
    xt::xtensor_fixed<float, xt::xshape<3>> minCorner;
    xt::xtensor_fixed<float, xt::xshape<3>> maxCorner;


public:
    BoundingBox(){
        minCorner = xt::zeros<float>({3});
        maxCorner = xt::zeros<float>({3});
    }

    BoundingBox(const xt::xtensor<float, 2> &nodes, 
                const xt::xtensor<std::size_t, 1> &nodeIndices){
        for (std::size_t i=0; i<3; i++){
            auto coords = xt::index_view(
                    xt::view(nodes, xt::all(), i),
                    nodeIndices);
            auto minmax = xt::minmax(coords)();
            minCorner(i) = minmax[0];
            maxCorner(i) = minmax[1];
        }

    }

    auto get_largest_extent_dimension() const {
        return xt::argmax( maxCorner - minCorner )(0);
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
    float min_squared_distance_from( const xt::xtensor<float, 1> &node) const {
        float squaredDist = 0;
        float tmp;
        for (std::size_t i=0; i<3; i++){
            tmp = 0;
            tmp = std::max(tmp, minCorner(i) - node(i));
            tmp = std::max(tmp, node(i) - maxCorner(i));
            squaredDist += tmp * tmp;
        }

        // std::cout<< "min corner: "<< minCorner << " max corner: " << maxCorner << 
                // " node: " << node << " min squared distance: " << squaredDist << std::endl;
        return squaredDist;
    }

}; // end of class


} // end of namespace
