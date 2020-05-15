#pragma once

#include <assert.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "reneu/type_aliase.hpp"
#include "reneu/utils/string.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"

using namespace reneu::utils;

using Attributes = xt::xtensor<int, 2>;

// use the c++17 nested namespace
namespace reneu {

class Skeleton {
 private:
  // point array (N x 4), the rows are points, the columns are x,y,z,r
  // normally the type is float
  Points points;

  // attributes of points (N x 4),
  // the columns are point type, parent, first child, sibling
  // the point types are defined in SWC format
  // 0 - undefined
  // 1 - soma
  // 2 - axon
  // 3 - (basal) dendrite
  // 4 - apical dendrite
  // 5 - fork point
  // 6 - end point
  // 7 - custom
  // normally the type is int
  Attributes attributes;

  inline auto get_classes() const { return xt::view(attributes, xt::all(), 0); }

  inline auto get_parents() const { return xt::view(attributes, xt::all(), 1); }

  inline auto get_childs() const { return xt::view(attributes, xt::all(), 2); }

  inline auto get_siblings() const {
    return xt::view(attributes, xt::all(), 3);
  }

  template <typename Ti>
  inline auto get_point(Ti pointIdx) const {
    return xt::view(points, pointIdx, xt::all());
  }

  inline auto squared_distance(int idx1, int idx2) const {
    auto point1 = get_point(idx1);
    auto point2 = get_point(idx2);
    return (point1(0) - point2(0)) * (point1(0) - point2(0)) +
           (point1(1) - point2(1)) * (point1(1) - point2(1)) +
           (point1(2) - point2(2)) * (point1(2) - point2(2));
  }

  template <typename Tn>
  inline auto initialize_points_and_attributes(Tn pointNum) {
    // create points and attributes
    Points::shape_type pointsShape = {pointNum, 4};
    points = xt::zeros<float>(pointsShape);
    Attributes::shape_type attributesShape = {pointNum, 4};
    // root point id is -2 rather than -1.
    attributes = xt::zeros<int>(attributesShape) - 2;
  }

  auto update_first_child_and_sibling() {
    auto parents = get_parents();

    auto childs = xt::view(attributes, xt::all(), 2);
    auto siblings = xt::view(attributes, xt::all(), 3);

    auto pointNum = parents.size();
    // the columns are class, parents, fist child, sibling

    // clean up the child and siblings
    childs = -2;
    siblings = -2;

    // update childs
    for (std::size_t pointIdx = 0; pointIdx < pointNum; pointIdx++) {
      auto parentPointIdx = parents(pointIdx);
      if (parentPointIdx >= 0) childs(parentPointIdx) = pointIdx;
    }

    // update siblings
    for (int pointIdx = 0; pointIdx < int(pointNum); pointIdx++) {
      auto parentPointIdx = parents(pointIdx);
      if (parentPointIdx >= 0) {
        auto currentSiblingPointIdx = childs(parentPointIdx);
        assert(currentSiblingPointIdx >= 0);
        if (currentSiblingPointIdx != pointIdx) {
          // look for an empty sibling spot to fill in
          auto nextSiblingPointIdx = siblings(currentSiblingPointIdx);
          while (nextSiblingPointIdx >= 0) {
            currentSiblingPointIdx = nextSiblingPointIdx;
            // move forward to look for empty sibling spot
            nextSiblingPointIdx = siblings(currentSiblingPointIdx);
          }
          // finally, we should find an empty spot
          siblings(currentSiblingPointIdx) = pointIdx;
        }
      }
    }
    return 0;
  }

 public:
  virtual ~Skeleton() = default;

  Skeleton(const xt::pytensor<float, 2> &points_,
           const xt::pytensor<int, 2> &attributes_)
      : points(points_), attributes(attributes_) {
    update_first_child_and_sibling();
  }

  Skeleton(const xt::pytensor<float, 2> &swcArray) {
    // this input array should follow the format of swc file
    assert(swcArray.shape(1) == 7);
    auto pointNum = swcArray.shape(0);
    initialize_points_and_attributes(pointNum);

    auto xids = xt::view(swcArray, xt::all(), 0);
    auto newIdx2oldIdx = xt::argsort(xids);

    // construct the points and attributes
    initialize_points_and_attributes(pointNum);

    for (std::size_t i = 0; i < pointNum; i++) {
      // we do sort here!
      auto oldIdx = newIdx2oldIdx(i);
      attributes(i, 0) = swcArray(oldIdx, 1);
      points(i, 0) = swcArray(oldIdx, 2);
      points(i, 1) = swcArray(oldIdx, 3);
      points(i, 2) = swcArray(oldIdx, 4);
      points(i, 3) = swcArray(oldIdx, 5);
      // swc point id is 1-based
      attributes(i, 1) = swcArray(oldIdx, 6) - 1;
    }
    update_first_child_and_sibling();
  }

  Skeleton(const std::string &file_name) {
    std::cerr << "this function is pretty slow and should be speed up using "
                 "memory map!"
              << std::endl;
    std::ifstream myfile(file_name, std::ios::in);

    if (myfile.is_open()) {
      std::string line;
      std::regex ws_re("\\s+");  // whitespace

      // initialize the data vectors
      std::vector<int> ids = {};
      std::vector<int> classes = {};
      std::vector<float> xs = {};
      std::vector<float> ys = {};
      std::vector<float> zs = {};
      std::vector<float> rs = {};
      std::vector<int> parents = {};

      while (std::getline(myfile, line)) {
        std::vector<std::string> parts = reneu::utils::split(line, "\\s+"_re);
        if (parts.size() == 7 && parts[0][0] != '#') {
          ids.push_back(std::stoi(parts[0]));
          classes.push_back(std::stoi(parts[1]));
          xs.push_back(std::stof(parts[2]));
          ys.push_back(std::stof(parts[3]));
          zs.push_back(std::stof(parts[4]));
          rs.push_back(std::stof(parts[5]));
          // swc is 1-based, and we are 0-based here.
          parents.push_back(std::stoi(parts[6]) - 1);
        }
      }
      assert(ids.size() == parents.size());
      myfile.close();

      // the point index could be unordered
      // we should order them and will drop the ids column
      auto pointNum = ids.size();
      std::vector<std::size_t> shape = {
          pointNum,
      };
      auto xids = xt::adapt(ids, shape);
      auto newIdx2oldIdx = xt::argsort(xids);

      // construct the points and attributes
      initialize_points_and_attributes(pointNum);

      for (std::size_t i = 0; i < pointNum; i++) {
        // we do sort here!
        auto oldIdx = newIdx2oldIdx(i);
        points(i, 0) = xs[oldIdx];
        points(i, 1) = ys[oldIdx];
        points(i, 2) = zs[oldIdx];
        points(i, 3) = rs[oldIdx];
        attributes(i, 0) = classes[oldIdx];
        // swc point id is 1-based
        attributes(i, 1) = parents[oldIdx];
      }
      update_first_child_and_sibling();

      auto childs = get_childs();
      assert(xt::any(childs >= 0));
    } else {
      std::cout << "can not open file: " << file_name << std::endl;
    }
  }

  inline auto get_points() { return points; }

  inline auto get_py_points() const { return PyPoints(points); }

  void set_py_points(const PyPoints &points_) {
    points = Points(points_);
    return;
  }

  inline auto get_attributes() { return attributes; }

  inline auto get_py_attributes() const { return PyPoints(attributes); }

  inline bool is_root_point(int pointIdx) const {
    return attributes(pointIdx, 1) < 0;
  }

  inline bool is_terminal_point(int pointIdx) const {
    return attributes(pointIdx, 2) < 0;
  }

  inline auto get_point_num() const { return points.shape(0); }

  bool is_branching_point(int pointIdx) const {
    auto childs = get_childs();
    auto childPointIdx = childs(pointIdx);
    if (childPointIdx < 0) {
      // no child, this is a terminal point
      return false;
    } else {
      // if child have sibling, then this is a branching point
      auto siblings = get_siblings();
      return siblings(childPointIdx) > 0;
    }
  }

  auto get_edge_num() const {
    auto pointNum = get_point_num();
    auto parents = get_parents();
    std::size_t edgeNum = 0;
    for (std::size_t i = 0; i < pointNum; i++) {
      if (parents(i) >= 0) {
        edgeNum += 1;
      }
    }
    return edgeNum;
  }

  auto get_edges() {
    auto edgeNum = get_edge_num();
    xt::xtensor<std::uint32_t, 2>::shape_type edgesShape = {edgeNum, 2};
    xt::xtensor<std::uint32_t, 2> edges = xt::zeros<std::uint32_t>(edgesShape);

    auto parents = get_parents();
    std::size_t edgeIdx = 0;
    for (std::size_t i = 0; i < get_point_num(); i++) {
      auto parentIdx = parents(i);
      if (parentIdx >= 0) {
        edges(edgeIdx, 0) = i;
        edges(edgeIdx, 1) = parentIdx;
        edgeIdx += 1;
      }
    }
    return edges;
  }

  std::vector<int> get_children_point_indexes(int pointIdx) const {
    auto childs = get_childs();
    auto siblings = get_siblings();

    std::vector<int> childrenPointIdxes = {};
    auto childPointIdx = childs(pointIdx);
    while (childPointIdx >= 0) {
      childrenPointIdxes.push_back(childPointIdx);
      childPointIdx = siblings(childPointIdx);
    }
    return childrenPointIdxes;
  }

  void translate_centroid_to_origin() { points -= xt::mean(points, {0}); }

  auto downsample(const float step) {
    auto att = attributes;

    auto classes = get_classes();
    auto parents = get_parents();
    auto childs = get_childs();
    auto siblings = get_siblings();

    auto pointNum = get_point_num();

    // std::cout<< "downsampling step: " << step <<std::endl;
    auto stepSquared = step * step;

    // start from the root points
    std::vector<int> rootPointIdxes = {};
    for (std::size_t pointIdx = 0; pointIdx < pointNum; pointIdx++) {
      if (is_root_point(pointIdx)) {
        rootPointIdxes.push_back(pointIdx);
      }
    }
    assert(!rootPointIdxes.empty());

    auto seedPointIdxes = rootPointIdxes;
    // initialize the seed point parent indexes
    std::vector<int> seedPointParentIdxes = {};
    seedPointParentIdxes.assign(seedPointIdxes.size(), -2);

    // we select some points out as new neuron
    // this selection should include all the seed point indexes
    std::vector<int> selectedPointIdxes = {};
    // we need to update the parents when we add new selected points
    std::vector<int> selectedParentPointIdxes = {};

    while (!seedPointIdxes.empty()) {
      auto startPointIdx = seedPointIdxes.back();
      auto startPointParentIdx = seedPointParentIdxes.back();
      seedPointIdxes.pop_back();
      seedPointParentIdxes.pop_back();
      selectedPointIdxes.push_back(startPointIdx);
      selectedParentPointIdxes.push_back(startPointParentIdx);

      // the start of measurement
      auto walkingPointIdx = startPointIdx;

      // walk through a segment
      while (!is_branching_point(walkingPointIdx) &&
             !is_terminal_point(walkingPointIdx)) {
        // walk to next point
        walkingPointIdx = childs[walkingPointIdx];
        // use squared distance to avoid sqrt computation.
        float d2 = squared_distance(startPointIdx, walkingPointIdx);
        if (d2 >= stepSquared) {
          // have enough walking distance, will include this point in new
          // skeleton
          selectedPointIdxes.push_back(walkingPointIdx);
          selectedParentPointIdxes.push_back(startPointIdx);

          // now we restart walking again
          startPointIdx = walkingPointIdx;
          // adjust the coordinate and radius by mean of nearest points;
          auto parentPointIdx = parents(walkingPointIdx);
          auto parentPoint = get_point(parentPointIdx);
          auto walkingPoint = xt::view(points, walkingPointIdx, xt::all());
          if (!is_terminal_point(walkingPointIdx)) {
            auto childPointIdx = childs(walkingPointIdx);
            auto childPoint = get_point(childPointIdx);
            // compute the mean of x,y,z,r to smooth the skeleton
            walkingPoint = (parentPoint + walkingPoint + childPoint) / 3;
          } else {
            // this point is the terminal point, do not have child
            walkingPoint = (parentPoint + walkingPoint) / 2;
          }
        }
      }
      // add current point
      selectedPointIdxes.push_back(walkingPointIdx);
      selectedParentPointIdxes.push_back(startPointIdx);

      // reach a branching/terminal point
      // add all children points as seeds
      // if reaching the terminal and there is no children
      // nothing will be added
      std::vector<int> childrenPointIdxes =
          get_children_point_indexes(walkingPointIdx);
      for (std::size_t i = 0; i < childrenPointIdxes.size(); i++) {
        seedPointIdxes.push_back(childrenPointIdxes[i]);
        seedPointParentIdxes.push_back(walkingPointIdx);
      }
    }

    assert(selectedPointIdxes.size() == selectedParentPointIdxes.size());
    assert(selectedPointIdxes.size() > 0);

    auto newPointNum = selectedPointIdxes.size();
    // std::cout<< "downsampled point number from "<< pointNum << " to " <<
    // newPointNum << std::endl;

    Attributes::shape_type newAttShape = {newPointNum, 4};
    Attributes newAtt = xt::zeros<int>(newAttShape) - 2;

    // find new point classes
    for (std::size_t i = 0; i < newPointNum; i++) {
      auto oldPointIdx = selectedPointIdxes[i];
      newAtt(i, 0) = att(oldPointIdx, 0);
    }

    // find new point parent index
    // the parent point index is pointing to old points
    // we need to update the point index to new points
    // update_parents(selectedParentPointIdxes, selectedPointIdxes);
    std::map<int, int> oldPointIdx2newPointIdx = {{-2, -2}};
    for (std::size_t newPointIdx = 0; newPointIdx < newPointNum;
         newPointIdx++) {
      auto oldPointIdx = selectedPointIdxes[newPointIdx];
      oldPointIdx2newPointIdx[oldPointIdx] = newPointIdx;
    }
    for (std::size_t i = 0; i < newPointNum; i++) {
      auto oldPointIdx = selectedParentPointIdxes[i];
      auto newPointIdx = oldPointIdx2newPointIdx[oldPointIdx];
      newAtt(i, 1) = newPointIdx;
    }

    // create new points
    Points::shape_type newPointsShape = {newPointNum, 4};
    Points newPoints = xt::zeros<float>(newPointsShape);
    for (std::size_t i = 0; i < newPointNum; i++) {
      auto oldPointIdx = selectedPointIdxes[i];
      auto oldPoint = xt::view(points, oldPointIdx, xt::all());
      newPoints(i, 0) = oldPoint(0);
      newPoints(i, 1) = oldPoint(1);
      newPoints(i, 2) = oldPoint(2);
      newPoints(i, 3) = oldPoint(3);
    }

    // find new first child and siblings.
    points = newPoints;
    attributes = newAtt;
    update_first_child_and_sibling();
    assert(any(xt::view(newAtt, xt::all(), 2) >= 0));
    return 0;
  }

  auto get_path_length() const {
    float pathLength = 0;
    auto parents = get_parents();
    for (std::size_t i = 0; i < get_point_num(); i++) {
      auto parentIdx = parents(i);
      if (parentIdx >= 0)
        pathLength += std::sqrt(squared_distance(i, parentIdx));
    }
    return pathLength;
  }

  std::string to_swc_str(const int precision = 3) const {
    std::ostringstream swc;
    swc << std::fixed;
    swc << std::setprecision(precision);
    auto classes = get_classes();
    auto parents = get_parents();
    auto pointNum = points.shape(0);

    // add some commented header
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    swc << "# Created using reneu at " << std::ctime(&now_t)
        << "# https://github.com/jingpengw/reneu \n";

    for (std::size_t pointIdx = 0; pointIdx < pointNum; pointIdx++) {
      // index, class, x, y, z, r, parent
      swc << pointIdx + 1 << " " << classes(pointIdx) << " " << std::fixed
          << points(pointIdx, 0) << " " << std::fixed << points(pointIdx, 1)
          << " " << std::fixed << points(pointIdx, 2) << " " << std::fixed
          << points(pointIdx, 3) << " " << parents(pointIdx) + 1 << "\n";
    }

    return swc.str();
  }

  int write_swc(std::string file_name, const int precision = 3) const {
    std::ofstream myfile(file_name, std::ios::out);
    myfile.precision(precision);

    if (myfile.is_open()) {
      auto classes = get_classes();
      auto parents = get_parents();
      auto pointNum = points.shape(0);

      // add some commented header
      auto now = std::chrono::system_clock::now();
      auto now_t = std::chrono::system_clock::to_time_t(now);
      myfile << "# Created using reneu at " << std::ctime(&now_t)
             << "# https://github.com/jingpengw/reneu \n";

      for (std::size_t pointIdx = 0; pointIdx < pointNum; pointIdx++) {
        // index, class, x, y, z, r, parent
        myfile << pointIdx + 1 << " " << classes(pointIdx) << " " << std::fixed
               << points(pointIdx, 0) << " " << std::fixed
               << points(pointIdx, 1) << " " << std::fixed
               << points(pointIdx, 2) << " " << std::fixed
               << points(pointIdx, 3) << " " << parents(pointIdx) + 1 << "\n";
      }
    } else {
      std::cout << "can not open file: " << file_name << std::endl;
    }
    return 0;
  }
};  // Skeleton class
}  // namespace reneu
