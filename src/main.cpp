#include <string>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <tuple>
#include <ctime>
#include <chrono>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

namespace py = pybind11;
using namespace xt::placeholders;  // required for `_` to work

template<typename T>
auto pytensor_to_numpy(xt::pytensor<T, 2>& tensor){
    return py::buffer_info(tensor.data(), sizeof(T), py::format_descriptor<T>::format(), 2,
                {tensor.shape(0), tensor.shape(1)}, {sizeof(T) * tensor.shape(1), sizeof(T)});

}

float squared_distance(xt::pytensor<float, 2>& nodes, int idx1, int idx2){
    auto node1 = xt::view(nodes, idx1, xt::range(_, 3));
    auto node2 = xt::view(nodes, idx2, xt::range(_, 3));
    return (node1(0) - node2(0)) * (node1(0) - node2(0)) + 
           (node1(1) - node2(1)) * (node1(1) - node2(1)) +
           (node1(2) - node2(2)) * (node1(2) - node2(2));
}

auto split_attributes(xt::pytensor<int, 2>& att){
    auto classes = xt::view(att, xt::all(), 0);
    auto parents = xt::view(att, xt::all(), 1);
    auto childs = xt::view(att, xt::all(), 2);
    auto siblings = xt::view(att, xt::all(), 3);
    // this only works in C++17
    return std::make_tuple(classes, parents, childs, siblings);
}

template<typename Ta>
auto update_first_child_and_sibling(xt::pytensor<Ta, 2>& att){
    auto [classes, parents, childs, siblings] = split_attributes(att);
    auto nodeNum = parents.size();
    // the columns are class, parents, fist child, sibling
    assert( att.shape(1) == 4 );

    // clean up the child and siblings
    for (std::size_t nodeIdx = 0; nodeIdx<nodeNum; nodeIdx++){
        childs(nodeIdx) = -1;
        siblings(nodeIdx) = -1;
    }

    // update childs
    for (std::size_t nodeIdx = 0; nodeIdx<nodeNum; nodeIdx++){
        auto parentNodeIdx = parents( nodeIdx );
        childs( parentNodeIdx ) = nodeIdx;
    }

    // update siblings
    for (int nodeIdx = 0; nodeIdx < int(nodeNum); nodeIdx++){
        auto parentNodeIdx = parents( nodeIdx );
        auto currentSiblingNodeIdx = childs( parentNodeIdx );
        assert( currentSiblingNodeIdx >= 0 );
        if (currentSiblingNodeIdx != nodeIdx){
            // look for an empty sibling spot to fill in
            auto nextSiblingNodeIdx = siblings(currentSiblingNodeIdx);
            while (nextSiblingNodeIdx >= 0){
                currentSiblingNodeIdx = nextSiblingNodeIdx;
                // move forward to look for empty sibling spot
                nextSiblingNodeIdx = siblings( currentSiblingNodeIdx );
            }
            // finally, we should find an empty spot
            siblings( currentSiblingNodeIdx ) = nodeIdx; 
        }
    }
    return 0;
}

bool is_root_node(xt::pytensor<int, 2>& att, int nodeIdx){
    return att(nodeIdx, 1) < 0;
}

bool is_terminal_node(xt::pytensor<int, 2>& att, int nodeIdx){
    return att(nodeIdx, 2 ) < 0;
}

bool is_branching_node(xt::pytensor<int, 2>& att, int nodeIdx){
    auto childNodeIdx = att(nodeIdx, 2 );
    // if child have sibling, then this is a branching node
    return  att(childNodeIdx, 3) > 0;
}

std::vector<int> get_children_node_indexes(xt::pytensor<int, 2>& att, int nodeIdx){
    auto childs = xt::view(att, xt::all(), 2);
    auto siblings = xt::view(att, xt::all(), 3);

    std::vector<int> childrenNodeIdxes = {};
    auto childNodeIdx = childs( nodeIdx );
    while (childNodeIdx >= 0){
        childrenNodeIdxes.push_back( childNodeIdx );
        childNodeIdx = siblings( childNodeIdx );
    } 
    return childrenNodeIdxes;
}

auto downsample(xt::pytensor<float, 2>& nodes, xt::pytensor<int, 2>& att, float step){
    std::cout<< "number of input nodes: " << nodes.shape(0) << std::endl;
    // this only works in C++17
    auto [classes, parents, childs, siblings] = split_attributes(att); 

    auto stepSquared = step * step;

    // start from the root nodes
    std::vector<int> rootNodeIdxes = {};
    for (std::size_t nodeIdx=0; nodeIdx<parents.shape(0); nodeIdx++){
        if (is_root_node(att, nodeIdx)){
            rootNodeIdxes.push_back( nodeIdx );
        }
    }
    auto seedNodeIdxes = rootNodeIdxes;
    // initialize the seed node parent indexes
    std::vector<int> seedNodeParentIdxes = {};
    seedNodeParentIdxes.assign( seedNodeIdxes.size(), -1 );

    // we select some nodes out as new neuron
    // this selection should include all the seed node indexes
    std::vector<int> selectedNodeIdxes = {};
    // we need to update the parents when we add new selected nodes
    std::vector<int> selectedParentNodeIdxes = {};

    while( !seedNodeIdxes.empty() ){
        auto walkingNodeIdx = seedNodeIdxes.back();
        seedNodeIdxes.pop_back();
        auto parentNodeIdx = seedNodeParentIdxes.back();
        seedNodeParentIdxes.pop_back();

        auto startNodeIdx = walkingNodeIdx;

        // walk through a segment
        while (!is_branching_node(att, walkingNodeIdx) && 
                !is_terminal_node(att, walkingNodeIdx)){
            // walk to next node
            walkingNodeIdx = childs[ walkingNodeIdx ];
            // use squared distance to avoid sqrt computation.
            float d2 = squared_distance(nodes, startNodeIdx, walkingNodeIdx);
            if (d2 < stepSquared){
                // keep walking and continue search
                // current node becomes parent and will give birth of children
                parentNodeIdx = walkingNodeIdx;
            } else {
                // have enough walking distance, will include this node in new skeleton
                selectedNodeIdxes.push_back( walkingNodeIdx );
                selectedParentNodeIdxes.push_back( startNodeIdx );

                // now we restart walking again
                startNodeIdx = walkingNodeIdx;
                // adjust the coordinate and radius by mean of nearest nodes;
                auto parentNode = xt::view(nodes, parentNodeIdx, xt::all());
                auto walkingNode = xt::view(nodes, walkingNodeIdx, xt::all());
                if (!is_branching_node(att, walkingNodeIdx) && 
                    !is_terminal_node(att, walkingNodeIdx)){
                    auto childNodeIdx = childs( walkingNodeIdx );
                    auto childNode = xt::view(nodes, childNodeIdx, xt::all());
                    // compute the mean of r,z,y,x to smooth the skeleton
                    walkingNode = (parentNode + walkingNode + childNode) / 3;
                } else {
                    // this node is the terminal node, do not have child
                    walkingNode = (parentNode + walkingNode) / 2;
                }
            }
        }

        // add all children nodes as seeds
        // if reaching the terminal and there is no children
        // nothing will be added
        std::vector<int> childrenNodeIdxes = get_children_node_indexes(att, walkingNodeIdx);
        for (std::size_t i=0; i<childrenNodeIdxes.size(); i++){
            seedNodeIdxes.push_back( childrenNodeIdxes[i] );
            seedNodeParentIdxes.push_back( parentNodeIdx );
        }
    }

    assert( selectedNodeIdxes.size() == selectedParentNodeIdxes.size() );
    auto newNodeNum = selectedNodeIdxes.size();
    xt::pytensor<int, 2> newAtt = xt::zeros<int>({newNodeNum}) - 1;

    // find new node classes
    for (std::size_t i=0; i<newNodeNum; i++){
        auto oldNodeIdx = selectedNodeIdxes[ i ];
        newAtt(i, 0) = att(oldNodeIdx, 0);
    }

    // find new node parent index
    // the parent node index is pointing to old nodes
    // we need to update the node index to new nodes
    // update_parents(selectedParentNodeIdxes, selectedNodeIdxes);
    std::map<int, int> oldNodeIdx2newNodeIdx = {};
    for (std::size_t newNodeIdx=0; newNodeIdx<newNodeNum; newNodeIdx++){
        auto oldNodeIdx = selectedNodeIdxes[ newNodeIdx ];
        oldNodeIdx2newNodeIdx[ oldNodeIdx ] = newNodeIdx;
    }
    for (std::size_t i = 0; i<newNodeNum; i++){
        auto oldNodeIdx = selectedParentNodeIdxes[i];
        auto newNodeIdx = oldNodeIdx2newNodeIdx[ oldNodeIdx ];
        selectedParentNodeIdxes[i] = newNodeIdx;
    }

    // find new first child and siblings.
    update_first_child_and_sibling(newAtt);

    // create new nodes
    xt::pytensor<float, 2>::shape_type newNodesShape = {newNodeNum, 4};
    xt::pytensor<float, 2> newNodes = xt::zeros<float>( newNodesShape );
    for (std::size_t i = 0; i<newNodeNum; i++){
        auto oldNodeIdx = selectedNodeIdxes[i];
        auto oldNode = xt::view(nodes, oldNodeIdx, xt::all());
        auto newNode = xt::view(newNodes, i, xt::all());
        for (std::size_t k=0; k<oldNode.size(); k++){
            newNode(k) = oldNode(k);
        }
    }

    std::cout<< "shape of nodes: " << newNodes.shape(0) << std::endl;
    std::cout<< "shape of attributes: " << newAtt.shape(0) << std::endl;
    //auto pyNewNodes = pytensor_to_numpy<float>(newNodes);
    //auto pyNewAtt = pytensor_to_numpy<int>(newAtt);
    //auto pyNewNodes = py::array_t<float>(newNodes);
    //auto pyNewAtt = py::array_t<int>(newAtt);
    //return py::make_tuple(pyNewNodes, pyNewAtt);
    return py::make_tuple(newNodes, newAtt);
    //return py::make_tuple( newNodes.python_array(), newAtt.python_array() );
}

std::string skeleton_to_swc_string( xt::pytensor<float, 2> nodes, xt::pytensor<int, 2> att,
        const int precision = 3){
    std::stringstream ss;
    ss.precision(precision);
    auto nodeNum = nodes.shape(0);

    // add some commented header
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t( now );
    ss << "# Created using reneu at " << std::ctime(&now_t) << 
        "# https://github.com/jingpengw/reneu \n";

    for (std::size_t nodeIdx = 0; nodeIdx<nodeNum; nodeIdx++ ){
        // index, class, x, y, z, r, parent
        ss << nodeIdx+1 << " " << att(nodeIdx, 0) << " " 
            << std::fixed << nodes(nodeIdx, 0) << " " << std::fixed << nodes(nodeIdx, 1) << " " 
            << std::fixed << nodes(nodeIdx, 2) << " " << std::fixed << nodes(nodeIdx, 3) << " " 
            << att(nodeIdx, 1)+1 << "\n";
    }
    return ss.str();
}

PYBIND11_MODULE(libreneu, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("update_first_child_and_sibling", &update_first_child_and_sibling<int>, R"pbdoc(
        update first child and sibling
    )pbdoc");

    m.def("downsample", &downsample, R"pbdoc(
        downsample
    )pbdoc");

    m.def("skeleton_to_swc_string", &skeleton_to_swc_string, R"pbdoc(
        skeleton_to_swc_string
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
