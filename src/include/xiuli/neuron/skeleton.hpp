#pragma once

#include "xiuli/utils/string.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <tuple>
#include <ctime>
#include <chrono>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
//#define FORCE_IMPORT_ARRAY
//#include "xtensor-python/pytensor.hpp"     // Numpy bindings

namespace py = pybind11;
using namespace xt::placeholders;  // required for `_` to work

using namespace xiuli::utils;

// use the c++17 nested namespace
namespace xiuli::neuron{

class Skeleton{

private:
    // node array (N x 4), the rows are nodes, the columns are x,y,z,r
    // normally the type is float
    xt::xtensor<float, 2> nodes;
    
    // attributes of nodes (N x 4), 
    // the columns are node type, parent, first child, sibling
    // the node types are defined in SWC format
    // 0 - undefined
    // 1 - soma
    // 2 - axon
    // 3 - (basal) dendrite
    // 4 - apical dendrite
    // 5 - fork point
    // 6 - end point
    // 7 - custom 
    // normally the type is int
    xt::xtensor<int, 2> attributes;
    
    auto get_classes(){
        return xt::view(this->attributes, xt::all(), 0);
    }

    auto get_parents(){
        return xt::view(this->attributes, xt::all(), 1);
    }

    auto get_childs(){
        return xt::view(this->attributes, xt::all(), 2);
    }

    auto get_siblings(){
        return xt::view(this->attributes, xt::all(), 3);
    }

    auto update_first_child_and_sibling(){
        auto parents = this->get_parents();
        auto childs = this->get_childs();
        auto siblings = this->get_siblings();

        auto nodeNum = parents.size();
        // the columns are class, parents, fist child, sibling

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
    
    template<typename T>
    auto xtensor_to_numpy(xt::xtensor<T, 2>& tensor){
        return py::buffer_info(tensor.data(), sizeof(T), py::format_descriptor<T>::format(), 2,
                    {tensor.shape(0), tensor.shape(1)}, {sizeof(T) * tensor.shape(1), sizeof(T)});

    }

    float squared_distance(int idx1, int idx2){
        auto node1 = xt::view(this->nodes, idx1, xt::range(_, 3));
        auto node2 = xt::view(this->nodes, idx2, xt::range(_, 3));
        return (node1(0) - node2(0)) * (node1(0) - node2(0)) + 
            (node1(1) - node2(1)) * (node1(1) - node2(1)) +
            (node1(2) - node2(2)) * (node1(2) - node2(2));
    }

public:
    virtual ~Skeleton() = default;

    Skeleton( xt::xtensor<float, 2> nodes_, xt::xtensor<int, 2> attributes_):
        nodes( nodes_ ), attributes( attributes_ ){}

    Skeleton( xt::xtensor<double, 2> data ){
        std::logic_error("Function not yet implemented");
    }

    Skeleton( std::string file_name ){
        std::ifstream myfile(file_name, std::ios::in);

        if (myfile.is_open()){
            std::string line;
            std::regex ws_re("\\s+"); // whitespace
            
            // initialize the data vectors
            std::vector<int> ids = {};
            std::vector<int> classes = {};
            std::vector<float> xs = {};
            std::vector<float> ys = {};
            std::vector<float> zs = {};
            std::vector<float> rs = {};
            std::vector<int> parents = {};

            while ( std::getline(myfile, line) ){
                std::vector<std::string> parts = xiuli::utils::split(line, "\\s+"_re);
                if (!parts.empty() && parts[0][0] != '#'){
                    ids.push_back( std::stoi( parts[0] ) );
                    classes.push_back( std::stoi( parts[1] ) );
                    xs.push_back( std::stof( parts[2] ) );
                    ys.push_back( std::stof( parts[3] ) );
                    zs.push_back( std::stof( parts[4] ) );
                    rs.push_back( std::stof( parts[5] ) );
                    parents.push_back( std::stoi( parts[6] ) );
                }                
            }
            assert(ids.size() == parents.size());
            myfile.close();
            
            // the node index could be unordered
            // we should order them and will drop the ids column
            auto nodeNum = ids.size();
            std::vector<std::size_t> shape = { nodeNum, };
            auto xids = xt::adapt(ids, shape);
            auto newIdx2oldIdx = xt::argsort(xids);

            // create nodes and attributes
            xt::xtensor<float, 2>::shape_type nodesShape = {nodeNum, 4};
            nodes = xt::zeros<float>( nodesShape );
            xt::xtensor<int, 2>::shape_type attributesShape = {nodeNum, 4};
            attributes = xt::zeros<int>( attributesShape ) - 1;
            for (std::size_t i = 0; i<nodeNum; i++){
                // we do sort here!
                auto oldIdx = newIdx2oldIdx( i );
                nodes(i, 0) = xs[ oldIdx ];
                nodes(i, 1) = ys[ oldIdx ];
                nodes(i, 2) = zs[ oldIdx ];
                nodes(i, 3) = rs[ oldIdx ];
                attributes(i, 0) = classes[i];
                // swc node id is 1-based
                attributes(i, 1) = parents[i] - 1;
            }
            this->update_first_child_and_sibling();

            auto childs = this->get_childs();
            assert( xt::any( childs >= 0 ) );
        }else{
            std::cout << "can not open file: " << file_name << std::endl;
        }
    }


//    auto get_nodes_as_numpy_array(){
//        xt::pytensor<float, 2> pynodes = this->nodes;
//        return pynodes;
//    }
//
//    auto get_attributes_as_numpy_array(){
//        xt::pytensor<int, 2> pyattributes = this->attributes;
//        return pyattributes;
//    }

    auto get_nodes(){
        return this->nodes;
    }

    auto get_attributes(){
        return this->attributes;
    }

    bool is_root_node(int nodeIdx){
        return this->attributes(nodeIdx, 1) < 0;
    }

    bool is_terminal_node(int nodeIdx){
        return this->attributes(nodeIdx, 2 ) < 0;
    }

    bool is_branching_node(int nodeIdx){
        auto att = this->attributes;
        auto childNodeIdx = att(nodeIdx, 2 );
        // if child have sibling, then this is a branching node
        return  att(childNodeIdx, 3) > 0;
    }

    auto get_node_num(){
        return this->nodes.shape(0);
    }

    std::vector<int> get_children_node_indexes(int nodeIdx){
        auto att = this->attributes;
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

    auto downsample(float step){
        auto nodes = this->nodes;
        auto att = this->attributes;

        auto classes = this->get_classes();
        auto parents = this->get_parents();
        auto childs = this->get_childs();
        auto siblings = this->get_siblings();

        auto nodeNum = this->get_node_num();
        auto stepSquared = step * step;

        // start from the root nodes
        std::vector<int> rootNodeIdxes = {};
        for (std::size_t nodeIdx=0; nodeIdx<nodeNum; nodeIdx++){
            if (is_root_node(nodeIdx)){
                rootNodeIdxes.push_back( nodeIdx );
            }
        }
        assert( !rootNodeIdxes.empty() );

        auto seedNodeIdxes = rootNodeIdxes;

        // we select some nodes out as new neuron
        // this selection should include all the seed node indexes
        std::vector<int> selectedNodeIdxes = {};
        // we need to update the parents when we add new selected nodes
        std::vector<int> selectedParentNodeIdxes = {};

        while( !seedNodeIdxes.empty() ){
            auto walkingNodeIdx = seedNodeIdxes.back();
            seedNodeIdxes.pop_back();
            
            // the start of measurement
            auto startNodeIdx = walkingNodeIdx;

            // walk through a segment
            while (!is_branching_node(walkingNodeIdx) && 
                    !is_terminal_node(walkingNodeIdx)){
                // walk to next node
                walkingNodeIdx = childs[ walkingNodeIdx ];
                // use squared distance to avoid sqrt computation.
                float d2 = squared_distance(startNodeIdx, walkingNodeIdx);
                if (d2 > stepSquared){
                    // have enough walking distance, will include this node in new skeleton
                    selectedNodeIdxes.push_back( walkingNodeIdx );
                    selectedParentNodeIdxes.push_back( startNodeIdx );

                    // now we restart walking again
                    startNodeIdx = walkingNodeIdx;
                    // adjust the coordinate and radius by mean of nearest nodes;
                    auto parentNodeIdx = parents( walkingNodeIdx );
                    auto parentNode = xt::view(nodes, parentNodeIdx, xt::all());
                    auto walkingNode = xt::view(nodes, walkingNodeIdx, xt::all());
                    if (!is_terminal_node(walkingNodeIdx)){
                        auto childNodeIdx = childs( walkingNodeIdx );
                        auto childNode = xt::view(nodes, childNodeIdx, xt::all());
                        // compute the mean of x,y,z,r to smooth the skeleton
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
            std::vector<int> childrenNodeIdxes = this->get_children_node_indexes(walkingNodeIdx);
            for (std::size_t i=0; i<childrenNodeIdxes.size(); i++){
                seedNodeIdxes.push_back( childrenNodeIdxes[i] );
            }
        }

        assert( selectedNodeIdxes.size() == selectedParentNodeIdxes.size() );
        assert( selectedNodeIdxes.size() > 0 );

        auto newNodeNum = selectedNodeIdxes.size();
        std::cout<< "node number after downsampling: " << newNodeNum << std::endl;

        xt::xtensor<int, 2> newAtt = xt::zeros<int>({newNodeNum}) - 1;

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

        // create new nodes
        xt::xtensor<float, 2>::shape_type newNodesShape = {newNodeNum, 4};
        xt::xtensor<float, 2> newNodes = xt::zeros<float>( newNodesShape );
        for (std::size_t i = 0; i<newNodeNum; i++){
            auto oldNodeIdx = selectedNodeIdxes[i];
            auto oldNode = xt::view(nodes, oldNodeIdx, xt::all());
            newNodes(i, 0) = oldNode(0);
            newNodes(i, 1) = oldNode(1);
            newNodes(i, 2) = oldNode(2);
            newNodes(i, 3) = oldNode(3);
        }

        std::cout<< "number of new nodes: " << newNodes.shape(0) << std::endl;
        //auto pyNewNodes = xtensor_to_numpy<float>(newNodes);
        //auto pyNewAtt = xtensor_to_numpy<int>(newAtt);
        //auto pyNewNodes = py::array_t<float>(newNodes);
        //auto pyNewAtt = py::array_t<int>(newAtt);
        //return py::make_tuple(pyNewNodes, pyNewAtt);
        //return py::make_tuple(newNodes, newAtt);
        //return py::make_tuple( newNodes.python_array(), newAtt.python_array() );
        
        // find new first child and siblings.
        this->nodes = newNodes;
        this->attributes = newAtt;
        this->update_first_child_and_sibling();
        assert( any(xt::view(newAtt, xt::all(), 2) >= 0 ) );
        return 0;
    }
        
    int write_swc( std::string file_name, const int precision = 3){
        std::ofstream myfile (file_name, std::ios::out);
        myfile.precision(precision);

        if (myfile.is_open()){       
            auto classes = this->get_classes();
            auto parents = this->get_parents();
            auto nodes = this->nodes; 
            auto nodeNum = nodes.shape(0);

            // add some commented header
            auto now = std::chrono::system_clock::now();
            auto now_t = std::chrono::system_clock::to_time_t( now );
            myfile << "# Created using reneu at " << std::ctime(&now_t) << 
                "# https://github.com/jingpengw/reneu \n";

            for (std::size_t nodeIdx = 0; nodeIdx<nodeNum; nodeIdx++ ){
                // index, class, x, y, z, r, parent
                myfile  << nodeIdx+1 << " " << classes(nodeIdx) << " " 
                        << std::fixed << nodes(nodeIdx, 0) << " " << std::fixed << nodes(nodeIdx, 1) << " " 
                        << std::fixed << nodes(nodeIdx, 2) << " " << std::fixed << nodes(nodeIdx, 3) << " " 
                        << parents(nodeIdx)+1 << "\n";
            }
        }else{
            std::cout << "can not open file: " << file_name <<std::endl;
        }
        return 0;
    }
}; // Skeleton class
} // namespace