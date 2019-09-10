#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <tuple>
#include <ctime>
#include <chrono>
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
#include "xiuli/utils/string.hpp"
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

using namespace xiuli::utils;
using namespace xt::placeholders;  // required for `_` to work

using NodesType = xt::xtensor<float, 2>;
using AttributesType = xt::xtensor<int, 2>;

// use the c++17 nested namespace
namespace xiuli::neuron{

class Skeleton{

private:
    // node array (N x 4), the rows are nodes, the columns are x,y,z,r
    // normally the type is float
    NodesType nodes;
    
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
    AttributesType attributes;

    inline auto get_classes(){
        return xt::view(attributes, xt::all(), 0);
    }

    inline auto get_parents(){
        return xt::view(attributes, xt::all(), 1);
    }

    inline auto get_childs(){
        return xt::view(attributes, xt::all(), 2);
    }

    inline auto get_siblings(){
        return xt::view(attributes, xt::all(), 3);
    }

    template<typename Ti>
    inline auto get_node( Ti nodeIdx ){
        return xt::view(nodes, nodeIdx, xt::all());
    } 

    inline auto squared_distance(int idx1, int idx2){
        auto node1 = get_node(idx1);
        auto node2 = get_node(idx2);
        return  (node1(0) - node2(0)) * (node1(0) - node2(0)) + 
                (node1(1) - node2(1)) * (node1(1) - node2(1)) +
                (node1(2) - node2(2)) * (node1(2) - node2(2));
    }

    template<typename Tn>
    inline auto initialize_nodes_and_attributes(Tn nodeNum){
        // create nodes and attributes
        NodesType::shape_type nodesShape = {nodeNum, 4};
        nodes = xt::zeros<float>( nodesShape );
        AttributesType::shape_type attributesShape = {nodeNum, 4};
        // root node id is -2 rather than -1.
        attributes = xt::zeros<int>( attributesShape ) - 2;
    }
    
    auto update_first_child_and_sibling(){
        auto parents = get_parents();
        auto childs = get_childs();
        auto siblings = get_siblings();

        auto nodeNum = parents.size();
        // the columns are class, parents, fist child, sibling

        // clean up the child and siblings
        childs = -2;
        siblings = -2;

        // update childs
        for (std::size_t nodeIdx = 0; nodeIdx<nodeNum; nodeIdx++){
            auto parentNodeIdx = parents( nodeIdx );
            if (parentNodeIdx >= 0)
                childs( parentNodeIdx ) = nodeIdx;
        }

        // update siblings
        for (int nodeIdx = 0; nodeIdx < int(nodeNum); nodeIdx++){
            auto parentNodeIdx = parents( nodeIdx );
            if (parentNodeIdx >= 0){
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
        }
        return 0;
    }
 
public:
    virtual ~Skeleton() = default;

    Skeleton( xt::pytensor<float, 2> nodes_, xt::pytensor<int, 2> attributes_):
        nodes( nodes_ ), attributes( attributes_ ){
            update_first_child_and_sibling();
        }

    Skeleton( xt::pytensor<float, 2> swcArray ){
        // this input array should follow the format of swc file
        assert(swcArray.shape(1) == 7);
        auto nodeNum = swcArray.shape(0);
        initialize_nodes_and_attributes( nodeNum );
        
        auto xids = xt::view(swcArray, xt::all(), 0);
        auto newIdx2oldIdx = xt::argsort(xids);

        // construct the nodes and attributes
        initialize_nodes_and_attributes(nodeNum);
        
        for (std::size_t i = 0; i<nodeNum; i++){
            // we do sort here!
            auto oldIdx = newIdx2oldIdx( i );
            attributes(i, 0) = swcArray( oldIdx, 1 );
            nodes(i, 0) = swcArray( oldIdx, 2 );
            nodes(i, 1) = swcArray( oldIdx, 3 );
            nodes(i, 2) = swcArray( oldIdx, 4 );
            nodes(i, 3) = swcArray( oldIdx, 5 );
            // swc node id is 1-based
            attributes(i, 1) = swcArray( oldIdx, 6 ) - 1;
        }
        update_first_child_and_sibling();
    }

    Skeleton( std::string file_name ){
        std::cerr<< "this function is pretty slow and should be speed up using memory map!" << std::endl;
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
                if (parts.size()==7 && parts[0][0] != '#'){
                    ids.push_back( std::stoi( parts[0] ) );
                    classes.push_back( std::stoi( parts[1] ) );
                    xs.push_back( std::stof( parts[2] ) );
                    ys.push_back( std::stof( parts[3] ) );
                    zs.push_back( std::stof( parts[4] ) );
                    rs.push_back( std::stof( parts[5] ) );
                    // swc is 1-based, and we are 0-based here.
                    parents.push_back( std::stoi( parts[6] ) - 1);
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

            // construct the nodes and attributes
            initialize_nodes_and_attributes(nodeNum);
            
            for (std::size_t i = 0; i<nodeNum; i++){
                // we do sort here!
                auto oldIdx = newIdx2oldIdx( i );
                nodes(i, 0) = xs[ oldIdx ];
                nodes(i, 1) = ys[ oldIdx ];
                nodes(i, 2) = zs[ oldIdx ];
                nodes(i, 3) = rs[ oldIdx ];
                attributes(i, 0) = classes[ oldIdx ];
                // swc node id is 1-based
                attributes(i, 1) = parents[ oldIdx ];
            }
            update_first_child_and_sibling();

            auto childs = get_childs();
            assert( xt::any( childs >= 0 ) );
        }else{
            std::cout << "can not open file: " << file_name << std::endl;
        }
    }
    
    inline auto get_nodes(){
        return nodes;
    }

    inline auto get_attributes(){
        return attributes;
    }

    inline bool is_root_node(int nodeIdx){
        return attributes(nodeIdx, 1) < 0;
    }

    inline bool is_terminal_node(int nodeIdx){
        return attributes(nodeIdx, 2 ) < 0;
    }

    inline bool is_branching_node(int nodeIdx){
        auto childNodeIdx = attributes(nodeIdx, 2 );
        // if child have sibling, then this is a branching node
        return  attributes(childNodeIdx, 3) > 0;
    }

    inline auto get_node_num(){
        return nodes.shape(0);
    }

    auto get_edge_num(){
        auto nodeNum = get_node_num();
        auto parents = get_parents();
        std::size_t edgeNum = 0;
        for (std::size_t i = 0; i<nodeNum; i++){
            if (parents( i ) >= 0){
                edgeNum += 1;
            }
        }
        return edgeNum;
    }

    auto get_edges(){
        auto edgeNum = get_edge_num();
        xt::xtensor<std::uint32_t, 2>::shape_type edgesShape = {edgeNum, 2};
        xt::xtensor<std::uint32_t, 2> edges = xt::zeros<std::uint32_t>( edgesShape );

        auto parents = get_parents();
        std::size_t edgeIdx = 0;
        for (std::size_t i = 0; i<get_node_num(); i++){
            auto parentIdx = parents( i );
            if (parentIdx >= 0 ){
                edges( edgeIdx, 0 ) = i;
                edges( edgeIdx, 1 ) = parentIdx;
                edgeIdx += 1;
            }
        }
        return edges;
    }

    std::vector<int> get_children_node_indexes(int nodeIdx){
        auto childs = get_childs();
        auto siblings = get_siblings();

        std::vector<int> childrenNodeIdxes = {};
        auto childNodeIdx = childs( nodeIdx );
        while (childNodeIdx >= 0){
            childrenNodeIdxes.push_back( childNodeIdx );
            childNodeIdx = siblings( childNodeIdx );
        } 
        return childrenNodeIdxes;
    }

    auto downsample(float step){
        auto att = attributes;

        auto classes = get_classes();
        auto parents = get_parents();
        auto childs = get_childs();
        auto siblings = get_siblings();

        auto nodeNum = get_node_num();

        //std::cout<< "downsampling step: " << step <<std::endl;
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
        // initialize the seed node parent indexes
        std::vector<int> seedNodeParentIdxes = {};
        seedNodeParentIdxes.assign( seedNodeIdxes.size(), -2 );
        
        // we select some nodes out as new neuron
        // this selection should include all the seed node indexes
        std::vector<int> selectedNodeIdxes = {};
        // we need to update the parents when we add new selected nodes
        std::vector<int> selectedParentNodeIdxes = {};

        while( !seedNodeIdxes.empty() ){
            auto startNodeIdx = seedNodeIdxes.back();
            auto startNodeParentIdx = seedNodeParentIdxes.back();
            seedNodeIdxes.pop_back();
            seedNodeParentIdxes.pop_back();
            selectedNodeIdxes.push_back( startNodeIdx );
            selectedParentNodeIdxes.push_back( startNodeParentIdx );
           
            // the start of measurement
            auto walkingNodeIdx = startNodeIdx;

            // walk through a segment
            while (!is_branching_node(walkingNodeIdx) && 
                    !is_terminal_node(walkingNodeIdx)){
                // walk to next node
                walkingNodeIdx = childs[ walkingNodeIdx ];
                // use squared distance to avoid sqrt computation.
                float d2 = squared_distance(startNodeIdx, walkingNodeIdx);
                if (d2 >= stepSquared){
                    // have enough walking distance, will include this node in new skeleton
                    selectedNodeIdxes.push_back( walkingNodeIdx );
                    selectedParentNodeIdxes.push_back( startNodeIdx );

                    // now we restart walking again
                    startNodeIdx = walkingNodeIdx;
                    // adjust the coordinate and radius by mean of nearest nodes;
                    auto parentNodeIdx = parents( walkingNodeIdx );
                    auto parentNode = get_node(parentNodeIdx);
                    auto walkingNode = get_node(walkingNodeIdx);
                    if (!is_terminal_node(walkingNodeIdx)){
                        auto childNodeIdx = childs( walkingNodeIdx );
                        auto childNode = get_node(childNodeIdx);
                        // compute the mean of x,y,z,r to smooth the skeleton
                        walkingNode = (parentNode + walkingNode + childNode) / 3;
                    } else {
                        // this node is the terminal node, do not have child
                        walkingNode = (parentNode + walkingNode) / 2;
                    }
                }
            }
            // add current node 
            selectedNodeIdxes.push_back( walkingNodeIdx );
            selectedParentNodeIdxes.push_back( startNodeIdx );

            // reach a branching/terminal node 
            // add all children nodes as seeds
            // if reaching the terminal and there is no children
            // nothing will be added
            std::vector<int> childrenNodeIdxes = get_children_node_indexes(walkingNodeIdx);
            for (std::size_t i=0; i<childrenNodeIdxes.size(); i++){
                seedNodeIdxes.push_back( childrenNodeIdxes[i] );
                seedNodeParentIdxes.push_back( walkingNodeIdx );
            }
        }

        assert( selectedNodeIdxes.size() == selectedParentNodeIdxes.size() );
        assert( selectedNodeIdxes.size() > 0 );

        auto newNodeNum = selectedNodeIdxes.size();
        //std::cout<< "downsampled node number from "<< nodeNum << " to " << newNodeNum << std::endl;

        AttributesType::shape_type newAttShape = {newNodeNum, 4};
        AttributesType newAtt = xt::zeros<int>(newAttShape) - 2;

        // find new node classes
        for (std::size_t i=0; i<newNodeNum; i++){
            auto oldNodeIdx = selectedNodeIdxes[ i ];
            newAtt(i, 0) = att(oldNodeIdx, 0);
        }

        // find new node parent index
        // the parent node index is pointing to old nodes
        // we need to update the node index to new nodes
        // update_parents(selectedParentNodeIdxes, selectedNodeIdxes);
        std::map<int, int> oldNodeIdx2newNodeIdx = {{-2, -2}};
        for (std::size_t newNodeIdx=0; newNodeIdx<newNodeNum; newNodeIdx++){
            auto oldNodeIdx = selectedNodeIdxes[ newNodeIdx ];
            oldNodeIdx2newNodeIdx[ oldNodeIdx ] = newNodeIdx;
        }
        for (std::size_t i = 0; i<newNodeNum; i++){
            auto oldNodeIdx = selectedParentNodeIdxes[i];
            auto newNodeIdx = oldNodeIdx2newNodeIdx[ oldNodeIdx ];
            newAtt(i, 1) = newNodeIdx;
        }

        // create new nodes
        NodesType::shape_type newNodesShape = {newNodeNum, 4};
        NodesType newNodes = xt::zeros<float>( newNodesShape );
        for (std::size_t i = 0; i<newNodeNum; i++){
            auto oldNodeIdx = selectedNodeIdxes[i];
            auto oldNode = xt::view(nodes, oldNodeIdx, xt::all());
            newNodes(i, 0) = oldNode(0);
            newNodes(i, 1) = oldNode(1);
            newNodes(i, 2) = oldNode(2);
            newNodes(i, 3) = oldNode(3);
        }

        // find new first child and siblings.
        nodes = newNodes;
        attributes = newAtt;
        update_first_child_and_sibling();
        assert( any(xt::view(newAtt, xt::all(), 2) >= 0 ) );
        return 0;
    }

    auto get_path_length(){
        float pathLength = 0;
        auto parents = get_parents();
        for (std::size_t i = 0; i<get_node_num(); i++){
            auto parentIdx = parents( i );
            pathLength += std::sqrt( squared_distance( i, parentIdx ) ); 
        }
        return pathLength;
    }
        
    int write_swc( std::string file_name, const int precision = 3){
        std::ofstream myfile (file_name, std::ios::out);
        myfile.precision(precision);

        if (myfile.is_open()){       
            auto classes = get_classes();
            auto parents = get_parents();
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