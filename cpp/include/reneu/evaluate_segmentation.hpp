#pragma once

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "type_aliase.hpp"

namespace reneu{

namespace bu = boost::numeric::ublas;


// A contingency table or confusion table to compute rand error (RE)
// and variation of information (VI).
// Reference:
// Meilă, Marina. "Comparing clusterings—an information based distance." Journal of multivariate analysis 98.5 (2007): 873-895.
class ContingencyTable{
private:
    bu::compressed_matrix<uint32_t> table;

// relabel segmentation id from 0 to K
auto remap_seg_id(Segmentation& seg){
    // keep the segid 0 unchanged since it is the background
    std::map<segid_t, segid_t> lookup_table = {{0,0}};
    
}

public:
// assume that the id in groundtruth (gt) and segmentation (seg) 
// is continuous and start from 0. The size of groundtruth and 
// segmentation should be the same.
ContingencyTable(const Segmentation& gt, const Segmentation& seg){

    table = bu::compressed_matrix()
}
ContingencyTable(const PySegmentation& gt, const PySegmentation& seg){
    ContingencyTable(std::move(Segmentation(gt)), std::move(Segmentation(seg)) );    
} 


};

} // end of reneu namespace
