#include <queue>
#include "../type_aliase.hpp"


namespace reneu{


struct EdgeInQueue{
    segid_t segid0;
    segid_t segid1;
    aff_edge_t aff;
    size_t version;

    // constructor for emplace operation
    EdgeInQueue(const segid_t& segid0_, const segid_t& segid1_, const aff_edge_t& aff_, const std::size_t& version_):
        segid0(segid0_), segid1(segid1_), aff(aff_), version(version_){}

    bool operator<(const EdgeInQueue& other) const {
        return aff < other.aff;
    }
};
// struct LessThanByAff{
//     bool operator()(const EdgeInQueue& lhs, const EdgeInQueue& rhs) const {
//         return lhs.aff < rhs.aff;
//     }
// };
using PriorityQueue = std::priority_queue<EdgeInQueue, std::vector<EdgeInQueue>, std::less<EdgeInQueue>>;


}// namespace reneu