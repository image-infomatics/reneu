#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "reneu/types.hpp" 
#include "reneu/segmentation/watershed.hpp"
#include "reneu/segmentation/dendrogram.hpp"
#include "reneu/segmentation/disjoint_sets.hpp"
#include "reneu/segmentation/region_graph.hpp"
#include "reneu/segmentation/preprocess.hpp"
#include "reneu/segmentation/seeded_watershed.hpp"
#include "reneu/segmentation/fragments.hpp"
#include "reneu/segmentation/utils.hpp"


namespace py = pybind11;
using namespace reneu;
using namespace pybind11::literals;

PYBIND11_MODULE(segmentation, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        segmentation package
        -----------------------
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("get_label_map", &get_label_map);

    m.def("watershed", &py_watershed);

    m.def("fill_background_with_affinity_guidance", &fill_background_with_affinity_guidance, "fill the background with affinity guidance.");

    m.def("remove_contact", &remove_contact, "remove object contacts.");
    
    m.def("seeded_watershed", &seeded_watershed, "watershed with a segmentation as seed");
    
    m.def("agglomerated_segmentation_to_merge_pairs", &agglomerated_segmentation_to_merge_pairs, "based on the fragments/supervoxels and agglomerated segmentation, get the corresponding merged object pairs.");
    
    py::class_<Dendrogram>(m, "Dendrogram")
        .def(py::init())
        .def(py::init<const aff_edge_t&>())
        .def_property_readonly("array", &Dendrogram::as_array)
        .def_property_readonly("edge_num", &Dendrogram::get_edge_num)
        .def("__str__", &Dendrogram::as_string)
        .def("push_edge", &Dendrogram::push_edge)
        .def("keep_only_contacting_edges", &Dendrogram::py_keep_only_contacting_edges)
        .def("merge", &Dendrogram::merge)
        .def(py::pickle(
            [](const Dendrogram& dend){ // __getstate__
                std::stringstream ss;
                boost::archive::text_oarchive oa(ss);
                oa << dend;
                return ss.str();
            },
            [](const std::string str){ // __setstate__
                std::stringstream ss(str);
                boost::archive::text_iarchive ia(ss);
                Dendrogram dend;
                ia >> (dend);
                return dend;
            }
        ))
        .def("to_disjoint_sets", &Dendrogram::to_disjoint_sets)
        .def("split_objects", &Dendrogram::split_objects)
        .def("materialize", &Dendrogram::py_materialize);

    py::class_<RegionGraph>(m, "RegionGraph")
        .def(py::init())
        .def(py::init<const PyAffinityMap&, const PySegmentation&>())
        .def_property_readonly("arrays", &RegionGraph::to_arrays)
        .def("merge_arrays", &RegionGraph::merge_arrays)
        .def("__repr__", &RegionGraph::as_string)
        .def(py::pickle(
            [](const RegionGraph& rg){ // __getstate__
                std::stringstream ss;
                boost::archive::text_oarchive oa(ss);
                oa << rg;
                return ss.str();
            },
            [](const std::string str){
                std::stringstream ss(str);
                boost::archive::text_iarchive ia(ss);
                RegionGraph rg;
                ia >> (rg);
                return rg;
            }
        ))
        .def("greedy_mean_affinity_agglomeration", 
            &RegionGraph::greedy_mean_affinity_agglomeration,
            "seg"_a, 
            "affinity_threshold"_a=0., 
            "min_voxel_num_threshold"_a=std::numeric_limits<size_t>::max(),
            "max_voxel_num_threshold"_a=std::numeric_limits<size_t>::max());

    py::class_<DisjointSets<segid_t>>(m, "DisjointSets")
        .def(py::init())
        .def(py::init<const PySegmentation&>())
        .def("make_set", 
            &DisjointSets<segid_t>::make_set)
        .def("union_set", 
            &DisjointSets<segid_t>::union_set,
            "id0"_a, "id1"_a, "by_size"_a=true)
        .def("find_set", 
            &DisjointSets<segid_t>::find_set)
        .def("make_and_union_set", 
            &DisjointSets<segid_t>::make_and_union_set,
            "id0"_a, "id1"_a, "by_size"_a=true)
        .def(py::pickle(
            [](const DisjointSets<segid_t>& djs){ // __getstate__
                std::stringstream ss;
                boost::archive::text_oarchive oa(ss);
                oa << djs;
                return ss.str();
            },
            [](const std::string str){ // __setstate__
                std::stringstream ss(str);
                boost::archive::text_iarchive ia(ss);
                DisjointSets<segid_t> djs;
                ia >> (djs);
                return djs;
            }
        ))
        .def("merge_array", 
            &DisjointSets<segid_t>::merge_array,
            "arr"_a, 
            "has_root"_a=false)
        .def_property_readonly("array", 
            &DisjointSets<segid_t>::to_array)
        .def("relabel", 
            &DisjointSets<segid_t>::py_relabel);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
