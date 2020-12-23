#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "reneu/type_aliase.hpp" 
#include "reneu/segmentation/watershed.hpp"
#include "reneu/segmentation/dendrogram.hpp"
#include "reneu/segmentation/region_graph.hpp"
#include "reneu/segmentation/region_graph_chunk.hpp"
#include "reneu/segmentation/fill_background_with_affinity_guidance.hpp"

namespace py = pybind11;
using namespace reneu;

PYBIND11_MODULE(segmentation, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        segmentation package
        -----------------------
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("watershed", &py_watershed);
    m.def("fill_background_with_affinity_guidance", &py_fill_background_with_affinity_guidance);

    py::class_<Dendrogram>(m, "Dendrogram")
        .def(py::init())
        .def(py::init<const aff_edge_t&>())
        .def_property_readonly("array", &Dendrogram::as_array)
        .def_property_readonly("edge_num", &Dendrogram::get_edge_num)
        .def("print", &Dendrogram::print)
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
        .def("materialize", &Dendrogram::py_materialize);

    py::class_<RegionGraph>(m, "RegionGraph")
        .def(py::init<const PyAffinityMap&, const PySegmentation&>())
        .def_property_readonly("array", &RegionGraph::as_array)
        .def("print", &RegionGraph::print)
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
        .def("greedy_merge", &RegionGraph::py_greedy_merge);

    py::class_<RegionGraphChunk, RegionGraph>(m, "RegionGraphChunk")
        .def(py::init<const PyAffinityMap&, const PySegmentation&, const std::array<bool, 6>&>())
        .def(py::pickle(
            [](const RegionGraphChunk& rg){ // __getstate__
                std::stringstream ss;
                boost::archive::text_oarchive oa(ss);
                oa << rg;
                return ss.str();
            },
            [](const std::string str){
                std::stringstream ss(str);
                boost::archive::text_iarchive ia(ss);
                RegionGraphChunk rg;
                ia >> (rg);
                return rg;
            }
        ))
        .def("merge_in_leaf_chunk", &RegionGraphChunk::merge_in_leaf_chunk)
        .def("merge_upper_chunk", &RegionGraphChunk::merge_upper_chunk);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
