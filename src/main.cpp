#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY


#include "xiuli/xiuli.hpp"
#include "xiuli/utils/math.hpp"
#include "xiuli/type_aliase.hpp" 


namespace py = pybind11;
using namespace xiuli;

PYBIND11_MODULE(xiuli, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        xiuli package
        -----------------------
        .. currentclass:: Skeleton
        .. autosummary::
           :toctree: _generate
           points
           attributes
           downsample
           write_swc
    )pbdoc";

    m.def("pca_first_component", &py_pca_first_component); 

    //py::class_<neuron::Skeleton, PySkeleton>(m, "Skeleton")
    py::class_<Skeleton>(m, "XSkeleton")
        .def(py::init<const PyPoints &, const PyPoints &>())
        .def(py::init<const PyPoints &>())
        .def(py::init<const std::string>())
        .def_property_readonly("points", &Skeleton::get_points)
        .def_property_readonly("attributes", &Skeleton::get_attributes)
        .def_property_readonly("path_length", &Skeleton::get_path_length)
        .def_property_readonly("edges", &Skeleton::get_edges)
        .def("__len__", &Skeleton::get_point_num)
        .def("downsample", &Skeleton::downsample)
        .def("to_swc_str", &Skeleton::to_swc_str)
        .def("write_swc", &Skeleton::write_swc);

    py::class_<KDTree>(m, "XKDTree")
        .def(py::init<const PyPoints &, const Index &>())
        .def("knn", &KDTree::py_knn);
        

    py::class_<ScoreTable>(m, "XNBLASTScoreTable")
        .def(py::init())
        .def(py::init<const std::string &>())
        .def(py::init<const PyPoints &>())
        .def_property_readonly("table", &ScoreTable::get_pytable)
        // python do not have single precision number!
        .def("__getitem__", py::overload_cast<const std::tuple<float, float>&>(
                                    &ScoreTable::operator()), "get table item");
    
    py::class_<VectorCloud>(m, "XVectorCloud")
        .def(py::init<const PyPoints &, const Index &, const Index &>())
        .def_property_readonly("vectors", &VectorCloud::get_vectors)
        .def("__len__", &VectorCloud::size)
        .def("query_by", &VectorCloud::query_by);

    py::class_<NBLASTScoreMatrix>(m, "XNBLASTScoreMatrix")
        //.def(py::init<const py::list &, const ScoreTable &>())
        // Note that the conversion from python list to std::vector has copy overhead
        .def(py::init<const std::vector<VectorCloud> &, const ScoreTable &>())
        .def_property_readonly("raw_score_matrix", &NBLASTScoreMatrix::get_raw_score_matrix)
        .def_property_readonly("normalized_score_matrix", 
                                &NBLASTScoreMatrix::get_normalized_score_matrix)
        .def_property_readonly("mean_score_matrix", 
                                &NBLASTScoreMatrix::get_mean_score_matrix);
        


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
