#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include "xiuli/xiuli.hpp"
#include "xiuli/utils/math.hpp"


namespace py = pybind11;
namespace xn = xiuli::neuron;
namespace xnn = xn::nblast;

PYBIND11_MODULE(libxiuli, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        xiuli package
        -----------------------
        .. currentclass:: Skeleton
        .. autosummary::
           :toctree: _generate
           nodes
           attributes
           downsample
           write_swc
    )pbdoc";

    m.def("pca_first_component", &xiuli::utils::py_pca_first_component); 

    //py::class_<xiuli::neuron::Skeleton, PySkeleton>(m, "Skeleton")
    py::class_<xn::Skeleton>(m, "XSkeleton")
        .def(py::init<const xt::pytensor<float, 2>, const xt::pytensor<int, 2>>())
        .def(py::init<const xt::pytensor<float, 2>>())
        .def(py::init<const std::string>())
        .def_property_readonly("nodes", &xn::Skeleton::get_nodes)
        .def_property_readonly("attributes", &xn::Skeleton::get_attributes)
        .def_property_readonly("path_length", &xn::Skeleton::get_path_length)
        .def_property_readonly("edges", &xn::Skeleton::get_edges)
        .def("__len__", &xn::Skeleton::get_node_num)
        .def("downsample", &xn::Skeleton::downsample)
        .def("write_swc", &xn::Skeleton::write_swc);

    py::class_<xnn::ScoreTable>(m, "XNBLASTScoreTable")
        .def(py::init())
        .def(py::init<const std::string>())
        .def(py::init<const xt::pytensor<float, 2>>())
        .def_property_readonly("table", &xnn::ScoreTable::get_pytable)
        // python do not have single precision number!
        .def("__getitem__", py::overload_cast<const std::tuple<float, float>&>(&xnn::ScoreTable::operator()), "get table item");
    
    py::class_<xnn::VectorCloud>(m, "XVectorCloud")
        .def(py::init<const xt::pytensor<float, 2>, const std::size_t>())
        .def_property_readonly("vectors", &xnn::VectorCloud::get_vectors)
        .def("__len__", &xnn::VectorCloud::size);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
