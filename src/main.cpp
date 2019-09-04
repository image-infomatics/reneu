#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include "xiuli/neuron/skeleton.hpp"

namespace py = pybind11;

//class PySkeleton : public xiuli::neuron::Skeleton{
//public:
//    // inherit the constructor
//    using Skeleton::Skeleton;
//};
//class PySkeleton : public xiuli::neuron::Skeleton {
//public:
//    using xiuli::neuron::Skeleton;
//    PySkeleton(xiuli::neuron::Skeleton &&base) : xiuli::neuron::Skeleton(std::move(base)) {}
//};

PYBIND11_MODULE(xiuli, m) {
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

    //py::class_<xiuli::neuron::Skeleton, PySkeleton>(m, "Skeleton")
    py::class_<xiuli::neuron::Skeleton>(m, "Skeleton")
        .def(py::init<std::string>())
        .def_property_readonly("nodes", &xiuli::neuron::Skeleton::get_nodes)
        .def_property_readonly("attributes", &xiuli::neuron::Skeleton::get_attributes)
        .def("__len__", &xiuli::neuron::Skeleton::get_node_num)
        .def("downsample", &xiuli::neuron::Skeleton::downsample)
        .def("write_swc", &xiuli::neuron::Skeleton::write_swc);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
