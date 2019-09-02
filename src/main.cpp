#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
#include "xiuli/neuron/skeleton.hpp"

namespace py = pybind11;

template<typename T>
auto pytensor_to_numpy(xt::pytensor<T, 2>& tensor){
    return py::buffer_info(tensor.data(), sizeof(T), py::format_descriptor<T>::format(), 2,
                {tensor.shape(0), tensor.shape(1)}, {sizeof(T) * tensor.shape(1), sizeof(T)});

}

template<typename T, std::size_t N>
auto xtensor_to_numpy_array(xt::xtensor<T, N>& tensor){
    xt::pytensor<T, N> pyarray = tensor;
    return pyarray;
}

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
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    //py::class_<xiuli::neuron::Skeleton, PySkeleton>(m, "Skeleton")
    py::class_<xiuli::neuron::Skeleton>(m, "Skeleton")
        .def(py::init<std::string>())
        .def("__len__", &xiuli::neuron::Skeleton::get_nodes_num)
        .def("downsample", &xiuli::neuron::Skeleton::downsample)
        .def("write_swc", &xiuli::neuron::Skeleton::write_swc);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
