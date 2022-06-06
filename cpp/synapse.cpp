#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY


#include "reneu/types.hpp" 
#include "reneu/synapse/detect_tbars.hpp"


namespace py = pybind11;
using namespace reneu;

PYBIND11_MODULE(synapse, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        skeleton package
        -----------------------
        .. currentclass:: Skeleton
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("detect_points", &py_detect_points);
    m.def("get_object_average_intensity", &py_get_object_average_intensity); 


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
