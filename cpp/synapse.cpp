#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY


#include "reneu/type_aliase.hpp" 
#include "reneu/synapse/detect_tbars.hpp"


namespace py = pybind11;
using namespace reneu;

PYBIND11_MODULE(skeleton, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        skeleton package
        -----------------------
        .. currentclass:: Skeleton
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("detect_points", &py_detect_points); 


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
