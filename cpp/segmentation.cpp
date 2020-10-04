#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY


#include "reneu/utils/math.hpp"
#include "reneu/type_aliase.hpp" 
#include "reneu/segmentation/watershed.hpp"
#include "reneu/segmentation/agglomeration.hpp"
#include "reneu/segmentation/dilation.hpp"

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
    m.def("dilate", &py_dilate);

    // agglomeration
    py::class_<SupervoxelDendrogram>(m, "XSupervoxelDendrogram")
        .def(py::init<const PyAffinityMap &, const PySegmentation &, aff_edge_t &>())
        .def("segment", &SupervoxelDendrogram::segment);



#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
