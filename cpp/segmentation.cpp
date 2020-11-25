#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY


#include "reneu/utils/math.hpp"
#include "reneu/type_aliase.hpp" 
#include "reneu/segmentation/watershed.hpp"
#include "reneu/segmentation/region_graph.hpp"
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

    py::class_<RegionGraph>(m, "RegionGraph")
        .def(py::init<const PyAffinityMap&, const PySegmentation&>())
        .def("print", &RegionGraph::print)
        .def("greedy_merge_until", &RegionGraph::py_greedy_merge_until);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
