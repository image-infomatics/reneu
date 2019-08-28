#include <iostream>
#include <assert.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pybind11.h>
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

namespace py = pybind11;

double sum_of_sines(xt::pytensor<double, 2>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}


int update_first_child_and_sibling(xt::pytensor<int, 2>& attributes){
    std::size_t node_num = attributes.shape(0);
    // the columns are class, parents, fist child, sibling
    assert( attributes.shape(1) == 4 );

    auto parents = xt::view(attributes, xt::all(), 1);
    auto childs = xt::view(attributes, xt::all(), 2);
    auto siblings = xt::view(attributes, xt::all(), 3);

    for (std::size_t nodeIdx = 0; nodeIdx < node_num; nodeIdx++){
        auto parentNodeIdx = parents( nodeIdx );
        auto siblingNodeIdx = childs( parentNodeIdx );
        if (siblingNodeIdx < 0){
            childs(parentNodeIdx) = nodeIdx;
        } else {
            // look for an empty sibling spot
            auto lastSiblingNodeIdx = siblingNodeIdx;
            auto nextSiblingNodeIdx = siblings(siblingNodeIdx);
            while (nextSiblingNodeIdx >= 0){
                lastSiblingNodeIdx = nextSiblingNodeIdx;
                // move forward to look for empty sibling spot
                nextSiblingNodeIdx = siblings( lastSiblingNodeIdx );
            }
            siblings( lastSiblingNodeIdx ) = nodeIdx; 
        }
    }
    return 0;
}


PYBIND11_MODULE(libreneu, m) {
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

    m.def("update_first_child_and_sibling", &update_first_child_and_sibling, R"pbdoc(
        update first child and sibling
    )pbdoc");

    m.def("sum_of_sines", &sum_of_sines, R"pbdoc(
        sun of sines
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
