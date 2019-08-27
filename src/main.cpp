#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pybind11.h>
#include <iostream>
#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings

namespace py = pybind11;

double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}


int test_xtensor(){
    xt::xtensor<double, 2> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xtensor<double, 1> arr2
      {5.0, 6.0, 7.0};

    xt::xtensor<double, 1> res = xt::view(arr1, 1) + arr2;

    std::cout << "output: "<< res << std::endl;

    return 0;
}

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

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

    m.def("test_xtensor", &test_xtensor, R"pbdoc(
        test xtensor
    )pbdoc");

    m.def("sum_of_sines", &sum_of_sines, R"pbdoc(
        sun of sines
    )pbdoc");

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
