#include "../src/capitulo2_algebra_lineal/algebra_lineal.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mathml_cpp, m) {
  m.doc() = "ML-Math-cpp";

  auto submodule =
      m.def_submodule("linear_algebra", "Capitulo 2: Algebra Lineal");

  // Vincular las funciones C++

  submodule.def("project_vector", &MathMl::LinearAlgebra::ProjectVector,
                "Proyecta el vector x sobre el vector y", py::arg("x"),
                py::arg("y"));

  submodule.def("mat_mul", &MathMl::LinearAlgebra::MatMul,
                "Multiplica dos matrices", py::arg("A"), py::arg("B"));
}
