#include "../src/capitulo2_algebra_lineal/algebra_lineal.h"
#include "../src/capitulo3_geometria_analitica/geometria.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mathml_cpp, m) {
  m.doc() =
      "ML-Math-cpp: Implementacion de Matematicas para Machine Learning en C++";

  // --- Capitulo 2: Algebra Lineal ---
  auto linear_algebra =
      m.def_submodule("linear_algebra", "Capitulo 2: Algebra Lineal");

  linear_algebra.def("project_vector", &MathMl::LinearAlgebra::ProjectVector,
                     "Proyecta el vector x sobre el vector y", py::arg("x"),
                     py::arg("y"));

  linear_algebra.def("mat_mul", &MathMl::LinearAlgebra::MatMul,
                     "Multiplica dos matrices", py::arg("A"), py::arg("B"));

  linear_algebra.def("solve_linear_system",
                     &MathMl::LinearAlgebra::SolveLinearSystem,
                     "Resuelve un sistema de ecuaciones lineales Ax = b",
                     py::arg("A"), py::arg("b"));

  linear_algebra.def("calculate_inverse",
                     &MathMl::LinearAlgebra::CalculateInverse,
                     "Calcula la inversa de una matriz", py::arg("A"));

  linear_algebra.def("calculate_determinant",
                     &MathMl::LinearAlgebra::CalculateDeterminant,
                     "Calcula el determinante de una matriz", py::arg("A"));

  linear_algebra.def("calculate_trace", &MathMl::LinearAlgebra::CalculateTrace,
                     "Calcula la traza de una matriz", py::arg("A"));

  linear_algebra.def("calculate_rank", &MathMl::LinearAlgebra::CalculateRank,
                     "Calcula el rango de una matriz", py::arg("A"));
}
