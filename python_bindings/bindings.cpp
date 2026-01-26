#include "../src/capitulo2_algebra_lineal/algebra_lineal.h"
#include "../src/capitulo3_geometria_analitica/geometria.h"
#include "../src/capitulo4_descomposiciones_matrices/descomposiciones.h"
#include "../src/capitulo5_calculo_vectorial/calculo.h"
#include "../src/capitulo6_probabilidad/probabilidad.h"
#include "../src/capitulo7_optimizacion/optimizacion.h"
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


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

  linear_algebra.def(
      "is_linearly_independent", &MathMl::LinearAlgebra::IsLinearlyIndependent,
      "Comprueba si las columnas de una matriz son linealmente independientes",
      py::arg("A"));

  linear_algebra.def("affine_forward", &MathMl::LinearAlgebra::AffineForward,
                     "Realiza la transformacion afin y = Ax + b", py::arg("A"),
                     py::arg("x"), py::arg("b"));

  // --- Capitulo 3: Geometria Analitica ---
  auto analytic_geometry =
      m.def_submodule("analytic_geometry", "Capitulo 3: Geometria Analitica");

  analytic_geometry.def(
      "euclidean_norm", &MathMl::AnalyticGeometry::EuclideanNorm,
      "Calcula la norma euclidiana de un vector", py::arg("v"));

  analytic_geometry.def(
      "manhattan_norm", &MathMl::AnalyticGeometry::ManhattanNorm,
      "Calcula la norma manhattan de un vector", py::arg("v"));

  analytic_geometry.def(
      "chebyshev_norm", &MathMl::AnalyticGeometry::ChebyshevNorm,
      "Calcula la norma chebyshev de un vector", py::arg("v"));

  analytic_geometry.def("euclidean_distance",
                        &MathMl::AnalyticGeometry::EuclideanDistance,
                        "Calcula la distancia euclidiana entre dos vectores",
                        py::arg("u"), py::arg("v"));

  analytic_geometry.def("manhattan_distance",
                        &MathMl::AnalyticGeometry::ManhattanDistance,
                        "Calcula la distancia manhattan entre dos vectores",
                        py::arg("u"), py::arg("v"));

  analytic_geometry.def("cosine_similarity",
                        &MathMl::AnalyticGeometry::CosineSimilarity,
                        "Calcula la similitud coseno entre dos vectores",
                        py::arg("u"), py::arg("v"));

  analytic_geometry.def("vector_angle", &MathMl::AnalyticGeometry::VectorAngle,
                        "Calcula el angulo entre dos vectores", py::arg("u"),
                        py::arg("v"));

  // --- Capitulo 4: Descomposiciones de Matrices ---
  auto matrix_decompositions = m.def_submodule(
      "matrix_decompositions", "Capitulo 4: Descomposiciones de Matrices");

  // Incluir cabeceras nuevas (al inicio del archivo se insertaran, pero los
  // enlaces aqui) Nota: Necesitamos incluir las cabeceras al principio del
  // archivo.

  matrix_decompositions.def(
      "get_eigenvalues", &MathMl::MatrixDecompositions::GetEigenvalues,
      "Obtiene los autovalores de una matriz", py::arg("A"));

  matrix_decompositions.def(
      "get_eigenvectors", &MathMl::MatrixDecompositions::GetEigenvectors,
      "Obtiene los autovectores de una matriz", py::arg("A"));

  matrix_decompositions.def(
      "cholesky_decomposition",
      &MathMl::MatrixDecompositions::CholeskyDecomposition,
      "Calcula la descomposicion de Cholesky A = LL^T", py::arg("A"));

  py::class_<MathMl::MatrixDecompositions::SVDResult>(matrix_decompositions,
                                                      "SVDResult")
      .def_readonly("U", &MathMl::MatrixDecompositions::SVDResult::U)
      .def_readonly("S", &MathMl::MatrixDecompositions::SVDResult::S)
      .def_readonly("V", &MathMl::MatrixDecompositions::SVDResult::V);

  matrix_decompositions.def("compute_svd",
                            &MathMl::MatrixDecompositions::ComputeSVD,
                            "Calcula SVD returning U, S, V", py::arg("A"));

  matrix_decompositions.def("pca", &MathMl::MatrixDecompositions::PCA,
                            "Realiza PCA sobre los datos", py::arg("data"),
                            py::arg("k"));

  // --- Capitulo 5: Calculo Vectorial ---
  auto vector_calculus =
      m.def_submodule("vector_calculus", "Capitulo 5: Calculo Vectorial");

  vector_calculus.def(
      "quadratic_function", &MathMl::VectorCalculus::QuadraticFunction,
      "Evalua funcion cuadratica", py::arg("A"), py::arg("b"), py::arg("x"));

  vector_calculus.def(
      "numerical_gradient",
      [](const Eigen::VectorXd &x, py::function func, double epsilon) {
        auto wrapped_func = [func](const Eigen::VectorXd &v) -> double {
          return func(v).cast<double>();
        };
        return MathMl::VectorCalculus::NumericalGradient(x, wrapped_func,
                                                         epsilon);
      },
      "Calcula gradiente numerico", py::arg("x"), py::arg("func"),
      py::arg("epsilon") = 1e-5);

  vector_calculus.def(
      "numerical_hessian",
      [](const Eigen::VectorXd &x, py::function func, double epsilon) {
        auto wrapped_func = [func](const Eigen::VectorXd &v) -> double {
          return func(v).cast<double>();
        };
        return MathMl::VectorCalculus::NumericalHessian(x, wrapped_func,
                                                        epsilon);
      },
      "Calcula Hessiano numerico", py::arg("x"), py::arg("func"),
      py::arg("epsilon") = 1e-5);

  // --- Capitulo 6: Probabilidad ---
  auto probability = m.def_submodule("probability", "Capitulo 6: Probabilidad");

  probability.def("calculate_mean", &MathMl::Probability::CalculateMean,
                  "Calcula media por columnas", py::arg("data"));

  probability.def("calculate_covariance",
                  &MathMl::Probability::CalculateCovariance,
                  "Calcula matriz de covarianza", py::arg("data"));

  probability.def("multivariate_gaussian_pdf",
                  &MathMl::Probability::MultivariateGaussianPDF,
                  "Evalua PDF Gaussiana", py::arg("x"), py::arg("mean"),
                  py::arg("covariance"));

  // --- Capitulo 7: Optimizacion ---
  auto optimization =
      m.def_submodule("optimization", "Capitulo 7: Optimizacion");

  optimization.def("rosenbrock", &MathMl::Optimization::Rosenbrock,
                   "Evalua la funcion de Rosenbrock", py::arg("x"));

  optimization.def("rosenbrock_gradient",
                   &MathMl::Optimization::RosenbrockGradient,
                   "Calcula el gradiente de Rosenbrock", py::arg("x"));

  optimization.def(
      "gradient_descent",
      [](const Eigen::VectorXd &start_point, double learning_rate,
         int iterations, py::function grad_func) {
        auto wrapped_func =
            [grad_func](const Eigen::VectorXd &v) -> Eigen::VectorXd {
          return grad_func(v).cast<Eigen::VectorXd>();
        };
        return MathMl::Optimization::GradientDescent(start_point, learning_rate,
                                                     iterations, wrapped_func);
      },
      "Ejecuta Gradient Descent", py::arg("start_point"),
      py::arg("learning_rate"), py::arg("iterations"), py::arg("grad_func"));

  optimization.def(
      "momentum_gradient_descent",
      [](const Eigen::VectorXd &start_point, double learning_rate,
         double momentum, int iterations, py::function grad_func) {
        auto wrapped_func =
            [grad_func](const Eigen::VectorXd &v) -> Eigen::VectorXd {
          return grad_func(v).cast<Eigen::VectorXd>();
        };
        return MathMl::Optimization::MomentumGradientDescent(
            start_point, learning_rate, momentum, iterations, wrapped_func);
      },
      "Ejecuta Momentum Gradient Descent", py::arg("start_point"),
      py::arg("learning_rate"), py::arg("momentum"), py::arg("iterations"),
      py::arg("grad_func"));

  // Wrappers for C++ internal testing (faster, no Python callback overhead)
  optimization.def(
      "run_gd_on_rosenbrock",
      [](const Eigen::VectorXd &start_point, double learning_rate,
         int iterations) {
        return MathMl::Optimization::GradientDescent(
            start_point, learning_rate, iterations,
            MathMl::Optimization::RosenbrockGradient);
      },
      "Ejecuta GD en Rosenbrock (puro C++)", py::arg("start_point"),
      py::arg("learning_rate"), py::arg("iterations"));

  optimization.def(
      "run_momentum_on_rosenbrock",
      [](const Eigen::VectorXd &start_point, double learning_rate,
         double momentum, int iterations) {
        return MathMl::Optimization::MomentumGradientDescent(
            start_point, learning_rate, momentum, iterations,
            MathMl::Optimization::RosenbrockGradient);
      },
      "Ejecuta Momentum en Rosenbrock (puro C++)", py::arg("start_point"),
      py::arg("learning_rate"), py::arg("momentum"), py::arg("iterations"));
}
