#include "algebra_lineal.h"

namespace MathMl {
namespace LinearAlgebra {

// ## Proyecciones ##

// Proyeccion Ortogonal
// ## Operaciones ##

// Producto Escalar
Eigen::VectorXd ProjectVector(const Eigen::VectorXd &x,
                              const Eigen::VectorXd &y) {
  double scalar = x.dot(y) / y.squaredNorm();
  return scalar * y;
}

// Producto Matricial
Eigen::MatrixXd MatMul(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
  return A * B;
}

// ## Resolucion de Sistemas ##

// Resolucion de Sistemas
Eigen::VectorXd SolveLinearSystem(const Eigen::MatrixXd &A,
                                  const Eigen::VectorXd &b) {
  return A.colPivHouseholderQr().solve(b);
}

// ## Inversa ##

// Inversa
Eigen::MatrixXd CalculateInverse(const Eigen::MatrixXd &A) {
  return A.inverse();
}

// ## Determinante ##

// Determinante
double CalculateDeterminant(const Eigen::MatrixXd &A) {
  return A.determinant();
}

// ## Traza ##

// Traza
double CalculateTrace(const Eigen::MatrixXd &A) { return A.trace(); }

// ## Rango ##

// Rango
int CalculateRank(const Eigen::MatrixXd &A) {
  return A.colPivHouseholderQr().rank();
}

// ## Independencia Lineal ##

// Comprueba si las columnas de una matriz son linealmente independientes
bool IsLinearlyIndependent(const Eigen::MatrixXd &A) {
  return CalculateRank(A) == A.cols();
}

// ## Transformaciones Afines ##

// Transformacion Afin y = Ax + b
Eigen::VectorXd AffineForward(const Eigen::MatrixXd &A,
                              const Eigen::VectorXd &x,
                              const Eigen::VectorXd &b) {
  // Validacion de dimensiones
  if (A.cols() != x.size()) {
    throw std::invalid_argument(
        "Dimension mismatch: A.cols() must equal x.size()");
  }
  if (A.rows() != b.size()) {
    throw std::invalid_argument(
        "Dimension mismatch: A.rows() must equal b.size()");
  }

  return A * x + b;
}

} // namespace LinearAlgebra
} // namespace MathMl
