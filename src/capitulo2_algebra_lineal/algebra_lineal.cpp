#include "algebra_lineal.h"

namespace MathMl {
namespace LinearAlgebra {

// Proyección de x sobre y
Eigen::VectorXd ProjectVector(const Eigen::VectorXd &x,
                              const Eigen::VectorXd &y) {
  double scalar = x.dot(y) / y.squaredNorm();
  return scalar * y;
}

// Multiplicación de matrices
Eigen::MatrixXd MatMul(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
  return A * B;
}

} // namespace LinearAlgebra
} // namespace MathMl
