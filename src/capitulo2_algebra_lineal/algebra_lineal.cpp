#include "algebra_lineal.h"

namespace MathMl {
namespace LinearAlgebra {

Eigen::VectorXd ProjectVector(const Eigen::VectorXd &x,
                              const Eigen::VectorXd &y) {
  double scalar = x.dot(y) / y.squaredNorm();
  return scalar * y;
}

Eigen::MatrixXd MatMul(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
  return A * B;
}

Eigen::VectorXd SolveLinearSystem(const Eigen::MatrixXd &A,
                                  const Eigen::VectorXd &b) {
  return A.colPivHouseholderQr().solve(b);
}

Eigen::MatrixXd CalculateInverse(const Eigen::MatrixXd &A) {
  return A.inverse();
}

double CalculateDeterminant(const Eigen::MatrixXd &A) {
  return A.determinant();
}

double CalculateTrace(const Eigen::MatrixXd &A) { return A.trace(); }

int CalculateRank(const Eigen::MatrixXd &A) {
  return A.colPivHouseholderQr().rank();
}

} // namespace LinearAlgebra
} // namespace MathMl
