#include "descomposiciones.h"
#include <iostream>
#include <stdexcept>


namespace MathMl {
namespace MatrixDecompositions {

Eigen::VectorXcd GetEigenvalues(const Eigen::MatrixXd &A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("La matriz debe ser cuadrada.");
  }
  Eigen::EigenSolver<Eigen::MatrixXd> es(A);
  return es.eigenvalues();
}

Eigen::MatrixXcd GetEigenvectors(const Eigen::MatrixXd &A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("La matriz debe ser cuadrada.");
  }
  Eigen::EigenSolver<Eigen::MatrixXd> es(A);
  return es.eigenvectors();
}

Eigen::MatrixXd CholeskyDecomposition(const Eigen::MatrixXd &A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("La matriz debe ser cuadrada.");
  }
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  if (llt.info() == Eigen::NumericalIssue) {
    throw std::runtime_error(
        "La matriz no es definida positiva (posiblemente).");
  }
  return llt.matrixL();
}

SVDResult ComputeSVD(const Eigen::MatrixXd &A) {
  Eigen::BDCSVD<Eigen::MatrixXd> svd(A,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
  SVDResult res;
  res.U = svd.matrixU();
  res.S = svd.singularValues();
  res.V = svd.matrixV();
  return res;
}

Eigen::MatrixXd PCA(const Eigen::MatrixXd &data, int k) {
  // 1. Centrar los datos (restar la media de cada columna)
  Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

  // 2. Calcular matriz de covarianza
  // Cov = (X^T * X) / (n - 1)
  Eigen::MatrixXd cov =
      (centered.adjoint() * centered) / double(data.rows() - 1);

  // 3. Obtener autovalores y autovectores
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      cov); // Usamos SelfAdjoint porque Cov es simetrica

  // Los autovalores/vectores en Eigen estan ordenados de menor a mayor.
  // Necesitamos los k mayores (ultimos k).
  Eigen::MatrixXd eigenvectors = es.eigenvectors().rightCols(k);

  // Invertir el orden de las columnas para que esten de mayor a menor autovalor
  // (opcional pero estandar)
  Eigen::MatrixXd eigenvectors_sorted = eigenvectors.rowwise().reverse();

  // 4. Proyectar los datos
  // Proyeccion = DatosCentrados * Autovectores
  return centered * eigenvectors_sorted;
}

} // namespace MatrixDecompositions
} // namespace MathMl
