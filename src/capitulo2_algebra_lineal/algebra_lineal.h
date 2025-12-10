#pragma once

#include <Eigen/Dense>

namespace MathMl {
namespace LinearAlgebra {

/**
 * @brief Proyecta el vector x sobre el vector y.
 *
 * Calcula la proyección ortogonal de un vector sobre otro.
 *
 * @param x Vector a proyectar.
 * @param y Vector sobre el cual se proyecta.
 * @return Eigen::VectorXd El vector proyección resultante.
 */
Eigen::VectorXd ProjectVector(const Eigen::VectorXd &x,
                              const Eigen::VectorXd &y);

/**
 * @brief Multiplica dos matrices.
 *
 * Realiza la multiplicación de matrices A * B.
 *
 * @param A Primera matriz.
 * @param B Segunda matriz.
 * @return Eigen::MatrixXd La matriz resultante de la multiplicación.
 */
Eigen::MatrixXd MatMul(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

} // namespace LinearAlgebra
} // namespace MathMl
