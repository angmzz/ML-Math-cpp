#pragma once

#include <Eigen/Dense>
#include <functional>

namespace MathMl {
namespace VectorCalculus {

/**
 * @brief Evalua una funcion cuadratica f(x) = 0.5 * x^T * A * x + b^T * x
 *
 * @param A Matriz simetrica.
 * @param b Vector.
 * @param x Vector donde evaluar.
 * @return double Valor de la funcion.
 */
double QuadraticFunction(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                         const Eigen::VectorXd &x);

/**
 * @brief Aproxima el gradiente de una funcion usando diferencias finitas.
 *
 * @param x Punto donde evaluar el gradiente.
 * @param func Funcion f(x) -> double.
 * @param epsilon Paso para diferencias finitas.
 * @return Eigen::VectorXd Gradiente aproximado.
 */
Eigen::VectorXd
NumericalGradient(const Eigen::VectorXd &x,
                  std::function<double(const Eigen::VectorXd &)> func,
                  double epsilon = 1e-5);

/**
 * @brief Aproxima el Hessiano de una funcion usando diferencias finitas.
 *
 * @param x Punto donde evaluar el Hessiano.
 * @param func Funcion f(x) -> double.
 * @param epsilon Paso para diferencias finitas.
 * @return Eigen::MatrixXd Hessiano aproximado.
 */
Eigen::MatrixXd
NumericalHessian(const Eigen::VectorXd &x,
                 std::function<double(const Eigen::VectorXd &)> func,
                 double epsilon = 1e-5);

} // namespace VectorCalculus
} // namespace MathMl
