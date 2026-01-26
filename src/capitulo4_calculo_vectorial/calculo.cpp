#include "calculo.h"

namespace MathMl {
namespace VectorCalculus {

double QuadraticFunction(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                         const Eigen::VectorXd &x) {
  // 0.5 * x^T * A * x + b^T * x
  return 0.5 * x.transpose() * A * x + b.dot(x);
}

Eigen::VectorXd
NumericalGradient(const Eigen::VectorXd &x,
                  std::function<double(const Eigen::VectorXd &)> func,
                  double epsilon) {
  int n = x.size();
  Eigen::VectorXd grad(n);
  Eigen::VectorXd x_plus = x;
  Eigen::VectorXd x_minus = x;

  for (int i = 0; i < n; ++i) {
    double original_val = x(i);

    x_plus(i) = original_val + epsilon;
    x_minus(i) = original_val - epsilon;

    grad(i) = (func(x_plus) - func(x_minus)) / (2.0 * epsilon);

    // Restaurar
    x_plus(i) = original_val;
    x_minus(i) = original_val;
  }
  return grad;
}

Eigen::MatrixXd
NumericalHessian(const Eigen::VectorXd &x,
                 std::function<double(const Eigen::VectorXd &)> func,
                 double epsilon) {
  int n = x.size();
  Eigen::MatrixXd hessian(n, n);

  // El Hessiano es la matriz Jacobiana del Gradiente.
  // Calculamos el gradiente en x.
  // Pero para diferencias finitas de segundo orden directas:
  // d^2f/dx_i dx_j = (f(x + ei*h + ej*h) - f(x + ei*h - ej*h) - f(x - ei*h +
  // ej*h) + f(x - ei*h - ej*h)) / (4*h*h) O mas simple: calcular gradiente en
  // x+h y x-h y diferenciar.

  // Vamos a usar la aproximacion simple iterando sobre el gradiente numerico
  // Esto es mas costoso pero mas facil de implementar reutilizando
  // NumericalGradient.

  // Definimos una funcion envoltorio para el gradiente i-esimo
  auto grad_func = [&](const Eigen::VectorXd &v) {
    return NumericalGradient(v, func, epsilon);
  };

  // Calculamos gradiente en puntos perturbados para obtener la matriz Jacobiana
  // del gradiente
  for (int j = 0; j < n; ++j) {
    Eigen::VectorXd x_plus = x;
    Eigen::VectorXd x_minus = x;

    x_plus(j) += epsilon;
    x_minus(j) -= epsilon;

    Eigen::VectorXd grad_plus = grad_func(x_plus);
    Eigen::VectorXd grad_minus = grad_func(x_minus);

    Eigen::VectorXd diff = (grad_plus - grad_minus) / (2.0 * epsilon);
    hessian.col(j) = diff;
  }

  // Asegurar simetria (ruido numerico)
  return 0.5 * (hessian + hessian.transpose());
}

} // namespace VectorCalculus
} // namespace MathMl
