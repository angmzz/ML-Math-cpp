#include "optimizacion.h"
#include <iostream>

namespace MathMl {
namespace Optimization {

double Rosenbrock(const Eigen::VectorXd &x) {
  // Función de Rosenbrock estándar: f(x,y) = (1-x)^2 + 100(y-x^2)^2
  // Asumimos que x tiene al menos 2 dimensiones.
  if (x.size() < 2)
    return 0.0;
  double x_val = x(0);
  double y_val = x(1);
  return std::pow(1.0 - x_val, 2) + 100.0 * std::pow(y_val - x_val * x_val, 2);
}

Eigen::VectorXd RosenbrockGradient(const Eigen::VectorXd &x) {
  if (x.size() < 2)
    return Eigen::VectorXd::Zero(x.size());
  double x_val = x(0);
  double y_val = x(1);

  Eigen::VectorXd grad(2);
  // df/dx = -2(1-x) - 400x(y-x^2)
  grad(0) = -2.0 * (1.0 - x_val) - 400.0 * x_val * (y_val - x_val * x_val);
  // df/dy = 200(y-x^2)
  grad(1) = 200.0 * (y_val - x_val * x_val);

  return grad;
}

std::vector<Eigen::VectorXd> GradientDescent(
    const Eigen::VectorXd &start_point, double learning_rate, int iterations,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_func) {
  std::vector<Eigen::VectorXd> trajectory;
  Eigen::VectorXd current = start_point;
  trajectory.push_back(current);

  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd grad = grad_func(current);
    current = current - learning_rate * grad;
    trajectory.push_back(current);
  }
  return trajectory;
}

std::vector<Eigen::VectorXd> MomentumGradientDescent(
    const Eigen::VectorXd &start_point, double learning_rate,
    double momentum, // Usualmente alrededor de 0.9
    int iterations,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_func) {
  std::vector<Eigen::VectorXd> trajectory;
  Eigen::VectorXd current = start_point;
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(start_point.size());
  trajectory.push_back(current);

  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd grad = grad_func(current);

    // v_{t+1} = gamma * v_t + eta * grad
    velocity = momentum * velocity + learning_rate * grad;

    // x_{t+1} = x_t - v_{t+1}
    current = current - velocity;
    trajectory.push_back(current);
  }
  return trajectory;
}

} // namespace Optimization
} // namespace MathMl
