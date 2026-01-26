#ifndef OPTIMIZACION_H
#define OPTIMIZACION_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace MathMl {
namespace Optimization {

// Evalúa la función de Rosenbrock en un punto x
double Rosenbrock(const Eigen::VectorXd &x);

// Calcula el gradiente analítico de la función de Rosenbrock en x
Eigen::VectorXd RosenbrockGradient(const Eigen::VectorXd &x);

// Descenso de Gradiente Estándar
// Retorna una lista de puntos visitados (La trayectoria)
std::vector<Eigen::VectorXd> GradientDescent(
    const Eigen::VectorXd &start_point, double learning_rate, int iterations,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_func);

// Descenso de Gradiente con Momento
// Retorna una lista de puntos visitados (la trayectoria)
std::vector<Eigen::VectorXd> MomentumGradientDescent(
    const Eigen::VectorXd &start_point, double learning_rate,
    double momentum, // Valores típicos: 0.9
    int iterations,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> grad_func);

} // namespace Optimization
} // namespace MathMl

#endif // OPTIMIZACION_H
