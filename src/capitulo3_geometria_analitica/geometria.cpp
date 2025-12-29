#include "geometria.h"
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MathMl {
namespace AnalyticGeometry {

// ## Normas ##

// Norma Euclidiana
double EuclideanNorm(const Eigen::VectorXd &v) { return v.norm(); }

// Norma Manhattan
double ManhattanNorm(const Eigen::VectorXd &v) { return v.lpNorm<1>(); }

// Norma Chebyshev
double ChebyshevNorm(const Eigen::VectorXd &v) {
  return v.lpNorm<Eigen::Infinity>();
}

// ## Distancias ##

// Distancia Euclidiana
double EuclideanDistance(const Eigen::VectorXd &u, const Eigen::VectorXd &v) {
  if (u.size() != v.size()) {
    throw std::invalid_argument("Los vectores deben tener la misma dimension");
  }
  return (u - v).norm();
}

// Distancia Manhattan
double ManhattanDistance(const Eigen::VectorXd &u, const Eigen::VectorXd &v) {
  if (u.size() != v.size()) {
    throw std::invalid_argument("Los vectores deben tener la misma dimension");
  }
  return (u - v).lpNorm<1>();
}

// ## Similitud ##

// Similitud Coseno
double CosineSimilarity(const Eigen::VectorXd &u, const Eigen::VectorXd &v) {
  if (u.size() != v.size()) {
    throw std::invalid_argument("Los vectores deben tener la misma dimension");
  }
  double dot = u.dot(v);
  double norm_u = u.norm();
  double norm_v = v.norm();

  if (norm_u == 0 || norm_v == 0) {
    // Manejo de vectores nulos
    return 0.0;
  }

  return dot / (norm_u * norm_v);
}

// Angulo entre vectores
double VectorAngle(const Eigen::VectorXd &u, const Eigen::VectorXd &v) {
  double similarity = CosineSimilarity(u, v);

  // Los valores clamped de [-1, 1] son para evitar errores de punto flotante
  // Punto flotante es una representacion de numeros reales en computadora
  // A veces puede haber errores de redondeo
  // Por lo tanto, si el valor es muy cercano a 1 o -1, lo clamo a 1 o -1
  if (similarity > 1.0)
    similarity = 1.0;
  if (similarity < -1.0)
    similarity = -1.0;

  double radians = std::acos(similarity);
  return radians * (180.0 / M_PI);
}

} // namespace AnalyticGeometry
} // namespace MathMl
