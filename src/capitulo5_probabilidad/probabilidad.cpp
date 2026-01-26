#include "probabilidad.h"
#include <cmath>
#include <stdexcept>

// Definir PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MathMl {
namespace Probability {

Eigen::VectorXd CalculateMean(const Eigen::MatrixXd &data) {
  return data.colwise().mean();
}

Eigen::MatrixXd CalculateCovariance(const Eigen::MatrixXd &data) {
  if (data.rows() < 2) {
    throw std::runtime_error(
        "Se requieren al menos 2 muestras para calcular la covarianza.");
  }

  // Centrar los datos
  Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

  // Covarianza muestral insesgada
  return (centered.adjoint() * centered) / double(data.rows() - 1);
}

double MultivariateGaussianPDF(const Eigen::VectorXd &x,
                               const Eigen::VectorXd &mean,
                               const Eigen::MatrixXd &covariance) {
  if (x.size() != mean.size() || x.size() != covariance.rows() ||
      covariance.rows() != covariance.cols()) {
    throw std::invalid_argument(
        "Dimensiones incompatibles en MultivariateGaussianPDF.");
  }

  int k = x.size();
  double det = covariance.determinant();

  if (det <= 0) {
    throw std::runtime_error("La matriz de covarianza debe ser definida "
                             "positiva (determinante > 0).");
  }

  double norm_const =
      1.0 / (std::pow(2 * M_PI, double(k) / 2.0) * std::sqrt(det));

  Eigen::VectorXd diff = x - mean;
  // exponent = -0.5 * (x - mu)^T * Sigma^-1 * (x - mu)
  double exponent = -0.5 * diff.transpose() * covariance.inverse() * diff;

  return norm_const * std::exp(exponent);
}

} // namespace Probability
} // namespace MathMl
