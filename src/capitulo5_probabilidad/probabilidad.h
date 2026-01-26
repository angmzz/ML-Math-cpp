#pragma once

#include <Eigen/Dense>

namespace MathMl {
namespace Probability {

/**
 * @brief Calcula la media de cada columna de una matriz de datos.
 *
 * @param data Matriz donde cada fila es una muestra y cada columna una
 * caracteristica.
 * @return Eigen::VectorXd Vector de medias.
 */
Eigen::VectorXd CalculateMean(const Eigen::MatrixXd &data);

/**
 * @brief Calcula la matriz de covarianza de los datos.
 *
 * @param data Matriz de datos (n_samples x n_features).
 * @return Eigen::MatrixXd Matriz de covarianza (n_features x n_features).
 */
Eigen::MatrixXd CalculateCovariance(const Eigen::MatrixXd &data);

/**
 * @brief Evalua la funcion de densidad de probabilidad (PDF) Gaussiana
 * Multivariada.
 *
 * @param x Punto donde evaluar (vector).
 * @param mean Vector de medias.
 * @param covariance Matriz de covarianza.
 * @return double Valor de la densidad.
 */
double MultivariateGaussianPDF(const Eigen::VectorXd &x,
                               const Eigen::VectorXd &mean,
                               const Eigen::MatrixXd &covariance);

} // namespace Probability
} // namespace MathMl
