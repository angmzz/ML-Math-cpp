#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace MathMl {
namespace AnalyticGeometry {

/**
 * @brief Calcula la norma Euclidiana (L2) de un vector.
 *
 * @param v Vector de entrada.
 * @return double Norma L2.
 */
double EuclideanNorm(const Eigen::VectorXd &v);

/**
 * @brief Calcula la norma Manhattan (L1) de un vector.
 *
 * @param v Vector de entrada.
 * @return double Norma L1.
 */
double ManhattanNorm(const Eigen::VectorXd &v);

/**
 * @brief Calcula la norma Chebyshev (L-infinito) de un vector.
 *
 * @param v Vector de entrada.
 * @return double Norma L-infinito.
 */
double ChebyshevNorm(const Eigen::VectorXd &v);

/**
 * @brief Calcula la distancia Euclidiana entre dos vectores.
 *
 * @param u Primer vector.
 * @param v Segundo vector.
 * @return double Distancia Euclidiana.
 */
double EuclideanDistance(const Eigen::VectorXd &u, const Eigen::VectorXd &v);

/**
 * @brief Calcula la distancia Manhattan entre dos vectores.
 *
 * @param u Primer vector.
 * @param v Segundo vector.
 * @return double Distancia Manhattan.
 */
double ManhattanDistance(const Eigen::VectorXd &u, const Eigen::VectorXd &v);

/**
 * @brief Calcula la similitud coseno entre dos vectores.
 *
 * @param u Primer vector.
 * @param v Segundo vector.
 * @return double Similitud coseno [-1, 1].
 */
double CosineSimilarity(const Eigen::VectorXd &u, const Eigen::VectorXd &v);

/**
 * @brief Calcula el angulo entre dos vectores en grados.
 *
 * @param u Primer vector.
 * @param v Segundo vector.
 * @return double Angulo en grados.
 */
double VectorAngle(const Eigen::VectorXd &u, const Eigen::VectorXd &v);

} // namespace AnalyticGeometry
} // namespace MathMl
