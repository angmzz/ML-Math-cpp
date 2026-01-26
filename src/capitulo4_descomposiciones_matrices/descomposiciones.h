#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>


namespace MathMl {
namespace MatrixDecompositions {

/**
 * @brief Obtiene los autovalores de una matriz.
 *
 * @param A Matriz cuadrada.
 * @return Eigen::VectorXcd Vector de autovalores (complejos).
 */
Eigen::VectorXcd GetEigenvalues(const Eigen::MatrixXd &A);

/**
 * @brief Obtiene los autovectores de una matriz.
 *
 * @param A Matriz cuadrada.
 * @return Eigen::MatrixXcd Matriz de autovectores (columnas).
 */
Eigen::MatrixXcd GetEigenvectors(const Eigen::MatrixXd &A);

/**
 * @brief Calcula la descomposicion de Cholesky A = LL^T.
 *
 * @param A Matriz definida positiva.
 * @return Eigen::MatrixXd Matriz triangular inferior L.
 */
Eigen::MatrixXd CholeskyDecomposition(const Eigen::MatrixXd &A);

/**
 * @brief Estructura para almacenar resultados de SVD.
 */
struct SVDResult {
  Eigen::MatrixXd U;
  Eigen::VectorXd S;
  Eigen::MatrixXd V; // V^T en Eigen es matrixV().transpose()
};

/**
 * @brief Calcula la Descomposicion en Valores Singulares (SVD).
 *
 * @param A Matriz de entrada.
 * @return SVDResult Estructura con U, S y V.
 */
SVDResult ComputeSVD(const Eigen::MatrixXd &A);

/**
 * @brief Realiza Analisis de Componentes Principales (PCA).
 *
 * @param data Matriz de datos (filas = muestras, columnas = caracteristicas).
 * @param k Numero de componentes principales a mantener.
 * @return Eigen::MatrixXd Matriz proyectada en k dimensiones.
 */
Eigen::MatrixXd PCA(const Eigen::MatrixXd &data, int k);

} // namespace MatrixDecompositions
} // namespace MathMl
