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

/**
 * @brief Resuelve un sistema de ecuaciones lineales Ax = b.
 *
 * @param A Matriz de coeficientes.
 * @param b Vector de constantes.
 * @return Eigen::VectorXd Vector solucion x.
 */
Eigen::VectorXd SolveLinearSystem(const Eigen::MatrixXd &A,
                                  const Eigen::VectorXd &b);

/**
 * @brief Calcula la inversa de una matriz cuadrada.
 *
 * @param A Matriz a invertir.
 * @return Eigen::MatrixXd Matriz inversa.
 */
Eigen::MatrixXd CalculateInverse(const Eigen::MatrixXd &A);

/**
 * @brief Calcula el determinante de una matriz cuadrada.
 *
 * @param A Matriz de entrada.
 * @return double Determinante.
 */
double CalculateDeterminant(const Eigen::MatrixXd &A);

/**
 * @brief Calcula la traza de una matriz cuadrada.
 *
 * Suma de los elementos de la diagonal principal.
 *
 * @param A Matriz de entrada.
 * @return double Traza.
 */
double CalculateTrace(const Eigen::MatrixXd &A);

/**
 * @brief Calcula el rango de una matriz.
 *
 * Numero de filas o columnas linealmente independientes.
 *
 * @param A Matriz de entrada.
 * @return int Rango.
 */
int CalculateRank(const Eigen::MatrixXd &A);

/**
 * @brief Comprueba si las columnas de una matriz son linealmente
 * independientes.
 *
 * @param A Matriz de entrada (vectores columna).
 * @return true Si las columnas son linealmente independientes.
 * @return false Si las columnas son linealmente dependientes.
 */
bool IsLinearlyIndependent(const Eigen::MatrixXd &A);

/**
 * @brief Realiza una transformacion afin y = Ax + b.
 *
 * Usada tipicamente en capas densas de redes neuronales.
 *
 * @param A Matriz de pesos.
 * @param x Vector de entrada.
 * @param b Vector de sesgo (bias).
 * @return Eigen::VectorXd Vector resultante y.
 */
Eigen::VectorXd AffineForward(const Eigen::MatrixXd &A,
                              const Eigen::VectorXd &x,
                              const Eigen::VectorXd &b);

} // namespace LinearAlgebra
} // namespace MathMl
