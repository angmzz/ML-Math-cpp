cmake_minimum_required(VERSION 3.10)
project(ML-Math-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32 AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
    message(STATUS "Adding Conda Library path: $ENV{CONDA_PREFIX}/Library")
endif()

find_package(Eigen3 3.3 REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(mathml_cpp 
    python_bindings/bindings.cpp
    src/capitulo1_algebra_lineal/algebra_lineal.cpp
    src/capitulo2_geometria_analitica/geometria.cpp
    src/capitulo3_descomposiciones_matrices/descomposiciones.cpp
    src/capitulo4_calculo_vectorial/calculo.cpp
    src/capitulo5_probabilidad/probabilidad.cpp
    src/capitulo6_optimizacion/optimizacion.cpp
)

target_link_libraries(mathml_cpp PRIVATE Eigen3::Eigen)
target_include_directories(mathml_cpp PRIVATE 
    src/capitulo1_algebra_lineal
    src/capitulo2_geometria_analitica
    src/capitulo3_descomposiciones_matrices
    src/capitulo4_calculo_vectorial
    src/capitulo5_probabilidad
    src/capitulo6_optimizacion
)

set_target_properties(mathml_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
