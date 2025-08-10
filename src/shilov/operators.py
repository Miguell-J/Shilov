# src/shilov/operators.py

"""
Módulo para operadores lineares.

Este módulo implementa conceitos relacionados a transformações lineares,
autovalores, autovetores e diagonalização, seguindo a teoria apresentada
no livro de Shilov.
"""

from typing import List, Sequence, Tuple, Optional, Union
import math
from .matrices import (matrix, identity, zeros, transpose, determinant, 
                      inverse, multm, addm, subm, scalar_mult)
from .vectors import vector, magnitude, norm, add, sub
from .linear_systems import solve_system


def apply_linear_operator(A: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    """
    Aplica um operador linear (representado por uma matriz) a um vetor.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz que representa o operador linear.
    x : sequence of float
        Vetor de entrada.
        
    Returns
    -------
    list of float
        Vetor resultante Ax.
    """
    from .matrices import matrix_vector_mult
    return matrix_vector_mult(A, x)


def characteristic_polynomial_2x2(A: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """
    Calcula os coeficientes do polinômio característico de uma matriz 2×2.
    Para uma matriz 2×2, o polinômio é λ² - tr(A)λ + det(A).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    tuple of float
        Coeficientes (a, b, c) do polinômio aλ² + bλ + c.
        
    Raises
    ------
    ValueError
        Se a matriz não for 2×2.
    """
    if len(A) != 2 or len(A[0]) != 2:
        raise ValueError("Esta função é apenas para matrizes 2×2")
    
    from .matrices import trace
    
    tr_A = trace(A)
    det_A = determinant(A)
    
    return (1.0, -tr_A, det_A)


def eigenvalues_2x2(A: Sequence[Sequence[float]]) -> List[complex]:
    """
    Calcula os autovalores de uma matriz 2×2 usando a fórmula quadrática.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    list of complex
        Lista dos autovalores (podem ser complexos).
    """
    a, b, c = characteristic_polynomial_2x2(A)
    
    # Resolve aλ² + bλ + c = 0
    discriminant = b*b - 4*a*c
    
    if discriminant >= 0:
        sqrt_discriminant = math.sqrt(discriminant)
        lambda1 = (-b + sqrt_discriminant) / (2*a)
        lambda2 = (-b - sqrt_discriminant) / (2*a)
        return [complex(lambda1, 0), complex(lambda2, 0)]
    else:
        sqrt_discriminant = math.sqrt(-discriminant)
        real_part = -b / (2*a)
        imag_part = sqrt_discriminant / (2*a)
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]


def power_method(A: Sequence[Sequence[float]], max_iter: int = 1000, 
                tolerance: float = 1e-10, initial_vector: Optional[Sequence[float]] = None) -> Tuple[float, List[float]]:
    """
    Calcula o maior autovalor (em módulo) e seu autovetor usando o método da potência.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    max_iter : int
        Número máximo de iterações.
    tolerance : float
        Tolerância para convergência.
    initial_vector : sequence of float, optional
        Vetor inicial. Se None, usa vetor aleatório.
        
    Returns
    -------
    tuple of (float, list of float)
        Tupla (autovalor, autovetor).
        
    Raises
    ------
    ValueError
        Se o método não convergir.
    """
    n = len(A)
    
    # Vetor inicial
    if initial_vector is None:
        v = [1.0 if i == 0 else 0.1 for i in range(n)]
    else:
        v = list(initial_vector)
    
    # Normaliza o vetor inicial
    v = norm(v)
    
    eigenvalue = 0.0
    
    for iteration in range(max_iter):
        # v_new = Av / ||Av||
        Av = apply_linear_operator(A, v)
        v_new = norm(Av)
        
        # Estima o autovalor usando o quociente de Rayleigh
        from .forms import inner_product
        eigenvalue_new = inner_product(v, Av) / inner_product(v, v)
        
        # Verifica convergência
        if abs(eigenvalue_new - eigenvalue) < tolerance and magnitude(sub(v_new, v)) < tolerance:
            return eigenvalue_new, v_new
        
        eigenvalue = eigenvalue_new
        v = v_new
    
    raise ValueError(f"Método da potência não convergiu em {max_iter} iterações")


def rayleigh_quotient(A: Sequence[Sequence[float]], x: Sequence[float]) -> float:
    """
    Calcula o quociente de Rayleigh R(x) = (x^T A x) / (x^T x).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
    x : sequence of float
        Vetor não-nulo.
        
    Returns
    -------
    float
        Quociente de Rayleigh.
        
    Raises
    ------
    ValueError
        Se x for vetor nulo.
    """
    from .vectors import dot
    
    if magnitude(x) < 1e-10:
        raise ValueError("Vetor não pode ser nulo")
    
    Ax = apply_linear_operator(A, x)
    numerator = dot(x, Ax)
    denominator = dot(x, x)
    
    return numerator / denominator


def is_eigenvalue(A: Sequence[Sequence[float]], lambda_val: float, tolerance: float = 1e-10) -> bool:
    """
    Verifica se um valor é autovalor de uma matriz.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    lambda_val : float
        Valor candidato a autovalor.
    tolerance : float
        Tolerância numérica.
        
    Returns
    -------
    bool
        True se lambda_val for autovalor de A.
    """
    n = len(A)
    I = identity(n)
    
    # Calcula A - λI
    lambda_I = scalar_mult(I, lambda_val)
    A_minus_lambda_I = subm(A, lambda_I)
    
    # Verifica se det(A - λI) ≈ 0
    det = determinant(A_minus_lambda_I)
    return abs(det) < tolerance


def find_eigenvector(A: Sequence[Sequence[float]], eigenvalue: float, 
                    tolerance: float = 1e-10) -> List[float]:
    """
    Encontra um autovetor correspondente a um autovalor dado.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    eigenvalue : float
        Autovalor conhecido.
    tolerance : float
        Tolerância numérica.
        
    Returns
    -------
    list of float
        Autovetor correspondente.
        
    Raises
    ------
    ValueError
        Se o valor dado não for um autovalor.
    """
    if not is_eigenvalue(A, eigenvalue, tolerance):
        raise ValueError("Valor fornecido não é um autovalor")
    
    n = len(A)
    I = identity(n)
    
    # Calcula A - λI
    lambda_I = scalar_mult(I, eigenvalue)
    A_minus_lambda_I = subm(A, lambda_I)
    
    # Encontra o espaço nulo de (A - λI)
    from .linear_systems import homogeneous_solution
    null_space = homogeneous_solution(A_minus_lambda_I)
    
    if not null_space:
        raise ValueError("Erro na determinação do autovetor")
    
    return null_space[0]


def geometric_multiplicity(A: Sequence[Sequence[float]], eigenvalue: float) -> int:
    """
    Calcula a multiplicidade geométrica de um autovalor.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    eigenvalue : float
        Autovalor.
        
    Returns
    -------
    int
        Multiplicidade geométrica (dimensão do autoespaço).
    """
    n = len(A)
    I = identity(n)
    
    # Calcula A - λI
    lambda_I = scalar_mult(I, eigenvalue)
    A_minus_lambda_I = subm(A, lambda_I)
    
    # A multiplicidade geométrica é a dimensão do espaço nulo
    from .matrices import rank
    return n - rank(A_minus_lambda_I)


def is_diagonalizable_2x2(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz 2×2 é diagonalizável.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    bool
        True se a matriz for diagonalizável.
    """
    #CONTINUA