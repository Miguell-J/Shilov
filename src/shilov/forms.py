# src/shilov/forms.py

"""
Módulo para formas bilineares e quadráticas.

Este módulo implementa conceitos relacionados a formas bilineares, formas quadráticas,
produtos internos e ortogonalização, seguindo a teoria apresentada no livro de Shilov.
"""

from typing import List, Sequence, Tuple, Optional
import math
from .vectors import vector, magnitude, add, sub, scalar_mult_vector
from .matrices import matrix, transpose, multm


def inner_product(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Calcula o produto interno padrão (euclidiano) entre dois vetores.
    
    Parameters
    ----------
    u, v : sequence of float
        Vetores de entrada.
        
    Returns
    -------
    float
        Produto interno <u, v>.
        
    Raises
    ------
    ValueError
        Se os vetores têm dimensões diferentes.
    """
    if len(u) != len(v):
        raise ValueError("Vetores devem ter a mesma dimensão")
    
    return sum(u[i] * v[i] for i in range(len(u)))


def bilinear_form(A: Sequence[Sequence[float]], u: Sequence[float], v: Sequence[float]) -> float:
    """
    Calcula a forma bilinear B(u,v) = u^T A v.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz que define a forma bilinear.
    u, v : sequence of float
        Vetores de entrada.
        
    Returns
    -------
    float
        Valor da forma bilinear.
    """
    from .matrices import matrix_vector_mult
    Av = matrix_vector_mult(A, v)
    return inner_product(u, Av)


def quadratic_form(A: Sequence[Sequence[float]], x: Sequence[float]) -> float:
    """
    Calcula a forma quadrática Q(x) = x^T A x.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica que define a forma quadrática.
    x : sequence of float
        Vetor de entrada.
        
    Returns
    -------
    float
        Valor da forma quadrática.
    """
    return bilinear_form(A, x, x)


def is_positive_definite(A: Sequence[Sequence[float]], tolerance: float = 1e-10) -> bool:
    """
    Verifica se uma matriz simétrica é positiva definida.
    
    Uma matriz é positiva definida se todos os seus autovalores são positivos,
    equivalentemente, se x^T A x > 0 para todo x ≠ 0.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
    tolerance : float
        Tolerância numérica.
        
    Returns
    -------
    bool
        True se a matriz for positiva definida.
    """
    from .operators import eigenvalues_2x2
    
    # Para matrizes 2×2
    if len(A) == 2:
        eigenvals = eigenvalues_2x2(A)
        return (eigenvals[0].real > tolerance and eigenvals[1].real > tolerance and
                eigenvals[0].imag == 0 and eigenvals[1].imag == 0)
    
    # Critério de Sylvester para caso geral
    return all(leading_principal_minor(A, k) > tolerance for k in range(1, len(A) + 1))


def leading_principal_minor(A: Sequence[Sequence[float]], k: int) -> float:
    """
    Calcula o k-ésimo menor principal de uma matriz.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    k : int
        Ordem do menor principal (1 ≤ k ≤ n).
        
    Returns
    -------
    float
        Valor do k-ésimo menor principal.
    """
    from .matrices import determinant
    
    # Extrai a submatriz k×k do canto superior esquerdo
    submatrix = [[A[i][j] for j in range(k)] for i in range(k)]
    return determinant(submatrix)


def gram_schmidt(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Aplica o processo de ortogonalização de Gram-Schmidt.
    
    Parameters
    ----------
    vectors : sequence of sequence of float
        Lista de vetores linearmente independentes.
        
    Returns
    -------
    list of list of float
        Lista de vetores ortonormais.
    """
    orthogonal = []
    
    for v in vectors:
        # Cópia do vetor atual
        w = list(v)
        
        # Subtrai as projeções nos vetores anteriores
        for u in orthogonal:
            proj_coeff = inner_product(v, u) / inner_product(u, u)
            projection = scalar_mult_vector(u, proj_coeff)
            w = sub(w, projection)
        
        # Normaliza o vetor
        norm_w = magnitude(w)
        if norm_w > 1e-10:
            w = scalar_mult_vector(w, 1.0 / norm_w)
            orthogonal.append(w)
    
    return orthogonal


def qr_decomposition_2x2(A: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Calcula a decomposição QR de uma matriz 2×2.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    tuple of (list of list of float, list of list of float)
        Tupla (Q, R) onde Q é ortogonal e R é triangular superior.
    """
    if len(A) != 2 or len(A[0]) != 2:
        raise ValueError("Esta função é apenas para matrizes 2×2")
    
    # Colunas da matriz A
    col1 = [A[0][0], A[1][0]]
    col2 = [A[0][1], A[1][1]]
    
    # Aplica Gram-Schmidt às colunas
    orthonormal = gram_schmidt([col1, col2])
    
    if len(orthonormal) < 2:
        raise ValueError("Colunas são linearmente dependentes")
    
    q1, q2 = orthonormal[0], orthonormal[1]
    
    # Matriz Q (colunas são os vetores ortonormais)
    Q = [[q1[0], q2[0]], 
         [q1[1], q2[1]]]
    
    # Matriz R = Q^T A
    Q_T = transpose(Q)
    R = multm(Q_T, A)
    
    return Q, R


def angle_between_vectors(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Calcula o ângulo entre dois vetores em radianos.
    
    Parameters
    ----------
    u, v : sequence of float
        Vetores de entrada.
        
    Returns
    -------
    float
        Ângulo em radianos (0 ≤ θ ≤ π).
    """
    dot_product = inner_product(u, v)
    norm_u = magnitude(u)
    norm_v = magnitude(v)
    
    if norm_u < 1e-10 or norm_v < 1e-10:
        raise ValueError("Vetores não podem ser nulos")
    
    cos_angle = dot_product / (norm_u * norm_v)
    # Garante que cos_angle está no intervalo [-1, 1]
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.acos(cos_angle)


def projection(u: Sequence[float], v: Sequence[float]) -> List[float]:
    """
    Calcula a projeção ortogonal de u sobre v.
    
    Parameters
    ----------
    u : sequence of float
        Vetor a ser projetado.
    v : sequence of float
        Vetor sobre o qual projetar.
        
    Returns
    -------
    list of float
        Projeção de u sobre v.
    """
    if magnitude(v) < 1e-10:
        raise ValueError("Vetor v não pode ser nulo")
    
    coeff = inner_product(u, v) / inner_product(v, v)
    return scalar_mult_vector(v, coeff)


def orthogonal_complement_2d(v: Sequence[float]) -> List[float]:
    """
    Encontra um vetor ortogonal a v no R².
    
    Parameters
    ----------
    v : sequence of float
        Vetor 2D.
        
    Returns
    -------
    list of float
        Vetor ortogonal a v.
        
    Raises
    ------
    ValueError
        Se v não for 2D ou for nulo.
    """
    if len(v) != 2:
        raise ValueError("Vetor deve ser 2D")
    
    if magnitude(v) < 1e-10:
        raise ValueError("Vetor não pode ser nulo")
    
    # Se v = (a, b), então (-b, a) é ortogonal
    return [-v[1], v[0]]


def matrix_of_bilinear_form(basis_u: Sequence[Sequence[float]], 
                           basis_v: Sequence[Sequence[float]],
                           form_func) -> List[List[float]]:
    """
    Calcula a matriz de uma forma bilinear em relação a bases dadas.
    
    Parameters
    ----------
    basis_u, basis_v : sequence of sequence of float
        Bases para os dois espaços vetoriais.
    form_func : callable
        Função que calcula a forma bilinear B(u, v).
        
    Returns
    -------
    list of list of float
        Matriz da forma bilinear.
    """
    m, n = len(basis_u), len(basis_v)
    matrix_form = []
    
    for i in range(m):
        row = []
        for j in range(n):
            value = form_func(basis_u[i], basis_v[j])
            row.append(value)
        matrix_form.append(row)
    
    return matrix_form


def is_orthogonal_matrix(Q: Sequence[Sequence[float]], tolerance: float = 1e-10) -> bool:
    """
    Verifica se uma matriz é ortogonal (Q^T Q = I).
    
    Parameters
    ----------
    Q : sequence of sequence of float
        Matriz a ser verificada.
    tolerance : float
        Tolerância numérica.
        
    Returns
    -------
    bool
        True se a matriz for ortogonal.
    """
    from .matrices import identity
    
    Q_T = transpose(Q)
    product = multm(Q_T, Q)
    I = identity(len(Q))
    
    # Verifica se Q^T Q = I
    for i in range(len(Q)):
        for j in range(len(Q)):
            diff = abs(product[i][j] - I[i][j])
            if diff > tolerance:
                return False
    
    return True