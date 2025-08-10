# src/shilov/transformations.py

"""
Módulo para transformações lineares especiais.

Este módulo implementa transformações lineares específicas como rotações,
reflexões, projeções e cisalhamentos, seguindo a teoria apresentada
no livro de Shilov.
"""

from typing import List, Sequence, Tuple, Optional
import math
from .matrices import matrix, identity, zeros, multm
from .vectors import vector, magnitude


def rotation_matrix_2d(angle: float) -> List[List[float]]:
    """
    Cria uma matriz de rotação 2D.
    
    Parameters
    ----------
    angle : float
        Ângulo de rotação em radianos.
        
    Returns
    -------
    list of list of float
        Matriz de rotação 2×2.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    return [[cos_a, -sin_a],
            [sin_a, cos_a]]


def reflection_matrix_2d(axis: str = 'x') -> List[List[float]]:
    """
    Cria uma matriz de reflexão 2D.
    
    Parameters
    ----------
    axis : str
        Eixo de reflexão: 'x', 'y', 'origin', ou 'line_y_equals_x'.
        
    Returns
    -------
    list of list of float
        Matriz de reflexão 2×2.
        
    Raises
    ------
    ValueError
        Se o eixo especificado não for reconhecido.
    """
    if axis == 'x':
        return [[1.0, 0.0], [0.0, -1.0]]
    elif axis == 'y':
        return [[-1.0, 0.0], [0.0, 1.0]]
    elif axis == 'origin':
        return [[-1.0, 0.0], [0.0, -1.0]]
    elif axis == 'line_y_equals_x':
        return [[0.0, 1.0], [1.0, 0.0]]
    else:
        raise ValueError("Eixo deve ser 'x', 'y', 'origin' ou 'line_y_equals_x'")


def reflection_matrix_line_2d(a: float, b: float, c: float) -> List[List[float]]:
    """
    Cria matriz de reflexão sobre a reta ax + by + c = 0.
    
    Parameters
    ----------
    a, b, c : float
        Coeficientes da equação da reta.
        
    Returns
    -------
    list of list of float
        Matriz de reflexão 2×2.
        
    Raises
    ------
    ValueError
        Se a² + b² = 0 (reta indefinida).
    """
    norm_sq = a*a + b*b
    if norm_sq < 1e-10:
        raise ValueError("Coeficientes a e b não podem ser ambos zero")
    
    # Matriz de reflexão sobre ax + by = 0 (passando pela origem)
    factor = 1.0 / norm_sq
    
    return [[factor * (b*b - a*a), factor * (-2*a*b)],
            [factor * (-2*a*b), factor * (a*a - b*b)]]


def scaling_matrix_2d(sx: float, sy: float) -> List[List[float]]:
    """
    Cria uma matriz de escalonamento 2D.
    
    Parameters
    ----------
    sx, sy : float
        Fatores de escala nos eixos x e y.
        
    Returns
    -------
    list of list of float
        Matriz de escalonamento 2×2.
    """
    return [[sx, 0.0],
            [0.0, sy]]


def shear_matrix_2d(shx: float = 0.0, shy: float = 0.0) -> List[List[float]]:
    """
    Cria uma matriz de cisalhamento 2D.
    
    Parameters
    ----------
    shx : float
        Fator de cisalhamento horizontal.
    shy : float
        Fator de cisalhamento vertical.
        
    Returns
    -------
    list of list of float
        Matriz de cisalhamento 2×2.
    """
    return [[1.0, shx],
            [shy, 1.0]]


def projection_matrix_line_2d(direction: Sequence[float]) -> List[List[float]]:
    """
    Cria matriz de projeção ortogonal sobre uma reta que passa pela origem.
    
    Parameters
    ----------
    direction : sequence of float
        Vetor direção da reta (2D).
        
    Returns
    -------
    list of list of float
        Matriz de projeção 2×2.
        
    Raises
    ------
    ValueError
        Se o vetor direção for nulo ou não for 2D.
    """
    if len(direction) != 2:
        raise ValueError("Vetor direção deve ser 2D")
    
    if magnitude(direction) < 1e-10:
        raise ValueError("Vetor direção não pode ser nulo")
    
    # Normaliza o vetor direção
    norm = magnitude(direction)
    u = [direction[0] / norm, direction[1] / norm]
    
    # Matriz de projeção P = uu^T
    return [[u[0] * u[0], u[0] * u[1]],
            [u[1] * u[0], u[1] * u[1]]]


def projection_matrix_orthogonal_2d(direction: Sequence[float]) -> List[List[float]]:
    """
    Cria matriz de projeção sobre a reta ortogonal à direção dada.
    
    Parameters
    ----------
    direction : sequence of float
        Vetor direção (2D).
        
    Returns
    -------
    list of list of float
        Matriz de projeção 2×2.
    """
    # Projeção ortogonal = I - projeção na direção
    I = identity(2)
    P_parallel = projection_matrix_line_2d(direction)
    
    from .matrices import subm
    return subm(I, P_parallel)


def householder_reflection_2d(v: Sequence[float]) -> List[List[float]]:
    """
    Cria uma matriz de reflexão de Householder em 2D.
    
    A reflexão é sobre a reta perpendicular ao vetor v.
    H = I - 2(vv^T)/(v^T v)
    
    Parameters
    ----------
    v : sequence of float
        Vetor normal à reta de reflexão (2D).
        
    Returns
    -------
    list of list of float
        Matriz de reflexão de Householder 2×2.
    """
    if len(v) != 2:
        raise ValueError("Vetor deve ser 2D")
    
    if magnitude(v) < 1e-10:
        raise ValueError("Vetor não pode ser nulo")
    
    # Calcula v^T v
    v_dot_v = v[0]*v[0] + v[1]*v[1]
    
    # H = I - 2(vv^T)/(v^T v)
    factor = 2.0 / v_dot_v
    
    return [[1.0 - factor * v[0] * v[0], -factor * v[0] * v[1]],
            [-factor * v[1] * v[0], 1.0 - factor * v[1] * v[1]]]


def compose_transformations(*matrices) -> List[List[float]]:
    """
    Compõe múltiplas transformações lineares.
    
    Parameters
    ----------
    *matrices : sequence of sequence of float
        Matrizes de transformação (da direita para a esquerda).
        
    Returns
    -------
    list of list of float
        Matriz resultante da composição.
    """
    if not matrices:
        raise ValueError("Pelo menos uma matriz deve ser fornecida")
    
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = multm(matrices[i], result)
    
    return result


def inverse_transformation_2x2(T: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Calcula a transformação inversa de uma matriz 2×2.
    
    Parameters
    ----------
    T : sequence of sequence of float
        Matriz de transformação 2×2.
        
    Returns
    -------
    list of list of float
        Matriz de transformação inversa.
    """
    from .matrices import inverse
    return inverse(T)


def polar_decomposition_2x2(A: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Decomposição polar de uma matriz 2×2: A = UP onde U é ortogonal e P é positiva semidefinida.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2 invertível.
        
    Returns
    -------
    tuple of (list of list of float, list of list of float)
        Tupla (U, P) onde U é ortogonal e P é positiva semidefinida.
    """
    from .matrices import transpose, determinant
    from .operators import eigenvalues_2x2, is_diagonalizable_2x2, diagonalize_2x2
    
    if abs(determinant(A)) < 1e-10:
        raise ValueError("Matriz deve ser invertível")
    
    # Calcula A^T A
    A_T = transpose(A)
    ATA = multm(A_T, A)
    
    # P = sqrt(A^T A) - para matriz 2×2, calculamos usando diagonalização
    if not is_diagonalizable_2x2(ATA):
        raise ValueError("A^T A não é diagonalizável")
    
    Q, D = diagonalize_2x2(ATA)
    
    # P = Q * sqrt(D) * Q^(-1)
    sqrt_D = [[math.sqrt(max(0, D[0][0])), 0.0],
              [0.0, math.sqrt(max(0, D[1][1]))]]
    
    from .matrices import inverse
    Q_inv = inverse(Q)
    P = multm(multm(Q, sqrt_D), Q_inv)
    
    # U = A * P^(-1)
    P_inv = inverse(P)
    U = multm(A, P_inv)
    
    return U, P


def singular_values_2x2(A: Sequence[Sequence[float]]) -> List[float]:
    """
    Calcula os valores singulares de uma matriz 2×2.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    list of float
        Valores singulares em ordem decrescente.
    """
    from .matrices import transpose
    from .operators import eigenvalues_2x2
    
    # Valores singulares são as raízes dos autovalores de A^T A
    A_T = transpose(A)
    ATA = multm(A_T, A)
    
    eigenvals = eigenvalues_2x2(ATA)
    
    # Extrai apenas a parte real e toma a raiz quadrada
    singular_values = [math.sqrt(max(0, eigenval.real)) for eigenval in eigenvals]
    
    # Ordena em ordem decrescente
    singular_values.sort(reverse=True)
    
    return singular_values


def condition_number_2x2(A: Sequence[Sequence[float]]) -> float:
    """
    Calcula o número de condição de uma matriz 2×2.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    float
        Número de condição (razão entre maior e menor valor singular).
    """
    singular_vals = singular_values_2x2(A)
    
    if singular_vals[1] < 1e-10:
        return float('inf')  # Matriz singular
    
    return singular_vals[0] / singular_vals[1]


def is_linear_transformation(T, test_vectors: Optional[Sequence[Sequence[float]]] = None) -> bool:
    """
    Verifica se uma função T satisfaz as propriedades de linearidade.
    
    Parameters
    ----------
    T : callable
        Função a ser testada.
    test_vectors : sequence of sequence of float, optional
        Vetores de teste. Se None, usa vetores padrão.
        
    Returns
    -------
    bool
        True se T parece ser linear com os vetores testados.
    """
    if test_vectors is None:
        test_vectors = [[1, 0], [0, 1], [1, 1], [2, 3]]
    
    tolerance = 1e-10
    
    # Testa T(u + v) = T(u) + T(v)
    for i in range(len(test_vectors)):
        for j in range(i + 1, len(test_vectors)):
            u, v = test_vectors[i], test_vectors[j]
            
            from .vectors import add
            u_plus_v = add(u, v)
            
            T_u_plus_v = T(u_plus_v)
            T_u = T(u)
            T_v = T(v)
            T_u_plus_T_v = add(T_u, T_v)
            
            # Verifica se T(u + v) ≈ T(u) + T(v)
            from .vectors import sub
            diff = sub(T_u_plus_v, T_u_plus_T_v)
            if magnitude(diff) > tolerance:
                return False
    
    # Testa T(αu) = αT(u)
    scalars = [0, 1, -1, 2.5, -3.7]
    for alpha in scalars:
        for u in test_vectors:
            from .vectors import scalar_mult_vector
            alpha_u = scalar_mult_vector(u, alpha)
            
            T_alpha_u = T(alpha_u)
            T_u = T(u)
            alpha_T_u = scalar_mult_vector(T_u, alpha)
            
            # Verifica se T(αu) ≈ αT(u)
            diff = sub(T_alpha_u, alpha_T_u)
            if magnitude(diff) > tolerance:
                return False
    
    return True


def matrix_from_transformation(T, basis: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]:
    """
    Extrai a matriz de uma transformação linear aplicando-a aos vetores da base.
    
    Parameters
    ----------
    T : callable
        Transformação linear.
    basis : sequence of sequence of float, optional
        Base do espaço. Se None, usa base canônica 2D.
        
    Returns
    -------
    list of list of float
        Matriz da transformação linear.
    """
    if basis is None:
        basis = [[1, 0], [0, 1]]  # Base canônica 2D
    
    # Aplica T a cada vetor da base
    transformed_basis = [T(v) for v in basis]
    
    # Matriz tem as imagens dos vetores da base como colunas
    n = len(transformed_basis[0])  # Dimensão do espaço de chegada
    m = len(transformed_basis)     # Número de vetores da base
    
    matrix_T = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(transformed_basis[j][i])
        matrix_T.append(row)
    
    return matrix_T


def kernel_basis_2x2(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Encontra uma base para o kernel (núcleo) de uma matriz 2×2.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    list of list of float
        Base do kernel. Lista vazia se ker(A) = {0}.
    """
    from .matrices import determinant
    from .linear_systems import homogeneous_solution
    
    # Se det(A) ≠ 0, então ker(A) = {0}
    if abs(determinant(A)) > 1e-10:
        return []
    
    # Encontra soluções não-triviais de Ax = 0
    solutions = homogeneous_solution(A)
    return solutions


def image_basis_2x2(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Encontra uma base para a imagem (espaço coluna) de uma matriz 2×2.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz 2×2.
        
    Returns
    -------
    list of list of float
        Base da imagem.
    """
    from .matrices import rank
    from .vectors import are_linearly_independent
    
    # Colunas da matriz
    col1 = [A[0][0], A[1][0]]
    col2 = [A[0][1], A[1][1]]
    
    # Se as colunas são linearmente independentes
    if are_linearly_independent([col1, col2]):
        return [col1, col2]
    
    # Se uma coluna é nula, retorna a outra (se não-nula)
    if magnitude(col1) < 1e-10:
        return [col2] if magnitude(col2) > 1e-10 else []
    
    if magnitude(col2) < 1e-10:
        return [col1]
    
    # Se as colunas são proporcionais, retorna uma delas
    return [col1]


def nullity(A: Sequence[Sequence[float]]) -> int:
    """
    Calcula a nulidade de uma matriz (dimensão do kernel).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz.
        
    Returns
    -------
    int
        Nulidade da matriz.
    """
    from .matrices import rank
    return len(A[0]) - rank(A)


def rank_nullity_theorem_verify(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica o teorema do posto-nulidade: rank(A) + nullity(A) = n.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz.
        
    Returns
    -------
    bool
        True se o teorema é satisfeito.
    """
    from .matrices import rank
    n = len(A[0])  # Número de colunas
    return rank(A) + nullity(A) == n