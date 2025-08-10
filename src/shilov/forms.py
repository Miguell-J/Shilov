# src/shilov/forms.py

"""
Módulo para formas bilineares e quadráticas.

Este módulo implementa conceitos relacionados a formas bilineares, formas quadráticas,
produtos internos e suas propriedades, seguindo a abordagem do livro de Shilov.
"""

from typing import List, Sequence, Tuple, Callable
import math
from .vectors import dot, magnitude, vector
from .matrices import matrix, transpose, multm, determinant, inverse, eigenvalues


def bilinear_form(A: Sequence[Sequence[float]], x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calcula o valor de uma forma bilinear B(x, y) = x^T A y.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz que define a forma bilinear.
    x : sequence of float
        Primeiro vetor.
    y : sequence of float
        Segundo vetor.
        
    Returns
    -------
    float
        Valor da forma bilinear.
        
    Raises
    ------
    ValueError
        Se as dimensões forem incompatíveis.
    """
    if len(A) != len(x) or len(A[0]) != len(y):
        raise ValueError("Dimensões incompatíveis para forma bilinear")
    
    result = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * A[i][j] * y[j]
    
    return result


def quadratic_form(A: Sequence[Sequence[float]], x: Sequence[float]) -> float:
    """
    Calcula o valor de uma forma quadrática Q(x) = x^T A x.
    
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


def is_positive_definite(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz simétrica é positiva definida.
    Uma matriz é positiva definida se todos os seus autovalores são positivos.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
        
    Returns
    -------
    bool
        True se a matriz for positiva definida, False caso contrário.
    """
    # Método dos menores principais
    n = len(A)
    
    for k in range(1, n + 1):
        # Calcula o determinante do menor principal k×k
        minor = [[A[i][j] for j in range(k)] for i in range(k)]
        det = determinant(minor)
        
        if det <= 0:
            return False
    
    return True


def is_positive_semidefinite(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz simétrica é positiva semidefinida.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
        
    Returns
    -------
    bool
        True se a matriz for positiva semidefinida, False caso contrário.
    """
    # Método dos menores principais
    n = len(A)
    
    for k in range(1, n + 1):
        minor = [[A[i][j] for j in range(k)] for i in range(k)]
        det = determinant(minor)
        
        if det < 0:
            return False
    
    return True


def is_negative_definite(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz simétrica é negativa definida.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
        
    Returns
    -------
    bool
        True se a matriz for negativa definida, False caso contrário.
    """
    # Uma matriz é negativa definida se -A é positiva definida
    minus_A = [[-A[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return is_positive_definite(minus_A)


def is_indefinite(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz simétrica é indefinida.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica.
        
    Returns
    -------
    bool
        True se a matriz for indefinida, False caso contrário.
    """
    return not (is_positive_semidefinite(A) or is_negative_definite(A))


def gram_schmidt(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Aplica o processo de Gram-Schmidt para ortogonalizar um conjunto de vetores.
    
    Parameters
    ----------
    vectors : sequence of sequence of float
        Lista de vetores linearmente independentes.
        
    Returns
    -------
    list of list of float
        Lista de vetores ortogonais.
        
    Raises
    ------
    ValueError
        Se os vetores forem linearmente dependentes.
    """
    if not vectors:
        return []
    
    orthogonal = []
    
    for i, v in enumerate(vectors):
        # Começa com o vetor original
        u = list(v)
        
        # Subtrai as projeções nos vetores anteriores
        for j in range(i):
            proj_coeff = dot(v, orthogonal[j]) / dot(orthogonal[j], orthogonal[j])
            for k in range(len(u)):
                u[k] -= proj_coeff * orthogonal[j][k]
        
        # Verifica se o vetor resultante é não-nulo
        if magnitude(u) < 1e-10:
            raise ValueError("Vetores são linearmente dependentes")
        
        orthogonal.append(u)
    
    return orthogonal


def orthonormalize(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Aplica o processo de Gram-Schmidt e normaliza os vetores resultantes.
    
    Parameters
    ----------
    vectors : sequence of sequence of float
        Lista de vetores linearmente independentes.
        
    Returns
    -------
    list of list of float
        Lista de vetores ortonormais.
    """
    from .vectors import norm
    
    orthogonal = gram_schmidt(vectors)
    return [norm(v) for v in orthogonal]


def inner_product(x: Sequence[float], y: Sequence[float], A: Sequence[Sequence[float]] = None) -> float:
    """
    Calcula o produto interno generalizado <x, y>_A = x^T A y.
    Se A não for fornecida, calcula o produto interno euclidiano padrão.
    
    Parameters
    ----------
    x : sequence of float
        Primeiro vetor.
    y : sequence of float
        Segundo vetor.
    A : sequence of sequence of float, optional
        Matriz que define o produto interno. Se None, usa a identidade.
        
    Returns
    -------
    float
        Valor do produto interno.
    """
    if A is None:
        return dot(x, y)
    else:
        return bilinear_form(A, x, y)


def norm_induced(x: Sequence[float], A: Sequence[Sequence[float]] = None) -> float:
    """
    Calcula a norma induzida por um produto interno.
    
    Parameters
    ----------
    x : sequence of float
        Vetor de entrada.
    A : sequence of sequence of float, optional
        Matriz que define o produto interno. Se None, usa norma euclidiana.
        
    Returns
    -------
    float
        Norma do vetor.
    """
    return math.sqrt(inner_product(x, x, A))


def angle_generalized(x: Sequence[float], y: Sequence[float], A: Sequence[Sequence[float]] = None) -> float:
    """
    Calcula o ângulo entre dois vetores usando um produto interno generalizado.
    
    Parameters
    ----------
    x : sequence of float
        Primeiro vetor.
    y : sequence of float
        Segundo vetor.
    A : sequence of sequence of float, optional
        Matriz que define o produto interno.
        
    Returns
    -------
    float
        Ângulo entre os vetores em radianos.
    """
    inner_xy = inner_product(x, y, A)
    norm_x = norm_induced(x, A)
    norm_y = norm_induced(y, A)
    
    if norm_x == 0 or norm_y == 0:
        raise ValueError("Não é possível calcular ângulo com vetores nulos")
    
    cos_theta = inner_xy / (norm_x * norm_y)
    return math.acos(max(-1, min(1, cos_theta)))


def qr_decomposition(A: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Calcula a decomposição QR de uma matriz usando Gram-Schmidt.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz m×n com colunas linearmente independentes.
        
    Returns
    -------
    tuple of (list of list of float, list of list of float)
        Tupla (Q, R) onde Q tem colunas ortonormais e R é triangular superior.
    """
    if not A or not A[0]:
        return [], []
    
    m, n = len(A), len(A[0])
    
    # Converte colunas da matriz em vetores
    columns = [[A[i][j] for i in range(m)] for j in range(n)]
    
    # Aplica Gram-Schmidt
    Q_columns = orthonormalize(columns)
    
    # Constrói Q
    Q = [[Q_columns[j][i] for j in range(n)] for i in range(m)]
    
    # Calcula R = Q^T A
    Q_T = transpose(Q)
    R = multm(Q_T, A)
    
    return Q, R


def cholesky_decomposition(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Calcula a decomposição de Cholesky de uma matriz positiva definida (A = LL^T).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica positiva definida.
        
    Returns
    -------
    list of list of float
        Matriz triangular inferior L tal que A = LL^T.
        
    Raises
    ------
    ValueError
        Se a matriz não for positiva definida.
    """
    if not is_positive_definite(A):
        raise ValueError("Matriz deve ser positiva definida para decomposição de Cholesky")
    
    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Elementos da diagonal
                sum_squares = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = math.sqrt(A[i][i] - sum_squares)
            else:  # Elementos abaixo da diagonal
                sum_products = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (A[i][j] - sum_products) / L[j][j]
    
    return L


def sylvester_criterion(A: Sequence[Sequence[float]]) -> str:
    """
    Aplica o critério de Sylvester para classificar uma forma quadrática.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz simétrica que define a forma quadrática.
        
    Returns
    -------
    str
        Classificação: 'positive_definite', 'negative_definite', 'positive_semidefinite', 
        'negative_semidefinite', ou 'indefinite'.
    """
    if is_positive_definite(A):
        return 'positive_definite'
    elif is_negative_definite(A):
        return 'negative_definite'
    elif is_positive_semidefinite(A):
        return 'positive_semidefinite'
    elif determinant([[-A[i][j] for j in range(len(A[0]))] for i in range(len(A))]) >= 0:
        return 'negative_semidefinite'
    else:
        return 'indefinite'


def polarization(Q: Callable[[Sequence[float]], float], x: Sequence[float], y: Sequence[float]) -> float:
    """
    Aplica a identidade de polarização para recuperar a forma bilinear a partir da quadrática.
    B(x,y) = 1/4[Q(x+y) - Q(x-y)]
    
    Parameters
    ----------
    Q : callable
        Função que calcula a forma quadrática.
    x : sequence of float
        Primeiro vetor.
    y : sequence of float
        Segundo vetor.
        
    Returns
    -------
    float
        Valor da forma bilinear B(x,y).
    """
    from .vectors import add, sub
    
    x_plus_y = add(x, y)
    x_minus_y = sub(x, y)
    
    return 0.25 * (Q(x_plus_y) - Q(x_minus_y))