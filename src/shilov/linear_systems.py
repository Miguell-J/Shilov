# src/shilov/linear_systems.py

"""
Módulo para resolução de sistemas lineares.

Este módulo implementa vários métodos para resolver sistemas da forma Ax = b,
onde A é uma matriz, x é o vetor de incógnitas e b é o vetor de termos independentes.
"""

from typing import List, Sequence, Tuple, Optional
import math
from .matrices import matrix, determinant, zeros, identity, transpose, multm
from .vectors import vector, magnitude, sub, add


def gauss_elimination(A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """
    Resolve um sistema linear usando eliminação gaussiana com substituição regressiva.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes n×n.
    b : sequence of float
        Vetor de termos independentes de tamanho n.
        
    Returns
    -------
    list of float
        Vetor solução x.
        
    Raises
    ------
    ValueError
        Se o sistema não tiver solução única ou se as dimensões forem incompatíveis.
    """
    if len(A) != len(b) or len(A) != len(A[0]):
        raise ValueError("Dimensões incompatíveis ou matriz não quadrada")
    
    n = len(A)
    
    # Cria matriz aumentada [A|b]
    augmented = []
    for i in range(n):
        augmented.append(list(A[i]) + [b[i]])
    
    # Eliminação progressiva
    for i in range(n):
        # Encontra o pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Troca linhas se necessário
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Verifica se o pivot é zero
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Sistema não tem solução única (matriz singular)")
        
        # Eliminação
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]
    
    # Substituição regressiva
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]
    
    return x


def gauss_jordan(A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """
    Resolve um sistema linear usando eliminação de Gauss-Jordan.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes n×n.
    b : sequence of float
        Vetor de termos independentes de tamanho n.
        
    Returns
    -------
    list of float
        Vetor solução x.
    """
    if len(A) != len(b) or len(A) != len(A[0]):
        raise ValueError("Dimensões incompatíveis ou matriz não quadrada")
    
    n = len(A)
    
    # Cria matriz aumentada [A|b]
    augmented = []
    for i in range(n):
        augmented.append(list(A[i]) + [b[i]])
    
    # Eliminação de Gauss-Jordan
    for i in range(n):
        # Encontra o pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Troca linhas se necessário
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Verifica se o pivot é zero
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Sistema não tem solução única (matriz singular)")
        
        # Normaliza a linha do pivot
        pivot = augmented[i][i]
        for j in range(n + 1):
            augmented[i][j] /= pivot
        
        # Elimina a coluna i em todas as outras linhas
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(n + 1):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extrai a solução
    return [augmented[i][n] for i in range(n)]


def lu_decomposition(A: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Calcula a decomposição LU de uma matriz (A = LU).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada n×n.
        
    Returns
    -------
    tuple of (list of list of float, list of list of float)
        Tupla (L, U) onde L é triangular inferior e U é triangular superior.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matriz deve ser quadrada")
    
    n = len(A)
    L = zeros(n, n)
    U = zeros(n, n)
    
    for i in range(n):
        # Calcula U[i][j] para j >= i
        for j in range(i, n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
        
        # Calcula L[i][j] para j <= i
        for j in range(i + 1):
            if i == j:
                L[i][j] = 1.0  # Diagonal de L é 1
            else:
                L[i][j] = A[i][j]
                for k in range(j):
                    L[i][j] -= L[i][k] * U[k][j]
                L[i][j] /= U[j][j]
    
    return L, U


def solve_lu(L: Sequence[Sequence[float]], U: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """
    Resolve um sistema Ax = b usando decomposição LU (onde A = LU).
    
    Parameters
    ----------
    L : sequence of sequence of float
        Matriz triangular inferior.
    U : sequence of sequence of float
        Matriz triangular superior.
    b : sequence of float
        Vetor de termos independentes.
        
    Returns
    -------
    list of float
        Vetor solução x.
    """
    n = len(b)
    
    # Resolve Ly = b (substituição progressiva)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]
    
    # Resolve Ux = y (substituição regressiva)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    
    return x


def jacobi_method(A: Sequence[Sequence[float]], b: Sequence[float], 
                  x0: Optional[Sequence[float]] = None, max_iter: int = 1000, 
                  tolerance: float = 1e-10) -> List[float]:
    """
    Resolve um sistema linear usando o método iterativo de Jacobi.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes n×n.
    b : sequence of float
        Vetor de termos independentes.
    x0 : sequence of float, optional
        Estimativa inicial. Se None, usa vetor zero.
    max_iter : int
        Número máximo de iterações.
    tolerance : float
        Tolerância para convergência.
        
    Returns
    -------
    list of float
        Vetor solução aproximada.
        
    Raises
    ------
    ValueError
        Se o método não convergir ou se a matriz não for diagonalmente dominante.
    """
    if len(A) != len(b) or len(A) != len(A[0]):
        raise ValueError("Dimensões incompatíveis ou matriz não quadrada")
    
    n = len(A)
    
    # Verifica se a diagonal não tem zeros
    for i in range(n):
        if abs(A[i][i]) < 1e-10:
            raise ValueError("Elemento diagonal zero encontrado")
    
    # Estimativa inicial
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)
    
    for iteration in range(max_iter):
        x_new = [0.0] * n
        
        for i in range(n):
            sum_ax = 0.0
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_ax) / A[i][i]
        
        # Verifica convergência
        if magnitude(sub(x_new, x)) < tolerance:
            return x_new
        
        x = x_new
    
    raise ValueError(f"Método de Jacobi não convergiu em {max_iter} iterações")


def gauss_seidel_method(A: Sequence[Sequence[float]], b: Sequence[float], 
                        x0: Optional[Sequence[float]] = None, max_iter: int = 1000, 
                        tolerance: float = 1e-10) -> List[float]:
    """
    Resolve um sistema linear usando o método iterativo de Gauss-Seidel.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes n×n.
    b : sequence of float
        Vetor de termos independentes.
    x0 : sequence of float, optional
        Estimativa inicial. Se None, usa vetor zero.
    max_iter : int
        Número máximo de iterações.
    tolerance : float
        Tolerância para convergência.
        
    Returns
    -------
    list of float
        Vetor solução aproximada.
    """
    if len(A) != len(b) or len(A) != len(A[0]):
        raise ValueError("Dimensões incompatíveis ou matriz não quadrada")
    
    n = len(A)
    
    # Verifica se a diagonal não tem zeros
    for i in range(n):
        if abs(A[i][i]) < 1e-10:
            raise ValueError("Elemento diagonal zero encontrado")
    
    # Estimativa inicial
    if x0 is None:
        x = [0.0] * n
    else:
        x = list(x0)
    
    for iteration in range(max_iter):
        x_old = x[:]
        
        for i in range(n):
            sum_ax = 0.0
            for j in range(n):
                if i != j:
                    sum_ax += A[i][j] * x[j]
            x[i] = (b[i] - sum_ax) / A[i][i]
        
        # Verifica convergência
        if magnitude(sub(x, x_old)) < tolerance:
            return x
    
    raise ValueError(f"Método de Gauss-Seidel não convergiu em {max_iter} iterações")


def is_diagonally_dominant(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz é diagonalmente dominante.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
        
    Returns
    -------
    bool
        True se a matriz for diagonalmente dominante, False caso contrário.
    """
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i][i])
        sum_off_diagonal = sum(abs(A[i][j]) for j in range(n) if i != j)
        if diagonal <= sum_off_diagonal:
            return False
    return True


def residual(A: Sequence[Sequence[float]], x: Sequence[float], b: Sequence[float]) -> List[float]:
    """
    Calcula o resíduo r = b - Ax de um sistema linear.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes.
    x : sequence of float
        Vetor solução.
    b : sequence of float
        Vetor de termos independentes.
        
    Returns
    -------
    list of float
        Vetor resíduo r = b - Ax.
    """
    from .matrices import matrix_vector_mult
    Ax = matrix_vector_mult(A, x)
    return sub(b, Ax)


def condition_number(A: Sequence[Sequence[float]]) -> float:
    """
    Calcula o número de condição de uma matriz (aproximação usando norma de Frobenius).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada não-singular.
        
    Returns
    -------
    float
        Número de condição aproximado.
        
    Raises
    ------
    ValueError
        Se a matriz for singular.
    """
    from .matrices import inverse, frobenius_norm
    
    try:
        A_inv = inverse(A)
        return frobenius_norm(A) * frobenius_norm(A_inv)
    except ValueError:
        raise ValueError("Matriz é singular, número de condição é infinito")


def solve_system(A: Sequence[Sequence[float]], b: Sequence[float], method: str = 'gauss') -> List[float]:
    """
    Resolve um sistema linear usando o método especificado.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de coeficientes n×n.
    b : sequence of float
        Vetor de termos independentes.
    method : str
        Método a ser usado: 'gauss', 'gauss_jordan', 'lu', 'jacobi', 'gauss_seidel'.
        
    Returns
    -------
    list of float
        Vetor solução.
        
    Raises
    ------
    ValueError
        Se o método especificado não for válido.
    """
    method = method.lower()
    
    if method == 'gauss':
        return gauss_elimination(A, b)
    elif method == 'gauss_jordan':
        return gauss_jordan(A, b)
    elif method == 'lu':
        L, U = lu_decomposition(A)
        return solve_lu(L, U, b)
    elif method == 'jacobi':
        return jacobi_method(A, b)
    elif method == 'gauss_seidel':
        return gauss_seidel_method(A, b)
    else:
        raise ValueError(f"Método '{method}' não reconhecido. Use: 'gauss', 'gauss_jordan', 'lu', 'jacobi', 'gauss_seidel'")


def homogeneous_solution(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Encontra uma base para o espaço nulo (kernel) de uma matriz usando eliminação gaussiana.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz m×n.
        
    Returns
    -------
    list of list of float
        Base do espaço nulo como lista de vetores.
    """
    if not A or not A[0]:
        return []
    
    m, n = len(A), len(A[0])
    
    # Cria cópia da matriz para eliminação
    M = [row[:] for row in A]
    
    # Eliminação gaussiana para forma escalonada reduzida
    pivot_cols = []
    current_row = 0
    
    for col in range(n):
        # Encontra pivot na coluna atual
        pivot_row = -1
        for row in range(current_row, m):
            if abs(M[row][col]) > 1e-10:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue  # Coluna livre
        
        pivot_cols.append(col)
        
        # Move linha do pivot
        if pivot_row != current_row:
            M[current_row], M[pivot_row] = M[pivot_row], M[current_row]
        
        # Normaliza a linha do pivot
        pivot = M[current_row][col]
        for j in range(n):
            M[current_row][j] /= pivot
        
        # Elimina outras linhas
        for row in range(m):
            if row != current_row and abs(M[row][col]) > 1e-10:
                factor = M[row][col]
                for j in range(n):
                    M[row][j] -= factor * M[current_row][j]
        
        current_row += 1
    
    # Identifica variáveis livres
    free_vars = [i for i in range(n) if i not in pivot_cols]
    
    if not free_vars:
        return [[0.0] * n]  # Apenas solução trivial
    
    # Constrói base do espaço nulo
    null_space = []
    
    for free_var in free_vars:
        solution = [0.0] * n
        solution[free_var] = 1.0
        
        # Para cada variável pivot, calcula seu valor
        for i in range(len(pivot_cols) - 1, -1, -1):
            pivot_col = pivot_cols[i]
            value = 0.0
            
            for j in range(pivot_col + 1, n):
                value -= M[i][j] * solution[j]
            
            solution[pivot_col] = value
        
        null_space.append(solution)
    
    return null_space