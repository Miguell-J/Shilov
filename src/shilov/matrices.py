## src/shilov/matrices.py

"""
Módulo para operações com matrizes.

Aqui implementamos o conceito de matrizes como transformações lineares em vetores.
Uma matriz pode ser vista como uma transformação que mapeia vetores de um espaço
para outro, preservando as operações de soma e multiplicação por escalar.
"""

from typing import List, Sequence, Union, Optional
import math
from .vectors import vector, dot


def matrix(data: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Cria uma matriz a partir de uma sequência de sequências de números.

    Parameters
    ----------
    data : sequence of sequence of float
        Sequência de sequências de números representando as linhas da matriz.

    Returns
    -------
    list of list of float
        Lista de listas de floats representando a matriz criada.
    
    Raises
    ------
    ValueError
        Se as linhas não tiverem o mesmo tamanho.
    """
    if not data:
        return []
    
    # Verifica se todas as linhas têm o mesmo tamanho
    first_row_len = len(data[0])
    for i, row in enumerate(data):
        if len(row) != first_row_len:
            raise ValueError(f"Linha {i} tem tamanho {len(row)}, esperado {first_row_len}")
    
    return [list(row) for row in data]


def identity(n: int) -> List[List[float]]:
    """
    Cria uma matriz identidade de ordem n.
    
    Parameters
    ----------
    n : int
        Ordem da matriz identidade.
        
    Returns
    -------
    list of list of float
        Matriz identidade n×n.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(rows: int, cols: int) -> List[List[float]]:
    """
    Cria uma matriz de zeros.
    
    Parameters
    ----------
    rows : int
        Número de linhas.
    cols : int
        Número de colunas.
        
    Returns
    -------
    list of list of float
        Matriz de zeros rows×cols.
    """
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def ones(rows: int, cols: int) -> List[List[float]]:
    """
    Cria uma matriz de uns.
    
    Parameters
    ----------
    rows : int
        Número de linhas.
    cols : int
        Número de colunas.
        
    Returns
    -------
    list of list of float
        Matriz de uns rows×cols.
    """
    return [[1.0 for _ in range(cols)] for _ in range(rows)]


def transpose(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Calcula a transposta de uma matriz.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de entrada.
        
    Returns
    -------
    list of list of float
        Matriz transposta.
    """
    if not A or not A[0]:
        return []
    
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def addm(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Soma duas matrizes elemento a elemento.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Primeira matriz.
    B : sequence of sequence of float
        Segunda matriz.
        
    Returns
    -------
    list of list of float
        Resultado da soma A + B.
        
    Raises
    ------
    ValueError
        Se as matrizes não tiverem as mesmas dimensões.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrizes devem ter as mesmas dimensões para soma")
    
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def subm(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Subtrai duas matrizes elemento a elemento.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz minuendo.
    B : sequence of sequence of float
        Matriz subtraendo.
        
    Returns
    -------
    list of list of float
        Resultado da subtração A - B.
        
    Raises
    ------
    ValueError
        Se as matrizes não tiverem as mesmas dimensões.
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrizes devem ter as mesmas dimensões para subtração")
    
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def multm(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Multiplica duas matrizes.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Primeira matriz (m×n).
    B : sequence of sequence of float
        Segunda matriz (n×p).
        
    Returns
    -------
    list of list of float
        Resultado da multiplicação A × B (m×p).
        
    Raises
    ------
    ValueError
        Se o número de colunas de A não for igual ao número de linhas de B.
    """
    if len(A[0]) != len(B):
        raise ValueError(f"Não é possível multiplicar matriz {len(A)}×{len(A[0])} por {len(B)}×{len(B[0])}")
    
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    
    result = zeros(rows_A, cols_B)
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def scalar_mult(A: Sequence[Sequence[float]], scalar: float) -> List[List[float]]:
    """
    Multiplica uma matriz por um escalar.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de entrada.
    scalar : float
        Escalar multiplicador.
        
    Returns
    -------
    list of list of float
        Resultado da multiplicação escalar.
    """
    return [[scalar * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def powm(A: Sequence[Sequence[float]], n: int) -> List[List[float]]:
    """
    Eleva uma matriz quadrada a uma potência inteira.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
    n : int
        Expoente (deve ser não-negativo).
        
    Returns
    -------
    list of list of float
        Resultado de A^n.
        
    Raises
    ------
    ValueError
        Se a matriz não for quadrada ou n for negativo.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matriz deve ser quadrada para exponenciação")
    
    if n < 0:
        raise ValueError("Expoente deve ser não-negativo")
    
    if n == 0:
        return identity(len(A))
    
    result = [row[:] for row in A]  # Cópia da matriz
    
    for _ in range(n - 1):
        result = multm(result, A)
    
    return result


def trace(A: Sequence[Sequence[float]]) -> float:
    """
    Calcula o traço de uma matriz quadrada (soma dos elementos da diagonal principal).
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
        
    Returns
    -------
    float
        Traço da matriz.
        
    Raises
    ------
    ValueError
        Se a matriz não for quadrada.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matriz deve ser quadrada para calcular o traço")
    
    return sum(A[i][i] for i in range(len(A)))


def determinant(A: Sequence[Sequence[float]]) -> float:
    """
    Calcula o determinante de uma matriz quadrada usando eliminação gaussiana.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
        
    Returns
    -------
    float
        Determinante da matriz.
        
    Raises
    ------
    ValueError
        Se a matriz não for quadrada.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matriz deve ser quadrada para calcular determinante")
    
    n = len(A)
    
    # Casos especiais
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    # Cria uma cópia da matriz para não modificar a original
    M = [row[:] for row in A]
    det = 1.0
    
    # Eliminação gaussiana
    for i in range(n):
        # Encontra o pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        
        # Troca linhas se necessário
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            det *= -1  # Troca de linhas muda o sinal do determinante
        
        # Se o pivot é zero, determinante é zero
        if abs(M[i][i]) < 1e-10:
            return 0.0
        
        det *= M[i][i]
        
        # Eliminação
        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n):
                M[k][j] -= factor * M[i][j]
    
    return det


def inverse(A: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Calcula a inversa de uma matriz quadrada usando eliminação de Gauss-Jordan.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada não-singular.
        
    Returns
    -------
    list of list of float
        Matriz inversa.
        
    Raises
    ------
    ValueError
        Se a matriz não for quadrada ou for singular.
    """
    if len(A) != len(A[0]):
        raise ValueError("Matriz deve ser quadrada para calcular a inversa")
    
    n = len(A)
    det = determinant(A)
    
    if abs(det) < 1e-10:
        raise ValueError("Matriz é singular (determinante próximo de zero)")
    
    # Cria matriz aumentada [A | I]
    augmented = []
    for i in range(n):
        row = list(A[i]) + [0.0] * n
        row[n + i] = 1.0  # Adiciona a identidade
        augmented.append(row)
    
    # Eliminação de Gauss-Jordan
    for i in range(n):
        # Encontra o pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Troca linhas
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Normaliza a linha do pivot
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Elimina a coluna
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extrai a matriz inversa
    return [[augmented[i][j + n] for j in range(n)] for i in range(n)]


def rank(A: Sequence[Sequence[float]]) -> int:
    """
    Calcula o posto (rank) de uma matriz usando eliminação gaussiana.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de entrada.
        
    Returns
    -------
    int
        Posto da matriz.
    """
    if not A or not A[0]:
        return 0
    
    # Cria uma cópia da matriz
    M = [row[:] for row in A]
    rows, cols = len(M), len(M[0])
    
    rank_count = 0
    
    for col in range(min(rows, cols)):
        # Encontra pivot na coluna atual
        pivot_row = -1
        for row in range(rank_count, rows):
            if abs(M[row][col]) > 1e-10:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue  # Coluna toda zero
        
        # Move linha do pivot para a posição correta
        if pivot_row != rank_count:
            M[rank_count], M[pivot_row] = M[pivot_row], M[rank_count]
        
        # Normaliza a linha do pivot
        pivot = M[rank_count][col]
        for j in range(cols):
            M[rank_count][j] /= pivot
        
        # Elimina outras linhas
        for row in range(rows):
            if row != rank_count and abs(M[row][col]) > 1e-10:
                factor = M[row][col]
                for j in range(cols):
                    M[row][j] -= factor * M[rank_count][j]
        
        rank_count += 1
    
    return rank_count


def matrix_vector_mult(A: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    """
    Multiplica uma matriz por um vetor.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz m×n.
    v : sequence of float
        Vetor de tamanho n.
        
    Returns
    -------
    list of float
        Vetor resultante de tamanho m.
        
    Raises
    ------
    ValueError
        Se as dimensões não forem compatíveis.
    """
    if len(A[0]) != len(v):
        raise ValueError(f"Número de colunas da matriz ({len(A[0])}) deve ser igual ao tamanho do vetor ({len(v)})")
    
    return [dot(row, v) for row in A]


def is_symmetric(A: Sequence[Sequence[float]]) -> bool:
    """
    Verifica se uma matriz é simétrica.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz quadrada.
        
    Returns
    -------
    bool
        True se a matriz for simétrica, False caso contrário.
    """
    if len(A) != len(A[0]):
        return False
    
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-10:
                return False
    
    return True


def frobenius_norm(A: Sequence[Sequence[float]]) -> float:
    """
    Calcula a norma de Frobenius de uma matriz.
    
    Parameters
    ----------
    A : sequence of sequence of float
        Matriz de entrada.
        
    Returns
    -------
    float
        Norma de Frobenius da matriz.
    """
    return math.sqrt(sum(sum(x*x for x in row) for row in A))


# Aliases para compatibilidade com nomes mais curtos
prod = multm  # Produto de matrizes
divm = None   # Divisão de matrizes não está bem definida matematicamente