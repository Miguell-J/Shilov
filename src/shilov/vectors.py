# src/shilov/vectors.py

#Aqui vamos criar o conceito de vetores, ou seja, objetos matematicos que possuem magnitude e direção
#precisamos primeiro, definir o que é um vetor e então começar a codar as outras relações

#primeiro criamos uma função que irá construir o nosso vetor, ele irá receber uma base, por exemplo
#(1,0,3) e daí teremos a nossa lista que irá representar o vetor
from typing import List, Sequence
import math

def vector(base: Sequence[float]) -> List[float]:
    """
    Cria um vetor a partir de uma sequência de números.

    Parameters
    ----------
    base : sequence of float
        Sequência de números representando as coordenadas do vetor.

    Returns
    -------
    list of float
        Lista de floats representando o vetor criado.
    """
    return list(base)

def magnitude(v: Sequence[float]) -> float:
    """
    Calcula a magnitude (norma Euclidiana) de um vetor.

    Parameters
    ----------
    v : sequence of float
        Vetor de entrada.

    Returns
    -------
    float
        A magnitude (comprimento) do vetor.
    """
    return math.sqrt(sum(x*x for x in v))

def dot(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Calcula o produto escalar entre dois vetores.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    float
        Resultado do produto escalar.

    Raises
    ------
    ValueError
        Se os vetores não tiverem o mesmo tamanho.
    """
    if len(v1) != len(v2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    return sum(a*b for a, b in zip(v1, v2))

def add(v1: Sequence[float], v2: Sequence[float]) -> List[float]:
    """
    Soma elemento a elemento dois vetores.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    list of float
        Resultado da soma elemento a elemento.

    Raises
    ------
    ValueError
        Se os vetores não tiverem o mesmo tamanho.
    """
    if len(v1) != len(v2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    return [a+b for a, b in zip(v1, v2)]

def sub(v1: Sequence[float], v2: Sequence[float]) -> List[float]:
    """
    Subtrai elemento a elemento dois vetores.

    Parameters
    ----------
    v1 : sequence of float
        Vetor minuendo.
    v2 : sequence of float
        Vetor subtraendo.

    Returns
    -------
    list of float
        Resultado da subtração (v1 - v2).

    Raises
    ------
    ValueError
        Se os vetores não tiverem o mesmo tamanho.
    """
    if len(v1) != len(v2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    return [a - b for a, b in zip(v1, v2)]

def norm(v: Sequence[float]) -> List[float]:
    """
    Retorna o vetor normalizado (mesma direção e magnitude 1).

    Parameters
    ----------
    v : sequence of float
        Vetor a ser normalizado.

    Returns
    -------
    list of float
        Vetor normalizado.

    Raises
    ------
    ValueError
        Se o vetor for nulo (magnitude zero).
    """
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Não é possível normalizar um vetor nulo")
    return [x/mag for x in v]

def angle(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Calcula o ângulo entre dois vetores em radianos.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    float
        Ângulo entre os vetores, em radianos.

    Raises
    ------
    ValueError
        Se algum vetor for nulo (magnitude zero).
    """
    dot_product = dot(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        raise ValueError("Não é possível calcular o ângulo de vetores nulos")
    cos_theta = dot_product / (mag1 * mag2)
    return math.acos(max(-1, min(1, cos_theta)))

def distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Calcula a distância euclidiana entre dois vetores.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    float
        Distância entre os dois vetores.
    """
    return magnitude(sub(v1, v2))

def cross(v1, v2):
    """
    Calcula o produto vetorial entre dois vetores 3D.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    list of float
        Vetor resultante do produto vetorial.
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Produto vetorial só é definido para vetores 3D")
    return [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]

def angle_degrees(v1, v2):
    """
    Calcula o ângulo entre dois vetores, retornando em graus.

    Parameters
    ----------
    v1 : sequence of float
        Primeiro vetor.
    v2 : sequence of float
        Segundo vetor.

    Returns
    -------
    float
        Ângulo entre os vetores, em graus.
    """
    return math.degrees(angle(v1, v2))

def projection(v1, v2):
    """
    Calcula a projeção do vetor v1 sobre o vetor v2.

    Parameters
    ----------
    v1 : sequence of float
        Vetor a ser projetado.
    v2 : sequence of float
        Vetor de referência para projeção.

    Returns
    -------
    list of float
        Vetor resultante da projeção de v1 sobre v2.

    Raises
    ------
    ValueError
        Se v2 for um vetor nulo (magnitude zero).
    """
    # proj_v2(v1) = (dot(v1,v2)/||v2||²) * v2
    mag2_sq = magnitude(v2)**2
    if mag2_sq == 0:
        raise ValueError("Não é possível projetar em um vetor nulo")
    scalar = dot(v1, v2) / mag2_sq
    return [scalar*x for x in v2]


#Aqui finalizamos por hora, talvez ainda faça modificações ou melhorias no futuro, mas por hoje é só


