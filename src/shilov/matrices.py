# src/shilov/matrices.py

#Aqui iremos criar o conceito de matrizes, a ideia das matrizes é que elas são transformações lineares em vetores
#Uma coisa que ajuda é imaginar um vetor no R³, se quisessemos rotacionar esse vetor por exemplo, usarios uma matriz para fazer essa
#transformação. Ou seja, a ideia que devemos ter em mente é que matrizes transformam vetores

#let's go babys
from typing import List, Sequence
import math

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
    """
    return [list(row) for row in data]


def prod():
    return

def inverse():
    return

def transpose():
    return

def determinant():
    return

def addm():
    return

def subm():
    return

def multm():
    return

def divm():
    return

def powm():
    return

def trace():
    return

def rank():
    return

#Amanhâ (possivelmente) faremos todas essas, está ficando uma beleza (HUAHAHAHAAHAHHAHAHAHHAHAHAHAHHAHAHAHAHHA) *risada maligna