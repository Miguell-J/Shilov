# test_shilov.py

"""
Exemplos de uso da biblioteca shilov.

Este arquivo demonstra como usar as funcionalidades implementadas
baseadas nos conceitos do livro de Shilov.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from shilov.matrices import *
from shilov.vectors import *
from shilov.operators import *
from shilov.forms import *
from shilov.transformations import *
from shilov.linear_systems import *


def test_eigenvalues_and_diagonalization():
    """Exemplo de cálculo de autovalores e diagonalização."""
    print("=== AUTOVALORES E DIAGONALIZAÇÃO ===")
    
    # Matriz exemplo
    A = [[3, 1], [1, 3]]
    print(f"Matriz A: {A}")
    
    # Calcula autovalores
    eigenvals = eigenvalues_2x2(A)
    print(f"Autovalores: {[ev.real for ev in eigenvals]}")
    
    # Verifica se é diagonalizável
    is_diag = is_diagonalizable_2x2(A)
    print(f"É diagonalizável: {is_diag}")
    
    if is_diag:
        # Diagonaliza
        P, D = diagonalize_2x2(A)
        print(f"Matriz P (autovetores): {P}")
        print(f"Matriz D (diagonal): {D}")
        
        # Verifica: A = PDP^(-1)
        P_inv = inverse(P)
        PD = multm(P, D)
        reconstructed = multm(PD, P_inv)
        print(f"Verificação A = PDP^(-1): {reconstructed}")
    
    print()


def test_quadratic_forms():
    """Exemplo de formas quadráticas."""
    print("=== FORMAS QUADRÁTICAS ===")
    
    # Matriz simétrica para forma quadrática
    A = [[2, 1], [1, 2]]
    print(f"Matriz da forma quadrática: {A}")
    
    # Testa com um vetor
    x = [1, 1]
    q_value = quadratic_form(A, x)
    print(f"Q({x}) = {q_value}")
    
    # Verifica se é positiva definida
    is_pos_def = is_positive_definite(A)
    print(f"É positiva definida: {is_pos_def}")
    
    # Calcula autovalores para verificar
    eigenvals = eigenvalues_2x2(A)
    print(f"Autovalores: {[ev.real for ev in eigenvals]} (todos positivos = positiva definida)")
    
    print()


def test_gram_schmidt():
    """Exemplo de ortogonalização de Gram-Schmidt."""
    print("=== PROCESSO DE GRAM-SCHMIDT ===")
    
    # Vetores linearmente independentes
    vectors = [[1, 1], [1, 2]]
    print(f"Vetores originais: {vectors}")
    
    # Aplica Gram-Schmidt
    orthonormal = gram_schmidt(vectors)
    print(f"Vetores ortonormais: {orthonormal}")
    
    # Verifica ortogonalidade
    dot_product = inner_product(orthonormal[0], orthonormal[1])
    print(f"Produto interno entre os vetores: {dot_product} (deve ser ≈ 0)")
    
    # Verifica normalização
    norm1 = magnitude(orthonormal[0])
    norm2 = magnitude(orthonormal[1])
    print(f"Normas: {norm1}, {norm2} (devem ser ≈ 1)")
    
    print()


def test_linear_transformations():
    """Exemplo de transformações lineares."""
    print("=== TRANSFORMAÇÕES LINEARES ===")
    
    # Matriz de rotação de 90 graus
    angle = 3.14159/2  # 90 graus em radianos
    R = rotation_matrix_2d(angle)
    print(f"Matriz de rotação 90°: {R}")
    
    # Aplica a rotação a um vetor
    v = [1, 0]
    v_rotated = apply_linear_operator(R, v)
    print(f"Vetor {v} rotacionado: {v_rotated}")
    
    # Matriz de reflexão no eixo x
    Ref = reflection_matrix_2d('x')
    print(f"Matriz de reflexão no eixo x: {Ref}")
    
    # Aplica reflexão
    v_reflected = apply_linear_operator(Ref, v)
    print(f"Vetor {v} refletido: {v_reflected}")
    
    # Composição de transformações
    composition = compose_transformations(R, Ref)
    print(f"Composição (reflexão ∘ rotação): {composition}")
    
    print()


def test_power_method():
    """Exemplo do método da potência."""
    print("=== MÉTODO DA POTÊNCIA ===")
    
    # Matriz com autovalor dominante
    A = [[4, 1], [1, 2]]
    print(f"Matriz A: {A}")
    
    try:
        # Aplica método da potência
        eigenval, eigenvec = power_method(A, max_iter=100, tolerance=1e-8)
        print(f"Maior autovalor: {eigenval}")
        print(f"Autovetor correspondente: {eigenvec}")
        
        # Verifica: Av = λv
        Av = apply_linear_operator(A, eigenvec)
        lambda_v = scalar_mult_vector(eigenvec, eigenval)
        print(f"Verificação Av: {Av}")
        print(f"Verificação λv: {lambda_v}")
        
    except ValueError as e:
        print(f"Erro no método da potência: {e}")
    
    print()


def test_qr_decomposition():
    """Exemplo de decomposição QR."""
    print("=== DECOMPOSIÇÃO QR ===")
    
    # Matriz exemplo
    A = [[3, 1], [4, 2]]
    print(f"Matriz A: {A}")
    
    try:
        # Decomposição QR
        Q, R = qr_decomposition_2x2(A)
        print(f"Matriz Q (ortogonal): {Q}")
        print(f"Matriz R (triangular superior): {R}")
        
        # Verifica: A = QR
        reconstructed = multm(Q, R)
        print(f"Verificação A = QR: {reconstructed}")
        
        # Verifica se Q é ortogonal
        is_orth = is_orthogonal_matrix(Q)
        print(f"Q é ortogonal: {is_orth}")
        
    except ValueError as e:
        print(f"Erro na decomposição QR: {e}")
    
    print()


def test_matrix_powers():
    """Exemplo de potências de matriz."""
    print("=== POTÊNCIAS DE MATRIZ ===")
    
    # Matriz diagonalizável
    A = [[2, 1], [1, 2]]
    print(f"Matriz A: {A}")
    
    # Calcula A^3
    A_cubed = matrix_power(A, 3)
    print(f"A^3: {A_cubed}")
    
    # Verifica manualmente: A^3 = A * A * A
    A_squared = multm(A, A)
    A_cubed_manual = multm(A_squared, A)
    print(f"A^3 (manual): {A_cubed_manual}")
    
    # Exponencial de matriz
    try:
        exp_A = matrix_exponential_2x2(A, t=1.0)
        print(f"e^A: {exp_A}")
    except ValueError as e:
        print(f"Erro no cálculo de e^A: {e}")
    
    print()


def test_projections():
    """Exemplo de projeções."""
    print("=== PROJEÇÕES ===")
    
    # Vetor para projetar
    u = [3, 4]
    # Direção da projeção
    v = [1, 0]
    
    print(f"Vetor u: {u}")
    print(f"Direção v: {v}")
    
    # Projeção de u sobre v
    proj = projection(u, v)
    print(f"Projeção de u sobre v: {proj}")
    
    # Matriz de projeção
    P = projection_matrix_line_2d(v)
    print(f"Matriz de projeção: {P}")
    
    # Aplica a matriz
    proj_matrix = apply_linear_operator(P, u)
    print(f"Projeção usando matriz: {proj_matrix}")
    
    print()


def test_condition_number():
    """Exemplo de número de condição."""
    print("=== NÚMERO DE CONDIÇÃO ===")
    
    # Matriz bem condicionada
    A1 = [[2, 0], [0, 1]]
    cond1 = condition_number_2x2(A1)
    print(f"Matriz A1: {A1}")
    print(f"Número de condição: {cond1}")
    
    # Matriz mal condicionada
    A2 = [[1, 1], [1, 1.001]]
    cond2 = condition_number_2x2(A2)
    print(f"Matriz A2: {A2}")
    print(f"Número de condição: {cond2}")
    
    print()


def main():
    """Executa todos os exemplos."""
    print("BIBLIOTECA SHILOV - EXEMPLOS DE USO")
    print("==================================")
    print()
    
    test_eigenvalues_and_diagonalization()
    test_quadratic_forms()
    test_gram_schmidt()
    test_linear_transformations()
    test_power_method()
    test_qr_decomposition()
    test_matrix_powers()
    test_projections()
    test_condition_number()
    
    print("Todos os exemplos foram executados!")


if __name__ == "__main__":
    main()