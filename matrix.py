import numpy as np
import scipy.sparse as sp



def split_by_rows(A, r_cut):
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    return A[:r_cut, :], A[r_cut:, :]

def split_by_cols(B, c_cut):
    
    if not sp.isspmatrix_csc(B):
        B = B.tocsc()
    return B[:, :c_cut], B[:, c_cut:]

def recursive_gemm(A,B, threshold=1e6, depth=0, max_depth=20):
    
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    if not sp.isspmatrix_csc(B):
        B = B.tocsc()

    # Caso base: checar tamanho efetivo
    A_nnz_rows = np.count_nonzero(np.diff(A.indptr))  # linhas com nnz
    B_nnz_cols = np.count_nonzero(np.diff(B.indptr))  # colunas com nnz
    nnz_mat_size = A_nnz_rows * B_nnz_cols

    # CASO 1: parar recursão e multiplicar diretamente
    if nnz_mat_size <= threshold or depth >= max_depth:
        return A @ B  # o SciPy já usa kernels C otimizados

    # CASO 2: A "horizontal" (mais colunas que linhas)
    if A.shape[0] <= A.shape[1]:
        mid_A = A.shape[1] // 2
        A1, A2 = split_by_cols(A.tocsc(), mid_A)  # dividir por colunas
        B1, B2 = split_by_rows(B.tocsr(), mid_A)  # dividir por linhas
        C1 = recursive_gemm(A1.tocsr(), B1.tocsc(), threshold, depth + 1)
        C2 = recursive_gemm(A2.tocsr(), B2.tocsc(), threshold, depth + 1)
        C = C1 + C2  # somar blocos
        return C

    # CASO 3: A "vertical" (mais linhas que colunas)
    else:
        mid_A = A.shape[0] // 2
        mid_B = B.shape[1] // 2
        A1, A2 = split_by_rows(A, mid_A)
        B1, B2 = split_by_cols(B, mid_B)
        C11 = recursive_gemm(A1, B1, threshold, depth + 1)
        C12 = recursive_gemm(A1, B2, threshold, depth + 1)
        C21 = recursive_gemm(A2, B1, threshold, depth + 1)
        C22 = recursive_gemm(A2, B2, threshold, depth + 1)
        C_top = sp.hstack([C11, C12])
        C_bottom = sp.hstack([C21, C22])
        C = sp.vstack([C_top, C_bottom])
        return C