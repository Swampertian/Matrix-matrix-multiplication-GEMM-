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

## Funções que dividem com base no número de elementos não nulos
def split_by_rows_nnz(A):
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    nnz_per_row = np.diff(A.indptr)
    cumulative = np.cumsum(nnz_per_row)
    half_nnz = cumulative[-1] / 2
    r_cut = np.searchsorted(cumulative, half_nnz)
    return A[:r_cut, :], A[r_cut:, :]
def split_by_cols_nnz(B):
    if not sp.isspmatrix_csc(B):
        B = B.tocsc()
    nnz_per_col = np.diff(B.indptr)
    cumulative = np.cumsum(nnz_per_col)
    half_nnz = cumulative[-1] / 2
    c_cut = np.searchsorted(cumulative, half_nnz)
    return B[:, :c_cut], B[:, c_cut:]
############################################################

import numpy as np

def recur_gemm_simple(A, B, threshold=2, depth=0):
    indent = "  " * depth  # para visualizar recursão
    nrows, ncols = A.shape[0], B.shape[1]
    print(f"{indent}Nível {depth}: {nrows}x{ncols}")

    # CASO 1 — Base: se a matriz for pequena, multiplica direto
    if A.shape[0] <= threshold or A.shape[1] <= threshold or B.shape[1] <= threshold:
        print(f"{indent}Caso 1 → multiplicação direta")
        return A @ B

    # CASO 2 — A "horizontal": mais colunas que linhas
    if A.shape[0] <= A.shape[1]:
        print(f"{indent}Caso 2 → A horizontal (divide em 2)")
        mid = A.shape[1] // 2  # corta no meio das colunas
        A1, A2 = A[:, :mid], A[:, mid:]
        B1, B2 = B[:mid, :], B[mid:, :]
        C1 = recur_gemm_simple(A1, B1, threshold, depth + 1)
        C2 = recur_gemm_simple(A2, B2, threshold, depth + 1)
        return C1 + C2

    # CASO 3 — A "vertical": mais linhas que colunas
    else:
        print(f"{indent}Caso 3 → A vertical (divide em 4 blocos)")
        midA = A.shape[0] // 2
        midB = B.shape[1] // 2
        A1, A2 = A[:midA, :], A[midA:, :]
        B1, B2 = B[:, :midB], B[:, midB:]
        # 4 chamadas recursivas
        C11 = recur_gemm_simple(A1, B1, threshold, depth + 1)
        C12 = recur_gemm_simple(A1, B2, threshold, depth + 1)
        C21 = recur_gemm_simple(A2, B1, threshold, depth + 1)
        C22 = recur_gemm_simple(A2, B2, threshold, depth + 1)
        # Junta os blocos
        top = np.hstack([C11, C12])
        bottom = np.hstack([C21, C22])
        return np.vstack([top, bottom])
