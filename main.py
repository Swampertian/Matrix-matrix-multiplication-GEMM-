from matrix import recursive_gemm
import numpy as np
import scipy.sparse as sp
np.random.seed(0)
A = sp.random(6, 6, density=0.3, format='csr', dtype=np.float64)
B = sp.random(6, 6, density=0.3, format='csc', dtype=np.float64)

# Multiplicação recursiva
C_rec = recursive_gemm(A, B, threshold=4)
C_ref = A @ B

# Comparação
diff = (C_rec - C_ref).power(2).sum()
print("Erro quadrático total:", diff)
print("Formas:", C_rec.shape, C_ref.shape)
print("NNZ (não nulos):", C_rec.nnz, C_ref.nnz)
