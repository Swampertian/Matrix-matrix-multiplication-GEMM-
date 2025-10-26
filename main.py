from matrix import recur_gemm_simple
import numpy as np

np.random.seed(0)
A = np.random.randint(0, 10, (4, 4))
B = np.random.randint(0, 10, (4, 4))

print("Matriz A:\n", A)
print("Matriz B:\n", B)

C = recur_gemm_simple(A, B, threshold=2)



print("\nResultado final:\n", C)
print("\nConfere com np.dot?", np.allclose(C, A @ B))
