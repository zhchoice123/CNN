"""例题：A = [[2,1],[1,2]] 的特征值、特征向量与 A^10。"""
import numpy as np
A = np.array([[2., 1.], [1., 2.]])
w, Q = np.linalg.eigh(A)
print("特征值:", w)
print("特征向量(列):\n", np.round(Q, 4))
print("A^10 =\n", np.linalg.matrix_power(A, 10))
print("公式 (3^10 ± 1)/2 =", (3**10 + 1) / 2, (3**10 - 1) / 2)
