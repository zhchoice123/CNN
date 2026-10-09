"""特征值分解：A = S Λ S^{-1}，A^k = S Λ^k S^{-1}；对称矩阵的特征向量正交。"""
import numpy as np

A = np.array([[0.8, 0.3],
              [0.2, 0.7]])                            # 一个马尔可夫矩阵（每列和为 1）
lam, S = np.linalg.eig(A)
print("特征值:", np.round(lam, 4))
A100 = S @ np.diag(lam ** 100) @ np.linalg.inv(S)
print("用特征分解算 A^100:\n", np.round(A100, 4))
print("与直接连乘一致:", np.allclose(A100, np.linalg.matrix_power(A, 100)))
print("稳态向量(特征值 1 的特征向量, 归一化):", np.round(S[:, np.argmax(lam)] / S[:, np.argmax(lam)].sum(), 4))

B = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])  # 对称矩阵
w, Q = np.linalg.eigh(B)
print("\n对称矩阵特征值(全为实数):", np.round(w, 4))
print("Q^T Q = I (特征向量正交):", np.allclose(Q.T @ Q, np.eye(3)))
print("全部特征值 > 0，所以 B 正定:", bool(np.all(w > 0)))
