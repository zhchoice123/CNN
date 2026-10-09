"""例题：用最小二乘拟合经过 (0,6), (1,0), (2,0) 的最佳直线 b = C + D t。"""
import numpy as np
t = np.array([0., 1., 2.]); b = np.array([6., 0., 0.])
A = np.column_stack([np.ones(3), t])
print("A^T A =\n", A.T @ A, "\nA^T b =", A.T @ b)
C, D = np.linalg.solve(A.T @ A, A.T @ b)
print(f"最佳直线: b = {C:g} + ({D:g}) t")
e = b - A @ [C, D]
print("误差向量 e =", e, " 误差平方和 =", e @ e)
