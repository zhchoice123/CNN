"""最小二乘 = 把 b 投影到 A 的列空间：A^T A x̂ = A^T b，残差与列空间正交。"""
import numpy as np

rng = np.random.default_rng(1)
t = np.linspace(0, 5, 20)
b = 1.5 + 0.8 * t + rng.normal(0, 0.3, t.size)      # 带噪声的直线数据
A = np.column_stack([np.ones_like(t), t])            # 设计矩阵：截距列 + 斜率列

x_normal = np.linalg.solve(A.T @ A, A.T @ b)         # 正规方程
x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)      # 数值稳定的库函数
print("正规方程解 [截距, 斜率] =", np.round(x_normal, 4))
print("lstsq 解               =", np.round(x_lstsq, 4))

p = A @ x_normal                                      # b 在列空间上的投影
e = b - p                                             # 残差
print("残差与每一列的内积 (应约为 0):", np.round(A.T @ e, 10))
P = A @ np.linalg.inv(A.T @ A) @ A.T                  # 投影矩阵
print("投影矩阵满足 P^2 = P:", np.allclose(P @ P, P), " 且对称:", np.allclose(P, P.T))
