"""列视角：A @ x 等于 A 的各列按 x 的分量做线性组合；用 SVD 求秩与零空间。"""
import numpy as np

A = np.array([[1., 2., 3.],
              [2., 4., 6.],
              [1., 0., 1.]])
x = np.array([2., -1., 1.])

print("A @ x            =", A @ x)
print("2*a1 - 1*a2 + 1*a3 =", 2 * A[:, 0] - 1 * A[:, 1] + 1 * A[:, 2])

U, s, Vt = np.linalg.svd(A)
rank = int(np.sum(s > 1e-10))
print("奇异值:", np.round(s, 4))
print("秩 r =", rank, " (第二行是第一行的 2 倍，所以秩不满)")
null = Vt[rank:].T                       # 右奇异向量中对应零奇异值的部分张成零空间
print("零空间基向量:", np.round(null.ravel(), 4))
print("验证 A @ n ≈ 0:", np.allclose(A @ null, 0))
m, n = A.shape
print(f"四个子空间维数: 列空间 {rank}, 零空间 {n-rank}, 行空间 {rank}, 左零空间 {m-rank}")
