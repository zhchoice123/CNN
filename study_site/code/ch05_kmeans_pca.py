"""无监督学习：K 均值聚类 与 主成分分析（用 SVD 计算）。"""
import numpy as np

rng = np.random.default_rng(5)
centers = np.array([[0, 0], [5, 5], [0, 6]])
X = np.vstack([rng.normal(c, 0.8, (60, 2)) for c in centers])

C = X[rng.choice(len(X), 3, replace=False)]              # 随机选 3 个初始中心
for it in range(1, 11):
    lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)   # 分配步
    newC = np.array([X[lab == k].mean(0) for k in range(3)])    # 更新步
    inertia = ((X - newC[lab]) ** 2).sum()
    if np.allclose(newC, C): print(f"第 {it} 次迭代收敛"); break
    C = newC
print("聚类中心:\n", np.round(C[np.argsort(C[:, 0] + C[:, 1])], 2), "\n簇内平方和:", round(inertia, 1))

# PCA：三维数据其实主要分布在一个平面上
Z = rng.normal(size=(300, 2)) @ np.array([[3, 1, 0.5], [0, 1, 2]]) + 0.1 * rng.normal(size=(300, 3))
Zc = Z - Z.mean(0)
U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
ratio = s ** 2 / np.sum(s ** 2)
print("\n各主成分解释方差比例:", np.round(ratio, 4))
print("前两个主成分累计:", round(ratio[:2].sum(), 4), " -> 可以安全地降到 2 维")
