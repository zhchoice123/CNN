"""自编码器：编码器把 10 维数据压到 2 维瓶颈，解码器再重建。线性自编码器学到的子空间与 PCA 相同。"""
import numpy as np

rng = np.random.default_rng(0)
Z = rng.normal(size=(500, 2))
A = rng.normal(size=(2, 10))
X = Z @ A + 0.05 * rng.normal(size=(500, 10)); X -= X.mean(0)       # 10 维数据实际只有 2 个自由度

We = rng.normal(0, 0.1, (10, 2)); Wd = rng.normal(0, 0.1, (2, 10))
for step in range(1, 3001):
    code = X @ We; Xr = code @ Wd
    err = Xr - X
    gWd = code.T @ err / len(X); gWe = X.T @ (err @ Wd.T) / len(X)
    We -= 0.05 * gWe; Wd -= 0.05 * gWd
    if step in (1, 100, 3000):
        print(f"第 {step:4d} 步 重建均方误差 {np.mean(err ** 2):.5f}")

U, s, Vt = np.linalg.svd(X, full_matrices=False)
X_pca = X @ Vt[:2].T @ Vt[:2]
print(f"PCA 保留 2 个主成分的重建误差 {np.mean((X_pca - X) ** 2):.5f}")
P_ae = Wd.T @ np.linalg.pinv(Wd.T); P_pca = Vt[:2].T @ Vt[:2]            # 两个子空间的投影矩阵
print("自编码器学到的子空间与 PCA 子空间相同:", np.allclose(P_ae, P_pca, atol=1e-2))
