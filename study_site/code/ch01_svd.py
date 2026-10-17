"""奇异值分解与低秩近似：保留前 k 个奇异值就得到误差最小的秩 k 近似（Eckart–Young）。"""
import numpy as np

rng = np.random.default_rng(0)
# 构造一个“近似低秩”的 60x40 矩阵：3 个主成分 + 少量噪声
X = rng.normal(size=(60, 3)) @ rng.normal(size=(3, 40)) + 0.05 * rng.normal(size=(60, 40))
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("前 6 个奇异值:", np.round(s[:6], 2))
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
for k in (1, 2, 3, 5):
    Xk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    err = np.linalg.norm(X - Xk) / np.linalg.norm(X)
    print(f"秩 {k} 近似: 保留能量 {energy[k-1]:.4f}, 相对误差 {err:.4f}, 存储量 {k*(60+40+1)} vs {60*40}")
