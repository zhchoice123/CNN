"""熵、交叉熵与 KL 散度（单位：比特）。交叉熵 = 熵 + KL 散度。"""
import numpy as np


def H(p):
    p = np.asarray(p, float); p = p[p > 0]
    return -np.sum(p * np.log2(p))


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return -np.sum(p[p > 0] * np.log2(q[p > 0]))


def kl(p, q):
    return cross_entropy(p, q) - H(p)


print("公平硬币熵:", H([0.5, 0.5]), "比特")
print("偏硬币 (0.9, 0.1) 熵:", round(H([0.9, 0.1]), 4), "比特")
print("均匀的 8 种结果熵:", H(np.ones(8) / 8), "比特")

p = np.array([0.7, 0.2, 0.1])                # 真实分布（如真实标签）
q = np.array([0.5, 0.3, 0.2])                # 模型预测分布
print(f"\nH(p) = {H(p):.4f}, 交叉熵 H(p,q) = {cross_entropy(p, q):.4f}, KL(p||q) = {kl(p, q):.4f}")
print("验证 H(p,q) = H(p) + KL:", np.isclose(cross_entropy(p, q), H(p) + kl(p, q)))
print(f"KL 不对称: KL(p||q)={kl(p, q):.4f}, KL(q||p)={kl(q, p):.4f}")

# 分类中的交叉熵损失：真实标签 one-hot，只剩 -log q(正确类)
onehot = np.array([1, 0, 0])
print(f"one-hot 交叉熵 = -log2 q[正确类] = {cross_entropy(onehot, q):.4f}")
