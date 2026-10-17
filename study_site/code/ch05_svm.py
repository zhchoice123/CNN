"""线性 SVM：用次梯度下降最小化 合页损失 + L2 正则，观察间隔与支持向量。"""
import numpy as np

rng = np.random.default_rng(4)
X = np.vstack([rng.normal([-2, -2], 0.8, (50, 2)), rng.normal([2, 2], 0.8, (50, 2))])
y = np.r_[-np.ones(50), np.ones(50)]                     # SVM 标签用 ±1

w, b, C = np.zeros(2), 0.0, 1.0
for epoch in range(1, 2001):
    lr = 0.01
    margins = y * (X @ w + b)
    viol = margins < 1                                    # 违反间隔的样本才产生梯度
    gw = w - C * (y[viol, None] * X[viol]).sum(0) / len(y)
    gb = -C * y[viol].sum() / len(y)
    w -= lr * gw; b -= lr * gb
margins = y * (X @ w + b)
print("w =", np.round(w, 3), " b =", round(b, 3))
print(f"间隔宽度 2/||w|| = {2/np.linalg.norm(w):.3f}")
print("训练准确率:", np.mean(np.sign(X @ w + b) == y))
print("位于间隔边界上或以内的样本数(支持向量):", int(np.sum(margins <= 1.0 + 1e-3)))
