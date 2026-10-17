"""逻辑回归：sigmoid + 交叉熵损失 + 梯度下降，梯度形式为 X^T (p - y) / n。"""
import numpy as np

rng = np.random.default_rng(2)
n = 400
X0 = rng.normal([-1, -1], 1.0, (n // 2, 2)); X1 = rng.normal([1.5, 1.0], 1.0, (n // 2, 2))
X = np.vstack([X0, X1]); y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
Xb = np.column_stack([np.ones(n), X])
sigmoid = lambda z: 1 / (1 + np.exp(-z))

w = np.zeros(3)
for step in range(1, 2001):
    p = sigmoid(Xb @ w)
    w -= 0.1 * Xb.T @ (p - y) / n
    if step in (1, 10, 100, 2000):
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        print(f"第 {step:4d} 步: 交叉熵 {loss:.4f}")
acc = ((sigmoid(Xb @ w) > 0.5) == y).mean()
print("参数 [b, w1, w2] =", np.round(w, 3), " 训练准确率 =", acc)
print(f"w1 的含义: x1 增加 1，正类的几率(odds)乘以 e^w1 = {np.exp(w[1]):.2f}")
