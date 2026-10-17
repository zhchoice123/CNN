"""两层神经网络 + 反向传播，从零学会 XOR。"""
import numpy as np

rng = np.random.default_rng(1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y = np.array([[0], [1], [1], [0]], float)
sig = lambda z: 1 / (1 + np.exp(-z))

W1 = rng.normal(0, 1, (2, 8)); b1 = np.zeros(8)      # 隐藏层 8 个神经元
W2 = rng.normal(0, 1, (8, 1)); b2 = np.zeros(1)
lr = 1.0
for step in range(1, 5001):
    # 前向传播
    h = np.tanh(X @ W1 + b1)
    p = sig(h @ W2 + b2)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # 反向传播（链式法则，从输出往回）
    dz2 = (p - y) / len(X)                           # sigmoid + 交叉熵的梯度
    dW2 = h.T @ dz2; db2 = dz2.sum(0)
    dh = dz2 @ W2.T
    dz1 = dh * (1 - h ** 2)                          # tanh 的导数
    dW1 = X.T @ dz1; db1 = dz1.sum(0)
    for P, G in ((W1, dW1), (b1, db1), (W2, dW2), (b2, db2)):
        P -= lr * G
    if step in (1, 100, 1000, 5000):
        print(f"第 {step:4d} 步 损失 {loss:.4f}")
print("预测概率:", np.round(p.ravel(), 3), " 真实标签:", y.ravel().astype(int))
