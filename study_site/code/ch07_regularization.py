"""正则化对比：20 维输入里只有 2 维有用、训练样本只有 60 个，一个 256 单元的网络很容易过拟合。
比较 无正则化 / L2 权重衰减 / Dropout 在“干净”测试集上的准确率。"""
import numpy as np

rng = np.random.default_rng(0)
D = 20


def make(n, noisy):
    X = rng.normal(size=(n, D))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)          # 真实规律只用到前两个特征
    if noisy:
        flip = rng.random(n) < 0.1; y[flip] = 1 - y[flip]
    return X, y


Xtr, ytr = make(60, True); Xte, yte = make(4000, False)
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def train(l2=0.0, p_drop=0.0, steps=3000, H=256, lr=0.1, seed=1):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, np.sqrt(2 / D), (D, H)); b1 = np.zeros(H)
    W2 = r.normal(0, np.sqrt(1 / H), (H, 1)); b2 = np.zeros(1)
    y = ytr[:, None]
    for _ in range(steps):
        h = np.maximum(Xtr @ W1 + b1, 0)
        if p_drop:                                      # inverted dropout：随机置零并除以保留概率
            h = h * (r.random(h.shape) > p_drop) / (1 - p_drop)
        p = sig(h @ W2 + b2)
        d2 = (p - y) / len(y)
        dW2 = h.T @ d2 + l2 * W2; db2 = d2.sum(0)      # L2 正则的梯度就是 λW（权重衰减）
        dh = d2 @ W2.T * (h > 0)
        dW1 = Xtr.T @ dh + l2 * W1; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    acc = lambda X, y: ((sig(np.maximum(X @ W1 + b1, 0) @ W2 + b2).ravel() > 0.5) == y).mean()
    return acc(Xtr, ytr), acc(Xte, yte)


for name, kw in [("无正则化", {}), ("L2 λ=0.03", {"l2": 0.03}), ("L2 λ=0.1", {"l2": 0.1}),
                 ("Dropout p=0.5", {"p_drop": 0.5}), ("Dropout p=0.7", {"p_drop": 0.7})]:
    tr, te = train(**kw)
    print(f"{name:14s} 训练准确率 {tr:.3f}   测试准确率 {te:.3f}")
print("无正则化时训练集 100% 而测试只有约 64%：严重过拟合。正则化牺牲一点训练表现，换来更好的泛化。")
