"""线性回归的三种解法：正规方程、批量梯度下降、小批量随机梯度下降；以及学习率的影响。"""
import numpy as np

rng = np.random.default_rng(0)
n = 200
X = np.column_stack([np.ones(n), rng.uniform(-2, 2, (n, 2))])
w_true = np.array([1.0, 2.0, -3.0])
y = X @ w_true + rng.normal(0, 0.5, n)

w_closed = np.linalg.solve(X.T @ X, X.T @ y)
print("正规方程:", np.round(w_closed, 3))


def mse_grad(w, Xb, yb):
    return 2 / len(yb) * Xb.T @ (Xb @ w - yb)


def batch_gd(lr, epochs=200):
    w = np.zeros(3)
    for _ in range(epochs):
        w -= lr * mse_grad(w, X, y)
    return w


for lr in (0.01, 0.1, 0.8):
    w = batch_gd(lr)
    loss = np.mean((X @ w - y) ** 2)
    print(f"批量梯度下降 lr={lr:<4}: w={np.round(w, 3)}  MSE={loss:.4g}")

w = np.zeros(3)
for epoch in range(30):                                   # 小批量 SGD，批大小 16
    for idx in np.array_split(rng.permutation(n), n // 16):
        w -= 0.05 * mse_grad(w, X[idx], y[idx])
print("lr 超过 2/λ_max 时发散，λ_max 为 MSE 海森矩阵最大特征值:", round(float(np.linalg.eigvalsh(2 / n * X.T @ X).max()), 3))
print("小批量 SGD (30 轮):", np.round(w, 3))
