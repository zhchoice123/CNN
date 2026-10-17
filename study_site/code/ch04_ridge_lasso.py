"""岭回归 (L2) 与 Lasso (L1)：L2 让系数整体变小，L1 能把不重要的系数压成恰好为 0。"""
import numpy as np

rng = np.random.default_rng(1)
n, p = 60, 8
X = rng.normal(size=(n, p))
w_true = np.array([3, -2, 1.5, 0, 0, 0, 0, 0], float)       # 只有前 3 个特征真正有用
y = X @ w_true + rng.normal(0, 1, n)
X = (X - X.mean(0)) / X.std(0); y = y - y.mean()


def ridge(lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def lasso(lam, iters=500):                                     # 坐标下降 + 软阈值
    w = np.zeros(p)
    for _ in range(iters):
        for j in range(p):
            r = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / (X[:, j] @ X[:, j] / n)
    return w


np.set_printoptions(precision=2, suppress=True)
print("最小二乘     :", ridge(0))
print("岭回归 λ=50  :", ridge(50))
print("Lasso  λ=0.3 :", lasso(0.3))
print("Lasso 中恰好为 0 的系数个数:", int(np.sum(lasso(0.3) == 0)))
