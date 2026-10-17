"""Softmax 回归的梯度检验：解析梯度 X^T (P - Y) / n 与数值梯度对比。"""
import numpy as np

rng = np.random.default_rng(3)
n, d, k = 5, 4, 3
X = rng.normal(size=(n, d)); labels = rng.integers(0, k, n); Y = np.eye(k)[labels]
W = rng.normal(size=(d, k)) * 0.1


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)                 # 减去最大值防止指数溢出
    E = np.exp(Z); return E / E.sum(1, keepdims=True)


def loss(W):
    return -np.mean(np.sum(Y * np.log(softmax(X @ W)), 1))


grad = X.T @ (softmax(X @ W) - Y) / n
num = np.zeros_like(W); eps = 1e-5
for i in range(d):
    for j in range(k):
        E = np.zeros_like(W); E[i, j] = eps
        num[i, j] = (loss(W + E) - loss(W - E)) / (2 * eps)
print("解析梯度与数值梯度的最大差:", f"{np.abs(grad - num).max():.2e}")
print("每行 softmax 概率之和:", softmax(X @ W).sum(1))
