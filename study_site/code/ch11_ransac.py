"""例题：RANSAC 拟合直线，并计算所需的迭代次数 N = log(1-p) / log(1-w^s)。"""
import math
import numpy as np

rng = np.random.default_rng(2)
x = rng.uniform(0, 10, 100); y = 2 * x + 1 + rng.normal(0, 0.3, 100)
out = rng.random(100) < 0.4; y[out] = rng.uniform(-20, 40, out.sum())    # 40% 离群点

ls = np.polyfit(x, y, 1)
best = None
for _ in range(200):
    i, j = rng.choice(100, 2, replace=False)
    if x[i] == x[j]: continue
    k = (y[j] - y[i]) / (x[j] - x[i]); b = y[i] - k * x[i]
    inl = np.abs(y - (k * x + b)) < 1.0
    if best is None or inl.sum() > best.sum(): best = inl
rk = np.polyfit(x[best], y[best], 1)
print(f"真实直线 y = 2x + 1；最小二乘 y = {ls[0]:.2f}x + {ls[1]:.2f}；RANSAC y = {rk[0]:.2f}x + {rk[1]:.2f}")

for w, s in [(0.5, 2), (0.5, 4), (0.5, 8), (0.7, 8)]:
    N = math.log(1 - 0.99) / math.log(1 - w ** s)
    print(f"内点比例 {w}, 最小样本数 {s}: 需要约 {math.ceil(N)} 次迭代才能以 99% 概率抽到一组全是内点的样本")
