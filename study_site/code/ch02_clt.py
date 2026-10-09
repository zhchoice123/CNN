"""中心极限定理与置信区间：即使总体是偏态的指数分布，样本均值也近似正态。"""
import numpy as np

rng = np.random.default_rng(0)
mu = 2.0                                     # 指数分布的真实均值
for n in (2, 10, 50):
    means = rng.exponential(mu, size=(20000, n)).mean(axis=1)
    skew = np.mean(((means - means.mean()) / means.std()) ** 3)
    print(f"n={n:3d}: 样本均值的标准差 {means.std():.3f} (理论 {mu/np.sqrt(n):.3f}), 偏度 {skew:.3f}")

# 95% 置信区间的覆盖率：重复抽样 10000 次，区间包含真值的比例
n, hits = 50, 0
for _ in range(10000):
    x = rng.exponential(mu, n)
    half = 1.96 * x.std(ddof=1) / np.sqrt(n)
    hits += (x.mean() - half <= mu <= x.mean() + half)
print(f"n=50 时 95% 置信区间的实际覆盖率: {hits/10000:.3f}")
