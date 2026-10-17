"""上线之后：用群体稳定性指数 PSI 检测数据漂移（训练分布 vs 线上分布）。"""
import numpy as np

rng = np.random.default_rng(0)
train = rng.normal(0.45, 0.12, 20000)          # 例如：训练集图像的平均亮度
prod_same = rng.normal(0.45, 0.12, 5000)
prod_shift = rng.normal(0.55, 0.15, 5000)      # 换了新相机：亮度整体偏高


def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return np.sum((a - e) * np.log(a / e))


for name, d in [("同分布", prod_same), ("换了新相机", prod_shift)]:
    v = psi(train, d)
    level = "稳定" if v < 0.1 else ("轻微漂移" if v < 0.25 else "显著漂移，需要排查/重新训练")
    print(f"{name:6s}: PSI = {v:.3f} -> {level}")
