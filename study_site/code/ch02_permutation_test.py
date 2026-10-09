"""假设检验：两组均值是否有差异？用置换检验计算 p 值（不依赖正态假设）。"""
import numpy as np

rng = np.random.default_rng(7)
a = rng.normal(10.0, 2.0, 30)                # 对照组
b = rng.normal(11.2, 2.0, 30)                # 实验组，真实均值高 1.2
obs = b.mean() - a.mean()

pooled = np.concatenate([a, b])
diffs = np.empty(20000)
for i in range(diffs.size):                  # 零假设下组标签可以任意交换
    perm = rng.permutation(pooled)
    diffs[i] = perm[30:].mean() - perm[:30].mean()
p = np.mean(np.abs(diffs) >= abs(obs))
print(f"观测到的均值差 = {obs:.3f}")
print(f"双侧 p 值 = {p:.4f}")
print("结论:", "在 α=0.05 下拒绝零假设" if p < 0.05 else "不能拒绝零假设")

# Welch t 统计量
t = obs / np.sqrt(a.var(ddof=1) / 30 + b.var(ddof=1) / 30)
print(f"Welch t 统计量 = {t:.3f}")
