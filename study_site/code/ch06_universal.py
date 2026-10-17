"""万能近似定理的直观演示（Nielsen 第 4 章）：两个陡峭的 sigmoid 相减得到一个“台阶”，许多台阶拼成任意曲线。"""
import numpy as np

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60)))
target = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(9 * x)
x = np.linspace(0, 1, 1000)

for n_bumps in (5, 20, 100):
    edges = np.linspace(0, 1, n_bumps + 1)
    approx = np.zeros_like(x)
    for a, b in zip(edges[:-1], edges[1:]):
        h = target((a + b) / 2)                              # 台阶高度 = 区间中点的函数值
        approx += h * (sig(1000 * (x - a)) - sig(1000 * (x - b)))   # 每个台阶用 2 个隐藏神经元
    err = np.max(np.abs(approx - target(x))[5:-5])
    print(f"{2*n_bumps:4d} 个隐藏神经元: 最大误差 {err:.4f}")
print("隐藏神经元越多，单隐层网络就能把目标函数逼近得越精确")
