"""为什么残差连接能训练很深的网络：比较梯度在 50 层中的衰减。

把每一层近似看成线性映射 W（权重较小，模拟难训练的深网络）。
普通网络的梯度要连乘 W^T；残差网络 y = x + F(x) 的雅可比是 (I + W)，梯度多了一条“恒等高速通道”。
"""
import numpy as np

rng = np.random.default_rng(0)
d, depth = 64, 50
Ws = [rng.normal(0, 0.2 / np.sqrt(d), (d, d)) for _ in range(depth)]

g_plain = np.ones(d) / np.sqrt(d)          # 初始梯度范数为 1
g_res = np.ones(d) / np.sqrt(d)
for l, W in enumerate(reversed(Ws), 1):
    g_plain = W.T @ g_plain                    # 普通网络: dL/dx = W^T dL/dy
    g_res = g_res + W.T @ g_res                # 残差网络: dL/dx = (I + W)^T dL/dy
    if l in (1, 10, 25, 50):
        print(f"反传 {l:2d} 层后  普通网络梯度范数 {np.linalg.norm(g_plain):.2e}   残差网络 {np.linalg.norm(g_res):.2e}")
print("结论: 普通深网络的梯度指数级消失，残差网络依靠恒等映射把梯度直接传回浅层")
