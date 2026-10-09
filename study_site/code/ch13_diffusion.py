"""扩散模型：前向加噪有闭式解 x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε；再用（理想的）噪声预测器逐步去噪采样。
数据分布取 N(2, 0.5²)，此时最优噪声预测器有解析式，可以单独检验采样算法是否正确。"""
import numpy as np

rng = np.random.default_rng(3)
T = 1000
beta = np.linspace(1e-4, 0.02, T)                 # 线性噪声调度
alpha = 1 - beta; abar = np.cumprod(alpha)
for t in (0, 100, 500, 999):
    print(f"t={t:4d}: 信号保留系数 √ᾱ = {np.sqrt(abar[t]):.4f}, 噪声系数 √(1-ᾱ) = {np.sqrt(1 - abar[t]):.4f}")

mu0, s0 = 2.0, 0.5


def eps_pred(x, t):                               # E[ε | x_t]，对高斯数据精确成立；真实模型用神经网络学它
    var = abar[t] * s0 ** 2 + 1 - abar[t]
    return np.sqrt(1 - abar[t]) * (x - np.sqrt(abar[t]) * mu0) / var


x = rng.normal(size=20000)                        # 从纯噪声出发
for t in range(T - 1, -1, -1):                    # DDPM 逆向采样
    mean = (x - beta[t] / np.sqrt(1 - abar[t]) * eps_pred(x, t)) / np.sqrt(alpha[t])
    x = mean + (np.sqrt(beta[t]) * rng.normal(size=x.size) if t > 0 else 0)
print(f"\n从噪声生成的样本: 均值 {x.mean():.3f}, 标准差 {x.std():.3f}  (数据分布 2.000 / 0.500)")
