"""VAE 的两个关键：重参数化技巧 z = μ + σ·ε，以及 KL(N(μ,σ²) || N(0,1)) 的闭式解。"""
import numpy as np

rng = np.random.default_rng(1)


def kl_closed(mu, sigma):
    return 0.5 * (mu ** 2 + sigma ** 2 - 1 - 2 * np.log(sigma))


for mu, sigma in [(0, 1), (1, 1), (0, 0.5), (2, 0.3)]:
    eps = rng.normal(size=200000)
    z = mu + sigma * eps                                     # 重参数化：随机性只来自 ε，对 μ、σ 可导
    log_q = -0.5 * ((z - mu) / sigma) ** 2 - np.log(sigma)
    log_p = -0.5 * z ** 2
    kl_mc = np.mean(log_q - log_p)                           # 蒙特卡洛估计（常数项相消）
    print(f"μ={mu}, σ={sigma}: KL 闭式解 {kl_closed(mu, sigma):.4f}  蒙特卡洛估计 {kl_mc:.4f}")
print("KL 项把每个样本的后验 q(z|x) 拉向标准正态分布，让隐空间连续、可以随意采样")

# 重参数化梯度：d E[z^2] / dμ = 2μ，用 ε 采样估计
mu, sigma = 1.5, 0.8
eps = rng.normal(size=100000)
print(f"E[z²] 对 μ 的梯度: 重参数化估计 {np.mean(2 * (mu + sigma * eps)):.4f}, 理论值 {2 * mu}")
