"""一维 GAN：生成器 G(z) = a·z + b 学习模仿 N(4, 1.25²)，判别器是关于 x 的二次逻辑回归。"""
import numpy as np

rng = np.random.default_rng(2)
sig = lambda t: 1 / (1 + np.exp(-np.clip(t, -30, 30)))
a, b = 1.0, 0.0                                   # 生成器参数（初始输出 N(0,1)）
w = np.zeros(3)                                   # 判别器 D(x) = σ(w0 + w1 x + w2 x²)
feat = lambda x: np.stack([np.ones_like(x), x, x ** 2], 1)
lr_d, lr_g, n = 0.05, 0.02, 256
for step in range(1, 6001):
    real = rng.normal(4, 1.25, n); z = rng.normal(size=n); fake = a * z + b
    for _ in range(2):                            # 判别器：最大化 log D(real) + log(1 - D(fake))
        Fr, Ff = feat(real / 4), feat(fake / 4)
        w += lr_d * (Fr.T @ (1 - sig(Fr @ w)) - Ff.T @ sig(Ff @ w)) / n
    Ff = feat(fake / 4); d = sig(Ff @ w)
    dlogD_dx = (1 - d) * (w[1] + 2 * w[2] * fake / 4) / 4    # 生成器：最大化 log D(fake)（非饱和损失）
    a += lr_g * np.mean(dlogD_dx * z); b += lr_g * np.mean(dlogD_dx)
    if step in (1, 500, 1000, 2000, 3000, 4000, 5000, 6000):
        s = a * rng.normal(size=20000) + b
        print(f"第 {step:4d} 步: 生成分布 均值 {s.mean():5.2f}  标准差 {s.std():4.2f}   (目标 4.00 / 1.25)")
print("均值很快被学到，标准差却在目标附近来回震荡：生成器与判别器在“追逐”，这正是 GAN 训练不稳定的缩影")
