"""空间滤波：均值、高斯、中值滤波对高斯噪声与椒盐噪声的去噪效果（用 PSNR 衡量）。"""
import numpy as np

rng = np.random.default_rng(1)
H = W = 64
yy, xx = np.mgrid[:H, :W]
clean = np.where((xx - 32) ** 2 + (yy - 32) ** 2 < 300, 200.0, 50.0)       # 一个亮圆盘


def psnr(a, b): return 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))


def window_apply(img, k, fn):
    p = k // 2; pad = np.pad(img, p, mode="edge"); out = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = fn(pad[i:i + k, j:j + k])
    return out


g = np.exp(-(np.arange(5) - 2) ** 2 / (2 * 1.0 ** 2)); G = np.outer(g, g); G /= G.sum()
filters = {"3x3 均值": lambda x: window_apply(x, 3, np.mean),
           "5x5 高斯 σ=1": lambda x: window_apply(x, 5, lambda w: np.sum(w * G)),
           "3x3 中值": lambda x: window_apply(x, 3, np.median)}

gauss = clean + rng.normal(0, 20, clean.shape)
sp = clean.copy(); m = rng.random(clean.shape); sp[m < 0.05] = 0; sp[m > 0.95] = 255
for noise_name, noisy in [("高斯噪声", gauss), ("椒盐噪声", sp)]:
    print(f"{noise_name}: 去噪前 PSNR {psnr(noisy, clean):.2f} dB")
    for name, f in filters.items():
        print(f"   {name:10s} -> {psnr(f(noisy), clean):.2f} dB")
print("中值滤波是非线性滤波，特别擅长去除椒盐噪声，且对边缘的保持更好")
