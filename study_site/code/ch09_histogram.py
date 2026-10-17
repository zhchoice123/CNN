"""灰度变换与直方图均衡化：把挤在窄区间里的灰度拉开，增强对比度。"""
import numpy as np

rng = np.random.default_rng(0)
img = np.clip(rng.normal(110, 12, (128, 128)), 0, 255).astype(np.uint8)   # 低对比度图像
print("原图 灰度范围:", img.min(), "-", img.max(), " 标准差:", round(img.std(), 1))

neg = 255 - img                                                            # 反转
gamma = (255 * (img / 255) ** 0.5).astype(np.uint8)                        # 伽马校正 γ=0.5 提亮暗部
print("反转后均值:", round(neg.mean(), 1), "  伽马0.5后均值:", round(gamma.mean(), 1))

hist = np.bincount(img.ravel(), minlength=256)
cdf = np.cumsum(hist) / img.size                                           # 累积分布函数
lut = np.round(255 * (cdf - cdf.min()) / (1 - cdf.min())).astype(np.uint8)  # 映射表
eq = lut[img]
print("均衡化后 灰度范围:", eq.min(), "-", eq.max(), " 标准差:", round(eq.std(), 1))
h_eq = np.bincount(eq.ravel(), minlength=256)
print("均衡化后各 64 级区间的像素占比:", np.round([h_eq[i:i+64].sum() / eq.size for i in range(0, 256, 64)], 3))
