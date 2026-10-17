"""频率域：二维傅里叶变换、去除周期噪声，以及卷积定理的验证。"""
import numpy as np

rng = np.random.default_rng(2)
N = 64
yy, xx = np.mgrid[:N, :N]
img = np.where((abs(xx - 32) < 12) & (abs(yy - 32) < 12), 1.0, 0.0)       # 方块
noisy = img + 0.5 * np.sin(2 * np.pi * 16 * xx / N)                       # 叠加竖条纹周期噪声

F = np.fft.fftshift(np.fft.fft2(noisy))
mag = np.abs(F)
peaks = np.argwhere(mag > 0.5 * mag.max())
print("频谱中幅度最大的位置(中心为直流分量 32,32):", peaks.tolist())
notch = np.ones_like(F); notch[32, 32 - 16] = notch[32, 32 + 16] = 0     # 陷波滤波器去掉两个噪声峰
clean = np.real(np.fft.ifft2(np.fft.ifftshift(F * notch)))
print(f"去噪前误差 {np.abs(noisy - img).mean():.4f} -> 陷波滤波后 {np.abs(clean - img).mean():.4f}")

# 理想低通：只保留半径 8 以内的低频，图像变模糊（边缘属于高频）
r = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
low = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(img)) * (r < 8))))
print(f"低通后方块边缘处的值(本应是0/1的跳变): {np.round(low[32, 18:23], 3)}")

# 卷积定理：空间域循环卷积 = 频率域逐点相乘
k = np.zeros((N, N)); k[:3, :3] = 1 / 9
spatial = np.zeros((N, N))
for dy in range(3):
    for dx in range(3):
        spatial += np.roll(np.roll(img, dy, 0), dx, 1) / 9
freq = np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(k)))
print("卷积定理验证(两种方式结果一致):", np.allclose(spatial, freq))
