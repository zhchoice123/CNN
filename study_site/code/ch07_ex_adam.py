"""例题：手算 Adam 的第一步更新。"""
import math
g, lr, b1, b2, eps = 0.5, 0.001, 0.9, 0.999, 1e-8
m = (1 - b1) * g; v = (1 - b2) * g ** 2
m_hat = m / (1 - b1); v_hat = v / (1 - b2)
step = lr * m_hat / (math.sqrt(v_hat) + eps)
print(f"m={m:.4f}, v={v:.6f}, m_hat={m_hat:.4f}, v_hat={v_hat:.4f}")
print(f"参数更新量 = {step:.6f}  (第一步的步长约等于学习率，与梯度大小无关)")
print("He 初始化 fan_in=512 时标准差:", math.sqrt(2 / 512))
