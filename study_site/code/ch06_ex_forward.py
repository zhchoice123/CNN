"""例题：一个 2-2-1 网络的前向传播与参数量计算。"""
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
x = np.array([1.0, 2.0])
W1 = np.array([[0.5, -1.0], [0.25, 0.5]]); b1 = np.array([0.0, 0.5])   # 列对应隐藏神经元
W2 = np.array([1.0, -2.0]); b2 = 0.5
z1 = x @ W1 + b1; a1 = np.maximum(z1, 0)
z2 = a1 @ W2 + b2; out = sig(z2)
print("z1 =", z1, " a1 = ReLU(z1) =", a1)
print("z2 =", z2, " 输出 sigmoid(z2) =", round(out, 4))
print("784-100-10 网络参数量:", 784 * 100 + 100 + 100 * 10 + 10)
