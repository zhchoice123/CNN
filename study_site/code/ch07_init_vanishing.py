"""梯度消失与权重初始化：比较 20 层网络中激活值与梯度的尺度。"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 256))
L = 20
act = {"sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),
       "relu": (lambda z: np.maximum(z, 0), lambda a: (a > 0).astype(float))}


def run(name, std_fn):
    f, df = act[name]
    a, Ws, As = x, [], []
    for _ in range(L):
        W = rng.normal(0, std_fn(256), (256, 256)); a = f(a @ W)
        Ws.append(W); As.append(a)
    g = np.ones_like(a)                       # 反向传播一个全 1 的梯度
    norms = []
    for W, a in zip(reversed(Ws), reversed(As)):
        g = (g * df(a)) @ W.T; norms.append(np.abs(g).mean())
    return As[-1].std(), norms[-1], norms[0]


cases = [("sigmoid", "Xavier 1/√n", lambda n: np.sqrt(1 / n)),
         ("relu", "Xavier 1/√n", lambda n: np.sqrt(1 / n)), ("relu", "He √(2/n)", lambda n: np.sqrt(2 / n))]
print("激活    初始化          第20层激活标准差   梯度传回第1层后相对第20层的倍数")
for name, label, fn in cases:
    s, g_first, g_last = run(name, fn)
    print(f"{name:7s} {label:14s} {s:14.4g}   {g_first / g_last:12.3g}")
print("sigmoid: 导数最大只有 0.25，梯度传回浅层时缩小约 1e-12 倍 -> 梯度消失")
print("ReLU + Xavier: ReLU 砍掉一半信号，方差每层减半，20 层后激活值几乎为 0")
print("ReLU + He: 方差乘 2 正好补偿，激活和梯度的尺度在各层之间基本保持稳定")
