"""普通 RNN 的梯度随时间步指数衰减；LSTM 的细胞状态在遗忘门≈1 时可以把信息原样保留很久。"""
import numpy as np

rng = np.random.default_rng(0)
d = 32
W = rng.normal(0, 0.9 / np.sqrt(d), (d, d))                   # 循环权重
h = np.zeros(d); hs = []
xs = rng.normal(size=(60, d))
for x in xs:                                                    # h_t = tanh(W h_{t-1} + x_t)
    h = np.tanh(W @ h + 0.5 * x); hs.append(h)

g = np.eye(d)                                                   # dh_T / dh_t = Π W^T diag(1 - h^2)
for k, h_t in enumerate(reversed(hs[1:]), 1):
    g = g @ (np.diag(1 - h_t ** 2) @ W)
    if k in (1, 5, 10, 20, 40):
        print(f"普通 RNN: 梯度回传 {k:2d} 步后的范数 {np.linalg.norm(g):.2e}")

# LSTM 细胞状态 c_t = f_t * c_{t-1} + i_t * g_t：遗忘门接近 1、输入门接近 0 时，c 几乎不变
c = 1.0
for t in range(60):
    f_gate, i_gate, g_cand = 0.99, 0.01, rng.uniform(-1, 1)
    c = f_gate * c + i_gate * g_cand
print(f"LSTM: 第 0 步写入的信息 1.0，60 步后细胞状态仍为 {c:.3f}；且 dc_T/dc_0 = 0.99^60 = {0.99**60:.3f}")
