"""缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V，以及为什么要除以 sqrt(d_k)。"""
import numpy as np

rng = np.random.default_rng(1)


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None: scores = np.where(mask, scores, -1e9)
    w = softmax(scores); return w @ V, w


# 4 个词元的自注意力：Q、K、V 都由同一个输入 X 线性变换得到
n, d_model, d_k = 4, 8, 4
X = rng.normal(size=(n, d_model))
Wq, Wk, Wv = (rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_k)) for _ in range(3))
out, w = attention(X @ Wq, X @ Wk, X @ Wv)
np.set_printoptions(precision=3, suppress=True)
print("注意力权重 (每行和为 1):\n", w)
causal = np.tril(np.ones((n, n), bool))                         # 解码器的因果掩码：只能看见过去
_, wc = attention(X @ Wq, X @ Wk, X @ Wv, causal)
print("因果掩码后的注意力权重 (上三角为 0):\n", wc)

# 不缩放时，点积的方差随 d_k 增长，softmax 变得接近 one-hot（梯度很小）
for dk in (4, 64, 512):
    q, k = rng.normal(size=(1000, dk)), rng.normal(size=(1000, dk))
    dots = (q * k).sum(1)
    print(f"d_k={dk:3d}: 点积标准差 {dots.std():6.2f} (≈√d_k={np.sqrt(dk):5.2f}); 缩放后 {(dots/np.sqrt(dk)).std():.2f}")
