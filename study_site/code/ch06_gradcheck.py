"""反向传播的正确性检验：解析梯度 vs 数值梯度（中心差分）。"""
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(6, 3)); Y = rng.normal(size=(6, 2))
params = {"W1": rng.normal(size=(3, 5)), "b1": rng.normal(size=5),
          "W2": rng.normal(size=(5, 2)), "b2": rng.normal(size=2)}


def forward(p):
    z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)        # ReLU
    out = a1 @ p["W2"] + p["b2"]
    return 0.5 * np.mean(np.sum((out - Y) ** 2, 1)), (z1, a1, out)


def backward(p):
    _, (z1, a1, out) = forward(p)
    d_out = (out - Y) / len(X)                                 # 输出层误差 δ^L
    g = {"W2": a1.T @ d_out, "b2": d_out.sum(0)}
    d1 = (d_out @ p["W2"].T) * (z1 > 0)                        # δ^l = (W^{l+1} δ^{l+1}) ⊙ σ'(z^l)
    g["W1"] = X.T @ d1; g["b1"] = d1.sum(0)
    return g


g = backward(params)
eps = 1e-6
for k, v in params.items():
    num = np.zeros_like(v)
    for idx in np.ndindex(v.shape):
        old = v[idx]
        v[idx] = old + eps; lp, _ = forward(params)
        v[idx] = old - eps; lm, _ = forward(params)
        v[idx] = old; num[idx] = (lp - lm) / (2 * eps)
    rel = np.abs(num - g[k]).max() / (np.abs(num).max() + 1e-12)
    print(f"{k}: 相对误差 {rel:.2e}", "✓" if rel < 1e-6 else "✗")
