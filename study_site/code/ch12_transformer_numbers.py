"""正弦位置编码，以及一个 Transformer 编码器层的参数量。"""
import numpy as np


def positional_encoding(n_pos, d):
    pos = np.arange(n_pos)[:, None]; i = np.arange(d // 2)[None]
    angle = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n_pos, d)); pe[:, 0::2] = np.sin(angle); pe[:, 1::2] = np.cos(angle)
    return pe


pe = positional_encoding(50, 64)
print("位置 0 的前 6 维:", np.round(pe[0, :6], 3))
sim = pe @ pe[10]
print("位置 10 与其他位置编码的点积 (越近越大):", {p: round(float(sim[p]), 2) for p in (10, 11, 12, 20, 40)})

d, h, d_ff = 512, 8, 2048
attn = 4 * (d * d + d)                                  # Q、K、V、输出投影
ffn = (d * d_ff + d_ff) + (d_ff * d + d)
norm = 2 * 2 * d                                        # 两个 LayerNorm 的 γ、β
print(f"\n编码器层 (d_model={d}, 头数={h}, d_ff={d_ff}):")
print(f"  多头注意力 {attn:,}  前馈网络 {ffn:,}  层归一化 {norm:,}  合计 {attn + ffn + norm:,}")
print(f"  每个头的维度 d_k = {d // h}；6 层编码器约 {6 * (attn + ffn + norm) / 1e6:.1f}M 参数")
print(f"  注意力分数矩阵大小随序列长度平方增长: n=512 -> {512**2:,} 个元素/头, n=4096 -> {4096**2:,}")
