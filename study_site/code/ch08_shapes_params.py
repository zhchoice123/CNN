"""卷积层输出尺寸与参数量：用公式逐层推算 LeNet-5 与一个 VGG 模块。"""


def out_size(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def conv_params(cin, cout, k):
    return (k * k * cin + 1) * cout          # 每个输出通道: k*k*cin 个权重 + 1 个偏置


print("== LeNet-5 (输入 32x32x1) ==")
h, c = 32, 1
layers = [("C1 conv5x5 x6", "conv", 5, 6), ("S2 pool2x2", "pool", 2, None),
          ("C3 conv5x5 x16", "conv", 5, 16), ("S4 pool2x2", "pool", 2, None)]
total = 0
for name, kind, k, cout in layers:
    if kind == "conv":
        n = conv_params(c, cout, k)
        h, c = out_size(h, k), cout
        total += n
        print(f"{name:16s} -> {h}x{h}x{c}  参数 {n}")
    else:
        h = out_size(h, k, s=2)
        print(f"{name:16s} -> {h}x{h}x{c}  参数 0")
fc = (h * h * c + 1) * 120
print(f"展平 {h*h*c} -> FC120 参数 {fc}")
print("卷积层参数合计:", total)

print("\n== VGG 风格：两个 3x3 vs 一个 5x5 (通道 C=64) ==")
C = 64
two_3x3 = 2 * conv_params(C, C, 3)
one_5x5 = conv_params(C, C, 5)
print("两个 3x3 卷积参数:", two_3x3)
print("一个 5x5 卷积参数:", one_5x5)
print(f"感受野相同(5x5)，参数节省 {100*(1-two_3x3/one_5x5):.1f}%，且多了一次非线性")
