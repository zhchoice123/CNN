"""例题：堆叠卷积层的感受野计算。 r_l = r_{l-1} + (k_l - 1) * j_{l-1}，j 为累计步长。"""


def receptive_field(layers):
    r, j = 1, 1
    for name, k, s in layers:
        r = r + (k - 1) * j
        j = j * s
        print(f"{name:12s} k={k} s={s}  感受野 {r}x{r}  累计步长 {j}")
    return r


print("三个 3x3 卷积(步长1):")
receptive_field([("conv3x3", 3, 1)] * 3)
print("\nconv3x3 -> pool2x2 -> conv3x3 -> pool2x2 -> conv3x3:")
receptive_field([("conv3x3", 3, 1), ("maxpool2x2", 2, 2), ("conv3x3", 3, 1), ("maxpool2x2", 2, 2), ("conv3x3", 3, 1)])
