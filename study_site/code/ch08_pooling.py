"""最大池化：降采样 + 对小平移的鲁棒性。"""
import numpy as np


def max_pool(x, size=2, stride=2):
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i*stride:i*stride+size, j*stride:j*stride+size].max()
    return out


fmap = np.array([[1, 3, 2, 0],
                 [4, 6, 1, 1],
                 [0, 2, 9, 5],
                 [1, 1, 3, 8]], dtype=float)
print("特征图:\n", fmap)
print("2x2 最大池化:\n", max_pool(fmap))

# 平移 1 个像素后，池化结果大体保持不变
a = np.zeros((4, 4)); a[0, 0] = 1        # 特征出现在 (0,0)
b = np.zeros((4, 4)); b[1, 1] = 1        # 同一特征平移到 (1,1)
print("原位置池化:", max_pool(a).ravel(), " 平移后池化:", max_pool(b).ravel())
print("两者相同:", np.array_equal(max_pool(a), max_pool(b)))
