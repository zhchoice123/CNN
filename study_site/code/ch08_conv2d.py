"""手写二维卷积（深度学习框架里的“卷积”其实是互相关），并用边缘检测核验证。"""
import numpy as np


def conv2d(x, k, stride=1, pad=0):
    """x: (H, W) 单通道图像, k: (kh, kw) 卷积核。返回特征图。"""
    if pad:
        x = np.pad(x, pad)
    kh, kw = k.shape
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = x[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * k)          # 局部窗口与核逐元素相乘再求和
    return out


# 一张 6x6 的“图像”：左半边暗(0)，右半边亮(10) —— 中间有一条竖直边缘
img = np.zeros((6, 6))
img[:, 3:] = 10

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])                    # 对水平方向的亮度变化敏感 -> 检测竖直边缘

fmap = conv2d(img, sobel_x)
print("输入形状:", img.shape, " 核形状:", sobel_x.shape)
print("输出形状 (无填充, 步长1):", fmap.shape)
print(fmap)
print("same 填充后输出形状:", conv2d(img, sobel_x, pad=1).shape)
print("步长 2 输出形状:", conv2d(img, sobel_x, stride=2).shape)
print("结论: 只有窗口跨过边缘的位置响应为 40，平坦区域响应为 0")
