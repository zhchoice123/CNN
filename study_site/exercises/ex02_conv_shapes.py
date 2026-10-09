"""
@title: 卷积层与池化：把每一层的形状算清楚
@level: 入门
@dataset: 无（随机张量）
@chapter: ch08
@minutes: 25
@goal: 搭 CNN 最常见的错误是“形状对不上”。这一篇只做一件事：亲手创建卷积层、池化层，推算每一层输出的形状和参数量，最后接上全连接分类头。
@task: 从一批 (8, 3, 32, 32) 的图片出发，搭出“卷积 → 下采样 → 池化 → 全连接”的小网络，输出 (8, 10)。
@names: x, conv, out_size, down, pool, n_params, features, n_flat, model
"""
# @step input 图片张量的形状：NCHW
# @explain
# PyTorch 里一批图片是 4 维张量，顺序固定为 **N × C × H × W**：
#
# - N：一批有几张图（batch size）
# - C：通道数（彩色图 RGB 是 3，灰度图是 1）
# - H、W：高和宽
#
# TensorFlow/Keras 默认是 N × H × W × C，从网上抄代码时要特别注意。
# @end
import torch
import torch.nn as nn

torch.manual_seed(0)
# @todo: 造一批 8 张、3 通道、32×32 的随机“图片”
# @blank nchw mask=8, 3, 32, 32
# @hint: 顺序是 批大小、通道、高、宽
# @option: 8, 32, 32, 3
# @option: 3, 32, 32, 8
# @option: 32, 32, 3, 8
# @bad: x = torch.randn(8, 32, 32, 3)
# @why: PyTorch 用 NCHW：批大小在最前，通道紧跟其后。8, 32, 32, 3 是 TensorFlow 的 NHWC 顺序。
x = torch.randn(8, 3, 32, 32)
# @endblank
print("输入形状：", tuple(x.shape))
# @check: assert tuple(x.shape) == (8, 3, 32, 32), f"x 应该是 (8, 3, 32, 32)，实际 {tuple(x.shape)}"

# @step conv 创建卷积层：输入通道、输出通道、卷积核、填充
# @explain
# `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`：
#
# - `in_channels` 必须等于输入的通道数 C
# - `out_channels` 是卷积核的个数，也就是输出有几张特征图
# - 3×3 卷积核在边上会“吃掉”一圈像素；四周补 1 圈 0（`padding=1`）就能让输出的高宽不变
# @end
# @todo: 3 个输入通道 → 16 个输出通道，3×3 卷积核，输出高宽保持 32×32
conv = nn.Conv2d(
    # @blank conv_in mask=in_channels=3
    # @hint: 卷积层的输入通道数要和图片的哪个维度一致？
    # @option: in_channels=32
    # @option: in_channels=8
    # @option: in_channels=1
    # @accept: in_channels = 3
    # @bad: in_channels=1,
    # @why: in_channels 要等于输入张量的 C=3（RGB）；写错会报 “expected input to have 1 channels, but got 3”。
    in_channels=3,
    # @endblank
    out_channels=16,
    kernel_size=3,
    # @blank conv_pad mask=padding=1
    # @hint: 3×3 的核要让输出和输入一样大，四周要补几圈？
    # @option: padding=0
    # @option: padding=2
    # @option: padding=3
    # @accept: padding = 1
    # @bad: padding=0,
    # @why: 输出尺寸 = (32 + 2×1 − 3) / 1 + 1 = 32。通用规律：k×k 的核配 padding=(k−1)/2 就能保持尺寸。
    padding=1,
    # @endblank
)
y = conv(x)
print("卷积后：", tuple(y.shape))
# @check: _y = conv(x); assert tuple(_y.shape) == (8, 16, 32, 32), f"卷积输出应是 (8, 16, 32, 32)，实际 {tuple(_y.shape)}"

# @step formula 输出尺寸公式
# @explain
# 每一层的输出高宽都可以用同一个公式算：
#
# $$o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$
#
# n 是输入尺寸，k 是卷积核大小，s 是步长，p 是填充。池化层也用同一个公式。把它写成函数，以后搭网络前先算一遍。
# @end
# @todo: 实现输出尺寸公式；Python 里向下取整的整数除法是 //
def out_size(n, k, s=1, p=0):
    # @blank formula_expr mask=(n + 2 * p - k) // s + 1
    # @hint: 先加两边的填充，再减去核大小，除以步长，最后加 1
    # @option: (n - k) // s + 1
    # @option: (n + p - k) // s + 1
    # @option: (n + 2 * p - k) / s
    # @accept: (n+2*p-k)//s+1
    # @accept: (n - k + 2 * p) // s + 1
    # @bad: return (n - k) // s + 1
    # @why: 两边各补 p 圈，所以是 2p；结果要向下取整再加 1。
    return (n + 2 * p - k) // s + 1
    # @endblank


print("32×32，3×3 核，填充 1：", out_size(32, 3, 1, 1))
# @check: assert out_size(32, 3, 1, 1) == 32, f"out_size(32, 3, 1, 1) 应为 32，实际 {out_size(32, 3, 1, 1)}"
# @check: assert out_size(28, 5) == 24 and out_size(32, 3, 2, 1) == 16 and out_size(7, 3, 2) == 3, "out_size 的公式不对：试试 (n + 2p − k) // s + 1"

# @step stride 用步长 2 的卷积做下采样
# @explain
# 想让特征图变小（下采样），可以用池化，也可以直接让卷积的步长 `stride=2`：每隔一个像素算一次，高宽减半。ResNet 就是这样下采样的。
# @end
# @todo: 16 → 32 个通道，3×3 核，填充 1，并让高宽从 32 变成 16
# @blank stride2 mask=stride=2
# @hint: 每次滑动跨几个像素，高宽就会变成原来的一半？
# @option: stride=1
# @option: stride=4
# @option: dilation=2
# @accept: stride = 2
# @bad: down = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
# @why: (32 + 2 − 3) // 2 + 1 = 16。dilation=2 是空洞卷积，扩大感受野但不缩小尺寸。
down = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
# @endblank
print("下采样后：", tuple(down(conv(x)).shape))
# @check: _z = down(conv(x)); assert tuple(_z.shape) == (8, 32, 16, 16), f"下采样后应是 (8, 32, 16, 16)，实际 {tuple(_z.shape)}"

# @step pool 最大池化
# @explain
# 池化没有可学习的参数，只是在每个窗口里做一次汇总：
#
# - **最大池化** `nn.MaxPool2d(2)`：取 2×2 窗口里的最大值，高宽减半，保留最明显的特征
# - 平均池化 `nn.AvgPool2d(2)`：取平均值
#
# `kernel_size=2` 时步长默认也是 2，所以窗口互不重叠。
# @end
# @todo: 2×2 窗口、取最大值的池化层
# @blank maxpool mask=MaxPool2d
# @hint: 取窗口内最大值的池化层叫什么？
# @option: AvgPool2d
# @option: AdaptiveAvgPool2d
# @option: Dropout2d
# @bad: pool = nn.AvgPool2d(kernel_size=2)
# @why: MaxPool2d 取最大值；AvgPool2d 取平均值；AdaptiveAvgPool2d 直接把任意尺寸变成指定尺寸。
pool = nn.MaxPool2d(kernel_size=2)
# @endblank
demo = torch.arange(16.0).view(1, 1, 4, 4)
print(demo[0, 0], "\n→ 最大池化 →\n", pool(demo)[0, 0])
# @check: _p = pool(torch.arange(16.0).view(1, 1, 4, 4))[0, 0]; assert _p.tolist() == [[5.0, 7.0], [13.0, 15.0]], f"4×4 的 0..15 经过 2×2 最大池化应是 [[5, 7], [13, 15]]，实际 {_p.tolist()}"

# @step params 数一数参数量
# @explain
# 卷积层的参数量和图片多大**无关**，只和通道数、核大小有关：
#
# $$\text{参数量} = (k \times k \times C_{in} + 1) \times C_{out}$$
#
# “+1” 是每个输出通道的偏置。`p.numel()` 返回一个张量里元素的个数，把所有参数加起来就是总参数量。
# @end
# @todo: 把 conv 的所有参数张量的元素个数加起来
# @blank numel mask=numel()
# @hint: 张量的哪个方法返回元素总数？
# @option: size()
# @option: dim()
# @option: shape
# @why: numel() = number of elements；size()/shape 返回的是各维长度，dim() 是维数。
n_params = sum(p.numel() for p in conv.parameters())
# @endblank
print("conv 的参数量：", n_params, "= (3×3×3 + 1) × 16")
# @check: assert n_params == (3 * 3 * 3 + 1) * 16, f"conv 的参数量应为 (3×3×3+1)×16 = 448，实际 {n_params}"

# @step head 接上全连接分类头
# @explain
# 卷积部分输出的是 (N, C, H, W) 的特征图，分类头 `nn.Linear` 需要知道“拉平以后有多少个数”。手算容易错，最稳的办法是**用一张假图片跑一遍**，看输出有多少个元素。
#
# `nn.Flatten()` 把 (N, C, H, W) 变成 (N, C×H×W)，最后接 `nn.Linear(n_flat, 10)` 得到 10 类的分数。
# @end
features = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 32 → 16
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 16 → 8
)
# @todo: 用一张全 0 的假图片（1 张、3 通道、32×32）跑一遍 features，数出拉平后的长度
# @blank dummy mask=torch.zeros(1, 3, 32, 32)
# @hint: 假图片的形状要和真实输入一样，只是批大小为 1
# @option: torch.zeros(3, 32, 32)
# @option: torch.zeros(1, 32, 32, 3)
# @option: torch.zeros(32, 32)
# @accept: torch.zeros((1, 3, 32, 32))
# @why: 卷积层要求 4 维输入 (N, C, H, W)；少了批维度或通道顺序不对都会报错。
n_flat = features(torch.zeros(1, 3, 32, 32)).numel()
# @endblank
model = nn.Sequential(features, nn.Flatten(), nn.Linear(n_flat, 10))
print("拉平长度：", n_flat, "  模型输出：", tuple(model(x).shape))
# @check: assert n_flat == 32 * 8 * 8, f"拉平后的长度应为 32×8×8 = 2048，实际 {n_flat}"
# @check: assert tuple(model(x).shape) == (8, 10), f"整个模型的输出应是 (8, 10)，实际 {tuple(model(x).shape)}"
