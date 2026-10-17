"""小而清晰的 CNN 模型集合，适合在 MNIST / CIFAR 上做对比实验。每个模型都能适配不同通道数和类别数。"""
import torch
import torch.nn as nn


def _norm(kind, c):
    return nn.BatchNorm2d(c) if kind == "bn" else nn.Identity()


def _conv_bn_relu(cin, cout, norm, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=(norm != "bn")), _norm(norm, cout), nn.ReLU(inplace=True))


class MLP(nn.Module):
    """全连接基线：把图像拉直，再接两层隐藏层。"""
    def __init__(self, in_ch, num_classes, size, width=32, dropout=0.0, **_):
        super().__init__()
        h = width * 16
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_ch * size * size, h), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(h, h // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(h // 2, num_classes))

    def forward(self, x):
        return self.net(x)


class LeNet(nn.Module):
    """LeNet-5 风格：两组 卷积+池化，再接全连接。"""
    def __init__(self, in_ch, num_classes, size, dropout=0.0, norm="bn", **_):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5, padding=2), _norm(norm, 6), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), _norm(norm, 16), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_ch, size, size)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 120), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(120, 84), nn.ReLU(inplace=True), nn.Linear(84, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class SimpleCNN(nn.Module):
    """depth 个“卷积-归一化-ReLU-池化”块，通道数每块翻倍，最后全局平均池化 + 线性分类。"""
    def __init__(self, in_ch, num_classes, size, width=32, depth=3, dropout=0.0, norm="bn", **_):
        super().__init__()
        layers, c = [], in_ch
        for i in range(depth):
            out = width * (2 ** i)
            layers += [_conv_bn_relu(c, out, norm), nn.MaxPool2d(2) if size // (2 ** (i + 1)) >= 1 else nn.Identity()]   # 图太小就不再池化
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class VGGMini(nn.Module):
    """VGG 思路：每个阶段两个 3×3 卷积 + 池化，共 3 个阶段。"""
    def __init__(self, in_ch, num_classes, size, width=32, dropout=0.0, norm="bn", **_):
        super().__init__()
        layers, c = [], in_ch
        for out in (width, width * 2, width * 4):
            layers += [_conv_bn_relu(c, out, norm), _conv_bn_relu(out, out, norm), nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_ch, size, size)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=(norm != "bn"))
        self.n1 = _norm(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=(norm != "bn"))
        self.n2 = _norm(norm, cout)
        self.short = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=(norm != "bn")), _norm(norm, cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + self.short(x))   # 残差连接：输出 = F(x) + x


class ResNetMini(nn.Module):
    """CIFAR 版 ResNet：3 个阶段，每阶段 depth 个残差块（depth=3 即 ResNet-20）。"""
    def __init__(self, in_ch, num_classes, size, width=16, depth=3, dropout=0.0, norm="bn", **_):
        super().__init__()
        self.stem = _conv_bn_relu(in_ch, width, norm)
        blocks, c = [], width
        for i, out in enumerate((width, width * 2, width * 4)):
            for j in range(depth):
                blocks.append(BasicBlock(c, out, 2 if (j == 0 and i > 0) else 1, norm))
                c = out
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def resnet18(in_ch, num_classes, size, dropout=0.0, **_):
    """torchvision 的 ResNet-18，改成适合小图：3×3 卷积 stem、去掉最大池化，从零训练。"""
    import torchvision
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    if dropout > 0:
        m.fc = nn.Sequential(nn.Dropout(dropout), m.fc)
    return m


MODELS = {"mlp": MLP, "lenet": LeNet, "simple_cnn": SimpleCNN, "vgg_mini": VGGMini, "resnet_mini": ResNetMini, "resnet18": resnet18}


def build(name, in_ch, num_classes, size, **kw):
    return MODELS[name](in_ch, num_classes, size, **kw)
