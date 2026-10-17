"""
@title: 残差网络与断点续训：ResNet 积木块、保存与恢复训练
@level: 进阶
@dataset: cifar10
@chapter: ch08
@minutes: 40
@goal: 亲手写出 ResNet 的核心——残差块（输出 = F(x) + x），搭成一个小 ResNet；再学会保存“检查点”（模型 + 优化器 + 轮数），程序中断后能从断点接着训练。
@task: 搭一个小 ResNet 在 CIFAR-10 上训练，保存检查点，再从检查点恢复出一个预测完全相同的模型。
@names: device, BasicBlock, SmallResNet, model, train_loader, test_loader, criterion, optimizer, train_one_epoch, evaluate, ckpt, model2
@pass: 0.80
"""
# @step setup 导入、设备与数据
# @explain
# 数据部分沿用第 5 篇（为了评测更快，这里只用确定性的预处理）。
# @end
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from labkit import DATA_ROOT, log_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf)
test_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf)
print("设备：", device)
# @check: assert len(train_set) == 50000, f"CIFAR-10 训练集应有 50000 张，实际 {len(train_set)}"
# @fast: train_set = torch.utils.data.Subset(train_set, range(8000)); test_set = torch.utils.data.Subset(test_set, range(1000))

# @step loader DataLoader
# @explain
# 和前几篇一样：训练集打乱，测试集不打乱。
# @end
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500)
# @check: assert len(train_loader) > 0 and len(test_loader) > 0

# @step block 残差块：输出 = F(x) + x
# @explain
# 网络很深时，梯度一层层往回传会越来越小，深层网络反而训练不好。ResNet 的办法是给每两层加一条**捷径（shortcut）**：
#
# $$\text{输出} = \text{ReLU}\big(F(x) + x\big)$$
#
# F(x) 只需要学“在 x 基础上要改多少”（残差）。梯度可以沿着 “+x” 这条路直接回传。
#
# 当这个块要**改变形状**（步长 2 缩小尺寸，或通道数变化）时，x 和 F(x) 形状对不上，捷径上要放一个 1×1 卷积把 x 变成相同形状。
# @end
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # @todo: 形状会变（步长不为 1，或输入输出通道不同）时，捷径用 1×1 卷积 + BN；否则直接恒等映射
        # @blank need_proj mask=stride != 1 or in_ch != out_ch
        # @hint: 什么时候 x 和 F(x) 的形状会不一样？
        # @option: stride == 1 and in_ch == out_ch
        # @option: in_ch != out_ch
        # @option: stride != 1
        # @accept: in_ch != out_ch or stride != 1
        # @bad: if in_ch != out_ch:
        # @why: 两种情况都会让形状变化：步长 2 让高宽减半，通道数变化让 C 不同。只判断其中一个，另一种情况下相加会报形状错误。
        if stride != 1 or in_ch != out_ch:
        # @endblank
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # @todo: 残差连接：把捷径的输出加回来，再过 ReLU
        # @blank residual mask=out + self.shortcut(x)
        # @hint: ResNet 的灵魂就是这一个加号
        # @option: out * self.shortcut(x)
        # @option: torch.cat([out, self.shortcut(x)], 1)
        # @option: self.shortcut(out)
        # @accept: self.shortcut(x) + out
        # @bad: return F.relu(out)
        # @why: 逐元素相加是 ResNet；沿通道拼接 torch.cat 是 DenseNet 的做法；相乘会破坏“恒等映射”的性质。
        return F.relu(out + self.shortcut(x))
        # @endblank


_b1 = BasicBlock(16, 16)
_b2 = BasicBlock(16, 32, stride=2)
print("同形状块：", tuple(_b1(torch.randn(2, 16, 32, 32)).shape), " 下采样块：", tuple(_b2(torch.randn(2, 16, 32, 32)).shape))
# @check: assert tuple(BasicBlock(16, 32, stride=2)(torch.randn(2, 16, 32, 32)).shape) == (2, 32, 16, 16), "BasicBlock(16, 32, stride=2) 应输出 (2, 32, 16, 16)"
# @check: assert tuple(BasicBlock(16, 32)(torch.randn(2, 16, 8, 8)).shape) == (2, 32, 8, 8), "只改变通道数时，捷径也需要 1×1 卷积"
# @check: assert tuple(BasicBlock(32, 32, stride=2)(torch.randn(2, 32, 16, 16)).shape) == (2, 32, 8, 8), "通道数不变、只有步长为 2 时，捷径也需要 1×1 卷积（条件要同时判断 stride 和通道数）"
# @check: assert isinstance(BasicBlock(16, 16).shortcut, nn.Identity), "形状不变时捷径应是 nn.Identity()"
# @check: _bb = BasicBlock(8, 8).eval(); torch.nn.init.zeros_(_bb.bn2.weight); _xx = torch.rand(1, 8, 4, 4); assert torch.allclose(_bb(_xx), F.relu(_xx)), "残差块的输出应是 relu(F(x) + x)：当 F(x)=0 时输出应等于 relu(x)"

# @step net 搭一个小 ResNet
# @explain
# 结构：3×3 卷积 stem → 3 个阶段（每个阶段 2 个残差块，通道 32 → 64 → 128，后两个阶段第一个块步长 2）→ **全局平均池化** → 全连接。
#
# 全局平均池化 `AdaptiveAvgPool2d(1)` 把每个通道的整张特征图平均成 1 个数，不管输入多大都输出 (N, C, 1, 1)，所以不用手算拉平长度。
# @end
class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            BasicBlock(32, 32), BasicBlock(32, 32),
            BasicBlock(32, 64, stride=2), BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2), BasicBlock(128, 128),
        )
        # @todo: 全局平均池化：每个通道变成 1×1
        # @blank gap mask=AdaptiveAvgPool2d(1)
        # @hint: 不管输入多大，都输出 1×1 的池化层
        # @option: MaxPool2d(1)
        # @option: AvgPool2d(2)
        # @option: Flatten(1)
        # @accept: AdaptiveAvgPool2d((1, 1))
        # @why: AdaptiveAvgPool2d(1) 自动选窗口大小，输出固定为 1×1；AvgPool2d(2) 只是把尺寸减半。
        self.pool = nn.AdaptiveAvgPool2d(1)
        # @endblank
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


model = SmallResNet().to(device)
print("参数量：", sum(p.numel() for p in model.parameters()))
# @check: _o = model(torch.zeros(2, 3, 32, 32, device=device)); assert tuple(_o.shape) == (2, 10), f"模型输出应是 (2, 10)，实际 {tuple(_o.shape)}"
# @check: _o2 = model(torch.zeros(2, 3, 64, 64, device=device)); assert tuple(_o2.shape) == (2, 10), "用了全局平均池化后，64×64 的输入也应输出 (2, 10)"

# @step optim 损失、优化器与轮数
# @explain
# 带动量的 SGD 是训练 ResNet 的经典选择（论文里用的就是它），学习率 0.05、动量 0.9、权重衰减 5e-4。
# @end
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
EPOCHS = 10
# @check: assert optimizer.param_groups[0]["momentum"] == 0.9
# @fast: EPOCHS = 2

# @step train 把一轮训练写成函数
# @explain
# 把“训练一轮”写成函数 `train_one_epoch()`，后面恢复训练时可以直接再调用。五个固定动作，这次全部自己写。
# @end


def train_one_epoch():
    model.train()
    seen, loss_sum, correct = 0, 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # @todo: 前向 → 损失 → 清梯度 → 反向 → 更新
        # @blank five mask=criterion(logits, yb)
        # @hint: 先算出 logits，再用损失函数和标签比较
        # @option: criterion(yb, logits)
        # @option: criterion(logits)
        # @option: criterion(model, yb)
        # @bad: logits = model(xb); loss = criterion(logits, yb); optimizer.zero_grad(); optimizer.step()
        # @why: 这五行是所有 PyTorch 训练代码的骨架；漏掉 backward 时 step 拿不到梯度，参数不会变化。
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # @endblank
        seen += len(yb)
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate(m):
    m.eval()
    correct, total = 0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (m(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate(model)
    print(f"第 {epoch} 轮  训练准确率 {train_acc:.4f}  测试准确率 {test_acc:.4f}  用时 {time.time() - t0:.1f}s")
    log_epoch(epoch, train_loss=train_loss, train_acc=train_acc, test_acc=test_acc, epoch_time=time.time() - t0)
# @check: assert test_acc > 0.3, f"训练后测试准确率只有 {test_acc:.2%}（随机猜是 10%），检查 train_one_epoch 里的五个动作"

# @step save 保存检查点：不只是模型
# @explain
# 只保存 `model.state_dict()` 够用来**预测**，但不够用来**继续训练**：
#
# - 优化器也有状态（SGD 的动量、Adam 的一阶二阶矩），丢了它，恢复后的前几步会走歪
# - 还要记下训练到了第几轮
#
# 所以检查点是一个字典：`{"epoch", "model", "optimizer"}`。
# @end
# @todo: 保存轮数、模型参数和优化器状态
ckpt = {
    "epoch": EPOCHS,
    "model": model.state_dict(),
    # @blank opt_state mask=optimizer.state_dict()
    # @hint: 动量等优化器内部状态也要保存
    # @option: optimizer.param_groups
    # @option: optimizer
    # @option: model.parameters()
    # @why: optimizer.state_dict() 包含动量缓存和超参数；直接保存 optimizer 对象会依赖代码结构，不推荐。
    "optimizer": optimizer.state_dict(),
    # @endblank
}
torch.save(ckpt, "resnet_ckpt.pt")
print("检查点已保存：resnet_ckpt.pt")
# @check: import os as _o; assert _o.path.exists("resnet_ckpt.pt"), "没有找到 resnet_ckpt.pt"
# @check: _c = torch.load("resnet_ckpt.pt"); assert {"epoch", "model", "optimizer"} <= set(_c), "检查点里应包含 epoch、model、optimizer 三项"
# @check: assert len(_c["optimizer"]["state"]) > 0, "optimizer 的状态是空的：应该保存 optimizer.state_dict()"

# @step resume 从检查点恢复
# @explain
# 恢复的顺序：先按同样的结构**新建**模型和优化器，再把保存的状态**加载**进去。加载后的模型应该和原来的模型预测完全一致。
#
# 跨设备加载（例如在 RTX 3060 上训练、拿到 Mac 上用）时，加 `map_location=device` 把张量放到当前设备上。
# @end
model2 = SmallResNet().to(device)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
saved = torch.load("resnet_ckpt.pt", map_location=device)
# @todo: 把保存的参数加载进新模型，再恢复优化器状态
# @blank load_sd mask=load_state_dict
# @hint: 和 state_dict() 对应的加载方法
# @option: state_dict
# @option: load
# @option: from_state_dict
# @bad: model2 = SmallResNet().to(device)
# @why: model.load_state_dict(字典) 把参数逐个拷进新模型；参数名和形状必须完全对应，所以要先建同样结构的模型。
model2.load_state_dict(saved["model"])
# @endblank
optimizer2.load_state_dict(saved["optimizer"])
start_epoch = saved["epoch"] + 1
print("从第", start_epoch, "轮继续训练；恢复后的测试准确率：", evaluate(model2))
# @check: model.eval(); model2.eval(); _x = next(iter(test_loader))[0][:16].to(device); assert torch.allclose(model(_x), model2(_x), atol=1e-5), "恢复出来的模型预测应与原模型完全一致：检查 load_state_dict"
