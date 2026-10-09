"""
@title: 第一个卷积网络：LeNet-5 识别手写数字
@level: 入门
@dataset: mnist
@chapter: ch08
@minutes: 35
@goal: 把第 1 篇的全连接网络换成卷积网络 LeNet-5：卷积 → 池化 → 卷积 → 池化 → 全连接。训练流程的五个固定动作再写一遍，直到不用想就能写出来。
@task: 用 LeNet-5 在 MNIST 上训练，测试准确率达到 98% 左右。
@names: device, train_set, test_set, train_loader, test_loader, LeNet, model, criterion, optimizer, EPOCHS, evaluate, test_acc
@pass: 0.97
"""
# @step setup 导入、设备与数据
# @explain
# 这一步和第 1 篇一样：选设备、定义 transform、读取本地的 MNIST。已经写过的部分直接给出，只留一个空复习一下。
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
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# @todo: 读取本地训练集与测试集（不联网）
# @blank download mask=download=False
# @hint: 数据已经预先下载好了，这个参数应该设成什么？
# @option: download=True
# @option: offline=True
# @option: cache=True
# @accept: download = False
# @why: download=False 只读本地文件；设为 True 时如果文件已存在也会先校验，没联网时可能报错。
train_set = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=False, transform=transform)
test_set = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=False, transform=transform)
# @endblank
print("设备：", device, "  训练样本：", len(train_set))
# @check: assert len(train_set) == 60000 and len(test_set) == 10000, "训练集 60000、测试集 10000"
# @fast: train_set = torch.utils.data.Subset(train_set, range(20000)); test_set = torch.utils.data.Subset(test_set, range(2000))

# @step loader DataLoader
# @explain
# 训练集打乱、测试集不打乱，这是固定写法。
# @end
# @blank loader2 mask=shuffle=True
# @hint: 训练集要不要打乱？
# @option: shuffle=False
# @option: sampler=None
# @option: drop_last=True
# @accept: shuffle = True
# @why: 训练集每轮打乱顺序；测试集顺序不影响结果，不需要打乱。
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# @endblank
test_loader = DataLoader(test_set, batch_size=1000)
# @check: import torch.utils.data as _tud; assert isinstance(train_loader.sampler, _tud.RandomSampler), "训练集的 DataLoader 要 shuffle=True"

# @step model LeNet-5 的结构
# @explain
# LeNet-5（1998）是最早成功的卷积网络，结构非常规整：
#
# | 层 | 输出形状 |
# |---|---|
# | 输入 | 1 × 28 × 28 |
# | 卷积 5×5，6 个核，填充 2 | 6 × 28 × 28 |
# | 最大池化 2×2 | 6 × 14 × 14 |
# | 卷积 5×5，16 个核 | 16 × 10 × 10 |
# | 最大池化 2×2 | 16 × 5 × 5 |
# | 全连接 → 120 → 84 → 10 | |
#
# 第二个池化之后拉平，长度是 16 × 5 × 5 = 400，这就是第一个全连接层的输入大小。
# @end
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # @todo: 第一层卷积：MNIST 是灰度图，输入几个通道？
        # @blank lenet_in mask=nn.Conv2d(1, 6
        # @hint: 灰度图有几个通道？
        # @option: nn.Conv2d(3, 6
        # @option: nn.Conv2d(6, 1
        # @option: nn.Conv2d(28, 6
        # @bad: self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        # @why: 灰度图只有 1 个通道；彩色图（CIFAR）才是 3。顺序是 (输入通道, 输出通道)。
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # @endblank
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        # @todo: 拉平后的长度 = 16 个通道 × 5 × 5
        # @blank lenet_flat mask=16 * 5 * 5
        # @hint: 第二次池化后的形状是 16 × 5 × 5
        # @option: 16 * 7 * 7
        # @option: 6 * 14 * 14
        # @option: 16 * 10 * 10
        # @accept: 400
        # @accept: 16*5*5
        # @bad: self.fc1 = nn.Linear(16 * 7 * 7, 120)
        # @why: 28 →（卷积 5×5 填充 2）28 →（池化）14 →（卷积 5×5 不填充）10 →（池化）5，所以是 16×5×5 = 400。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # @endblank
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # @todo: 卷积 → ReLU → 池化，做两次
        # @blank lenet_fwd mask=self.pool(F.relu(self.conv1(x)))
        # @hint: 由内向外：先卷积，再激活，最后池化
        # @option: F.relu(self.pool(self.conv1(x)))
        # @option: self.conv1(self.pool(F.relu(x)))
        # @option: self.pool(self.conv1(F.relu(x)))
        # @why: 标准顺序是 卷积 → 激活 → 池化。先池化再卷积会丢掉一半分辨率。（对最大池化来说 ReLU 和池化交换顺序结果相同，但习惯写法是先激活。）
        x = self.pool(F.relu(self.conv1(x)))
        # @endblank
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = LeNet().to(device)
print("参数量：", sum(p.numel() for p in model.parameters()))
# @check: _o = model(torch.zeros(2, 1, 28, 28, device=device)); assert tuple(_o.shape) == (2, 10), f"LeNet 的输出应是 (2, 10)，实际 {tuple(_o.shape)}"
# @check: assert sum(p.numel() for p in model.parameters()) == 61706, f"LeNet-5 的参数量应为 61706，实际 {sum(p.numel() for p in model.parameters())}；检查各层的通道数和全连接大小"

# @step optim 损失与优化器
# @explain
# 这次用 **Adam** 优化器：它为每个参数自动调整步长，对学习率没那么敏感，`lr=1e-3` 是最常用的起点。
# @end
criterion = nn.CrossEntropyLoss()
# @todo: 用 Adam 优化模型全部参数，学习率 0.001
# @blank adam mask=Adam
# @hint: 自适应学习率、最常用的那个优化器
# @option: SGD
# @option: Adagrad
# @option: LBFGS
# @why: Adam 结合了动量和自适应学习率，收敛快、调参简单；SGD 不加动量时在这里会慢很多。
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# @endblank
EPOCHS = 3
# @check: assert isinstance(optimizer, torch.optim.Adam), "这里要用 torch.optim.Adam"
# @fast: EPOCHS = 1

# @step evaluate 评估函数
# @explain
# 和第 1 篇一样：`@torch.no_grad()` + `model.eval()` + `argmax(1)`。
# @end
@torch.no_grad()
def evaluate(loader):
    # @blank eval2 mask=model.eval()
    # @hint: 评估前切换模式
    # @option: model.train()
    # @option: model.test()
    # @option: torch.eval()
    # @why: model.eval() 切到评估模式，Dropout/BatchNorm 会改用评估时的行为。
    model.eval()
    # @endblank
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total
# @check: _a = evaluate(test_loader); assert 0.0 <= _a <= 1.0 and not model.training, "evaluate() 应返回 0–1 的准确率，并调用 model.eval()"

# @step train 训练循环（五个固定动作）
# @explain
# 不看第 1 篇，自己写出这五步：
#
# 1. 前向：`logits = model(xb)`
# 2. 损失：`loss = criterion(logits, yb)`
# 3. 清梯度：`optimizer.zero_grad()`
# 4. 反向：`loss.backward()`
# 5. 更新：`optimizer.step()`
# @end
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, seen, loss_sum, correct = time.time(), 0, 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        # @todo: 清梯度 → 反向传播 → 更新参数（顺序很重要）
        # @blank three mask=optimizer.zero_grad()
        # @hint: 三步里的第一步：先把上一批留下的梯度清掉
        # @option: optimizer.step()
        # @option: loss.backward()
        # @option: model.zero_grad
        # @accept: model.zero_grad()
        # @bad: optimizer.step(); loss.backward(); optimizer.zero_grad()
        # @why: 顺序固定为 zero_grad → backward → step。如果先 step 再 backward，这一步用的是上一批的梯度（甚至被清空的梯度）。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # @endblank
        seen += len(yb)
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
    epoch_time = time.time() - t0
    test_acc = evaluate(test_loader)
    print(f"第 {epoch} 轮  训练准确率 {correct / seen:.4f}  测试准确率 {test_acc:.4f}  用时 {epoch_time:.1f}s")
    log_epoch(epoch, train_loss=loss_sum / seen, train_acc=correct / seen, test_acc=test_acc, epoch_time=epoch_time, img_per_s=seen / epoch_time)
# @check: assert test_acc > 0.85, f"测试准确率只有 {test_acc:.2%}，正常应超过 85%：检查训练循环的三个动作顺序"
