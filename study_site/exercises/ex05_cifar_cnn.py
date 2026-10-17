"""
@title: 彩色图片分类：数据增强、BatchNorm、Dropout 与学习率调度
@level: 进阶
@dataset: cifar10
@chapter: ch07
@minutes: 40
@goal: 在 CIFAR-10 上把训练做得更“像样”：只对训练集做数据增强、用 BatchNorm 稳定训练、用 Dropout 抑制过拟合、用学习率调度器逐步降低学习率，并弄清楚为什么评估前一定要 model.eval()。
@task: 训练一个带 BatchNorm 和 Dropout 的 CNN，CIFAR-10 测试准确率达到 75% 以上（完整训练约 15 轮）。
@names: device, train_tf, test_tf, train_set, test_set, train_loader, test_loader, SmallCNN, model, criterion, optimizer, scheduler, EPOCHS, evaluate, test_acc
@pass: 0.75
"""
# @step setup 导入与设备
# @explain
# 和前几篇一样的开头。CIFAR-10 是 10 类、32×32 的**彩色**图片（飞机、汽车、鸟、猫……），比 MNIST 难得多。
# @end
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from labkit import DATA_ROOT, log_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
print("设备：", device)
# @check: assert isinstance(device, torch.device)

# @step transforms 数据增强：只用在训练集
# @explain
# **数据增强**在每次取图时随机变一变（平移、翻转），相当于免费得到更多训练样本，是抑制过拟合最有效的手段之一。
#
# - `RandomCrop(32, padding=4)`：四周补 4 像素再随机裁回 32×32，相当于随机平移
# - `RandomHorizontalFlip()`：随机左右翻转。汽车、猫左右翻转后还是汽车、猫；但**上下翻转**会造出现实中不存在的图片
#
# **测试集绝不能做随机增强**：评估要的是稳定、可复现的结果。
# @end
# @todo: 训练集：随机裁剪（补 4 像素）+ 随机左右翻转 + 转张量 + 标准化
train_tf = transforms.Compose([
    # @blank crop_pad mask=padding=4
    # @hint: 四周补几个像素再随机裁回 32×32？
    # @option: padding=0
    # @option: padding=32
    # @option: size=4
    # @accept: padding = 4
    # @why: padding=0 时随机裁 32×32 等于原图，没有任何增强效果；补 4 像素允许最多 ±4 像素的平移。
    transforms.RandomCrop(32, padding=4),
    # @endblank
    # @blank hflip mask=RandomHorizontalFlip()
    # @hint: 哪种翻转不会造出“不真实”的图片？
    # @option: RandomVerticalFlip()
    # @option: RandomRotation(180)
    # @option: RandomInvert()
    # @why: 左右镜像的汽车仍然是汽车；上下颠倒或旋转 180° 的图片在真实数据里几乎不会出现，反而会干扰学习。
    transforms.RandomHorizontalFlip(),
    # @endblank
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# @todo: 测试集只转张量 + 标准化，不做任何随机变换
# @blank test_tf_b mask=transforms.ToTensor(), transforms.Normalize(MEAN, STD)
# @hint: 测试集只保留确定性的变换
# @option: transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
# @option: transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
# @option: transforms.ToTensor()
# @bad: test_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# @why: 测试时做随机增强会让同一张图每次的预测不一样；同时别忘了标准化，训练和测试的预处理必须一致。
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# @endblank
# @check: assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_tf.transforms), "训练集需要 RandomHorizontalFlip"
# @check: assert any(isinstance(t, transforms.RandomCrop) for t in train_tf.transforms), "训练集需要 RandomCrop"
# @check: assert not any("Random" in type(t).__name__ for t in test_tf.transforms), "测试集不能有随机变换"
# @check: assert any(isinstance(t, transforms.Normalize) for t in test_tf.transforms), "测试集也要标准化（和训练集一致）"

# @step data 数据集：训练集和测试集用不同的 transform
# @explain
# 训练集和测试集用**不同的** transform。CIFAR-10 已经在本地，`download=False`。
# @end
train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=train_tf)
test_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=test_tf)
# @check: assert len(train_set) == 50000 and len(test_set) == 10000, f"CIFAR-10 训练集 50000、测试集 10000，实际 {len(train_set)} / {len(test_set)}"
# @fast: train_set = torch.utils.data.Subset(train_set, range(8000)); test_set = torch.utils.data.Subset(test_set, range(2000))

# @step loader DataLoader
# @explain
# 训练集打乱，测试集不打乱；测试时批可以大一些（不需要反向传播，占用显存少）。
# @end
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500)
print("训练样本：", len(train_set), "  测试样本：", len(test_set))
# @check: _x, _y = next(iter(train_loader)); assert tuple(_x.shape[1:]) == (3, 32, 32), f"CIFAR 图片应是 (3, 32, 32)，实际 {tuple(_x.shape[1:])}"

# @step model 带 BatchNorm 和 Dropout 的 CNN
# @explain
# 每个卷积块：**卷积 → BatchNorm → ReLU**，两次后接最大池化。
#
# - **BatchNorm2d(C)**：把每个通道在一个批次里标准化，让每层的输入分布稳定，可以用更大的学习率、收敛更快。参数是通道数 C，要和前面卷积的输出通道一致
# - **Dropout(p)**：训练时随机把 p 比例的神经元置 0，逼网络不依赖个别特征，抑制过拟合
# - 卷积后面有 BatchNorm 时，卷积的偏置是多余的（BN 会减掉均值），所以 `bias=False`
#
# 三个块后，32×32 变成 4×4，通道数 128，拉平长度 128×4×4 = 2048。
# @end
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        # @blank bn mask=BatchNorm2d(cout)
        # @hint: 对卷积输出的每个通道做批量归一化，参数是通道数
        # @option: BatchNorm2d(cin)
        # @option: BatchNorm1d(cout)
        # @option: LayerNorm(cout)
        # @accept: BatchNorm2d(num_features=cout)
        # @bad: nn.BatchNorm2d(cin),
        # @why: BN 跟在卷积后面，要归一化的是卷积的输出，所以通道数是 cout；BatchNorm1d 用于全连接层的输出。
        nn.BatchNorm2d(cout),
        # @endblank
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # @todo: 输入是 RGB 彩色图
        # @blank rgb mask=block(3, 32)
        # @hint: 彩色图有几个通道？
        # @option: block(1, 32)
        # @option: block(32, 3)
        # @option: block(32, 32)
        # @bad: self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        # @why: CIFAR 是 RGB 三通道；MNIST 才是 1 通道。
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        # @endblank
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # @blank dropout mask=Dropout(0.3)
            # @hint: 训练时随机丢弃 30% 的神经元
            # @option: Dropout2d(3)
            # @option: ReLU(0.3)
            # @option: BatchNorm1d(0.3)
            # @accept: Dropout(p=0.3)
            # @why: nn.Dropout(p) 的 p 是丢弃概率（0~1）；它只在 model.train() 模式下生效，eval 模式下什么都不做。
            nn.Dropout(0.3),
            # @endblank
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SmallCNN().to(device)
print("参数量：", sum(p.numel() for p in model.parameters()))
# @check: _o = model(torch.zeros(2, 3, 32, 32, device=device)); assert tuple(_o.shape) == (2, 10), f"模型输出应是 (2, 10)，实际 {tuple(_o.shape)}"
# @check: assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules()) and any(isinstance(m, nn.Dropout) for m in model.modules()), "模型里需要 BatchNorm2d 和 Dropout"

# @step optim 优化器、权重衰减与学习率调度
# @explain
# - `weight_decay`：权重衰减（L2 正则），每步把权重往 0 拉一点，进一步防止过拟合，常用 5e-4
# - **学习率调度器**：前期用大学习率快速下降，后期用小学习率精细调整。`CosineAnnealingLR(optimizer, T_max=EPOCHS)` 让学习率在 EPOCHS 轮里按余弦曲线从初始值降到 0
# - 调度器要在**每轮结束时**调用一次 `scheduler.step()`
# @end
criterion = nn.CrossEntropyLoss()
EPOCHS = 15
# @todo: Adam，学习率 1e-3，权重衰减 5e-4
# @blank wd mask=weight_decay=5e-4
# @hint: L2 正则在优化器里叫什么参数？
# @option: momentum=5e-4
# @option: dropout=5e-4
# @option: lr_decay=5e-4
# @accept: weight_decay=0.0005
# @accept: weight_decay = 5e-4
# @why: weight_decay 在每次更新时额外减去 wd × 参数，等价于在损失里加 L2 正则项。
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
# @endblank
# @todo: 余弦退火：在 EPOCHS 轮内把学习率降到 0
# @blank cosine mask=CosineAnnealingLR
# @hint: 学习率按余弦曲线下降的调度器
# @option: StepLR
# @option: ExponentialLR
# @option: ReduceLROnPlateau
# @why: CosineAnnealingLR 平滑地降到 0；StepLR 每隔固定轮数乘一个系数（阶梯形）。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# @endblank
# @check: assert optimizer.param_groups[0]["weight_decay"] > 0, "优化器需要设置 weight_decay"
# @check: assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR), "这里要用 CosineAnnealingLR"
# @fast: EPOCHS = 1; scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# @step evaluate 为什么评估前一定要 model.eval()
# @explain
# BatchNorm 和 Dropout 在训练和评估时**行为不同**：
#
# - Dropout：训练时随机丢弃；评估时全部保留
# - BatchNorm：训练时用当前批次的均值方差；评估时用训练中累计的全局均值方差
#
# 忘了 `model.eval()`，同一张图两次预测结果会不一样，准确率也会偏低。评测会把同一批数据预测两次，检查结果是否完全一致。
# @end
@torch.no_grad()
def evaluate(loader):
    # @blank eval3 mask=model.eval()
    # @hint: 让 Dropout 关闭、BatchNorm 使用全局统计量
    # @option: model.train()
    # @option: torch.no_grad()
    # @option: model.requires_grad_(False)
    # @bad: pass
    # @why: no_grad 只是不算梯度，并不会切换 Dropout/BatchNorm 的行为；只有 model.eval() 会。
    model.eval()
    # @endblank
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total
# @check: _xb = next(iter(test_loader))[0][:64].to(device); evaluate([(_xb, torch.zeros(len(_xb), dtype=torch.long, device=device))]); assert torch.equal(model(_xb), model(_xb)), "评估模式下同一批图片两次预测应完全一致：evaluate() 里要调用 model.eval()"

# @step train 训练循环 + 每轮更新学习率
# @explain
# 五个固定动作之外，多了一件事：每轮结束调用 `scheduler.step()` 更新学习率。可以在日志里看到学习率逐轮下降。
#
# 想看完整效果，在页面上点“放进实验室”完整训练 15 轮（Apple M3 约 2–3 分钟，RTX 3060 更快）。
# @end
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, seen, loss_sum, correct = time.time(), 0, 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seen += len(yb)
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
    # @todo: 每轮结束更新一次学习率
    # @blank sched_step mask=scheduler.step()
    # @hint: 调度器在每轮结束时要做什么？
    # @option: optimizer.step()
    # @option: scheduler.zero_grad()
    # @option: scheduler.update()
    # @why: scheduler.step() 按调度规则修改 optimizer 里的 lr；它通常每轮调一次（OneCycleLR 这类是每个 batch 调一次）。
    scheduler.step()
    # @endblank
    epoch_time = time.time() - t0
    test_acc = evaluate(test_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"第 {epoch} 轮  训练准确率 {correct / seen:.4f}  测试准确率 {test_acc:.4f}  学习率 {lr_now:.6f}  用时 {epoch_time:.1f}s")
    log_epoch(epoch, train_loss=loss_sum / seen, train_acc=correct / seen, test_acc=test_acc, lr=lr_now, epoch_time=epoch_time, img_per_s=seen / epoch_time)
# @check: assert optimizer.param_groups[0]["lr"] < 1e-3, "训练结束时学习率应该已经下降：每轮末尾要调用 scheduler.step()"
# @check: assert test_acc > 0.3, f"测试准确率只有 {test_acc:.2%}（随机猜是 10%），检查模型和训练循环"
