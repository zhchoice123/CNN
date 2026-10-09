"""
@title: 第一个完整训练循环：全连接网络识别手写数字
@level: 入门
@dataset: mnist
@chapter: ch06
@minutes: 30
@goal: 亲手写出 PyTorch 训练代码的固定环节：选设备 → 数据集 → DataLoader → 模型 → 损失与优化器 → 评估函数 → 训练循环（前向、损失、清梯度、反向传播、更新）→ 保存。
@task: 用两层全连接网络在 MNIST 上训练，测试准确率达到 90% 以上。
@names: device, train_set, test_set, train_loader, test_loader, model, criterion, optimizer, EPOCHS, evaluate, test_acc
@pass: 0.90
"""
# @step device 导入库，选择计算设备
# @explain
# 训练代码的第一件事：决定**在哪块硬件上算**。同一份代码要能在不同电脑上跑：
#
# - NVIDIA 显卡（例如 Windows 上的 RTX 3060）→ `cuda`
# - Apple 芯片（M1/M2/M3）的 GPU → `mps`
# - 都没有就用 `cpu`
#
# 后面所有的模型和数据都要 `.to(device)` 放到同一个设备上，否则会报“张量不在同一设备”的错。
# @end
# @todo: 导入 torch / torchvision，并从 labkit 导入 DATA_ROOT（本地数据集目录）和 log_epoch（记录每轮指标，实验室据此画曲线）
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from labkit import DATA_ROOT, log_epoch

# @todo: 依次检查 NVIDIA 显卡（cuda）、Apple 芯片 GPU（mps），都没有就用 CPU，结果存进变量 device
# @blank device_sel mask=cuda
# @hint: NVIDIA 显卡在 PyTorch 里叫什么后端？
# @option: gpu
# @option: nvidia
# @option: mps
# @why: PyTorch 用 "cuda" 表示 NVIDIA GPU（你的 RTX 3060），用 "mps" 表示 Apple 芯片的 GPU（Metal Performance Shaders）。
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# @endblank
print("使用设备：", device)
# @check: assert isinstance(device, torch.device), "device 应该是 torch.device(...) 创建的对象"

# @step dataset 读取数据集：transform + Dataset
# @explain
# `torchvision.datasets.MNIST` 是一个 **Dataset**：按下标取出 `(图片, 标签)`。
#
# - `transform` 决定每张图片取出来时怎么处理：`ToTensor()` 把 0–255 的图片变成 0–1 的浮点张量，形状 `(1, 28, 28)`；`Normalize(均值, 标准差)` 再做标准化，让训练更稳定。
# - 数据集已经预先下载到本地 `DATA_ROOT`，所以 `download=False`，不用联网。
# @end
# @todo: 把图片转成张量，再用 MNIST 的均值 0.1307、标准差 0.3081 标准化
# @blank totensor mask=ToTensor
# @hint: 把 PIL 图片转成 0–1 浮点张量的变换叫什么？
# @option: PILToTensor
# @option: Resize
# @option: Grayscale
# @why: ToTensor 会把像素值除以 255 并把维度排成 (C, H, W)；PILToTensor 不会缩放到 0–1。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
# @endblank
# @todo: 从本地目录读取训练集和测试集，不要联网下载
# @blank ds_train mask=train=True
# @hint: 用哪个参数区分训练集和测试集？
# @option: split="train"
# @option: training=True
# @accept: train = True
# @why: torchvision 的 MNIST 用 train=True/False 选择训练集（6 万张）或测试集（1 万张）。
train_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=transform)
# @endblank
test_set = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transform)
# @check: assert len(train_set) == 60000 and len(test_set) == 10000, f"训练集应有 60000 张、测试集 10000 张，实际 {len(train_set)} / {len(test_set)}"
# @check: _x, _y = train_set[0]; assert tuple(_x.shape) == (1, 28, 28), f"单张图片应是 (1, 28, 28) 的张量，实际 {getattr(_x, 'shape', type(_x))}"
# @fast: train_set = torch.utils.data.Subset(train_set, range(6000)); test_set = torch.utils.data.Subset(test_set, range(2000))

# @step loader 用 DataLoader 分批、打乱
# @explain
# 一次把 6 万张图都塞进模型既慢又占内存，所以按**小批量（mini-batch）**训练。`DataLoader` 负责：
#
# - 按 `batch_size` 把样本拼成一批，形状变成 `(B, 1, 28, 28)`
# - `shuffle=True`：每一轮都打乱顺序，避免模型记住样本顺序
#
# 测试集不需要打乱。
#
# > 在 Windows 上如果要设置 `num_workers>0`，脚本必须放在 `if __name__ == "__main__":` 里；这里用默认的 0 就不会有问题。
# @end
# @todo: 训练集每批 128 张并打乱；测试集每批 1000 张、不打乱
# @blank shuffle mask=shuffle=True
# @hint: 训练时每一轮都应该打乱样本顺序吗？
# @option: shuffle=False
# @option: drop_last=True
# @accept: shuffle = True
# @bad: train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
# @why: 打乱让每个小批量都接近整体分布，梯度更“诚实”；不打乱时模型可能先学完所有 0 再学 1，训练会很不稳定。
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
# @endblank
test_loader = DataLoader(test_set, batch_size=1000)
# @check: _xb, _yb = next(iter(train_loader)); assert tuple(_xb.shape) == (128, 1, 28, 28), f"一批训练图片应是 (128, 1, 28, 28)，实际 {tuple(_xb.shape)}"
# @check: import torch.utils.data as _tud; assert isinstance(train_loader.sampler, _tud.RandomSampler), "训练集的 DataLoader 需要打乱（shuffle=True）"

# @step model 定义模型：继承 nn.Module
# @explain
# PyTorch 的模型都继承 `nn.Module`，固定写两件事：
#
# 1. `__init__` 里**创建层**（层里有可学习的参数）
# 2. `forward` 里**写数据怎么流过这些层**（前向传播）
#
# 全连接层 `nn.Linear(输入数, 输出数)` 只接受“拉平”的向量，所以要先把 `(B, 1, 28, 28)` 变成 `(B, 784)`。
# @end
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # @todo: 输入 28×28=784 个像素 → 256 个隐藏单元 → 10 个类别
        # @blank fc1 mask=784
        # @hint: 一张 28×28 的图片拉平以后有多少个数？
        # @option: 28
        # @option: 256
        # @option: 1
        # @accept: 28*28
        # @accept: 28 * 28
        # @bad: self.fc1 = nn.Linear(28, 256)
        # @why: 28 × 28 = 784，拉平后的每个像素都是一个输入特征。
        self.fc1 = nn.Linear(784, 256)
        # @endblank
        # @blank fc2 mask=10
        # @hint: MNIST 有几个类别？
        # @option: 1
        # @option: 784
        # @option: 9
        # @why: 数字 0–9 共 10 类，最后一层输出 10 个分数（logits），分数最大的就是预测结果。
        self.fc2 = nn.Linear(256, 10)
        # @endblank

    def forward(self, x):
        # @todo: 把 (B, 1, 28, 28) 拉平成 (B, 784)，再依次经过 fc1 → ReLU → fc2
        # @blank flatten mask=flatten
        # @hint: 哪个函数把多维张量“压扁”？第二个参数 1 表示保留第 0 维（批大小）
        # @option: reshape
        # @option: squeeze
        # @option: view_as
        # @accept: flatten
        # @why: torch.flatten(x, 1) 保留批维度，把其余维度合成一维；squeeze 只会去掉长度为 1 的维度。
        x = torch.flatten(x, 1)
        # @endblank
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# @todo: 创建模型并放到 device 上
# @blank model_to mask=.to(device)
# @hint: 模型的参数默认在 CPU 上，怎么搬到 GPU？
# @option: .cuda()
# @option: .device()
# @option: .gpu()
# @why: .to(device) 对 cuda / mps / cpu 都适用；.cuda() 只能用于 NVIDIA 显卡，在 Mac 上会报错。
model = MLP().to(device)
# @endblank
print(model)
# @check: _out = model(torch.zeros(4, 1, 28, 28, device=device)); assert tuple(_out.shape) == (4, 10), f"模型输出应是 (4, 10)，实际 {tuple(_out.shape)}"
# @check: assert next(model.parameters()).device.type == device.type, "模型参数不在 device 上，忘了 .to(device)？"

# @step optim 损失函数、优化器与超参数
# @explain
# - **损失函数**衡量预测有多差。多分类用交叉熵 `nn.CrossEntropyLoss`，它的输入是**未经 softmax 的分数**（logits）和类别下标。
# - **优化器**根据梯度更新参数，要告诉它“更新谁”——`model.parameters()`——以及学习率 `lr`。
# - `EPOCHS`：把训练集完整看几遍。
# @end
# @todo: 多分类用交叉熵损失
# @blank criterion mask=CrossEntropyLoss
# @hint: 多分类问题的标准损失函数
# @option: MSELoss
# @option: BCELoss
# @option: L1Loss
# @why: CrossEntropyLoss 内部先做 log-softmax 再算负对数似然，适合多分类；MSELoss 用于回归。
criterion = nn.CrossEntropyLoss()
# @endblank
# @todo: 用带动量的 SGD 更新模型的全部参数，学习率 0.1
# @blank opt_params mask=model.parameters()
# @hint: 优化器需要知道要更新哪些张量
# @option: model
# @option: model.state_dict()
# @option: model.forward
# @why: model.parameters() 返回所有可学习参数（这里是两层的权重和偏置共 4 个张量）。
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# @endblank
EPOCHS = 3
# @check: assert isinstance(criterion, nn.CrossEntropyLoss), "损失函数应该是 nn.CrossEntropyLoss()"
# @check: assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters())), "优化器里应该包含模型的全部参数"
# @fast: EPOCHS = 1

# @step evaluate 评估函数：关掉梯度，切到评估模式
# @explain
# 评估时不需要反向传播，所以：
#
# - `@torch.no_grad()`：不记录计算图，更快、更省内存
# - `model.eval()`：切到评估模式（Dropout、BatchNorm 在训练和评估时行为不同，这个模型虽然没有，但养成习惯）
#
# 预测类别 = 分数最大的那一类：`logits.argmax(1)`。
# @end
# @blank nograd mask=no_grad
# @hint: 评估时不需要梯度，用哪个装饰器关掉？
# @option: enable_grad
# @option: inference
# @option: grad
# @why: no_grad 让这段代码不构建计算图；enable_grad 恰好相反。
@torch.no_grad()
# @endblank
def evaluate(loader):
    # @blank eval_mode mask=eval
    # @hint: 切换到评估模式的方法
    # @option: train
    # @option: test
    # @option: predict
    # @why: model.eval() 切到评估模式；训练时要再调用 model.train() 切回来。
    model.eval()
    # @endblank
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * len(yb)
        # @blank argmax mask=argmax(1)
        # @hint: 每一行 10 个分数里，取最大值的“位置”作为预测类别
        # @option: max(1)
        # @option: argmax(0)
        # @option: softmax(1)
        # @why: argmax(1) 在类别维度（第 1 维）上找最大值的下标；argmax(0) 会在批维度上找，结果是错的。
        correct += (logits.argmax(1) == yb).sum().item()
        # @endblank
        total += len(yb)
    return loss_sum / total, correct / total
# @check: _l, _a = evaluate(test_loader); assert 0.0 <= _a <= 1.0, f"准确率应在 0–1 之间，实际 {_a}"
# @check: assert not model.training, "evaluate() 里需要调用 model.eval()"

# @step train 训练循环：前向 → 损失 → 清梯度 → 反向传播 → 更新
# @explain
# 这是所有训练代码里**最核心、永远不变**的五步，每个小批量都要做一遍：
#
# 1. **前向传播**：`logits = model(xb)`
# 2. **计算损失**：`loss = criterion(logits, yb)`
# 3. **清空梯度**：`optimizer.zero_grad()`——PyTorch 的梯度默认会**累加**，不清空就会把上一批的梯度也算进来
# 4. **反向传播**：`loss.backward()`——用链式法则算出每个参数的梯度，存进 `参数.grad`
# 5. **更新参数**：`optimizer.step()`——按梯度和学习率修改参数
#
# 每轮开始调用 `model.train()`（因为评估函数切到了 eval 模式）。每轮结束评估一次，并用 `log_epoch` 记录指标，这样实验室页面就能画出曲线。
# @end
for epoch in range(1, EPOCHS + 1):
    # @blank train_mode mask=train()
    # @hint: 评估后要切回哪种模式？
    # @option: eval()
    # @option: training()
    # @option: fit()
    # @why: evaluate() 把模型切到了 eval 模式，每轮训练前要用 model.train() 切回来。
    model.train()
    # @endblank
    t0 = time.time()
    loss_sum, correct, seen = 0.0, 0, 0
    for xb, yb in train_loader:
        # @blank to_dev mask=.to(device)
        # @hint: 数据也要和模型在同一个设备上
        # @option: .cuda()
        # @option: .cpu()
        # @option: .float()
        # @why: 数据和模型必须在同一设备，否则报 “Expected all tensors to be on the same device”。
        xb, yb = xb.to(device), yb.to(device)
        # @endblank
        # @todo: ① 前向传播  ② 计算损失
        # @blank forward mask=model(xb)
        # @hint: 把这一批图片送进模型，得到每类的分数
        # @option: model.forward
        # @option: model.predict(xb)
        # @option: model.parameters()
        # @accept: model.forward(xb)
        # @why: 直接调用 model(xb) 会执行 forward（并触发钩子），这是推荐写法。
        logits = model(xb)
        # @endblank
        # @blank loss mask=criterion(logits, yb)
        # @hint: 损失函数的两个参数：模型输出、真实标签
        # @option: criterion(yb, logits)
        # @option: criterion(logits)
        # @option: criterion(xb, yb)
        # @why: CrossEntropyLoss(输入, 目标)：第一个是 logits，第二个是类别下标，顺序不能反。
        loss = criterion(logits, yb)
        # @endblank
        # @todo: ③ 清空上一步的梯度  ④ 反向传播  ⑤ 更新参数
        # @blank zero_grad mask=zero_grad
        # @hint: 梯度会累加，每步开始前要做什么？
        # @option: step
        # @option: backward
        # @option: reset
        # @why: 不清空的话，这一步的梯度 = 本批梯度 + 之前所有批的梯度，更新方向就错了。
        optimizer.zero_grad()
        # @endblank
        # @blank backward mask=backward
        # @hint: 哪个调用会用链式法则计算所有参数的梯度？
        # @option: step
        # @option: grad
        # @option: forward
        # @bad: pass
        # @why: loss.backward() 从损失出发反向传播，把梯度存进每个参数的 .grad。
        loss.backward()
        # @endblank
        # @blank step mask=step
        # @hint: 有了梯度之后，哪个调用真正修改参数？
        # @option: update
        # @option: zero_grad
        # @option: backward
        # @bad: pass
        # @why: optimizer.step() 按 “参数 -= 学习率 × 梯度”（加上动量等）更新参数。
        optimizer.step()
        # @endblank
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        seen += len(yb)
    epoch_time = time.time() - t0
    test_loss, test_acc = evaluate(test_loader)
    print(f"第 {epoch} 轮  训练损失 {loss_sum / seen:.4f}  训练准确率 {correct / seen:.4f}  测试准确率 {test_acc:.4f}  用时 {epoch_time:.1f}s")
    log_epoch(epoch, train_loss=loss_sum / seen, train_acc=correct / seen, test_loss=test_loss, test_acc=test_acc,
              epoch_time=epoch_time, img_per_s=seen / epoch_time)
# @check: assert test_acc > 0.8, f"训练后测试准确率只有 {test_acc:.2%}，正常应该超过 80%。检查五步顺序和 zero_grad/backward/step 是否写对"

# @step save 保存训练好的模型
# @explain
# 训练好的参数存在 `model.state_dict()` 里（一个 “参数名 → 张量” 的字典）。保存它，以后可以加载回来直接预测，不用重新训练：
#
# ```python
# model = MLP().to(device)
# model.load_state_dict(torch.load("mnist_mlp.pt"))
# ```
# @end
# @blank save_sd mask=state_dict()
# @hint: 保存的是参数字典，而不是整个模型对象
# @option: parameters()
# @option: modules()
# @option: eval()
# @why: 推荐保存 state_dict：只含参数，文件小、跨代码版本更稳；直接 torch.save(model) 会依赖类定义的位置。
torch.save(model.state_dict(), "mnist_mlp.pt")
# @endblank
print("模型已保存到 mnist_mlp.pt")
# @check: import os as _o; assert _o.path.exists("mnist_mlp.pt"), "没有找到保存的 mnist_mlp.pt"
# @check: _sd = torch.load("mnist_mlp.pt"); assert "fc1.weight" in _sd, "保存的应是 model.state_dict()"
