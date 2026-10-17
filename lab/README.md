# 训练实验室（Training Lab）

在本机用 PyTorch 训练 CNN，**换参数跑几次，直接对比准确率和训练效率**；也可以把教材里的训练代码拿来跑，和内置训练一起对比。

## 启动

| | macOS（Apple 芯片，MPS） | Windows（NVIDIA，如 RTX 3060） |
|---|---|---|
| 一次性安装 | `python3 lab/setup.py`（或 `bash lab/setup.sh`） | 双击 `lab\setup_windows.bat`（或 `py -3.13 lab\setup.py`） |
| 启动 | `bash lab/start.sh` | 双击 `lab\start_windows.bat` |

`setup.py` 会按平台装对 PyTorch：macOS 用 PyPI（自带 MPS）；Windows 检测到 NVIDIA 显卡时从 `download.pytorch.org/whl/cu126` 安装 CUDA 版（显卡驱动 ≥ 560；`--cuda cu130` 可换新版，`--cpu` 只装 CPU 版，`--datasets` 顺便下载数据集）。

然后打开 <http://localhost:8766/#/lab>。这个端口同时提供整个学习站，所以不需要再单独启动 8765 那个静态服务。

> 在 Claude 桌面应用里，也可以直接让 Claude 用项目的 `cnn-lab` 配置启动（`.claude/launch.json`）。

## 页面功能

| 标签 | 作用 |
|---|---|
| 新建训练 | 表单里调所有超参数（数据集、模型、优化器、学习率、批大小、调度、增强、Dropout、BN…），每项都有说明。**参数对比**模式：只改一个参数（如学习率取 4 个值），依次各训练一次 |
| 运行记录与对比 | 勾选最多 8 次运行：曲线（准确率/损失 × 轮数/用时）、悬停十字线、表格视图、**效率对比表**（吞吐、每轮用时、达到共同目标准确率所需时间）和自动生成的“读图要点” |
| 数据集 | 本地数据集状态、一键下载、引用方式 |
| 自定义脚本 | 载入模板或代码库里的教材代码，改完直接运行，并参与对比 |

### 效率怎么比

* **稳态吞吐**（图/秒）：排除第 1 轮（包含 GPU 预热）。
* **达到共同目标准确率所需时间**：取所有选中运行里“最低的那个最佳准确率”作为目标，看谁最先到——这才是公平的“收敛速度”比较，吞吐高并不等于收敛快。
* 计时只统计训练（每轮前后同步设备），**不含**测试集评估。
* 同一时刻只运行一个任务（共用一块 GPU，并行会让计时不准）。

## 代码练习的评测

学习站的“代码练习”（`#/practice`）L3/L4 会把读者写的代码发到 `/api/exercise/check`：服务器按步骤标题切开代码、在每步末尾注入检查、在临时目录里用 `.venv` 运行（快速模式：缩小数据和轮数，几秒出结果），返回哪一步失败、失败原因和读者代码的行号。练习的写法见 `study_site/exercises/README.md`。

每台机器都要跑一次自动验收（答案全部通过、错误样例都能检出、单次评测 ≤ 30 s）：

```bash
python3 lab/verify_exercises.py
```

完整的两台机器验收清单见 [`ACCEPTANCE.md`](ACCEPTANCE.md)。

## 两台电脑之间对比

运行记录页“导出所选”得到一个 `.json`，拿到另一台电脑上“导入”。导入的运行会标出来源电脑，效率表多一列“机器”。准确率可以跨机器比较；速度取决于硬件，比较参数效率请用同一台电脑上的运行。

## 数据集

放在项目根目录 `datasets/`，torchvision 原生结构。**任何** PyTorch 代码里写
`torchvision.datasets.CIFAR10(root="…/datasets", download=False)` 就能离线用。
预装：MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100；KMNIST、SVHN 按需下载。命令行：

```bash
lab/.venv/bin/python -I lab/datasets_manager.py status
lab/.venv/bin/python -I lab/datasets_manager.py download svhn
```

> macOS 的官方 Python 没有系统根证书，下载会报 `CERTIFICATE_VERIFY_FAILED`。这里通过 `certifi` 提供证书，**保持 HTTPS 校验开启**，不要关闭校验。

## 让别的代码也能参与对比：`labkit`

```python
from labkit import DATA_ROOT, get_device, load_tensors, log_epoch

device = get_device()                                   # 有 Apple GPU 用 MPS
xtr, ytr, xte, yte = load_tensors("cifar10", device)    # 已归一化的本地数据
...
log_epoch(epoch, train_loss=..., train_acc=..., test_loss=..., test_acc=..., epoch_time=...)
```

* 脚本里已有 `torchvision.datasets.X(root=..., download=True)` 的教材代码：在代码库页点“复制（改用本地数据集）”，会自动把路径改成 `DATA_ROOT`、`download=False`。
* 不调用 `log_epoch` 也能运行，只是没有曲线（可在运行详情里看日志）。
* 模板见 `lab/templates/`，都是短小可读的完整训练循环。
* D2L 的代码依赖 `d2l` 库和上文变量，Keras/TensorFlow 代码本机没装 TF，这两类多数只能阅读参考。

## 目录

```
lab/
├── server.py            本地服务器（纯标准库）：提供网页 + /api，调度训练子进程
├── train.py             统一训练脚本（数据常驻设备、批量增强、逐轮写指标）
├── lab_models.py        模型：mlp / lenet / simple_cnn / vgg_mini / resnet_mini / resnet18
├── lab_schema.py        超参数定义与校验（网页表单由它生成）
├── datasets_manager.py  数据集检查 / 下载 / 离线加载
├── labkit.py            给自定义脚本用的小工具
├── templates/           可读的完整训练示例
├── platform_utils.py    跨平台：venv 路径、进程组与停止（POSIX 信号 / Windows CTRL_BREAK、taskkill）、UTF-8、硬件信息
├── exercise_core.py     代码练习：解析标记、生成四级练习、组装评测脚本、解析报错
├── ex_runner.py         在临时目录里执行评测（超时、输出截断）
├── verify_exercises.py  练习的自动验收
├── setup.py             跨平台安装（setup.sh / setup_windows.bat 只是包装）
├── start.sh / start_windows.bat
├── ACCEPTANCE.md        两台机器的验收清单
├── runs/                每次运行的配置、指标、日志（可随时删除）
└── .venv/               PyTorch 环境（不入库）
```

## 安全说明

服务器只监听 `127.0.0.1`；校验 `Host`（防 DNS 重绑定）与 `Origin`；所有写操作必须带自定义头 `X-Lab`，其他网站无法借你的浏览器触发训练。
“自定义脚本”会在你的机器上执行任意 Python——和你在终端里运行是一样的，请只运行你信任的代码。
