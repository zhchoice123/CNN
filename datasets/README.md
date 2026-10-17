# 本地数据集

由 `lab/datasets_manager.py` 下载，采用 torchvision 原生目录结构，任何 PyTorch 代码都可以离线直接使用：

```python
import torchvision
ds = torchvision.datasets.CIFAR10(root="<本目录绝对路径>", train=True, download=False)
```

| 数据集 | 目录 / 文件 | 说明 |
|---|---|---|
| MNIST | `MNIST/raw/` | 手写数字，灰度 28×28，6 万训练 + 1 万测试 |
| Fashion-MNIST | `FashionMNIST/raw/` | 10 类服饰，格式同 MNIST |
| CIFAR-10 | `cifar-10-batches-py/`（+ `cifar-10-python.tar.gz`） | 10 类彩色 32×32，5 万训练 + 1 万测试 |
| CIFAR-100 | `cifar-100-python/`（+ `cifar-100-python.tar.gz`） | 同图像，100 个细类 |
| KMNIST / SVHN | 按需下载 | 在网页「实验室 → 数据集」点下载 |

压缩包下载时 torchvision 会校验 MD5。查看状态：`lab/.venv/bin/python -I lab/datasets_manager.py status`。
