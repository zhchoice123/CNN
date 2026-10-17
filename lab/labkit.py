"""给“别的来源的训练代码”用的小工具：用本地数据集、自动选设备、统一记录每轮指标，
这样它们的训练过程也能出现在网页的“运行记录”里，和内置训练一起对比。

在脚本开头加：
    from labkit import DATA_ROOT, get_device, load_tensors, log_epoch

  DATA_ROOT      本地数据集目录（torchvision 原生结构）。教材代码里的
                 torchvision.datasets.CIFAR10(root=DATA_ROOT, download=False) 即可离线使用。
  get_device()   有 Apple GPU(MPS) 用 MPS，否则 CUDA / CPU
  load_tensors() 一次拿到已归一化的 (xtr, ytr, xte, yte) 张量
  log_epoch()    每个 epoch 末调用一次，记录指标（网页据此画曲线并对比）
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA_ROOT = os.environ.get("LAB_DATA", str(HERE.parent / "datasets"))
RUN_DIR = os.environ.get("LAB_RUN_DIR")
_T0 = time.time()
_state = {"train_elapsed": 0.0}


def get_device():
    """NVIDIA 显卡 → cuda；Apple 芯片 → mps；否则 cpu。"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device=None):
    """MPS/CUDA 是异步的：计时前后调用一次，耗时才准确。"""
    import torch
    d = device or get_device()
    if d.type == "mps":
        torch.mps.synchronize()
    elif d.type == "cuda":
        torch.cuda.synchronize()


def load_tensors(name, device=None, normalize=True):
    """name: mnist / fmnist / cifar10 / cifar100 / kmnist / svhn。返回 float32 的 N×C×H×W 与 int64 标签。"""
    import torch
    import datasets_manager as dm
    from train import MEAN_STD
    xtr, ytr = dm.load_arrays(name, "train")
    xte, yte = dm.load_arrays(name, "test")
    if normalize:
        m, s = MEAN_STD[name]
        m = torch.tensor(m).view(1, -1, 1, 1)
        s = torch.tensor(s).view(1, -1, 1, 1)
        xtr, xte = (xtr.float() / 255 - m) / s, (xte.float() / 255 - m) / s
    if device is not None:
        xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    return xtr, ytr, xte, yte


def log_epoch(epoch, **metrics):
    """记录一个 epoch 的指标。常用键：train_loss, train_acc, test_loss, test_acc, epoch_time, lr, img_per_s。"""
    row = dict(epoch=int(epoch), **{k: (float(v) if v is not None else None) for k, v in metrics.items()})
    if "epoch_time" in row:
        _state["train_elapsed"] += row["epoch_time"]
        row.setdefault("train_elapsed", _state["train_elapsed"])
    row.setdefault("wall", time.time() - _T0)
    line = json.dumps(row)
    print("@@epoch " + line, flush=True)
    if RUN_DIR and not _state.get("info"):           # 第一次记录时写下设备与硬件信息，跨机器对比时用来区分
        _state["info"] = True
        try:
            import platform_utils
            info = dict(device=str(get_device()), hw=platform_utils.hardware_info())
            (Path(RUN_DIR) / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        except Exception:      # noqa: BLE001  记录失败不影响训练
            pass
    if RUN_DIR:
        with open(Path(RUN_DIR) / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
