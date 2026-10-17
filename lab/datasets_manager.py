"""数据集管理：检查 / 下载 / 加载（离线可用）。

所有数据放在  <项目根>/datasets/  下，使用 torchvision 的原生目录结构：
    datasets/MNIST/raw/...            datasets/FashionMNIST/raw/...
    datasets/cifar-10-batches-py/...  datasets/cifar-100-python/...
因此任何教材/网上代码只要写  torchvision.datasets.CIFAR10(root=DATA_ROOT, download=False)
就能直接用，不必再下载。

命令行：
    .venv/bin/python -I datasets_manager.py status
    .venv/bin/python -I datasets_manager.py download mnist fmnist cifar10 cifar100
"""
import json
import os
import sys
import time
from pathlib import Path

try:  # macOS 官方 Python 没有系统根证书；用 certifi 提供，保持 HTTPS 校验开启
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("LAB_DATA", ROOT / "datasets"))

# name -> 说明。mb 为解压后大致占用，仅用于界面提示。
SPECS = {
    "mnist":    dict(cls="MNIST",        title="MNIST 手写数字",        classes=10,  shape=(1, 28, 28), n=(60000, 10000), mb=55,  core=True,
                     desc="0–9 手写数字，灰度 28×28。入门首选，几秒钟就能训练一轮。",
                     check="MNIST/raw/train-images-idx3-ubyte"),
    "fmnist":   dict(cls="FashionMNIST", title="Fashion-MNIST 服饰",    classes=10,  shape=(1, 28, 28), n=(60000, 10000), mb=55,  core=True,
                     desc="10 类服饰灰度图，格式同 MNIST 但更难，常用来替代 MNIST。",
                     check="FashionMNIST/raw/train-images-idx3-ubyte"),
    "cifar10":  dict(cls="CIFAR10",      title="CIFAR-10 物体",         classes=10,  shape=(3, 32, 32), n=(50000, 10000), mb=340, core=True,
                     desc="10 类彩色小图（飞机、汽车、鸟、猫…），CNN 经典基准。",
                     check="cifar-10-batches-py/data_batch_1"),
    "cifar100": dict(cls="CIFAR100",     title="CIFAR-100 物体（100 类）", classes=100, shape=(3, 32, 32), n=(50000, 10000), mb=340, core=True,
                     desc="同 CIFAR-10 的图像，但分成 100 个细类，更难。",
                     check="cifar-100-python/train"),
    # 按需下载（不预装）
    "kmnist":   dict(cls="KMNIST",       title="KMNIST 日文假名",       classes=10,  shape=(1, 28, 28), n=(60000, 10000), mb=55,  core=False,
                     desc="10 类日文平假名，格式同 MNIST。",
                     check="KMNIST/raw/train-images-idx3-ubyte"),
    "svhn":     dict(cls="SVHN",         title="SVHN 街景门牌号",       classes=10,  shape=(3, 32, 32), n=(73257, 26032), mb=330, core=False,
                     desc="街景门牌号数字，彩色 32×32，比 MNIST 更接近真实场景。",
                     check="train_32x32.mat"),
}


# 下载中途只有压缩包：用它的大小估算进度，也用来发现“终端里手动启动的下载”，避免重复下载同一个文件
ARCHIVES = {"cifar10": ("cifar-10-python.tar.gz", 170498071), "cifar100": ("cifar-100-python.tar.gz", 169001437)}


def partial(name):
    if name not in ARCHIVES or is_ready(name):
        return None
    fn, total = ARCHIVES[name]
    p = DATA_ROOT / fn
    if not p.exists():
        return None
    st = p.stat()
    return dict(percent=round(min(99.0, st.st_size / total * 100), 1), active=(time.time() - st.st_mtime) < 120)


def is_ready(name):
    return (DATA_ROOT / SPECS[name]["check"]).exists()


def dir_mb(name):
    top = SPECS[name]["check"].split("/")[0]
    p = DATA_ROOT / top
    if p.is_file():
        return round(p.stat().st_size / 1e6, 1)
    if not p.exists():
        return 0
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def status():
    out = []
    for k, s in SPECS.items():
        out.append(dict(name=k, title=s["title"], desc=s["desc"], classes=s["classes"], shape=list(s["shape"]),
                        n_train=s["n"][0], n_test=s["n"][1], approx_mb=s["mb"], core=s["core"],
                        ready=is_ready(k), size_mb=dir_mb(k), partial=partial(k)))
    return out


def download(name):
    import torchvision.datasets as D
    s = SPECS[name]
    cls = getattr(D, s["cls"])
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if name == "svhn":
        for split in ("train", "test"):
            cls(root=str(DATA_ROOT), split=split, download=True)
    else:
        for train in (True, False):
            cls(root=str(DATA_ROOT), train=train, download=True)
    return load_arrays(name, "test")  # 简单自检：能读出来


def load_arrays(name, split):
    """返回 (uint8 张量 N×C×H×W, int64 标签)。始终 download=False，离线读取。"""
    import numpy as np
    import torch
    import torchvision.datasets as D
    s = SPECS[name]
    cls = getattr(D, s["cls"])
    if name == "svhn":
        ds = cls(root=str(DATA_ROOT), split=split, download=False)
        x = torch.from_numpy(ds.data)                       # N,3,32,32 uint8
        y = torch.from_numpy(np.asarray(ds.labels)).long()
    else:
        ds = cls(root=str(DATA_ROOT), train=(split == "train"), download=False)
        d = ds.data
        x = torch.as_tensor(np.asarray(d))
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)   # 灰度 N,1,H,W / 彩色 N,3,H,W
        y = torch.as_tensor(np.asarray(ds.targets)).long()
    return x.contiguous(), y


def write_manifest():
    info = {k: {"title": v["title"], "ready": is_ready(k), "size_mb": dir_mb(k)} for k, v in SPECS.items()}
    (DATA_ROOT / "MANIFEST.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        for r in status():
            print(f"{r['name']:9s} {'✔' if r['ready'] else '✘'}  {r['size_mb']:>7} MB  {r['title']}")
    elif cmd == "download":
        names = sys.argv[2:] or [k for k, v in SPECS.items() if v["core"]]
        for n in names:
            if n not in SPECS:
                print("未知数据集", n); continue
            if is_ready(n):
                print(f"{n}: 已存在，跳过"); continue
            print(f"{n}: 下载中 …", flush=True)
            x, y = download(n)
            print(f"{n}: 完成，测试集 {tuple(x.shape)}，类别数 {int(y.max()) + 1}", flush=True)
        write_manifest()
