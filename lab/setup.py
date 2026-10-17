#!/usr/bin/env python3
"""一次性准备训练环境（macOS / Windows / Linux 通用，可重复运行）。

    macOS（Apple 芯片）：   python3 lab/setup.py
    Windows（RTX 3060 等）： py -3.13 lab\\setup.py        或双击 lab\\setup_windows.bat
    只装 CPU 版：            python lab/setup.py --cpu
    换 CUDA 版本：           python lab/setup.py --cuda cu130      （需要 580 以上的显卡驱动）
    顺便下载数据集：         python lab/setup.py --datasets

做的事：创建 lab/.venv → 按平台安装合适的 PyTorch → 安装 numpy / pillow / certifi → 打印检测结果。
Windows 上 PyTorch 的 CUDA 版不在 PyPI，需要从 download.pytorch.org 的对应目录安装；
torch 2.14.1 的 Windows 版提供 cu126 和 cu130 两种，默认 cu126（显卡驱动 ≥ 560 即可）。
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
VENV = HERE / ".venv"
VPY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def has_nvidia():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            print("  检测到 NVIDIA 显卡：", out.stdout.strip().splitlines()[0])
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="只安装 CPU 版 PyTorch")
    ap.add_argument("--cuda", default="cu126", help="Windows/Linux 的 CUDA 构建：cu126（默认）或 cu130")
    ap.add_argument("--datasets", action="store_true", help="安装完成后下载 MNIST / Fashion-MNIST / CIFAR-10 / CIFAR-100")
    a = ap.parse_args()

    v = sys.version_info
    print(f"使用的 Python：{sys.executable}（{v.major}.{v.minor}.{v.micro}）")
    if v < (3, 10):
        sys.exit("需要 Python 3.10 以上，推荐 3.12 或 3.13")
    if v >= (3, 14) and IS_WIN:
        print("  ⚠ Windows 上推荐用 Python 3.12 / 3.13（torchvision 的 CUDA 版这两个版本最齐全）：py -3.13 lab\\setup.py")

    if not VPY.exists():
        print("① 创建虚拟环境 lab/.venv")
        sh([sys.executable, "-m", "venv", VENV])
    else:
        print("① 虚拟环境已存在，跳过创建")
    sh([VPY, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("② 安装 PyTorch")
    torch_cmd = [VPY, "-m", "pip", "install", "torch", "torchvision"]
    if a.cpu:
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        print("  CPU 版")
    elif sys.platform == "darwin":
        print("  macOS：PyPI 上的 torch 自带 Apple GPU（MPS）支持")
    elif has_nvidia():
        if IS_WIN:
            torch_cmd += ["--index-url", f"https://download.pytorch.org/whl/{a.cuda}"]
            print(f"  Windows + NVIDIA：使用 {a.cuda} 构建")
        else:
            print("  Linux + NVIDIA：PyPI 上的 torch 默认带 CUDA")
    else:
        print("  没有检测到 NVIDIA 显卡（nvidia-smi 不可用），安装默认版本；训练会在 CPU 上运行")
    sh(torch_cmd)
    sh([VPY, "-m", "pip", "install", "-q", "numpy", "pillow", "certifi"])

    print("③ 检测")
    code = ("import torch, torchvision, platform_utils as p, json;"
            "print('PyTorch', torch.__version__, '| torchvision', torchvision.__version__);"
            "print('CUDA 可用:', torch.cuda.is_available(), '| Apple MPS 可用:', torch.backends.mps.is_available());"
            "print('硬件:', json.dumps(p.hardware_info(), ensure_ascii=False))")
    sh([VPY, "-X", "utf8", "-c", code], env=dict(os.environ, PYTHONPATH=str(HERE)))

    if a.datasets:
        print("④ 下载数据集（官方服务器较慢，可能需要几十分钟）")
        sh([VPY, "-X", "utf8", "-I", HERE / "datasets_manager.py", "download"])

    start = "lab\\start_windows.bat（或 python lab\\server.py）" if IS_WIN else "bash lab/start.sh（或 python3 lab/server.py）"
    print(f"\n✔ 环境就绪。启动：{start}，然后打开 http://localhost:8766")
    print("  验收练习：python lab/verify_exercises.py")


if __name__ == "__main__":
    main()
