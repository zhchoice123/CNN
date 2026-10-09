"""跨平台小工具：让实验室在 macOS（Apple 芯片 / MPS）和 Windows（NVIDIA / CUDA）上都能跑。

* venv_python()   虚拟环境里的 Python：macOS/Linux 是 .venv/bin/python，Windows 是 .venv\\Scripts\\python.exe
* popen_kwargs()  子进程放进独立的进程组，方便整组停止
* stop(proc)      先“温和”停止（让训练写完 summary），hard=True 时强制结束整棵进程树
* python_cmd()    统一加 -X utf8：Windows 默认编码不是 UTF-8，中文日志会乱码或报错
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"


def venv_python():
    p = HERE / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    return p


def python_cmd(*args, isolated=True):
    """组装子进程命令。isolated=True 用 -I（不读 PYTHON* 环境变量、不把当前目录加进路径）。"""
    cmd = [str(venv_python()), "-X", "utf8"]
    cmd += ["-I"] if isolated else ["-s"]
    return cmd + [str(a) for a in args]


def popen_kwargs():
    if IS_WIN:
        return dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return dict(start_new_session=True)


def stop(proc, hard=False):
    """温和停止：POSIX 发 SIGTERM 给整个进程组；Windows 发 CTRL_BREAK（train.py 会捕获并写入 stopped 状态）。
    hard=True：POSIX 发 SIGKILL；Windows 用 taskkill /T /F 结束整棵进程树。"""
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            if hard:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL if hard else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def install_stop_handlers(handler):
    """训练子进程里注册“收到停止请求”的处理函数（POSIX: SIGTERM/SIGINT；Windows: SIGBREAK/SIGINT）。"""
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def hardware_info():
    """记录在每次运行里，跨机器对比时用来区分“这是哪台电脑跑的”。"""
    import platform
    info = dict(host=platform.node(), os=f"{platform.system()} {platform.release()}", machine=platform.machine(), python=sys.version.split()[0])
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            chip = ""
            if sys.platform == "darwin":
                try:
                    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3).stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    pass
            info["gpu"] = f"{chip or 'Apple'} GPU (MPS)"
        else:
            info["gpu"] = "CPU"
    except ImportError:
        pass
    return info
