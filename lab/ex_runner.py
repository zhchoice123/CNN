"""在本机执行练习评测脚本：独立临时目录、超时、输出截断，执行完删除临时目录。"""
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import platform_utils as pu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / "runs" / "_exercise"
SCRIPT_NAME = "exercise_run.py"
MAX_OUT = 200_000
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_lock = threading.Lock()          # 同一时刻只评测一份代码（共用一块 GPU）


def run(code, timeout=120, keep=False):
    """返回 dict(stdout, returncode, timed_out, seconds)。"""
    WORK.mkdir(parents=True, exist_ok=True)
    d = WORK / uuid.uuid4().hex[:10]
    d.mkdir()
    (d / SCRIPT_NAME).write_text(code, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(HERE), LAB_DATA=os.environ.get("LAB_DATA", str(ROOT / "datasets")),
               PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTORCH_ENABLE_MPS_FALLBACK="1", EX_CHECK="1", PYTHON_COLORS="0", NO_COLOR="1")
    env.pop("FORCE_COLOR", None)
    out_path = d / "out.txt"
    t0 = time.time()
    timed_out = False
    with _lock, open(out_path, "wb") as out:
        proc = subprocess.Popen(pu.python_cmd(d / SCRIPT_NAME, isolated=False), cwd=str(d), env=env,
                                stdout=out, stderr=subprocess.STDOUT, **pu.popen_kwargs())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            pu.stop(proc, hard=True)
            proc.wait()
    raw = out_path.read_bytes()
    text = ANSI.sub("", raw[-MAX_OUT:].decode("utf-8", "replace"))
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return dict(stdout=text, returncode=proc.returncode, timed_out=timed_out, seconds=round(time.time() - t0, 2))
