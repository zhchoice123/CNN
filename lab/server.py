#!/usr/bin/env python3
"""本地训练实验室服务器（只用 Python 标准库；训练在 .venv 的子进程里跑）。

    python3 lab/server.py            # 然后打开 http://localhost:8766/#/lab

同一个端口同时提供：
  * 学习站网页（study_site/）            →  /
  * 实验室 API                            →  /api/...

安全：只监听 127.0.0.1；校验 Host（防 DNS 重绑定）和 Origin；写操作必须带 X-Lab 头
（浏览器里别的网站无法跨域带自定义头，所以不能借你的浏览器来触发训练）。
"""
import atexit
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SITE = ROOT / "study_site"
RUNS = HERE / "runs"
TEMPLATES = HERE / "templates"
PY = None   # 由 platform_utils.venv_python() 决定（macOS: .venv/bin/python；Windows: .venv\\Scripts\\python.exe）
PORT = int(os.environ.get("LAB_PORT", "8766"))
DATA_ROOT = Path(os.environ.get("LAB_DATA", ROOT / "datasets"))

sys.path.insert(0, str(HERE))
import datasets_manager as dm   # noqa: E402  纯标准库导入（torch 只在函数里延迟导入）
import ex_runner                # noqa: E402
import exercise_core as xc      # noqa: E402
import lab_schema               # noqa: E402
import platform_utils as pu     # noqa: E402

ALLOWED_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
ALLOWED_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}", "http://localhost:8765", "http://127.0.0.1:8765"}
MAX_BODY = 20_000_000          # 导入运行记录时文件可能较大
SCRIPT_TIMEOUT = 3 * 3600

RUNS.mkdir(exist_ok=True)
FINAL = {"done", "failed", "stopped", "diverged", "interrupted"}


# ───────────────────────── 运行管理 ─────────────────────────
class Runs:
    def __init__(self):
        self.lock = threading.RLock()
        self.queue = []          # 排队中的 run id
        self.procs = {}          # 运行中的 id -> Popen
        self.stop_requested = set()
        for p in RUNS.glob("r*/run.json"):   # 重启后，上次没结束的任务标记为中断
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                if r.get("status") in ("queued", "running"):
                    r["status"] = "interrupted"
                    r["ended"] = time.time()
                    p.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            except (OSError, ValueError):
                pass
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- 持久化 ----
    def _dir(self, rid):
        return RUNS / rid

    def _read(self, rid):
        return json.loads((self._dir(rid) / "run.json").read_text(encoding="utf-8"))

    def _write(self, r):
        p = self._dir(r["id"]) / "run.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ---- 创建 / 排队 ----
    def create(self, kind, name, group, config=None, code=None):
        rid = "r" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        d = self._dir(rid)
        d.mkdir(parents=True)
        r = dict(id=rid, kind=kind, name=name, group=group, config=config or {}, status="queued", created=time.time())
        if code is not None:
            (d / "script.py").write_text(code, encoding="utf-8")
        with self.lock:
            self._write(r)
            self.queue.append(rid)
        return r

    # ---- 调度线程：同一时刻只跑一个任务，保证计时对比公平 ----
    def _worker(self):
        while True:
            time.sleep(0.4)
            with self.lock:
                if self.procs or not self.queue:
                    continue
                rid = self.queue.pop(0)
            try:
                self._run(rid)
            except Exception as e:     # noqa: BLE001  调度线程不能死
                with self.lock:
                    r = self._read(rid)
                    r.update(status="failed", error=str(e), ended=time.time())
                    self._write(r)

    def _run(self, rid):
        with self.lock:
            r = self._read(rid)
            if r["status"] != "queued":
                return
            d = self._dir(rid)
            env = dict(os.environ, LAB_DATA=str(DATA_ROOT), LAB_RUN_DIR=str(d), PYTORCH_ENABLE_MPS_FALLBACK="1", PYTHONUNBUFFERED="1",
                       PYTHON_COLORS="0", NO_COLOR="1")
            env.pop("FORCE_COLOR", None)
            env["PYTHONUTF8"] = "1"
            if r["kind"] == "train":
                cmd = pu.python_cmd(HERE / "train.py", "--run-dir", d, "--config", json.dumps(r["config"]))
                cwd = str(d)
            else:   # 自定义脚本：不用 -I（要靠 PYTHONPATH 找到 labkit）；脚本放在独立的新目录里运行
                env["PYTHONPATH"] = str(HERE)
                cmd = pu.python_cmd(d / "script.py", isolated=False)
                cwd = str(d)
            log = open(d / "log.txt", "wb")
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, **pu.popen_kwargs())
            r.update(status="running", started=time.time(), pid=proc.pid)
            self._write(r)
            self.procs[rid] = proc
        threading.Thread(target=self._watch, args=(rid, proc, log), daemon=True).start()

    def _watch(self, rid, proc, log):
        timeout = SCRIPT_TIMEOUT
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill(proc, hard=True)
            proc.wait()
        log.close()
        with self.lock:
            self.procs.pop(rid, None)
            r = self._read(rid)
            d = self._dir(rid)
            if rid in self.stop_requested:
                status = "stopped"
            elif (d / "summary.json").exists():
                status = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("status", "done")
            else:
                status = "done" if proc.returncode == 0 else "failed"
            r.update(status=status, ended=time.time(), returncode=proc.returncode)
            self._write(r)

    @staticmethod
    def _kill(proc, hard=False):
        pu.stop(proc, hard=hard)

    def stop(self, rid):
        with self.lock:
            if rid in self.queue:
                self.queue.remove(rid)
                r = self._read(rid)
                r.update(status="stopped", ended=time.time())
                self._write(r)
                return True
            proc = self.procs.get(rid)
            if not proc:
                return False
            self.stop_requested.add(rid)
        self._kill(proc)
        threading.Timer(10, lambda: self._kill(proc, hard=True) if proc.poll() is None else None).start()
        return True

    def delete(self, rid):
        with self.lock:
            if rid in self.procs or rid in self.queue:
                return False
            d = self._dir(rid)
            if d.exists() and d.parent == RUNS:
                shutil.rmtree(d)
        return True

    def export(self, ids):
        """打包若干运行（配置、指标、硬件信息、汇总；不含日志），用于在另一台电脑上导入对比。"""
        out = []
        for rid in ids:
            d = self._dir(rid)
            if not (d / "run.json").exists():
                continue
            item = dict(run=self._read(rid), metrics=self._metrics(d))
            for k in ("info", "summary"):
                if (d / f"{k}.json").exists():
                    try:
                        item[k] = json.loads((d / f"{k}.json").read_text(encoding="utf-8"))
                    except ValueError:
                        pass
            if (d / "script.py").exists():
                item["script"] = (d / "script.py").read_text(encoding="utf-8")
            out.append(item)
        return dict(format="cnn-lab-runs", version=1, exported=time.time(), host=pu.hardware_info(), runs=out)

    def import_bundle(self, b):
        if not isinstance(b, dict) or b.get("format") != "cnn-lab-runs":
            raise ValueError("不是训练实验室导出的运行记录文件")
        have = set()
        for p in RUNS.glob("r*/run.json"):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                have.add(r.get("orig_id") or r["id"])
            except (OSError, ValueError, KeyError):
                pass
        made, skipped = [], 0
        src = (b.get("host") or {})
        for it in b.get("runs", [])[:500]:
            r = dict(it.get("run") or {})
            oid = r.get("orig_id") or r.get("id")
            if not oid or oid in have:
                skipped += 1
                continue
            rid = "r" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
            d = self._dir(rid)
            d.mkdir(parents=True)
            r.update(id=rid, orig_id=oid, imported=True, imported_at=time.time(),
                     group=f"{r.get('group', '')} · 来自 {src.get('host') or '其他电脑'}",
                     status=r.get("status") if r.get("status") in FINAL else "interrupted")
            self._write(r)
            (d / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in it.get("metrics", [])), encoding="utf-8")
            for k in ("info", "summary"):
                if it.get(k):
                    (d / f"{k}.json").write_text(json.dumps(it[k], ensure_ascii=False), encoding="utf-8")
            if it.get("script"):
                (d / "script.py").write_text(it["script"], encoding="utf-8")
            made.append(rid)
            have.add(oid)
        return dict(imported=made, skipped=skipped)

    def shutdown(self):
        for p in list(self.procs.values()):
            self._kill(p)

    # ---- 查询 ----
    @staticmethod
    def _metrics(d):
        rows = []
        p = d / "metrics.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass   # 正在写入的半行
        return rows

    def _brief(self, r, rows):
        d = self._dir(r["id"])
        best = max(rows, key=lambda m: m.get("test_acc", -1)) if rows else None
        steady = [m["img_per_s"] for m in rows[1:] if m.get("img_per_s")]   # 排除第 1 轮（含设备预热）
        out = dict(r, epochs_done=len(rows), best_test_acc=best.get("test_acc") if best else None,
                   best_epoch=best.get("epoch") if best else None, last=rows[-1] if rows else None,
                   train_time=rows[-1].get("train_elapsed") if rows else None,
                   steady_img_per_s=(sum(steady) / len(steady)) if steady else (rows[0].get("img_per_s") if rows else None))
        if (d / "info.json").exists():
            try:
                out["info"] = json.loads((d / "info.json").read_text(encoding="utf-8"))
            except ValueError:
                pass
        if r["status"] == "running" and (d / "progress.json").exists():
            try:
                out["progress"] = json.loads((d / "progress.json").read_text(encoding="utf-8"))
            except ValueError:
                pass
        return out

    def list(self):
        out = []
        for p in sorted(RUNS.glob("r*/run.json"), reverse=True)[:300]:     # 只返回最近 300 条，避免记录过多时轮询变慢
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(self._brief(r, self._metrics(p.parent)))
        return out

    def get(self, rid):
        d = self._dir(rid)
        if not (d / "run.json").exists():
            return None
        r = self._read(rid)
        rows = self._metrics(d)
        out = self._brief(r, rows)
        out["metrics"] = rows
        if (d / "script.py").exists():
            out["script"] = (d / "script.py").read_text(encoding="utf-8")
        log = d / "log.txt"
        out["log"] = log.read_bytes()[-12000:].decode("utf-8", "replace") if log.exists() else ""
        return out


RM = None
DL_JOBS = {}      # 数据集下载任务
ENV_INFO = {"torch": None}


def probe_env():
    if not pu.venv_python().exists():
        ENV_INFO.update(error="未找到 lab/.venv，请先按 README 创建虚拟环境并安装 torch")
        return
    try:
        out = subprocess.run(pu.python_cmd("-c", "import json,torch,torchvision,platform_utils as p;print(json.dumps(dict(torch=torch.__version__,"
                                           "torchvision=torchvision.__version__,mps=torch.backends.mps.is_available(),cuda=torch.cuda.is_available(),"
                                           "hw=p.hardware_info())))", isolated=False),
                             capture_output=True, text=True, timeout=120, env=dict(os.environ, PYTHONPATH=str(HERE)))
        ENV_INFO.update(json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception as e:   # noqa: BLE001
        ENV_INFO.update(error=f"无法导入 torch：{e}")


def dl_start(name):
    if name not in dm.SPECS:
        return None
    job = DL_JOBS.get(name)
    if job and job["state"] == "running":
        return job
    part = dm.partial(name)
    if part and part["active"]:          # 终端里已有一个下载在写这个文件
        return dict(state="running", external=True)
    log = RUNS / f"_download_{name}.log"
    f = open(log, "wb")
    p = subprocess.Popen(pu.python_cmd(HERE / "datasets_manager.py", "download", name), stdout=f, stderr=subprocess.STDOUT,
                         cwd=str(RUNS), **pu.popen_kwargs())
    DL_JOBS[name] = job = dict(state="running", log=str(log), started=time.time())

    def watch():
        p.wait()
        f.close()
        job["state"] = "done" if p.returncode == 0 and dm.is_ready(name) else "failed"
    threading.Thread(target=watch, daemon=True).start()
    return job


def dl_progress(job):
    try:
        tail = Path(job["log"]).read_bytes()[-400:].decode("utf-8", "replace")
    except OSError:
        return None
    m = re.findall(r"(\d+(?:\.\d+)?)%", tail)
    return float(m[-1]) if m else None


def templates():
    out = []
    for p in sorted(TEMPLATES.glob("*.py")):
        code = p.read_text(encoding="utf-8")
        doc = re.match(r'"""(.*?)"""', code, re.S)
        out.append(dict(id=p.stem, file=p.name, doc=(doc.group(1).strip() if doc else ""), code=code))
    return out


# ───────────────────────── HTTP ─────────────────────────
class Handler(SimpleHTTPRequestHandler):
    server_version = "CNNLab/1.0"

    def log_message(self, fmt, *a):          # 安静一点，只记录 API 错误
        if len(a) > 1 and str(a[1]).startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")     # 开发期不缓存，避免改了不生效
        super().end_headers()

    # ---- 工具 ----
    def _cors(self):
        o = self.headers.get("Origin")
        if o in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Lab")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _guard(self, write):
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            self._json({"error": "Host 不被允许"}, 403)
            return False
        o = self.headers.get("Origin")
        if o and o not in ALLOWED_ORIGINS:
            self._json({"error": "Origin 不被允许"}, 403)
            return False
        if write and self.headers.get("X-Lab") != "1":
            self._json({"error": "缺少 X-Lab 头"}, 403)
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("请求体过大")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    # ---- 路由 ----
    def do_OPTIONS(self):
        if not self._guard(False):
            return
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return super().do_GET()
        if not self._guard(False):
            return
        try:
            self._get_api(u.path, parse_qs(u.query))
        except Exception as e:   # noqa: BLE001
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        if not self._guard(True):
            return
        try:
            self._post_api(u.path, self._body())
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:   # noqa: BLE001
            self._json({"error": str(e)}, 500)

    def do_DELETE(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        if not self._guard(True):
            return
        m = re.fullmatch(r"/api/runs/(r[\w-]+)", u.path)
        if not m:
            return self._json({"error": "not found"}, 404)
        ok = RM.delete(m.group(1))
        self._json({"ok": ok} if ok else {"error": "任务仍在运行或排队，请先停止"}, 200 if ok else 409)

    def _get_api(self, path, q):
        if path == "/api/health":
            running = [r for r in RM.list() if r["status"] in ("running", "queued")]
            return self._json(dict(ok=True, env=ENV_INFO, data_root=str(DATA_ROOT), busy=len(running), port=PORT))
        if path == "/api/schema":
            return self._json(dict(fields=lab_schema.FIELDS, defaults=lab_schema.DEFAULTS))
        if path == "/api/templates":
            return self._json(templates())
        if path == "/api/datasets":
            rows = dm.status()
            for r in rows:
                job = DL_JOBS.get(r["name"])
                if job:
                    r["job"] = dict(state=job["state"], percent=dl_progress(job))
            return self._json(rows)
        if path == "/api/runs":
            return self._json(RM.list())
        m = re.fullmatch(r"/api/runs/(r[\w-]+)", path)
        if m:
            r = RM.get(m.group(1))
            return self._json(r if r else {"error": "not found"}, 200 if r else 404)
        if path == "/api/runs/export":
            ids = [i for i in (q.get("ids", [""])[0]).split(",") if re.fullmatch(r"r[\w-]+", i)]
            return self._json(RM.export(ids))
        if path == "/api/compare":
            ids = [i for i in (q.get("ids", [""])[0]).split(",") if re.fullmatch(r"r[\w-]+", i)]
            return self._json([r for r in (RM.get(i) for i in ids) if r])
        self._json({"error": "not found"}, 404)

    def _post_api(self, path, b):
        if path == "/api/runs":
            items = b.get("runs") or []
            if not items or len(items) > 60:
                raise ValueError("runs 数量应在 1–60 之间")
            group = b.get("group") or ("g" + time.strftime("%H%M%S"))
            made = []
            for it in items:
                cfg = lab_schema.normalize(it.get("config"))
                name = it.get("name") or f"{cfg['model']}·{cfg['dataset']}·{cfg['optimizer']}·lr{cfg['lr']:g}"
                made.append(RM.create("train", name[:80], group, config=cfg))
            return self._json({"runs": [m["id"] for m in made], "group": group})
        if path == "/api/runs/import":
            return self._json(RM.import_bundle(b.get("bundle")))
        if path == "/api/script":
            code = b.get("code") or ""
            if not code.strip() or len(code) > 200_000:
                raise ValueError("脚本为空或过长")
            r = RM.create("script", (b.get("name") or "自定义脚本")[:80], b.get("group") or "script", config={"source": b.get("source", "")}, code=code)
            return self._json({"runs": [r["id"]]})
        if path == "/api/exercise/check":
            ex = xc.find(str(b.get("id", "")))
            if not ex:
                raise ValueError("未知的练习")
            code = str(b.get("code", ""))
            if not code.strip() or len(code) > 200_000:
                raise ValueError("代码为空或过长")
            try:
                script, lmap = xc.assemble(ex, code, upto=b.get("upto") or None)
            except xc.ExerciseError as e:
                return self._json(dict(ok=False, passed=[], failed=dict(step=None, title="", kind="structure", message=str(e), line=None)))
            r = ex_runner.run(script, timeout=int(b.get("timeout") or 150))
            res = xc.interpret(ex, r["stdout"], lmap, ex_runner.SCRIPT_NAME)
            if r["timed_out"]:
                res["ok"] = False
                f = res.get("failed") or {}
                res["failed"] = dict(f, kind="timeout", message="运行超时：检查是否有死循环，或数据/轮数是否过大")
            if b.get("upto") and not r["timed_out"] and b["upto"] in res["passed"]:
                res.update(ok=True, failed=None)           # 只运行到某一步：该步之前全部通过就算通过
            out = "\n".join(l for l in r["stdout"].splitlines() if not l.startswith("@@"))
            res.update(seconds=r["seconds"], output=out[-6000:], upto=b.get("upto"))
            return self._json(res)
        m = re.fullmatch(r"/api/runs/(r[\w-]+)/stop", path)
        if m:
            return self._json({"ok": RM.stop(m.group(1))})
        m = re.fullmatch(r"/api/datasets/(\w+)/download", path)
        if m:
            job = dl_start(m.group(1))
            return self._json({"ok": bool(job)} if job else {"error": "未知数据集"}, 200 if job else 404)
        self._json({"error": "not found"}, 404)


def main():
    global RM
    RM = Runs()
    threading.Thread(target=probe_env, daemon=True).start()
    atexit.register(RM.shutdown)
    signal.signal(signal.SIGTERM, lambda *_: (RM.shutdown(), os._exit(0)))
    handler = functools.partial(Handler, directory=str(SITE))
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    print(f"训练实验室已启动： http://localhost:{PORT}/#/lab   （Ctrl+C 退出；运行记录保存在 {RUNS}）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RM.shutdown()


if __name__ == "__main__":
    main()
