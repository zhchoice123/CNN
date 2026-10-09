#!/usr/bin/env python3
"""练习的自动验收（在每台机器上都要跑一遍）：

    python3 lab/verify_exercises.py            # 全部练习
    python3 lab/verify_exercises.py ex01       # 只验某一篇（前缀匹配）

逐篇检查：
  1. 内容完整：每步有讲解；每个空有 ≥2 个选项、选项不重复、含正确答案、有解析
  2. 答案脚本按评测模式运行，所有步骤检查都通过
  3. 每个 @bad 错误样例都会在它所在的步骤被检出（证明检查真的有效）
  4. 空白的 L4 模板不会被误判为通过
全部通过返回 0，否则返回 1。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exercise_core as xc   # noqa: E402
import ex_runner             # noqa: E402
import platform_utils as pu  # noqa: E402

OK, BAD = "✔", "✘"
SLOW = 30          # 单次评测的时间上限（秒），目标是 20 s 内


def static_checks(ex):
    probs = []
    for s in ex["steps"]:
        if not s["explain"]:
            probs.append(f"步骤 {s['id']} 缺少讲解 @explain")
        for b in s["blanks"]:
            if len(b["options"]) < 2:
                probs.append(f"空 {b['id']} 选项少于 2 个")
            if len(set(b["options"])) != len(b["options"]):
                probs.append(f"空 {b['id']} 有重复选项")
            if not b["why"]:
                probs.append(f"空 {b['id']} 缺少解析 @why")
            if not b["hint"]:
                probs.append(f"空 {b['id']} 缺少提示 @hint")
    for k in ("title", "task", "goal"):
        if not ex["meta"].get(k):
            probs.append(f"缺少 @{k}")
    return probs


def check_run(ex, text, timeout):
    code, lmap = xc.assemble(ex, text)
    r = ex_runner.run(code, timeout=timeout)
    res = xc.interpret(ex, r["stdout"], lmap)
    res["seconds"] = r["seconds"]
    res["timed_out"] = r["timed_out"]
    res["stdout"] = r["stdout"]
    return res


def verify(ex):
    fails = 0
    print(f"\n━━ {ex['id']} · {ex['meta'].get('title', '')}")
    n_blank = sum(len(s["blanks"]) for s in ex["steps"])
    print(f"   {len(ex['steps'])} 步 · {n_blank} 个空 · {sum(len(s['checks']) for s in ex['steps'])} 条检查")
    probs = static_checks(ex)
    if probs:
        fails += len(probs)
        for p in probs:
            print(f"   {BAD} 内容：{p}")
    else:
        print(f"   {OK} 内容完整")

    res = check_run(ex, xc.render(ex, "answer"), timeout=600)
    if res["ok"] and res["seconds"] > SLOW:
        fails += 1
        print(f"   {BAD} 答案脚本通过，但快速评测用了 {res['seconds']} s（上限 {SLOW} s）：@fast 没有生效，或数据/轮数还要再缩小")
    elif res["ok"]:
        print(f"   {OK} 答案脚本全部 {len(res['passed'])} 步通过（{res['seconds']} s）")
    else:
        fails += 1
        f = res["failed"] or {}
        print(f"   {BAD} 答案脚本在步骤 [{f.get('step')}] 失败：{f.get('message')}（{res['seconds']} s）")
        print("     " + "\n     ".join(res["stdout"].strip().splitlines()[-15:]))

    for s in ex["steps"]:
        for b in s["blanks"]:
            for bad in b["bad"]:
                r = check_run(ex, xc.render(ex, "answer", blank_override={b["id"]: bad}), timeout=600)
                got = (r["failed"] or {}).get("step")
                if not r["ok"] and got == s["id"]:
                    print(f"   {OK} 错误样例 {b['id']}「{bad}」在步骤 [{got}] 被检出：{(r['failed'] or {}).get('message', '')[:60]}")
                else:
                    fails += 1
                    print(f"   {BAD} 错误样例 {b['id']}「{bad}」没有在步骤 [{s['id']}] 被检出（结果：{'通过' if r['ok'] else '在 ' + str(got) + ' 失败'}）")

    r = check_run(ex, xc.render(ex, "l4"), timeout=120)
    if r["ok"]:
        fails += 1
        print(f"   {BAD} 空白 L4 模板居然通过了评测，检查太弱")
    else:
        print(f"   {OK} 空白 L4 模板在步骤 [{(r['failed'] or {}).get('step')}] 被判为未完成")
    return fails


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 重定向到文件时也逐行输出，方便看进度
    except AttributeError:
        pass
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if not pu.venv_python().exists():
        print("未找到 lab/.venv，请先运行 python3 lab/setup.py")
        return 1
    import json, os, subprocess
    try:
        out = subprocess.run(pu.python_cmd("-c", "import json,platform_utils;print(json.dumps(platform_utils.hardware_info()))", isolated=False),
                             capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONPATH=str(HERE)))
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:      # noqa: BLE001
        info = pu.hardware_info()
    print(f"机器：{info.get('host')} · {info.get('os')} · {info.get('gpu', '?')}")
    t0 = time.time()
    total = 0
    exs = [e for e in xc.load_all() if e["id"].startswith(only)]
    if not exs:
        print("没有找到练习")
        return 1
    for ex in exs:
        total += verify(ex)
    print(f"\n{'全部通过' if not total else f'共 {total} 个问题'} · {len(exs)} 篇 · {time.time() - t0:.0f} s")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
