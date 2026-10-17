"""分步代码练习的核心：解析带标记的答案脚本，生成四级练习，并把读者代码组装成可评测的脚本。

纯标准库，构建器（study_site/tools/build_exercises.py）和服务器（lab/server.py）共用。

答案脚本 = 一份能直接运行的完整训练代码 + 注释标记（格式说明见 study_site/exercises/README.md）：

    # @step data 加载数据                ← 新步骤（必须顶格）：id + 标题
    # @explain … # @end                   ← 讲解（Markdown 轻量语法）
    # @todo: 用 DataLoader 包装训练集       ← 显示给读者的提示注释
    # @blank loader mask=shuffle=True      ← 挖空开始；mask 是 L1/L2 要填的片段
    # @hint: … / @option: … / @option*: … / @accept: … / @why: … / @bad: …
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    # @endblank
    # @check: assert len(train_loader) > 0, "提示信息"   ← 评测时在本步末尾执行
    # @fast: EPOCHS = 1                   ← 仅评测时执行，用来缩小数据/轮数

四级难度：
  L1 选择：mask 位置给出选项        L2 填空：手动输入 mask
  L3 写整块：挖空的整行代码由读者写   L4 默写：只剩步骤标题和提示，整份脚本自己写
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX_DIR = ROOT / "study_site" / "exercises"

RX_MARK = re.compile(r"^(\s*)# @(\w+\*?)(?::\s?(.*)|\s+(.*)|\s*)$")
RX_HEADER = re.compile(r"^# =+ 步骤 (\d+)[：:]\s*(.*?)\s*\[(\w+)\] =+\s*$")


class ExerciseError(ValueError):
    pass


def header_line(n, title, sid):
    return f"# ===================== 步骤 {n}：{title} [{sid}] ====================="


def _mark(line):
    m = RX_MARK.match(line)
    if not m:
        return None
    indent, key = m.group(1), m.group(2)
    val = m.group(3) if m.group(3) is not None else (m.group(4) or "")
    return indent, key, val.rstrip()


# ───────────────────────── 解析 ─────────────────────────
def parse(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    meta = {}
    m = re.match(r'\s*"""(.*?)"""\s*\n', text, re.S)
    body = text
    if m:
        for ln in m.group(1).splitlines():
            mm = re.match(r"\s*@(\w+):\s*(.*)$", ln)
            if mm:
                meta[mm.group(1)] = mm.group(2).strip()
        body = text[m.end():]
    ex = dict(id=path.stem, file=path.name, meta=meta, steps=[])
    step = None
    blank = None
    explain = None
    ids = set()
    for no, raw in enumerate(body.splitlines(), 1):
        line = raw.rstrip("\n")
        mk = _mark(line)
        if explain is not None:
            if mk and mk[1] == "end":
                step["explain"] = "\n".join(explain).strip()
                explain = None
            else:
                explain.append(re.sub(r"^\s*# ?", "", line))
            continue
        if mk:
            indent, key, val = mk
            if key == "step":
                if blank:
                    raise ExerciseError(f"{path.name}:{no} 上一个 @blank 没有 @endblank")
                if indent:
                    raise ExerciseError(f"{path.name}:{no} @step 必须顶格")
                parts = val.split(None, 1)
                if len(parts) < 2:
                    raise ExerciseError(f"{path.name}:{no} @step 需要 id 和标题")
                step = dict(id=parts[0], title=parts[1].strip(), explain="", items=[], checks=[], fasts=[], blanks=[])
                if step["id"] in ids:
                    raise ExerciseError(f"{path.name}:{no} 重复的步骤 id {step['id']}")
                ids.add(step["id"])
                ex["steps"].append(step)
                continue
            if step is None:
                raise ExerciseError(f"{path.name}:{no} 第一个 @step 之前不能有标记")
            if key == "explain":
                explain = []
            elif key == "todo":
                step["items"].append(("todo", indent, val))
            elif key == "check":
                step["checks"].append(val)
            elif key == "fast":
                step["fasts"].append(val)
                step["fast_at"] = (len(step["items"]), no)
            elif key == "blank":
                if blank:
                    raise ExerciseError(f"{path.name}:{no} @blank 不能嵌套")
                parts = val.split(None, 1)
                bid = parts[0]
                if bid in ids:
                    raise ExerciseError(f"{path.name}:{no} 重复的 id {bid}")
                ids.add(bid)
                mask = None
                if len(parts) > 1 and parts[1].startswith("mask="):
                    mask = parts[1][5:]
                blank = dict(id=bid, mask=mask, indent=indent, hint="", options=[], correct=None, accept=[], why="", bad=[], code=[], step=step["id"])
            elif blank is not None and key in ("hint", "why"):
                blank[key] = (blank[key] + " " + val).strip()
            elif blank is not None and key in ("option", "option*"):
                blank["options"].append(val)
                if key == "option*":
                    blank["correct"] = val
            elif blank is not None and key == "accept":
                blank["accept"].append(val)
            elif blank is not None and key == "bad":
                blank["bad"].append(val)
            elif key == "endblank":
                if not blank:
                    raise ExerciseError(f"{path.name}:{no} 多余的 @endblank")
                _finish_blank(path, no, blank)
                step["blanks"].append(blank)
                step["items"].append(("blank", blank["indent"], blank))
                blank = None
            else:
                raise ExerciseError(f"{path.name}:{no} 未知标记 @{key}")
            continue
        if blank is not None:
            blank["code"].append(line)
        elif step is not None:
            step["items"].append(("code", "", line))
        elif line.strip():
            raise ExerciseError(f"{path.name}:{no} 代码必须写在某个 @step 之后")
    if blank:
        raise ExerciseError(f"{path.name}: 文件结束时 @blank 未关闭")
    if explain is not None:
        raise ExerciseError(f"{path.name}: @explain 没有 @end")
    if not ex["steps"]:
        raise ExerciseError(f"{path.name}: 没有任何 @step")
    for s in ex["steps"]:     # 去掉每步末尾的空行
        while s["items"] and s["items"][-1][0] == "code" and not s["items"][-1][2].strip():
            s["items"].pop()
        if "fast_at" in s:     # @fast 在评测时注入到“步骤末尾”，所以它后面不能再有代码，否则缩小数据/轮数来不及生效
            idx, no = s["fast_at"]
            later = [it for it in s["items"][idx:] if it[0] != "todo" and (it[0] == "blank" or (it[2].strip() and not it[2].strip().startswith("#")))]
            if later:
                raise ExerciseError(f"{path.name}:{no} 步骤 [{s['id']}] 的 @fast 后面还有代码；请把后面的代码拆到下一个步骤（@fast 只在步骤末尾生效）")
    return ex


def _finish_blank(path, no, b):
    code = "\n".join(b["code"])
    if not code.strip():
        raise ExerciseError(f"{path.name}:{no} 挖空 {b['id']} 没有代码")
    if not b["mask"]:
        raise ExerciseError(f"{path.name}:{no} 挖空 {b['id']} 需要 mask=…")
    if b["mask"] not in code:
        raise ExerciseError(f"{path.name}:{no} 挖空 {b['id']} 的 mask「{b['mask']}」不在代码里")
    if b["correct"] is None:
        b["correct"] = b["mask"]
        b["options"].insert(0, b["mask"])
    if b["correct"] != b["mask"]:
        raise ExerciseError(f"{path.name}:{no} 挖空 {b['id']} 的 @option* 必须等于 mask")


# ───────────────────────── 生成各种视图 ─────────────────────────
def _todo_line(indent, text):
    return f"{indent}# {text}"


def step_code(step, mode, blank_override=None):
    """mode: answer（完整答案）/ view（L1/L2：mask 替换为占位符 ⟦id⟧）/ l3（挖空行换成 TODO）/ l4（只留提示）。"""
    out = []
    last_code = ""            # 上一行真正的代码（用来判断挖空是不是代码块的唯一内容）
    for kind, indent, val in step["items"]:
        if kind == "code":
            if mode != "l4":
                out.append(val)
            if val.strip() and not val.strip().startswith("#"):
                last_code = val
        elif kind == "todo":
            if mode == "l4":
                out.append(f"# TODO: {val}")                # L4 没有类/循环骨架，提示一律顶格
            else:
                out.append(f"{indent}# TODO: {val}" if mode == "l3" else _todo_line(indent, val))
        else:
            b = val
            if blank_override and b["id"] in blank_override:
                out.extend(_indent_block(blank_override[b["id"]], b["indent"]))
            elif mode in ("answer",):
                out.extend(b["code"])
            elif mode == "view":
                out.extend(ln.replace(b["mask"], f"⟦{b['id']}⟧") for ln in b["code"])
            elif mode == "l3":
                tip = b["hint"] or "补全这里的代码"
                out.append(f"{b['indent']}# ✏️ 在这里写代码（{b['id']}）：{tip}")
                out.append(f"{b['indent']}pass" if last_code.rstrip().endswith(":") else "")   # 冒号后面必须有语句，才补 pass
            elif mode == "l4" and b["hint"]:
                out.append(f"#   提示：{b['hint']}")
    return out


def _indent_block(code, indent):
    lines = code.splitlines() or [""]
    base = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return [indent + l[base:] if l.strip() else "" for l in lines]


def doc_header(ex, level):
    m = ex["meta"]
    lines = ['"""', f"{m.get('title', ex['id'])}", ""]
    if m.get("task"):
        lines.append(f"任务：{m['task']}")
    if m.get("names"):
        lines.append(f"请使用这些变量名（评测会检查）：{m['names']}")
    if level in ("l3", "l4"):
        lines.append("每个“步骤”标题行不要删除：评测按步骤运行检查，并告诉你哪一步出了问题。")
    lines.append('"""')
    return lines


def render(ex, mode, blank_override=None, with_doc=True):
    out = doc_header(ex, mode) if with_doc else []
    for n, s in enumerate(ex["steps"], 1):
        out.append("")
        out.append(header_line(n, s["title"], s["id"]))
        out.extend(step_code(s, mode, blank_override))
    return "\n".join(out).strip("\n") + "\n"


def answer_script(ex, with_headers=True):
    """干净的完整答案（可直接运行，可放进实验室）。"""
    if with_headers:
        return render(ex, "answer")
    return "\n".join(l for s in ex["steps"] for l in step_code(s, "answer")) + "\n"


def to_json(ex):
    """给前端用的数据（不含检查代码以外的内部细节；检查代码也给出，便于读者理解评测在查什么）。"""
    steps = []
    for n, s in enumerate(ex["steps"], 1):
        steps.append(dict(id=s["id"], n=n, title=s["title"], explain=s["explain"],
                          view="\n".join(step_code(s, "view")), answer="\n".join(step_code(s, "answer")),
                          checks=s["checks"],
                          blanks=[dict(id=b["id"], mask=b["mask"], hint=b["hint"], options=b["options"], accept=b["accept"], why=b["why"],
                                       lines=len(b["code"])) for b in s["blanks"]]))
    return dict(id=ex["id"], file=ex["file"], meta=ex["meta"], steps=steps,
                l3=render(ex, "l3"), l4=render(ex, "l4"), answer=answer_script(ex))


# ───────────────────────── 组装评测脚本 ─────────────────────────
PRELUDE = [
    "import os as _os, sys as _sys",
    "_os.environ['EX_CHECK'] = '1'",
    "EX_CHECK = True",
]


def split_user(ex, text):
    """按步骤标题行切分读者代码。返回 (前言行, {step_id: [(行号, 文本)]})。"""
    want = [s["id"] for s in ex["steps"]]
    segs, cur, pre = {}, None, []
    seen = []
    for i, line in enumerate(text.splitlines(), 1):
        m = RX_HEADER.match(line.strip())
        if m:
            sid = m.group(3)
            if sid not in want:
                raise ExerciseError(f"第 {i} 行：未知的步骤标记 [{sid}]，请不要修改步骤标题行")
            if sid in segs:
                raise ExerciseError(f"第 {i} 行：步骤 [{sid}] 出现了两次")
            segs[sid] = []
            seen.append(sid)
            cur = sid
            continue
        (segs[cur] if cur else pre).append((i, line))
    missing = [s for s in want if s not in segs]
    if missing:
        raise ExerciseError("缺少步骤标题行：" + "、".join(f"[{m}]" for m in missing) + "。请不要删除或修改“# ===== 步骤 … =====”这些行")
    if seen != want:
        raise ExerciseError("步骤标题的顺序被改动了，请保持原来的顺序")
    return pre, segs


def assemble(ex, text, upto=None):
    """把读者代码组装成评测脚本。upto=步骤 id 时只运行到该步（含）。
    返回 (脚本文本, 行号映射)；行号映射[i] = (step_id, 读者行号或 None)。"""
    pre, segs = split_user(ex, text)
    out, lmap = [], []

    def emit(line, sid=None, uline=None):
        out.append(line)
        lmap.append((sid, uline))
    for l in PRELUDE:
        emit(l)
    for i, l in pre:
        emit(l, None, i)
    for s in ex["steps"]:
        sid = s["id"]
        emit(f"print('@@step_begin {sid}', flush=True)", sid)
        for i, l in segs[sid]:
            emit(l, sid, i)
        emit("", sid)
        for c in s["checks"]:
            emit(c, sid)
        for f in s["fasts"]:
            emit(f, sid)
        emit(f"print('@@step_ok {sid}', flush=True)", sid)
        if upto and sid == upto:
            break
    return "\n".join(out) + "\n", lmap


def interpret(ex, stdout, lmap, script_name="exercise_run.py"):
    """解析评测输出：哪些步骤通过、第一个失败的步骤与原因、对应读者第几行。"""
    passed = re.findall(r"^@@step_ok (\w+)", stdout, re.M)
    begun = re.findall(r"^@@step_begin (\w+)", stdout, re.M)
    titles = {s["id"]: s["title"] for s in ex["steps"]}
    result = dict(passed=passed, ok=len(passed) == len(ex["steps"]), failed=None)
    if result["ok"]:
        return result
    fail_step = begun[-1] if begun else (ex["steps"][0]["id"])
    # 回溯里最后一个落在评测脚本里的行
    lines = [int(n) for n in re.findall(rf'File ".*?{re.escape(script_name)}", line (\d+)', stdout)]
    uline, kind = None, "error"
    if lines:
        idx = lines[-1] - 1
        if 0 <= idx < len(lmap):
            sid, ul = lmap[idx]
            fail_step = sid or fail_step
            uline = ul
            if ul is None:
                kind = "check"     # 出错位置是注入的检查代码 → 检查没通过
    err = ""
    tb = stdout.rfind("Traceback (most recent call last)")
    if tb >= 0:
        tail = [l for l in stdout[tb:].splitlines() if l.strip()]
        err = tail[-1] if tail else ""
    else:   # 编译期的语法/缩进错误没有 Traceback 开头，只有 File…/代码/^/错误名 几行
        errs = re.findall(r"^(\w*(?:Error|Exception)\b.*)$", stdout, re.M)
        err = errs[-1] if errs else ""
    if err.startswith("AssertionError"):
        kind = "check"
        err = err[len("AssertionError"):].lstrip(": ") or "检查没有通过"
    result["failed"] = dict(step=fail_step, title=titles.get(fail_step, fail_step), kind=kind, message=err, line=uline)
    return result


def load_all():
    out = []
    for p in sorted(EX_DIR.glob("ex*.py")):
        out.append(parse(p))
    return out


def find(ex_id):
    p = EX_DIR / f"{ex_id}.py"
    if not re.fullmatch(r"ex\w+", ex_id or "") or not p.exists():
        return None
    return parse(p)


if __name__ == "__main__":
    for e in load_all():
        print(e["id"], len(e["steps"]), "步", sum(len(s["blanks"]) for s in e["steps"]), "空")
        print(json.dumps(to_json(e), ensure_ascii=False)[:200])
