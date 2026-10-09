#!/usr/bin/env python3
"""Build the study site data.

1. Load every content/chXX.py (a CHAPTER dict).
2. Validate quizzes, vocab and cards.
3. Run every code/*.py referenced by a chapter and record its real output (cached by source hash).
4. Generate MP3 audio with edge-tts (vocab EN/ZH, vocab example sentences, reading passages).
5. Write assets/js/course-data.js  (window.COURSE = {...}) so index.html works from file://.

Usage:  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 tools/build.py [--no-audio] [--only ch08]
"""
import argparse
import hashlib
import html
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

PY = sys.executable
CACHE = ROOT / "code" / ".outputs.json"


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:60] or hashlib.md5(s.encode()).hexdigest()[:8]


# ---------- light markup -> HTML ----------
def inline(s):
    s = re.sub(r"`([^`]+)`", lambda m: "<code>" + html.escape(m.group(1), quote=False) + "</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def md(text):
    """Blank-line paragraphs, '- ' bullet lists, '1. ' numbered lists, '> ' notes, $$math$$ blocks kept."""
    if not text:
        return ""
    blocks = re.split(r"\n\s*\n", text.strip("\n"))
    out = []
    for b in blocks:
        lines = [l.rstrip() for l in b.strip("\n").split("\n")]
        stripped = [l.strip() for l in lines]
        if all(l.startswith("|") for l in stripped):
            rows = [[c.strip() for c in l.strip("|").split("|")] for l in stripped if not re.fullmatch(r"[|:\-\s]+", l)]
            head, body = rows[0], rows[1:]
            out.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in head) +
                       "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) +
                       "</tbody></table></div>")
        elif all(l.startswith("- ") for l in stripped):
            out.append("<ul>" + "".join(f"<li>{inline(l[2:])}</li>" for l in stripped) + "</ul>")
        elif all(re.match(r"\d+\. ", l) for l in stripped):
            out.append("<ol>" + "".join(f"<li>{inline(re.sub(r'^\d+\. ', '', l))}</li>" for l in stripped) + "</ol>")
        elif stripped[0].startswith("> "):
            out.append('<div class="note">' + inline(" ".join(l[2:] if l.startswith("> ") else l for l in stripped)) + "</div>")
        elif stripped[0].startswith("$$"):
            out.append('<div class="math-block">' + "\n".join(stripped) + "</div>")
        elif stripped[0].startswith("<"):
            out.append("\n".join(lines))
        else:
            out.append("<p>" + inline(" ".join(stripped)) + "</p>")
    return "\n".join(out)


# ---------- load & validate ----------
def load_chapters(only=None):
    chapters = []
    for f in sorted((ROOT / "content").glob("ch*.py")):
        if only and f.stem not in only:
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ch = mod.CHAPTER
        ch["id"] = f.stem
        chapters.append(ch)
    return chapters


def validate(ch):
    errs = []
    cid = ch["id"]
    for key in ("no", "title", "title_en", "group", "summary", "learn", "examples", "quiz", "vocab"):
        if key not in ch:
            errs.append(f"{cid}: missing {key}")
    q = ch.get("quiz", {})
    for i, m in enumerate(q.get("mc", [])):
        if not (0 <= m["answer"] < len(m["options"])):
            errs.append(f"{cid} mc{i}: answer index out of range")
    for i, t in enumerate(q.get("tf", [])):
        if not isinstance(t["answer"], bool):
            errs.append(f"{cid} tf{i}: answer must be bool")
    for i, f in enumerate(q.get("fill", [])):
        if "____" not in f["q"]:
            errs.append(f"{cid} fill{i}: no ____ blank")
        if isinstance(f["answer"], str):
            f["answer"] = [f["answer"]]
    for r in q.get("reading", []):
        for j, m in enumerate(r["questions"]):
            if not (0 <= m["answer"] < len(m["options"])):
                errs.append(f"{cid} reading q{j}: answer out of range")
    seen = set()
    for v in ch.get("vocab", []):
        if len(v) != 5:
            errs.append(f"{cid} vocab {v[0]}: need (en, zh, def_en, def_zh, example)")
        if v[0].lower() in seen:
            errs.append(f"{cid} vocab duplicate {v[0]}")
        seen.add(v[0].lower())
    for s in ch.get("learn", []):
        if s.get("code") and not (ROOT / "code" / s["code"]).exists():
            errs.append(f"{cid}: code file {s['code']} missing")
    for e in ch.get("examples", []):
        if e.get("code") and not (ROOT / "code" / e["code"]).exists():
            errs.append(f"{cid}: code file {e['code']} missing")
    return errs


# ---------- code verification ----------
def run_code(names):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    results, failures = {}, []
    for name in sorted(names):
        src = (ROOT / "code" / name).read_text(encoding="utf-8")
        h = hashlib.sha1(src.encode()).hexdigest()
        if name in cache and cache[name]["hash"] == h and cache[name]["ok"]:
            results[name] = cache[name]
            continue
        t0 = time.time()
        try:
            p = subprocess.run([PY, name], cwd=ROOT / "code", capture_output=True, text=True, timeout=120)
            ok = p.returncode == 0
            out = (p.stdout + ("\n" + p.stderr if p.stderr and not ok else "")).rstrip()
        except subprocess.TimeoutExpired:
            ok, out = False, "TIMEOUT"
        r = {"hash": h, "ok": ok, "output": out, "seconds": round(time.time() - t0, 2), "source": src}
        results[name] = cache[name] = r
        print(f"  ran {name}: {'ok' if ok else 'FAILED'} ({r['seconds']}s)")
        if not ok:
            failures.append((name, out[-800:]))
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return results, failures


# ---------- assemble ----------
def shuffle_options(m, key):
    """Deterministically shuffle choices (seeded by question) so the key is not always in the same slot."""
    idx = list(range(len(m["options"])))
    random.Random(hashlib.md5(key.encode()).hexdigest()).shuffle(idx)
    return [m["options"][i] for i in idx], idx.index(m["answer"])



def build(only=None, audio=True):
    from gen_audio import VOICE_EN, VOICE_ZH, generate
    chapters = load_chapters(only)
    errs = [e for ch in chapters for e in validate(ch)]
    if errs:
        print("VALIDATION ERRORS:\n  " + "\n  ".join(errs))
        sys.exit(1)
    code_names = {s["code"] for ch in chapters for s in ch["learn"] if s.get("code")}
    code_names |= {e["code"] for ch in chapters for e in ch["examples"] if e.get("code")}
    code, failures = run_code(code_names)
    if failures:
        for n, o in failures:
            print(f"CODE FAILED {n}:\n{o}")
        sys.exit(1)

    jobs = []
    out_chapters = []
    for ch in chapters:
        cid = ch["id"]
        vocab = []
        for en, zh, den, dzh, ex in ch["vocab"]:
            s = slug(en)
            base = f"audio/vocab/{cid}/{s}"
            a = {"en": f"{base}_en.mp3", "zh": f"{base}_zh.mp3", "ex": f"{base}_ex.mp3"}
            jobs += [(en, VOICE_EN, ROOT / a["en"], "-5%"), (zh, VOICE_ZH, ROOT / a["zh"], "+0%"),
                     (ex, VOICE_EN, ROOT / a["ex"], "-5%")]
            vocab.append({"en": en, "zh": zh, "defEn": den, "defZh": dzh, "example": ex, "audio": a, "id": s})
        reading = []
        for i, r in enumerate(ch["quiz"].get("reading", [])):
            a = f"audio/reading/{cid}_{i + 1}.mp3"
            plain = re.sub(r"<[^>]+>|\*\*|`", "", r["passage"])
            jobs.append((plain, VOICE_EN, ROOT / a, "-5%"))
            reading.append({**r, "passage": md(r["passage"]), "zh": md(r.get("zh", "")), "audio": a})
        learn = []
        for i, s in enumerate(ch["learn"]):
            item = {"id": f"{cid}-s{i + 1}", "title": s["title"], "titleEn": s.get("title_en", ""),
                    "body": md(s["body"]), "points": [inline(p) for p in s.get("points", [])],
                    "src": s.get("src", ""), "codeNote": inline(s.get("code_note", ""))}
            if s.get("code"):
                c = code[s["code"]]
                item["code"] = {"name": s["code"], "source": c["source"], "output": c["output"]}
            learn.append(item)
        examples = []
        for i, e in enumerate(ch["examples"]):
            item = {"id": f"{cid}-e{i + 1}", "title": e["title"], "problem": md(e["problem"]),
                    "steps": [md(x) for x in e["steps"]], "answer": md(e.get("answer", ""))}
            if e.get("code"):
                c = code[e["code"]]
                item["code"] = {"name": e["code"], "source": c["source"], "output": c["output"]}
            examples.append(item)
        q = ch["quiz"]
        quiz = {
            "mc": [{"q": inline(m["q"]), "options": [inline(o) for o in opts], "answer": ans,
                    "explain": inline(m.get("explain", ""))}
                   for m in q.get("mc", []) for opts, ans in [shuffle_options(m, cid + m["q"])]],
            "tf": [{"q": inline(t["q"]), "answer": t["answer"], "explain": inline(t.get("explain", ""))}
                   for t in q.get("tf", [])],
            "fill": [{"q": inline(f["q"]), "answer": f["answer"], "explain": inline(f.get("explain", ""))}
                     for f in q.get("fill", [])],
            "reading": [{"title": r["title"], "passage": r["passage"], "zh": r["zh"], "audio": r["audio"],
                         "questions": [{"q": inline(m["q"]), "options": [inline(o) for o in opts],
                                        "answer": ans, "explain": inline(m.get("explain", ""))}
                                       for m in r["questions"] for opts, ans in [shuffle_options(m, cid + m["q"])]]}
                        for r in reading],
        }
        cards = [{"front": inline(f), "back": inline(b)} for f, b in ch.get("cards", [])]
        out_chapters.append({"id": cid, "no": ch["no"], "title": ch["title"], "titleEn": ch["title_en"],
                             "group": ch["group"], "summary": inline(ch["summary"]), "sources": ch.get("sources", []),
                             "learn": learn, "examples": examples, "quiz": quiz, "vocab": vocab, "cards": cards})

    if audio:
        print(f"  audio jobs: {len(jobs)} ({sum(not j[2].exists() for j in jobs)} missing)")
        generate(jobs)
    for ch in out_chapters:  # drop audio paths that failed to generate
        for v in ch["vocab"]:
            v["audio"] = {k: p for k, p in v["audio"].items() if (ROOT / p).exists()}
        for r in ch["quiz"]["reading"]:
            if not (ROOT / r["audio"]).exists():
                r["audio"] = ""

    if only:  # merge into existing data
        existing = {}
        dp = ROOT / "assets" / "js" / "course-data.js"
        if dp.exists():
            txt = dp.read_text(encoding="utf-8")
            data = json.loads(txt[txt.index("{"):txt.rstrip().rstrip(";").rindex("}") + 1])
            existing = {c["id"]: c for c in data["chapters"]}
        for c in out_chapters:
            existing[c["id"]] = c
        out_chapters = list(existing.values())
    out_chapters.sort(key=lambda c: c["no"])
    data = {"title": "深度学习与计算机视觉 · 双语学习站", "builtAt": time.strftime("%Y-%m-%d %H:%M"),
            "chapters": out_chapters}
    (ROOT / "assets" / "js" / "course-data.js").write_text(
        "window.COURSE = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    n = lambda k: sum(len(c["quiz"][k]) for c in out_chapters)
    print(f"built {len(out_chapters)} chapters | learn {sum(len(c['learn']) for c in out_chapters)} | "
          f"examples {sum(len(c['examples']) for c in out_chapters)} | mc {n('mc')} tf {n('tf')} fill {n('fill')} "
          f"reading {n('reading')} | vocab {sum(len(c['vocab']) for c in out_chapters)} | code {len(code)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    build(a.only, not a.no_audio)
