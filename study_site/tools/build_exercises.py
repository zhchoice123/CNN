#!/usr/bin/env python3
"""把 study_site/exercises/ex*.py（带标记的答案脚本）构建成前端数据 assets/js/exercises-data.js。

    python3 tools/build_exercises.py

只做解析与格式校验（很快，不需要 torch）。答案是否真的能跑、检查是否有效，用
    python3 ../lab/verify_exercises.py
在每台机器上验收。
"""
import json
import sys
import time
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SITE.parent / "lab"))
import exercise_core as xc   # noqa: E402

OUT = SITE / "assets" / "js" / "exercises-data.js"


def main():
    items = []
    for ex in xc.load_all():
        j = xc.to_json(ex)
        items.append(j)
        nb = sum(len(s["blanks"]) for s in j["steps"])
        print(f"  {ex['id']:24s} {len(j['steps'])} 步 · {nb} 个空 · {j['meta'].get('title', '')}")
    OUT.write_text("window.EXERCISES=" + json.dumps(dict(built=time.strftime("%Y-%m-%d %H:%M"), items=items), ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"练习：{len(items)} 篇 → {OUT.relative_to(SITE)}（{OUT.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    try:
        main()
    except xc.ExerciseError as e:
        print("✘ 练习格式错误：", e)
        sys.exit(1)
