#!/usr/bin/env python3
"""Generate MP3 pronunciations with edge-tts (same approach as ~/Documents/EdgeTtsProject/edge_tts_demo.py).

Used by build.py; can also be run alone:  python3 tools/gen_audio.py "hello world" out.mp3 --voice en-US-AriaNeural
"""
import argparse
import asyncio
from pathlib import Path

import edge_tts

VOICE_EN = "en-US-AriaNeural"
VOICE_ZH = "zh-CN-XiaoxiaoNeural"


async def _one(text, voice, out: Path, rate, sem, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".part")
                await edge_tts.Communicate(text=text, voice=voice, rate=rate).save(str(tmp))
                if tmp.stat().st_size < 500:
                    raise RuntimeError("audio too small")
                tmp.rename(out)
                return True
            except Exception as e:  # network hiccups: retry
                if attempt == retries - 1:
                    print(f"  ! failed {out.name}: {e}")
                    return False
                await asyncio.sleep(1.5 * (attempt + 1))


def generate(jobs, concurrency=6):
    """jobs: list of (text, voice, Path, rate). Existing files are skipped."""
    todo = [j for j in jobs if not j[2].exists()]
    if not todo:
        return 0

    async def run():
        sem = asyncio.Semaphore(concurrency)
        res = await asyncio.gather(*[_one(t, v, p, r, sem) for t, v, p, r in todo])
        return sum(bool(x) for x in res)

    done = asyncio.run(run())
    print(f"  audio: generated {done}/{len(todo)} new files")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("output", type=Path)
    ap.add_argument("--voice", default=VOICE_EN)
    ap.add_argument("--rate", default="+0%")
    a = ap.parse_args()
    generate([(a.text, a.voice, a.output, a.rate)])
