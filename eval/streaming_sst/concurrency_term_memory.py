#!/usr/bin/env python3
"""Concurrency sweep: N simultaneous streaming sessions sharing the open memory.

Tests the framework's coalescing micro-batch scheduler + vLLM continuous batching
under load: N clients stream the same audio at once, against one shared term
memory + retriever. Reports per-session chunk latency and aggregate throughput,
to show that adding concurrent sessions does not blow up per-chunk latency.

    python eval/streaming_sst/concurrency_term_memory.py --base-url http://aries:8011 \
        --preset open_wiki_academic --levels 1,8,16,32 --max-segs 4
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import statistics
import time
import urllib.parse
import urllib.request
import wave
from typing import Any, Dict, List

import numpy as np
import websockets


def read_wav(p: str) -> np.ndarray:
    with wave.open(p) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def init(base_url: str, lang: str, preset: str) -> str:
    q = urllib.parse.urlencode({"agent_type": "RASST", "language_pair": lang, "glossary_preset": preset})
    with urllib.request.urlopen(urllib.request.Request(f"{base_url}/init?{q}", data=b""), timeout=30) as r:
        return json.load(r)["session_id"]


def delete(base_url: str, sid: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{base_url}/delete_session?{urllib.parse.urlencode({'session_id': sid})}", data=b""), timeout=15)
    except Exception:  # noqa: BLE001
        pass


async def one_session(base_url: str, lang: str, preset: str, pcm: np.ndarray, chunk: int) -> Dict[str, Any]:
    sid = await asyncio.to_thread(init, base_url, lang, preset)
    retr: List[float] = []
    gen: List[float] = []
    n = 0
    try:
        async with websockets.connect(f"{base_url.replace('http','ws')}/wss/{sid}?event_format=json", max_size=None) as ws:
            await ws.recv()

            async def feed() -> None:
                for i in range(0, len(pcm), chunk):
                    await ws.send(pcm[i:i + chunk].tobytes())
                    await asyncio.sleep(0.45)
                await ws.send("EOF")

            ft = asyncio.create_task(feed())
            idle = 0
            while idle < 1:
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=60.0)
                except asyncio.TimeoutError:
                    idle += 1
                    continue
                o = json.loads(m)
                if o.get("type") == "partial":
                    n += 1
                    meta = o.get("meta") or {}
                    if meta.get("retrieve_s") is not None:
                        retr.append(float(meta["retrieve_s"]))
                    if meta.get("elapsed_s") is not None:
                        gen.append(float(meta["elapsed_s"]))
            await ft
    finally:
        await asyncio.to_thread(delete, base_url, sid)
    return {"partials": n, "retr": retr, "gen": gen}


async def run_level(base_url: str, lang: str, preset: str, pcm: np.ndarray, chunk: int, n: int) -> Dict[str, Any]:
    t0 = time.time()
    results = await asyncio.gather(*[one_session(base_url, lang, preset, pcm, chunk) for _ in range(n)])
    wall = time.time() - t0
    retr = [x * 1000 for r in results for x in r["retr"][1:]]  # drop each session's cold chunk
    gen = [x * 1000 for r in results for x in r["gen"]]
    total = sum(r["partials"] for r in results)

    def pct(xs: List[float], p: float):
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 1)

    return {
        "N": n,
        "total_partials": total,
        "wall_s": round(wall, 1),
        "throughput_seg_s": round(total / wall, 2) if wall else 0.0,
        "retrieve_p50_ms": pct(retr, 50),
        "retrieve_p95_ms": pct(retr, 95),
        "gen_p50_ms": pct(gen, 50),
        "gen_p95_ms": pct(gen, 95),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://aries:8011")
    ap.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg")
    ap.add_argument("--preset", default="open_wiki_academic")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--max-segs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    fs = sorted(glob.glob(f"{args.seg_dir}/*.wav"))[: args.max_segs]
    pcm = np.concatenate([read_wav(f) for f in fs]).astype(np.float32)
    print(f"[conc] preset={args.preset} {len(fs)} segs {len(pcm)/16000:.1f}s; levels={args.levels}", flush=True)

    rows = []
    for n in [int(x) for x in args.levels.split(",") if x.strip()]:
        try:
            row = asyncio.run(run_level(args.base_url, args.language_pair, args.preset, pcm, args.chunk, n))
        except Exception as exc:  # noqa: BLE001
            row = {"N": n, "error": str(exc)[:200]}
        rows.append(row)
        print("[conc]", json.dumps(row), flush=True)

    cols = ["N", "total_partials", "wall_s", "throughput_seg_s", "retrieve_p50_ms", "retrieve_p95_ms", "gen_p50_ms", "gen_p95_ms"]
    print("\n=== concurrency sweep (" + args.preset + ") ===", flush=True)
    print(" | ".join(c.ljust(14) for c in cols), flush=True)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(14) for c in cols), flush=True)
    if args.out_json:
        json.dump(rows, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
