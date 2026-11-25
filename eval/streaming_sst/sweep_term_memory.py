#!/usr/bin/env python3
"""Term-memory scale sweep over the framework's JSON WebSocket protocol.

For each glossary/term-memory preset, opens a session, streams a fixed set of
16 kHz mono segment wavs, and records per-chunk retrieval/generation latency and
retrieved-term counts from the structured `meta`. Reports cold (first chunk,
includes index warm-load) separately from warm steady-state percentiles, so the
scale curve reflects realistic streaming latency rather than one-off index load.

This does NOT need SimulEval; it measures the terminology+latency axis
(retrieve p50/p95, refs/chunk, active_terms). BLEU/StreamLAAL stay with the
existing SimulEval harness.

    python eval/streaming_sst/sweep_term_memory.py \
        --base-url http://aries:8011 \
        --seg-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg \
        --presets none,acl_tagged_raw,open_wiki_academic,open_wiki_100k,open_wiki_1m \
        --language-pair "English -> Chinese" --max-segs 6
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import statistics
import urllib.parse
import urllib.request
import wave
from typing import Any, Dict, List, Optional

import numpy as np
import websockets


def _http_base(base_url: str) -> str:
    return base_url.rstrip("/")


def _ws_base(base_url: str) -> str:
    return _http_base(base_url).replace("http://", "ws://").replace("https://", "wss://")


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def http_get(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def init_session(base_url: str, lang: str, preset: str) -> Dict[str, Any]:
    q = urllib.parse.urlencode({"agent_type": "RASST", "language_pair": lang, "glossary_preset": preset})
    req = urllib.request.Request(f"{_http_base(base_url)}/init?{q}", data=b"")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def delete_session(base_url: str, sid: str) -> None:
    q = urllib.parse.urlencode({"session_id": sid})
    try:
        urllib.request.urlopen(urllib.request.Request(f"{_http_base(base_url)}/delete_session?{q}", data=b""), timeout=15)
    except Exception:  # noqa: BLE001
        pass


def _term_memory(base_url: str) -> Dict[str, Any]:
    try:
        return http_get(f"{_http_base(base_url)}/health").get("term_memory", {})
    except Exception:  # noqa: BLE001
        return {}


async def run_preset(base_url: str, lang: str, preset: str, pcm: np.ndarray, chunk: int) -> Dict[str, Any]:
    info = init_session(base_url, lang, preset)
    sid = info["session_id"]
    retr: List[float] = []
    gen: List[float] = []
    refs: List[int] = []
    used = 0
    out_text: List[str] = []
    try:
        async with websockets.connect(f"{_ws_base(base_url)}/wss/{sid}?event_format=json", max_size=None) as ws:
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
                    m = await asyncio.wait_for(ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    idle += 1
                    continue
                o = json.loads(m)
                if o.get("type") != "partial":
                    continue
                meta = o.get("meta") or {}
                if meta.get("retrieve_s") is not None:
                    retr.append(float(meta["retrieve_s"]))
                if meta.get("elapsed_s") is not None:
                    gen.append(float(meta["elapsed_s"]))
                rs = meta.get("references") or []
                refs.append(len(rs))
                txt = o.get("text") or ""
                out_text.append(txt)
                if "<t>" in txt:
                    used += txt.count("<t>")
            await ft
    finally:
        delete_session(base_url, sid)

    def pct(xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        idx = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
        return round(s[idx] * 1000.0, 1)

    warm = retr[1:] if len(retr) > 1 else retr  # drop cold first chunk (index warm-load)
    tm = _term_memory(base_url)
    return {
        "preset": preset,
        "preset_terms": info.get("preset_terms"),
        "active_terms": tm.get("active_terms"),
        "chunks": len(refs),
        "cold_retrieve_ms": round(retr[0] * 1000.0, 1) if retr else None,
        "warm_retrieve_p50_ms": pct(warm, 50),
        "warm_retrieve_p95_ms": pct(warm, 95),
        "gen_p50_ms": pct(gen, 50),
        "refs_per_chunk": round(statistics.mean(refs), 2) if refs else 0.0,
        "used_tag_count": used,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://aries:8011")
    ap.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg")
    ap.add_argument("--presets", default="none,acl_tagged_raw,open_wiki_academic")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--max-segs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    fs = sorted(glob.glob(f"{args.seg_dir}/*.wav"))[: args.max_segs]
    if not fs:
        raise SystemExit(f"no wavs in {args.seg_dir}")
    pcm = np.concatenate([read_wav(f) for f in fs]).astype(np.float32)
    print(f"[sweep] {len(fs)} segs, {len(pcm)/16000:.1f}s audio; base={args.base_url}")

    rows: List[Dict[str, Any]] = []
    for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
        try:
            row = asyncio.run(run_preset(args.base_url, args.language_pair, preset, pcm, args.chunk))
        except Exception as exc:  # noqa: BLE001
            row = {"preset": preset, "error": str(exc)[:200]}
        rows.append(row)
        print("[sweep]", json.dumps(row, ensure_ascii=False))

    cols = ["preset", "active_terms", "chunks", "cold_retrieve_ms", "warm_retrieve_p50_ms",
            "warm_retrieve_p95_ms", "gen_p50_ms", "refs_per_chunk", "used_tag_count"]
    print("\n=== term-memory scale sweep ===")
    print(" | ".join(c.ljust(12) for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(12) for c in cols))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as h:
            json.dump(rows, h, ensure_ascii=False, indent=2)
        print(f"\n[sweep] wrote {args.out_json}")


if __name__ == "__main__":
    main()
