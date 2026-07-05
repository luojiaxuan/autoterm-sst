#!/usr/bin/env python3
"""Evaluate adaptive working-glossary behavior over the JSON WS protocol.

This measures the terminology-routing side of the demo: retrieval latency,
reference volume, prompt-reference budget, switch count, and time to first
switch. Use ``score_auto_glossary.py`` to turn the JSON output into a compact
comparison table, optionally joined with term-recall rows from ``score_terms.py``.
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import websockets


def _http_base(base_url: str) -> str:
    return base_url.rstrip("/")


def _ws_base(base_url: str) -> str:
    return _http_base(base_url).replace("http://", "ws://").replace("https://", "wss://")


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def init_session(base_url: str, lang: str, preset: str) -> Dict[str, Any]:
    q = urllib.parse.urlencode({"agent_type": "RASST", "language_pair": lang, "glossary_preset": preset})
    req = urllib.request.Request(f"{_http_base(base_url)}/init?{q}", data=b"")
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def delete_session(base_url: str, session_id: str) -> None:
    q = urllib.parse.urlencode({"session_id": session_id})
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{_http_base(base_url)}/delete_session?{q}", data=b""),
            timeout=15,
        )
    except Exception:  # noqa: BLE001
        pass


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


async def run_preset(
    *,
    base_url: str,
    language_pair: str,
    preset: str,
    pcm: np.ndarray,
    chunk: int,
    feed_sleep: float,
) -> Dict[str, Any]:
    info = init_session(base_url, language_pair, preset)
    session_id = info["session_id"]
    retrieve_s: List[float] = []
    refs_per_chunk: List[int] = []
    prompt_refs_per_chunk: List[int] = []
    candidate_pool_counts: List[int] = []
    prompt_shortfalls: List[int] = []
    fixed_prompt_ks: List[int] = []
    rescue_chunks = 0
    active_slices_by_chunk: List[List[str]] = []
    domains: List[str] = []
    active_presets: List[str] = []
    router_actions: List[str] = []
    router_reasons: List[str] = []
    router_confidences: List[float] = []
    first_switch_chunk: Optional[int] = None
    first_switch_s: Optional[float] = None
    switch_count = 0
    chunks = 0
    try:
        async with websockets.connect(f"{_ws_base(base_url)}/wss/{session_id}?event_format=json", max_size=None) as ws:
            await ws.recv()

            async def feed() -> None:
                for start in range(0, len(pcm), chunk):
                    await ws.send(pcm[start:start + chunk].tobytes())
                    await asyncio.sleep(feed_sleep)
                await ws.send("EOF")

            feed_task = asyncio.create_task(feed())
            idle = 0
            while idle < 1:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=45.0)
                except asyncio.TimeoutError:
                    idle += 1
                    continue
                event = json.loads(msg)
                if event.get("type") != "partial":
                    continue
                chunks += 1
                meta = event.get("meta") or {}
                if meta.get("retrieve_s") is not None:
                    retrieve_s.append(float(meta["retrieve_s"]))
                refs = meta.get("references") or []
                refs_per_chunk.append(len(refs))
                prompt_refs_per_chunk.append(int(meta.get("prompt_reference_count") or 0))
                candidate_pool_counts.append(int(meta.get("candidate_pool_count") or len(refs)))
                prompt_shortfalls.append(int(meta.get("prompt_candidate_shortfall") or 0))
                fixed_prompt_ks.append(int(meta.get("fixed_prompt_k") or 0))
                if meta.get("open_wiki_rescue_triggered"):
                    rescue_chunks += 1
                active_slices_by_chunk.append([str(item) for item in (meta.get("active_slices") or [])])
                topic = meta.get("topic") or {}
                domains.append(str(topic.get("active_domain") or ""))
                active_presets.append(str(topic.get("active_glossary_preset") or ""))
                topic_router = meta.get("topic_router") or {}
                if topic_router:
                    router_actions.append(str(topic_router.get("action") or ""))
                    router_reasons.append(str(topic_router.get("reason") or ""))
                    if topic_router.get("confidence") is not None:
                        router_confidences.append(float(topic_router["confidence"]))
                current_switch_count = int(topic.get("switch_count") or 0)
                if current_switch_count > switch_count and first_switch_chunk is None:
                    first_switch_chunk = chunks
                    first_switch_s = (meta.get("cursor_samples") or 0) / 16000.0
                switch_count = max(switch_count, current_switch_count)
            await feed_task
    finally:
        delete_session(base_url, session_id)

    warm = retrieve_s[1:] if len(retrieve_s) > 1 else retrieve_s
    return {
        "preset": preset,
        "session_id": session_id,
        "initial_active_glossary": info.get("active_glossary_preset"),
        "preset_terms": info.get("preset_terms"),
        "chunks": chunks,
        "retrieve_p50_ms": round((percentile(warm, 50) or 0.0) * 1000.0, 2) if warm else None,
        "retrieve_p95_ms": round((percentile(warm, 95) or 0.0) * 1000.0, 2) if warm else None,
        "refs_per_chunk": round(statistics.mean(refs_per_chunk), 3) if refs_per_chunk else 0.0,
        "prompt_refs_per_chunk": round(statistics.mean(prompt_refs_per_chunk), 3) if prompt_refs_per_chunk else 0.0,
        "candidate_pool_per_chunk": round(statistics.mean(candidate_pool_counts), 3) if candidate_pool_counts else 0.0,
        "prompt_shortfall_chunks": sum(1 for item in prompt_shortfalls if item),
        "fixed_prompt_k": max(fixed_prompt_ks) if fixed_prompt_ks else None,
        "open_wiki_rescue_chunks": rescue_chunks,
        "active_slices_by_chunk": active_slices_by_chunk,
        "switch_count": switch_count,
        "first_switch_chunk": first_switch_chunk,
        "first_switch_s": first_switch_s,
        "router_actions": router_actions,
        "router_reasons": router_reasons[-8:],
        "router_confidence_avg": round(statistics.mean(router_confidences), 4) if router_confidences else None,
        "domains": domains,
        "active_presets": active_presets,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg")
    ap.add_argument(
        "--presets",
        default="none,open_wiki_100k,nlp_core_10k,medicine_core_10k,auto_working,acl_tagged_raw",
    )
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--max-segs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--feed-sleep", type=float, default=0.45)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    wavs = sorted(glob.glob(f"{args.seg_dir}/*.wav"))[: args.max_segs]
    if not wavs:
        raise SystemExit(f"no wavs found in {args.seg_dir}")
    pcm = np.concatenate([read_wav(path) for path in wavs]).astype(np.float32)
    rows: List[Dict[str, Any]] = []
    for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
        try:
            row = asyncio.run(
                run_preset(
                    base_url=args.base_url,
                    language_pair=args.language_pair,
                    preset=preset,
                    pcm=pcm,
                    chunk=args.chunk,
                    feed_sleep=args.feed_sleep,
                )
            )
        except Exception as exc:  # noqa: BLE001
            row = {"preset": preset, "error": str(exc)[:500]}
        rows.append(row)
        print("[auto-eval]", json.dumps(row, ensure_ascii=False))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[auto-eval] wrote {out}")


if __name__ == "__main__":
    main()
