#!/usr/bin/env python3
"""Capture one live RASST JSON-WS sample for the paper UI screenshot."""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as wav:
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def strip_tags(text: str) -> str:
    return text.replace("<t>", "").replace("</t>", "")


async def capture(args: argparse.Namespace) -> dict[str, Any]:
    files = sorted(glob.glob(f"{args.seg_dir}/*.wav"))[: args.max_segs]
    pcm = np.concatenate([read_wav(path) for path in files]).astype(np.float32)
    query = urllib.parse.urlencode(
        {
            "agent_type": "RASST",
            "language_pair": args.language_pair,
            "glossary_preset": args.glossary_preset,
        }
    )
    with urllib.request.urlopen(urllib.request.Request(f"{args.base_url}/init?{query}", data=b""), timeout=30) as resp:
        init = json.load(resp)
    sid = init["session_id"]
    parts: list[str] = []
    metas: list[dict[str, Any]] = []
    all_refs: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    try:
        ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{ws_url}/wss/{sid}?event_format=json", max_size=None) as ws:
            await ws.recv()

            async def feed() -> None:
                for offset in range(0, len(pcm), args.chunk):
                    await ws.send(pcm[offset : offset + args.chunk].tobytes())
                    await asyncio.sleep(args.feed_sleep)
                await ws.send("EOF")

            feeder = asyncio.create_task(feed())
            idle = 0
            while idle < 1:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0 if feeder.done() else 60.0)
                except asyncio.TimeoutError:
                    idle += 1
                    continue
                obj = json.loads(msg)
                if obj.get("type") != "partial":
                    continue
                text = strip_tags(str(obj.get("text") or "")).strip()
                meta = obj.get("meta") or {}
                if text:
                    parts.append(text)
                if meta.get("references"):
                    metas.append(meta)
                    for ref in meta.get("references") or []:
                        key = str(ref.get("key") or ref.get("term") or "").casefold()
                        if key and key not in seen_refs:
                            seen_refs.add(key)
                            all_refs.append(ref)
            await feeder
    finally:
        try:
            delete_q = urllib.parse.urlencode({"session_id": sid})
            urllib.request.urlopen(urllib.request.Request(f"{args.base_url}/delete_session?{delete_q}", data=b""), timeout=15)
        except Exception:
            pass

    latest = metas[-1] if metas else {}
    return {
        "session_id": sid,
        "init": init,
        "audio_files": files,
        "translation": "".join(parts),
        "latest_meta": latest,
        "references": latest.get("references") or [],
        "all_references": all_refs,
        "topic": latest.get("topic") or {},
        "topic_router": latest.get("topic_router") or {},
        "retrieve_ms": round(float(latest.get("retrieve_s") or 0.0) * 1000.0, 1),
        "elapsed_ms": round(float(latest.get("elapsed_s") or 0.0) * 1000.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--seg-dir", default="/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg")
    parser.add_argument("--language-pair", default="English -> Chinese")
    parser.add_argument("--glossary-preset", default="auto_working")
    parser.add_argument("--max-segs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=8000)
    parser.add_argument("--feed-sleep", type=float, default=0.45)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    sample = asyncio.run(capture(args))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "refs": len(sample["references"]), "retrieve_ms": sample["retrieve_ms"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
