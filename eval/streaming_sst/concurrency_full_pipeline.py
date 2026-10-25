#!/usr/bin/env python3
"""Client-measured latency of N concurrent real-time sessions with routing and retrieval on.

Every session streams a different excerpt of real lecture audio at real-time pace: chunk i
is sent at t0 + i * chunk_seconds regardless of server progress, as a live microphone would.
For each output event the client maps ``meta.cursor_samples`` back to the chunk that
completed that cursor and records the delay between sending that chunk and receiving the
event. It also records first-audio-to-first-text delay, the drain time after the last
chunk, retrieval/generation time reported by the server, whether each event carried
routing and retrieval evidence, and GPU memory sampled during the level.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import websockets

SAMPLE_RATE = 16000


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as handle:
        assert handle.getframerate() == SAMPLE_RATE and handle.getnchannels() == 1, path
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def init_session(base_url: str, language_pair: str, preset: str) -> str:
    query = urllib.parse.urlencode({"agent_type": "RASST", "language_pair": language_pair, "glossary_preset": preset})
    request = urllib.request.Request(f"{base_url}/init?{query}", data=b"")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["session_id"]


def delete_session(base_url: str, session_id: str) -> None:
    query = urllib.parse.urlencode({"session_id": session_id})
    urllib.request.urlopen(urllib.request.Request(f"{base_url}/delete_session?{query}", data=b""), timeout=30).read()


def percentile(values: List[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


async def run_session(args: argparse.Namespace, pcm: np.ndarray, start_delay: float) -> Dict[str, Any]:
    await asyncio.sleep(start_delay)
    session_id = await asyncio.to_thread(init_session, args.base_url, args.language_pair, args.preset)
    chunk = args.chunk_samples
    n_chunks = len(pcm) // chunk
    sent_at: List[float] = [0.0] * n_chunks
    events: List[Dict[str, Any]] = []
    url = f"{args.base_url.replace('http', 'ws')}/wss/{session_id}?event_format=json"
    async with websockets.connect(url, max_size=None) as ws:
        await ws.recv()
        t0 = time.monotonic()

        async def feed() -> None:
            for i in range(n_chunks):
                due = t0 + (i + 1) * args.chunk_seconds
                # note (luojiaxuan): a chunk becomes available only once its last sample has been spoken.
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                await ws.send(pcm[i * chunk:(i + 1) * chunk].tobytes())
                sent_at[i] = time.monotonic()
            await ws.send("EOF")

        feeder = asyncio.create_task(feed())
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.idle_timeout_sec)
            except asyncio.TimeoutError:
                break
            received = time.monotonic()
            if not isinstance(raw, str) or not raw.startswith("{"):
                if isinstance(raw, str) and raw.startswith("PROCESSING_COMPLETE"):
                    break
                continue
            message = json.loads(raw)
            meta = message.get("meta") or {}
            if "cursor_samples" not in meta:
                continue
            cursor = int(meta["cursor_samples"])
            index = cursor // chunk - 1
            if index < 0 or index >= n_chunks or sent_at[index] == 0.0:
                continue
            events.append({
                "type": message.get("type"),
                "chunk": index,
                "latency_s": received - sent_at[index],
                "has_text": bool(str(message.get("text") or "").strip()),
                "received_rel_s": received - t0,
                "retrieve_s": meta.get("retrieve_s"),
                "elapsed_s": meta.get("elapsed_s"),
                "references": len(meta.get("references") or []),
                "selected_slices": meta.get("selected_slice_presets"),
                "router_action": (meta.get("topic_router") or {}).get("action") if isinstance(meta.get("topic_router"), dict) else meta.get("router_action"),
            })
            if index == n_chunks - 1 and feeder.done():
                break
        await feeder
    await asyncio.to_thread(delete_session, args.base_url, session_id)
    first_chunk = min((e for e in events), key=lambda e: e["received_rel_s"], default=None)
    first_text = min((e["received_rel_s"] for e in events if e["has_text"]), default=None)
    last_send_rel = sent_at[n_chunks - 1] - t0 if n_chunks else 0.0
    last_event_rel = max((e["received_rel_s"] for e in events), default=None)
    return {
        "chunks": n_chunks,
        "events": events,
        "covered_chunks": len({e["chunk"] for e in events}),
        "first_text_after_audio_start_s": first_text,
        "first_event_after_audio_start_s": first_chunk["received_rel_s"] if first_chunk else None,
        "drain_after_last_chunk_s": (last_event_rel - last_send_rel) if last_event_rel is not None else None,
    }


def sample_gpu_memory(gpus: str, stop: asyncio.Event, out: List[List[int]]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", f"--id={gpus}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        out.append([int(x) for x in result.stdout.split()])
        time.sleep(2.0)


async def run_level(args: argparse.Namespace, excerpts: List[np.ndarray], n: int) -> Dict[str, Any]:
    stop = asyncio.Event()
    memory: List[List[int]] = []
    sampler = asyncio.get_running_loop().run_in_executor(None, sample_gpu_memory, args.gpus, stop, memory)
    # note (luojiaxuan): sessions start staggered across one chunk so their chunk boundaries do not align.
    tasks = [run_session(args, excerpts[i % len(excerpts)], args.chunk_seconds * i / max(1, n)) for i in range(n)]
    sessions = await asyncio.gather(*tasks)
    stop.set()
    await sampler
    warm = [e for s in sessions for e in s["events"] if e["chunk"] >= args.warmup_chunks]
    text_events = [e for e in warm if e["type"] == "partial"]
    latencies = [e["latency_s"] for e in warm]
    per_session_p95 = [percentile([e["latency_s"] for e in s["events"] if e["chunk"] >= args.warmup_chunks], 95) for s in sessions]
    deadline = args.chunk_seconds
    return {
        "sessions": n,
        "chunks_per_session": sessions[0]["chunks"],
        "coverage_min": min(s["covered_chunks"] / s["chunks"] for s in sessions),
        "e2e_latency_s": {p: percentile(latencies, q) for p, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "worst_session_p95_s": max(x for x in per_session_p95 if x is not None),
        "share_over_chunk_length": float(np.mean([x > deadline for x in latencies])) if latencies else None,
        "first_text_after_audio_start_s": {
            "median": statistics.median([s["first_text_after_audio_start_s"] for s in sessions if s["first_text_after_audio_start_s"] is not None]),
            "max": max(s["first_text_after_audio_start_s"] for s in sessions if s["first_text_after_audio_start_s"] is not None),
        },
        "drain_after_last_chunk_s": {
            "median": statistics.median([s["drain_after_last_chunk_s"] for s in sessions if s["drain_after_last_chunk_s"] is not None]),
            "max": max(s["drain_after_last_chunk_s"] for s in sessions if s["drain_after_last_chunk_s"] is not None),
        },
        "retrieve_s": {p: percentile([float(e["retrieve_s"]) for e in warm if e["retrieve_s"] is not None], q) for p, q in (("p50", 50), ("p95", 95))},
        "generation_s": {p: percentile([float(e["elapsed_s"]) for e in warm if e["elapsed_s"] is not None], q) for p, q in (("p50", 50), ("p95", 95))},
        "events_with_selected_slices": float(np.mean([bool(e["selected_slices"]) for e in text_events])) if text_events else None,
        "events_with_references": float(np.mean([e["references"] > 0 for e in text_events])) if text_events else None,
        "gpu_memory_mib_max": [max(col) for col in zip(*memory)] if memory else None,
        "raw_sessions": sessions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--preset", default="auto_working")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--audio-glob", action="append", required=True)
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--seconds-per-session", type=float, default=180.0)
    ap.add_argument("--chunk-samples", type=int, default=30720)
    ap.add_argument("--warmup-chunks", type=int, default=3)
    ap.add_argument("--idle-timeout-sec", type=float, default=60.0)
    ap.add_argument("--gpus", required=True, help="physical GPU ids for memory sampling, e.g. 4,5")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.chunk_seconds = args.chunk_samples / SAMPLE_RATE

    files = sorted({f for pattern in args.audio_glob for f in glob.glob(pattern)})
    length = int(args.seconds_per_session * SAMPLE_RATE)
    excerpts = []
    for k in range(32):
        pcm = read_wav(files[k % len(files)])
        offset = (k // len(files)) * length
        if offset + length > len(pcm):
            offset = max(0, len(pcm) - length)
        excerpts.append(pcm[offset:offset + length])
    result = {"audio_files": files, "seconds_per_session": args.seconds_per_session, "levels": []}
    for n in [int(x) for x in args.levels.split(",")]:
        row = asyncio.run(run_level(args, excerpts, n))
        summary = {k: v for k, v in row.items() if k != "raw_sessions"}
        print(json.dumps(summary), flush=True)
        result["levels"].append(row)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
