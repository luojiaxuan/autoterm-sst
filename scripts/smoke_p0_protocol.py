#!/usr/bin/env python3
"""P0 protocol smoke test for the RASST demo server."""

import argparse
import asyncio
import math
import time
import urllib.parse

import numpy as np
import requests
import soundfile as sf
import websockets


def init_session(base_url: str, idx: int, run_id: str) -> str:
    params = {
        "agent_type": "InfiniSST",
        "language_pair": "English -> Chinese",
        "latency_multiplier": 2,
        "client_id": f"p0_{run_id}_{idx:02d}",
    }
    response = requests.post(f"{base_url}/init", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "session_id" not in data:
        raise RuntimeError(f"missing session_id in /init response: {data}")
    if not data.get("scheduler_based", False):
        raise RuntimeError(f"session is not scheduler-based: {data}")
    return data["session_id"]


def delete_session(base_url: str, session_id: str) -> None:
    response = requests.post(f"{base_url}/delete_session", params={"session_id": session_id}, timeout=10)
    if response.status_code >= 400:
        print(f"cleanup_failed={session_id} status={response.status_code}")


def load_audio(args) -> np.ndarray:
    if not args.audio_file:
        return (0.1 * np.sin(np.linspace(0, 100, args.samples))).astype("float32")

    audio, sample_rate = sf.read(args.audio_file, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("scipy is required to resample non-16kHz audio") from exc
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype("float32")
    return np.asarray(audio, dtype="float32")


async def run_session(ws_base_url: str, session_id: str, audio: np.ndarray, timeout_s: float) -> str:
    uri = f"{ws_base_url}/wss/{urllib.parse.quote(session_id, safe='')}"
    async with websockets.connect(uri, open_timeout=timeout_s) as websocket:
        ready = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
        if not ready.startswith("READY"):
            raise RuntimeError(f"{session_id}: expected READY, got {ready}")
        await websocket.send(audio.tobytes())
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            message = await asyncio.wait_for(websocket.recv(), timeout=max(0.1, deadline - time.time()))
            if message.startswith("ERROR"):
                raise RuntimeError(f"{session_id}: {message}")
            if message == "":
                return "<empty translation>"
            if message and not message.startswith("READY"):
                return message
        raise TimeoutError(f"{session_id}: no translation within {timeout_s}s")


async def run_all(args) -> int:
    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    health_data = health.json()
    print(f"health={health_data}")
    if not health_data.get("scheduler_enabled"):
        raise RuntimeError("scheduler is not enabled")

    run_id = str(int(time.time() * 1000))
    session_ids = [init_session(args.base_url, idx, run_id) for idx in range(args.sessions)]
    audio = load_audio(args)
    print(f"audio_samples={len(audio)} audio_seconds={len(audio) / 16000:.3f} audio_file={args.audio_file or '<synthetic>'}")
    try:
        ws_base_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
        started = time.time()
        tasks = [run_session(ws_base_url, session_id, audio, args.timeout) for session_id in session_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - started
    finally:
        if args.cleanup:
            for session_id in session_ids:
                delete_session(args.base_url, session_id)

    failures = [result for result in results if isinstance(result, Exception)]
    successes = [result for result in results if not isinstance(result, Exception)]
    print(f"sessions={args.sessions} successes={len(successes)} failures={len(failures)} elapsed_s={elapsed:.3f}")
    if successes:
        print(f"sample_result={successes[0]}")
    for failure in failures[:5]:
        print(f"failure={failure}")
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run P0 RASST demo protocol smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--audio-file", default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup")
    parser.set_defaults(cleanup=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_all(parse_args())))
