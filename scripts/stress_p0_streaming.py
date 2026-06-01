#!/usr/bin/env python3
"""Long-running streaming stress test for the RASST demo protocol.

This follows the browser chunking behavior: float32 PCM at 16 kHz, with
15360 * latency_multiplier samples per WebSocket message.
"""

import argparse
import asyncio
import math
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import requests
import soundfile as sf
import websockets


BASE_CHUNK_SAMPLES = 960 * 16
TARGET_SAMPLE_RATE = 16000


@dataclass
class SessionStats:
    idx: int
    session_id: str
    ready_s: Optional[float] = None
    chunks_sent: int = 0
    bytes_sent: int = 0
    messages: int = 0
    errors: List[str] = field(default_factory=list)
    first_result_s: Optional[float] = None
    last_result_s: Optional[float] = None
    send_done_s: Optional[float] = None
    session_done_s: Optional[float] = None
    max_tbt_s: float = 0.0
    disconnected: bool = False
    result_times_s: List[float] = field(default_factory=list)

    def observe_message(self, elapsed_s: float, message: str) -> None:
        if message.startswith("ERROR"):
            self.errors.append(message)
            return
        if message.startswith("READY") or message.startswith("PROCESSING_COMPLETE"):
            return
        self.messages += 1
        self.result_times_s.append(elapsed_s)
        if self.first_result_s is None:
            self.first_result_s = elapsed_s
        if self.last_result_s is not None:
            self.max_tbt_s = max(self.max_tbt_s, elapsed_s - self.last_result_s)
        self.last_result_s = elapsed_s

    def stream_laal_s(self, audio_duration_s: float) -> Optional[float]:
        """Message-level streaming LAAL proxy using wall-clock emission times.

        This is not token-level SimulEval LAAL. It measures how far each emitted
        text update lags behind an even target-message schedule over the source
        audio duration, which is the signal this browser-style stress test can
        observe without token timestamps.
        """
        if not self.result_times_s:
            return None
        target_step_s = audio_duration_s / len(self.result_times_s)
        lags = [
            max(0.0, emit_s - ((idx + 1) * target_step_s))
            for idx, emit_s in enumerate(self.result_times_s)
        ]
        return float(sum(lags) / len(lags))


def load_audio(path: str, duration_sec: float) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("scipy is required to resample non-16kHz audio") from exc
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor).astype("float32")

    target_samples = int(duration_sec * TARGET_SAMPLE_RATE)
    if len(audio) < target_samples:
        repeats = math.ceil(target_samples / max(1, len(audio)))
        audio = np.tile(audio, repeats)
    return np.asarray(audio[:target_samples], dtype="float32")


def init_session(args, idx: int, run_id: str) -> str:
    payload = {
        "agent_type": args.agent_type,
        "language_pair": args.language_pair,
        "latency_multiplier": args.latency_multiplier,
        "client_id": f"p0stream_{run_id}_{idx:02d}",
        "glossary_preset": args.glossary_preset,
        "glossary_text": args.glossary_text,
    }
    response = requests.post(f"{args.base_url}/init", json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    if "session_id" not in data:
        raise RuntimeError(f"missing session_id in /init response: {data}")
    if args.require_scheduler and not data.get("scheduler_based", False) and not data.get("rasst_backend", False):
        raise RuntimeError(f"session is not scheduler-based: {data}")
    return data["session_id"]


def delete_session(base_url: str, session_id: str) -> None:
    try:
        response = requests.post(f"{base_url}/delete_session", params={"session_id": session_id}, timeout=10)
        if response.status_code >= 400:
            print(f"cleanup_failed={session_id} status={response.status_code}")
    except Exception as exc:
        print(f"cleanup_exception={session_id} error={exc}")


async def reader(websocket, stats: SessionStats, start_time: float, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            stats.observe_message(time.time() - start_time, str(message))
        except asyncio.TimeoutError:
            continue
        except websockets.ConnectionClosed as exc:
            stats.disconnected = True
            stats.errors.append(f"connection_closed code={exc.code} reason={exc.reason}")
            return
        except Exception as exc:
            stats.errors.append(f"reader_error={exc}")
            return


async def run_streaming_session(args, session_id: str, idx: int, audio: np.ndarray, start_barrier: asyncio.Event) -> SessionStats:
    stats = SessionStats(idx=idx, session_id=session_id)
    ws_base_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://")
    uri = f"{ws_base_url}/wss/{urllib.parse.quote(session_id, safe='')}"
    chunk_samples = args.chunk_samples or BASE_CHUNK_SAMPLES * args.latency_multiplier
    chunk_period_s = chunk_samples / TARGET_SAMPLE_RATE
    chunks = [audio[offset:offset + chunk_samples] for offset in range(0, len(audio), chunk_samples)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]
    stop_event = asyncio.Event()
    start_time = time.time()

    async with websockets.connect(uri, open_timeout=args.open_timeout, max_size=None) as websocket:
        ready = await asyncio.wait_for(websocket.recv(), timeout=args.open_timeout)
        if not str(ready).startswith("READY"):
            raise RuntimeError(f"{session_id}: expected READY, got {ready}")
        stats.ready_s = time.time() - start_time

        await start_barrier.wait()
        stream_start = time.time()
        reader_task = asyncio.create_task(reader(websocket, stats, stream_start, stop_event))
        try:
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) < chunk_samples:
                    padded = np.zeros(chunk_samples, dtype="float32")
                    padded[:len(chunk)] = chunk
                    chunk = padded
                await websocket.send(np.asarray(chunk, dtype="float32").tobytes())
                stats.chunks_sent += 1
                stats.bytes_sent += int(chunk.nbytes)

                if args.realtime:
                    next_send_at = stream_start + (chunk_idx + 1) * chunk_period_s
                    sleep_s = next_send_at - time.time()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
            stats.send_done_s = time.time() - stream_start

            if args.send_eof:
                await websocket.send("EOF")
            await asyncio.sleep(args.drain_sec)
            stats.session_done_s = time.time() - stream_start
        finally:
            stop_event.set()
            await asyncio.gather(reader_task, return_exceptions=True)

    return stats


async def run_all(args) -> int:
    health = requests.get(f"{args.base_url}/health", timeout=20)
    health.raise_for_status()
    before_health = health.json()
    print(f"health_before={before_health}")
    if args.fail_on_mock and before_health.get("mock_mode"):
        raise RuntimeError("server is in mock_mode; refusing real stress test")

    audio = load_audio(args.audio_file, args.duration_sec)
    chunk_samples = args.chunk_samples or BASE_CHUNK_SAMPLES * args.latency_multiplier
    expected_chunks = math.ceil(len(audio) / chunk_samples)
    print(
        "stress_config="
        f"sessions={args.sessions} duration_s={args.duration_sec:.1f} "
        f"chunk_samples={chunk_samples} chunk_s={chunk_samples / TARGET_SAMPLE_RATE:.3f} "
        f"expected_chunks_per_session={expected_chunks} "
        f"audio_file={args.audio_file}"
    )

    run_id = str(int(time.time() * 1000))
    session_ids = [init_session(args, idx, run_id) for idx in range(args.sessions)]
    start_barrier = asyncio.Event()
    tasks = [
        asyncio.create_task(run_streaming_session(args, session_id, idx, audio, start_barrier))
        for idx, session_id in enumerate(session_ids)
    ]

    wall_start = time.time()
    try:
        await asyncio.sleep(args.connect_grace_sec)
        start_barrier.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if args.cleanup:
            for session_id in session_ids:
                delete_session(args.base_url, session_id)

    wall_elapsed = time.time() - wall_start
    after_health = requests.get(f"{args.base_url}/health", timeout=20).json()

    failures = []
    stats_list: List[SessionStats] = []
    for result in results:
        if isinstance(result, Exception):
            failures.append(result)
        else:
            stats_list.append(result)
            if result.errors:
                failures.append(RuntimeError(f"{result.session_id}: {result.errors[:3]}"))
            if result.chunks_sent != expected_chunks:
                failures.append(RuntimeError(f"{result.session_id}: chunks_sent={result.chunks_sent}, expected={expected_chunks}"))

    total_chunks = sum(item.chunks_sent for item in stats_list)
    total_messages = sum(item.messages for item in stats_list)
    first_result_values = [item.first_result_s for item in stats_list if item.first_result_s is not None]
    max_tbt_values = [item.max_tbt_s for item in stats_list if item.max_tbt_s > 0]
    send_done_values = [item.send_done_s for item in stats_list if item.send_done_s is not None]
    session_done_values = [item.session_done_s for item in stats_list if item.session_done_s is not None]
    last_result_lag_values = [
        item.last_result_s - args.duration_sec
        for item in stats_list
        if item.last_result_s is not None
    ]
    stream_laal_values = [
        value
        for value in (item.stream_laal_s(args.duration_sec) for item in stats_list)
        if value is not None
    ]
    disconnected = sum(1 for item in stats_list if item.disconnected)
    if args.realtime and args.require_realtime_wallclock:
        min_send_done_s = max(0.0, args.duration_sec - (chunk_samples / TARGET_SAMPLE_RATE) - args.realtime_tolerance_sec)
        for item in stats_list:
            if item.send_done_s is None or item.send_done_s < min_send_done_s:
                failures.append(
                    RuntimeError(
                        f"{item.session_id}: audio was not streamed in real time; "
                        f"send_done_s={item.send_done_s}, expected>={min_send_done_s:.3f}"
                    )
                )
    print(f"health_after={after_health}")
    print(
        "stress_result="
        f"sessions={args.sessions} completed={len(stats_list)} failures={len(failures)} "
        f"wall_elapsed_s={wall_elapsed:.3f} total_chunks={total_chunks} total_messages={total_messages} "
        f"disconnected={disconnected} realtime={args.realtime}"
    )
    if send_done_values:
        print(
            "audio_send_elapsed_s="
            f"avg={float(np.mean(send_done_values)):.3f} "
            f"p50={np.percentile(send_done_values, 50):.3f} "
            f"min={min(send_done_values):.3f} max={max(send_done_values):.3f} "
            f"target={args.duration_sec:.3f}"
        )
    if session_done_values:
        print(
            "session_elapsed_s="
            f"avg={float(np.mean(session_done_values)):.3f} "
            f"p50={np.percentile(session_done_values, 50):.3f} "
            f"min={min(session_done_values):.3f} max={max(session_done_values):.3f}"
        )
    if first_result_values:
        print(
            "first_result_s="
            f"min={min(first_result_values):.3f} p50={np.percentile(first_result_values, 50):.3f} "
            f"max={max(first_result_values):.3f}"
        )
    else:
        print("first_result_s=none")
    if max_tbt_values:
        print(
            "max_tbt_s="
            f"p50={np.percentile(max_tbt_values, 50):.3f} "
            f"p95={np.percentile(max_tbt_values, 95):.3f} "
            f"max={max(max_tbt_values):.3f}"
        )
    else:
        print("max_tbt_s=none")
    if stream_laal_values:
        print(
            "stream_laal_s="
            f"avg={float(np.mean(stream_laal_values)):.3f} "
            f"p50={np.percentile(stream_laal_values, 50):.3f} "
            f"p95={np.percentile(stream_laal_values, 95):.3f} "
            f"max={max(stream_laal_values):.3f}"
        )
    else:
        print("stream_laal_s=none")
    if last_result_lag_values:
        print(
            "last_result_lag_s="
            f"avg={float(np.mean(last_result_lag_values)):.3f} "
            f"p50={np.percentile(last_result_lag_values, 50):.3f} "
            f"min={min(last_result_lag_values):.3f} max={max(last_result_lag_values):.3f}"
        )

    for item in stats_list[:5]:
        stream_laal = item.stream_laal_s(args.duration_sec)
        print(
            f"session_sample idx={item.idx} chunks={item.chunks_sent} messages={item.messages} "
            f"send_done_s={item.send_done_s} session_done_s={item.session_done_s} "
            f"first_result_s={item.first_result_s} last_result_s={item.last_result_s} "
            f"stream_laal_s={stream_laal} max_tbt_s={item.max_tbt_s:.3f} errors={item.errors[:2]}"
        )
    for failure in failures[:10]:
        print(f"failure={failure}")

    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run 32-session long streaming P0 stress test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--audio-file", required=True)
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--latency-multiplier", type=int, default=2)
    parser.add_argument("--chunk-samples", type=int, default=None)
    parser.add_argument("--agent-type", default="RASST")
    parser.add_argument("--language-pair", default="English -> Chinese")
    parser.add_argument("--glossary-preset", default="acl_tagged_raw")
    parser.add_argument("--glossary-text-file", default=None)
    parser.add_argument("--open-timeout", type=float, default=60.0)
    parser.add_argument("--drain-sec", type=float, default=30.0)
    parser.add_argument("--connect-grace-sec", type=float, default=1.0)
    parser.add_argument("--send-eof", action="store_true")
    parser.add_argument("--no-realtime", action="store_false", dest="realtime")
    parser.add_argument("--allow-fast-feed", action="store_false", dest="require_realtime_wallclock")
    parser.add_argument("--realtime-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup")
    parser.add_argument("--allow-mock", action="store_false", dest="fail_on_mock")
    parser.add_argument("--allow-traditional", action="store_false", dest="require_scheduler")
    parser.set_defaults(
        realtime=True,
        require_realtime_wallclock=True,
        cleanup=True,
        fail_on_mock=True,
        require_scheduler=True,
    )
    args = parser.parse_args()
    if args.glossary_text_file:
        with open(args.glossary_text_file, "r", encoding="utf-8") as handle:
            args.glossary_text = handle.read()
    else:
        args.glossary_text = ""
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(asyncio.run(run_all(parse_args())))
