#!/usr/bin/env python3
"""Long-running streaming stress test for the RASST demo protocol.

This follows the browser chunking behavior: float32 PCM at 16 kHz, with
15360 * latency_multiplier samples per WebSocket message.
"""

import argparse
import asyncio
import json
import math
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
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
    processing_complete_s: Optional[float] = None
    max_tbt_s: float = 0.0
    disconnected: bool = False
    result_times_s: List[float] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)

    def observe_message(self, elapsed_s: float, message: str) -> bool:
        if message.startswith("ERROR"):
            self.errors.append(message)
            return False
        if message.startswith("READY"):
            return False
        if message.startswith("PROCESSING_COMPLETE"):
            self.processing_complete_s = elapsed_s
            return True
        self.messages += 1
        self.texts.append(message)
        self.result_times_s.append(elapsed_s)
        if self.first_result_s is None:
            self.first_result_s = elapsed_s
        if self.last_result_s is not None:
            self.max_tbt_s = max(self.max_tbt_s, elapsed_s - self.last_result_s)
        self.last_result_s = elapsed_s
        return False

    def wallclock_laal_proxy_s(self, audio_duration_s: float) -> Optional[float]:
        """Message-level wall-clock LAAL proxy using emission times.

        This is not token-level SimulEval StreamLAAL and must not be reported
        as StreamLAAL. It measures how far each emitted text update lags behind
        an even target-message schedule over the source audio duration, which
        is the signal this browser-style stress test can observe without token
        timestamps.
        """
        if not self.result_times_s:
            return None
        target_step_s = audio_duration_s / len(self.result_times_s)
        lags = [
            max(0.0, emit_s - ((idx + 1) * target_step_s))
            for idx, emit_s in enumerate(self.result_times_s)
        ]
        return float(sum(lags) / len(lags))


def join_hypothesis(texts: List[str], language_pair: str) -> str:
    cleaned = [clean_text_for_bleu(text) for text in texts if clean_text_for_bleu(text)]
    lower_pair = language_pair.lower()
    if "chinese" in lower_pair or "japanese" in lower_pair:
        return "".join(cleaned)
    return " ".join(cleaned)


def clean_text_for_bleu(text: str) -> str:
    text = re.sub(r"</?t>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _parse_audio_yaml(path: Path) -> List[dict]:
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    except Exception:
        pass

    items: List[dict] = []
    current = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("- "):
            if current:
                items.append(current)
            current = {}
            line = raw_line[2:]
        else:
            line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        items.append(current)
    return items


def load_reference_text(args) -> Optional[str]:
    if args.reference_text_file:
        return "".join(
            clean_text_for_bleu(line)
            for line in Path(args.reference_text_file).read_text(encoding="utf-8").splitlines()
        )
    if not args.reference_audio_yaml or not args.reference_ref_file:
        return None

    items = _parse_audio_yaml(Path(args.reference_audio_yaml))
    refs = Path(args.reference_ref_file).read_text(encoding="utf-8").splitlines()
    target_wav = Path(args.reference_wav or args.audio_file).name
    selected_lines = []
    for idx, item in enumerate(items):
        wav = str(item.get("wav", ""))
        if Path(wav).name == target_wav and idx < len(refs):
            selected_lines.append(clean_text_for_bleu(refs[idx]))
    if not selected_lines:
        raise RuntimeError(
            f"no reference segments matched wav={target_wav} in {args.reference_audio_yaml}"
        )
    return "".join(selected_lines)


def compute_sentence_bleu(hypothesis: str, reference: str, tokenize: str) -> float:
    import sacrebleu

    try:
        return float(
            sacrebleu.sentence_bleu(
                hypothesis,
                [reference],
                tokenize=tokenize,
                use_effective_order=True,
            ).score
        )
    except TypeError:
        return float(sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=tokenize).score)


def compute_corpus_bleu(hypotheses: List[str], reference: str, tokenize: str) -> float:
    import sacrebleu

    references = [[reference for _ in hypotheses]]
    try:
        return float(
            sacrebleu.corpus_bleu(
                hypotheses,
                references,
                tokenize=tokenize,
                use_effective_order=True,
            ).score
        )
    except TypeError:
        return float(sacrebleu.corpus_bleu(hypotheses, references, tokenize=tokenize).score)


def write_bleu_artifacts(args, stats_list: List[SessionStats], reference_text: str) -> List[float]:
    hypotheses = [join_hypothesis(item.texts, args.language_pair) for item in stats_list]
    scores = [
        compute_sentence_bleu(hypothesis, reference_text, args.bleu_tokenize)
        for hypothesis in hypotheses
    ]
    corpus_score = compute_corpus_bleu(hypotheses, reference_text, args.bleu_tokenize) if hypotheses else 0.0
    print(
        "sentence_bleu="
        f"avg={float(np.mean(scores)):.3f} p50={np.percentile(scores, 50):.3f} "
        f"min={min(scores):.3f} max={max(scores):.3f} corpus={corpus_score:.3f} "
        f"tokenize={args.bleu_tokenize}"
    )
    for item, hypothesis, score in list(zip(stats_list, hypotheses, scores))[:5]:
        print(
            f"bleu_sample idx={item.idx} score={score:.3f} "
            f"hyp_chars={len(hypothesis)} ref_chars={len(reference_text)} "
            f"hyp_preview={hypothesis[:120]}"
        )

    if args.save_dir:
        output_dir = Path(args.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "session_predictions.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item, hypothesis, score in zip(stats_list, hypotheses, scores):
                handle.write(
                    json.dumps(
                        {
                            "idx": item.idx,
                            "session_id": item.session_id,
                            "sentence_bleu": score,
                            "hypothesis": hypothesis,
                            "reference": reference_text,
                            "messages": item.texts,
                            "result_times_s": item.result_times_s,
                            "wallclock_laal_proxy_s": item.wallclock_laal_proxy_s(args.duration_sec),
                            "max_tbt_s": item.max_tbt_s,
                            "first_result_s": item.first_result_s,
                            "last_result_s": item.last_result_s,
                            "send_done_s": item.send_done_s,
                            "session_done_s": item.session_done_s,
                            "processing_complete_s": item.processing_complete_s,
                            "errors": item.errors,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "sessions": args.sessions,
                    "audio_file": args.audio_file,
                    "duration_sec": args.duration_sec,
                    "language_pair": args.language_pair,
                    "glossary_preset": args.glossary_preset,
                    "bleu_tokenize": args.bleu_tokenize,
                    "metric_notes": {
                        "wallclock_laal_proxy_s": (
                            "Browser-observed message-level wall-clock proxy; "
                            "not SimulEval StreamLAAL."
                        )
                    },
                    "sentence_bleu_avg": float(np.mean(scores)) if scores else None,
                    "sentence_bleu_p50": float(np.percentile(scores, 50)) if scores else None,
                    "sentence_bleu_min": min(scores) if scores else None,
                    "sentence_bleu_max": max(scores) if scores else None,
                    "corpus_bleu": corpus_score,
                    "reference_chars": len(reference_text),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"bleu_artifacts={output_dir}")
    return scores


def _metric_summary(values: List[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "avg": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def write_stress_artifacts(
    args,
    stats_list: List[SessionStats],
    failures: List[Exception],
    *,
    before_health: dict,
    after_health: dict,
    wall_elapsed_s: float,
    expected_chunks: int,
    total_chunks: int,
    total_messages: int,
    disconnected: int,
) -> None:
    if not args.save_dir:
        return

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "session_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in stats_list:
            handle.write(
                json.dumps(
                    {
                        "idx": item.idx,
                        "session_id": item.session_id,
                        "hypothesis": join_hypothesis(item.texts, args.language_pair),
                        "messages": item.texts,
                        "result_times_s": item.result_times_s,
                        "wallclock_laal_proxy_s": item.wallclock_laal_proxy_s(args.duration_sec),
                        "max_tbt_s": item.max_tbt_s,
                        "first_result_s": item.first_result_s,
                        "last_result_s": item.last_result_s,
                        "send_done_s": item.send_done_s,
                        "session_done_s": item.session_done_s,
                        "processing_complete_s": item.processing_complete_s,
                        "chunks_sent": item.chunks_sent,
                        "errors": item.errors,
                        "disconnected": item.disconnected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    wallclock_laal_proxy_values = [
        value
        for value in (item.wallclock_laal_proxy_s(args.duration_sec) for item in stats_list)
        if value is not None
    ]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "sessions": args.sessions,
                "completed": len(stats_list),
                "failures": [str(failure) for failure in failures[:20]],
                "wall_elapsed_s": wall_elapsed_s,
                "total_chunks": total_chunks,
                "expected_chunks_per_session": expected_chunks,
                "total_messages": total_messages,
                "disconnected": disconnected,
                "realtime": args.realtime,
                "audio_file": args.audio_file,
                "duration_sec": args.duration_sec,
                "language_pair": args.language_pair,
                "glossary_preset": args.glossary_preset,
                "metric_notes": {
                    "wallclock_laal_proxy_s": (
                        "Browser-observed message-level wall-clock proxy; "
                        "not SimulEval StreamLAAL."
                    ),
                    "session_elapsed_s": (
                        "Client WebSocket lifetime from stream start through EOF/drain; "
                        "use result_stream_elapsed_s and last_result_lag_s for translation timing."
                    ),
                },
                "audio_send_elapsed_s": _metric_summary(
                    [item.send_done_s for item in stats_list if item.send_done_s is not None]
                ),
                "session_elapsed_s": _metric_summary(
                    [item.session_done_s for item in stats_list if item.session_done_s is not None]
                ),
                "result_stream_elapsed_s": _metric_summary(
                    [item.last_result_s for item in stats_list if item.last_result_s is not None]
                ),
                "processing_complete_s": _metric_summary(
                    [
                        item.processing_complete_s
                        for item in stats_list
                        if item.processing_complete_s is not None
                    ]
                ),
                "post_send_wait_s": _metric_summary(
                    [
                        item.session_done_s - item.send_done_s
                        for item in stats_list
                        if item.session_done_s is not None and item.send_done_s is not None
                    ]
                ),
                "first_result_s": _metric_summary(
                    [item.first_result_s for item in stats_list if item.first_result_s is not None]
                ),
                "max_tbt_s": _metric_summary([item.max_tbt_s for item in stats_list if item.max_tbt_s > 0]),
                "wallclock_laal_proxy_s": _metric_summary(wallclock_laal_proxy_values),
                "last_result_lag_s": _metric_summary(
                    [
                        item.last_result_s - args.duration_sec
                        for item in stats_list
                        if item.last_result_s is not None
                    ]
                ),
                "health_before": before_health,
                "health_after": after_health,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"stress_artifacts={output_dir}")


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
    if response.status_code == 422:
        response = requests.post(f"{args.base_url}/init", params=payload, timeout=20)
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
            if stats.observe_message(time.time() - start_time, str(message)):
                return
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
                try:
                    await asyncio.wait_for(reader_task, timeout=args.drain_sec)
                except asyncio.TimeoutError:
                    pass
            else:
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
    reference_text = load_reference_text(args)
    if reference_text is not None:
        print(
            "reference_config="
            f"chars={len(reference_text)} "
            f"audio_yaml={args.reference_audio_yaml} ref_file={args.reference_ref_file} "
            f"reference_wav={args.reference_wav or args.audio_file}"
        )
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
    result_stream_values = [item.last_result_s for item in stats_list if item.last_result_s is not None]
    processing_complete_values = [
        item.processing_complete_s for item in stats_list if item.processing_complete_s is not None
    ]
    post_send_wait_values = [
        item.session_done_s - item.send_done_s
        for item in stats_list
        if item.session_done_s is not None and item.send_done_s is not None
    ]
    last_result_lag_values = [
        item.last_result_s - args.duration_sec
        for item in stats_list
        if item.last_result_s is not None
    ]
    wallclock_laal_proxy_values = [
        value
        for value in (item.wallclock_laal_proxy_s(args.duration_sec) for item in stats_list)
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
    if result_stream_values:
        print(
            "result_stream_elapsed_s="
            f"avg={float(np.mean(result_stream_values)):.3f} "
            f"p50={np.percentile(result_stream_values, 50):.3f} "
            f"min={min(result_stream_values):.3f} max={max(result_stream_values):.3f} "
            f"target={args.duration_sec:.3f}"
        )
    if processing_complete_values:
        print(
            "processing_complete_s="
            f"avg={float(np.mean(processing_complete_values)):.3f} "
            f"p50={np.percentile(processing_complete_values, 50):.3f} "
            f"min={min(processing_complete_values):.3f} max={max(processing_complete_values):.3f}"
        )
    if post_send_wait_values:
        print(
            "post_send_wait_s="
            f"avg={float(np.mean(post_send_wait_values)):.3f} "
            f"p50={np.percentile(post_send_wait_values, 50):.3f} "
            f"min={min(post_send_wait_values):.3f} max={max(post_send_wait_values):.3f}"
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
    if wallclock_laal_proxy_values:
        print(
            "wallclock_laal_proxy_s="
            f"avg={float(np.mean(wallclock_laal_proxy_values)):.3f} "
            f"p50={np.percentile(wallclock_laal_proxy_values, 50):.3f} "
            f"p95={np.percentile(wallclock_laal_proxy_values, 95):.3f} "
            f"max={max(wallclock_laal_proxy_values):.3f}"
        )
    else:
        print("wallclock_laal_proxy_s=none")
    if last_result_lag_values:
        print(
            "last_result_lag_s="
            f"avg={float(np.mean(last_result_lag_values)):.3f} "
            f"p50={np.percentile(last_result_lag_values, 50):.3f} "
            f"min={min(last_result_lag_values):.3f} max={max(last_result_lag_values):.3f}"
        )

    for item in stats_list[:5]:
        wallclock_laal_proxy = item.wallclock_laal_proxy_s(args.duration_sec)
        print(
            f"session_sample idx={item.idx} chunks={item.chunks_sent} messages={item.messages} "
            f"send_done_s={item.send_done_s} session_done_s={item.session_done_s} "
            f"processing_complete_s={item.processing_complete_s} "
            f"first_result_s={item.first_result_s} last_result_s={item.last_result_s} "
            f"wallclock_laal_proxy_s={wallclock_laal_proxy} max_tbt_s={item.max_tbt_s:.3f} "
            f"errors={item.errors[:2]}"
        )
    for failure in failures[:10]:
        print(f"failure={failure}")

    write_stress_artifacts(
        args,
        stats_list,
        failures,
        before_health=before_health,
        after_health=after_health,
        wall_elapsed_s=wall_elapsed,
        expected_chunks=expected_chunks,
        total_chunks=total_chunks,
        total_messages=total_messages,
        disconnected=disconnected,
    )

    if reference_text is not None and stats_list:
        write_bleu_artifacts(args, stats_list, reference_text)

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
    parser.add_argument("--reference-text-file", default=None)
    parser.add_argument("--reference-audio-yaml", default=None)
    parser.add_argument("--reference-ref-file", default=None)
    parser.add_argument("--reference-wav", default=None)
    parser.add_argument("--bleu-tokenize", default="zh")
    parser.add_argument("--save-dir", default=None)
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
