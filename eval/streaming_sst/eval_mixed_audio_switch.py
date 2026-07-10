#!/usr/bin/env python3
"""Run a mixed ACL/medicine audio playlist through the JSON WS demo server.

This is the deployable follow-up to ``eval_mixed_domain_switch.py``: it streams
real ACL and medicine audio into one ``auto_working`` session, records the
runtime router/probe metadata per partial output, and scores active-domain
switch behavior against the known playlist spans. It does not use ASR text or
source transcripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.eval_mixed_domain_switch import (  # noqa: E402
    DEFAULT_ACL_ROOT,
    DEFAULT_MEDICINE_AUDIO_DIR,
)
from framework.agents.term_memory.domain_taxonomy import domain_for_preset  # noqa: E402

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioBlock:
    item_id: str
    expected_domain: str
    corpus: str
    wav_paths: Sequence[str]


@dataclass(frozen=True)
class AudioBlockSpan:
    block_index: int
    item_id: str
    corpus: str
    expected_domain: str
    start_sample: int
    end_sample: int
    sample_count: int
    wav_count: int

    @property
    def start_s(self) -> float:
        return self.start_sample / TARGET_SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end_sample / TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class OracleChunkPlan:
    chunk_index: int
    start_sample: int
    future_cursor_samples: int
    expected_domain: str
    glossary_preset: str
    chunk: np.ndarray


def read_acl_audio_blocks(
    acl_root: str,
    *,
    limit_items: int,
    max_segs_per_talk: int = 0,
) -> List[AudioBlock]:
    if int(limit_items) <= 0:
        return []
    root = Path(acl_root)
    meta_path = root / "segments.meta.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"ACL segment metadata not found: {meta_path}")
    by_talk: "OrderedDict[str, List[str]]" = OrderedDict()
    for raw in meta_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            continue
        talk_id = str(meta.get("talk") or meta.get("talk_id") or "").strip()
        wav_path = str(meta.get("seg_wav") or "").strip()
        if not talk_id or not wav_path:
            continue
        by_talk.setdefault(talk_id, []).append(wav_path)

    blocks: List[AudioBlock] = []
    for talk_id, wavs in by_talk.items():
        selected = [path for path in wavs if Path(path).is_file()]
        if int(max_segs_per_talk) > 0:
            selected = selected[: int(max_segs_per_talk)]
        if selected:
            blocks.append(AudioBlock(talk_id, "nlp", "acl", selected))
        if len(blocks) >= max(0, int(limit_items)):
            break
    return blocks


def read_medicine_audio_blocks(
    audio_dir: str,
    *,
    limit_items: int,
    medicine_ids: Optional[Sequence[str]] = None,
) -> List[AudioBlock]:
    if int(limit_items) <= 0:
        return []
    root = Path(audio_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"medicine audio directory not found: {root}")
    paths = sorted(root.glob("sample_*_v2/*_v2.wav"), key=lambda path: _medicine_audio_sort_key(path))
    by_id: Dict[str, Path] = {}
    for path in paths:
        medicine_id = path.stem[:-3] if path.stem.endswith("_v2") else path.stem
        by_id[medicine_id] = path
    if medicine_ids:
        selected_paths: List[Path] = []
        missing: List[str] = []
        for raw_id in medicine_ids:
            medicine_id = str(raw_id).strip().removeprefix("medicine_")
            if not medicine_id:
                continue
            path = by_id.get(medicine_id)
            if path is None:
                missing.append(medicine_id)
            else:
                selected_paths.append(path)
        if missing:
            raise FileNotFoundError(f"medicine audio id(s) not found: {', '.join(missing)}")
        paths = selected_paths
    blocks: List[AudioBlock] = []
    for path in paths[: max(0, int(limit_items))]:
        medicine_id = path.stem[:-3] if path.stem.endswith("_v2") else path.stem
        blocks.append(AudioBlock(f"medicine_{medicine_id}", "medicine", "medicine", [str(path)]))
    return blocks


def build_schedule(
    acl_blocks: Sequence[AudioBlock],
    medicine_blocks: Sequence[AudioBlock],
    *,
    schedule: str,
    seed: int = 20260707,
) -> List[AudioBlock]:
    acl = list(acl_blocks)
    medicine = list(medicine_blocks)
    if schedule == "alternating":
        out: List[AudioBlock] = []
        for idx in range(max(len(acl), len(medicine))):
            if idx < len(acl):
                out.append(acl[idx])
            if idx < len(medicine):
                out.append(medicine[idx])
        return out
    if schedule == "random":
        out = acl + medicine
        random.Random(int(seed)).shuffle(out)
        return out
    if schedule == "acl_then_medicine":
        return acl + medicine
    if schedule == "medicine_then_acl":
        return medicine + acl
    raise ValueError(f"unknown schedule: {schedule}")


def build_spans(
    blocks: Sequence[AudioBlock],
    *,
    max_seconds_per_item: float = 0.0,
) -> List[AudioBlockSpan]:
    spans: List[AudioBlockSpan] = []
    cursor = 0
    max_samples = int(round(float(max_seconds_per_item) * TARGET_SAMPLE_RATE)) if max_seconds_per_item > 0 else 0
    for idx, block in enumerate(blocks, start=1):
        total = 0
        for path in block.wav_paths:
            if max_samples and total >= max_samples:
                break
            frames = wav_num_frames(path)
            if max_samples:
                frames = min(frames, max_samples - total)
            total += max(0, frames)
        spans.append(
            AudioBlockSpan(
                block_index=idx,
                item_id=block.item_id,
                corpus=block.corpus,
                expected_domain=block.expected_domain,
                start_sample=cursor,
                end_sample=cursor + total,
                sample_count=total,
                wav_count=len(block.wav_paths),
            )
        )
        cursor += total
    return spans


def resolve_chunk_samples(chunk: int, base_segment_sec: float, latency_multiplier: int) -> int:
    if int(chunk) > 0:
        return int(chunk)
    lm = max(1, min(4, int(latency_multiplier)))
    return max(1, int(round(float(base_segment_sec) * lm * TARGET_SAMPLE_RATE)))


def resolve_max_switch_events(max_switch_events: int, max_switch_seconds: float, chunk_samples: int) -> int:
    if float(max_switch_seconds) > 0:
        chunk_seconds = max(1e-9, int(chunk_samples) / TARGET_SAMPLE_RATE)
        return max(1, int(math.ceil(float(max_switch_seconds) / chunk_seconds)))
    return max(1, int(max_switch_events))


def wav_num_frames(path: str) -> int:
    with wave.open(path) as handle:
        if handle.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"expected {TARGET_SAMPLE_RATE} Hz wav, got {handle.getframerate()}: {path}")
        if handle.getnchannels() != 1:
            raise ValueError(f"expected mono wav, got {handle.getnchannels()} channels: {path}")
        return int(handle.getnframes())


def read_wav(path: str, *, max_frames: int = 0) -> np.ndarray:
    with wave.open(path) as handle:
        if handle.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"expected {TARGET_SAMPLE_RATE} Hz wav, got {handle.getframerate()}: {path}")
        frames = int(handle.getnframes())
        if max_frames > 0:
            frames = min(frames, max_frames)
        raw = handle.readframes(frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def iter_pcm_chunks(
    blocks: Sequence[AudioBlock],
    *,
    chunk_samples: int,
    max_seconds_per_item: float = 0.0,
) -> Iterator[np.ndarray]:
    carry = np.zeros((0,), dtype=np.float32)
    max_samples = int(round(float(max_seconds_per_item) * TARGET_SAMPLE_RATE)) if max_seconds_per_item > 0 else 0
    for block in blocks:
        emitted_for_block = 0
        for path in block.wav_paths:
            remaining = max_samples - emitted_for_block if max_samples else 0
            if max_samples and remaining <= 0:
                break
            pcm = read_wav(path, max_frames=remaining if max_samples else 0)
            emitted_for_block += int(pcm.shape[0])
            if carry.size:
                pcm = np.concatenate([carry, pcm]).astype(np.float32, copy=False)
                carry = np.zeros((0,), dtype=np.float32)
            while int(pcm.shape[0]) >= int(chunk_samples):
                yield pcm[:chunk_samples].astype(np.float32, copy=False)
                pcm = pcm[chunk_samples:]
            carry = pcm.astype(np.float32, copy=False)
    if carry.size:
        yield carry.astype(np.float32, copy=False)


def _http_base(base_url: str) -> str:
    return base_url.rstrip("/")


def _ws_base(base_url: str) -> str:
    return _http_base(base_url).replace("http://", "ws://").replace("https://", "wss://")


def parse_oracle_preset_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item.count("=") != 1:
            raise ValueError(
                "oracle preset entries must use DOMAIN=PRESET, "
                f"got {item!r}"
            )
        raw_domain, raw_preset = item.split("=", 1)
        domain = raw_domain.strip().lower()
        preset = raw_preset.strip()
        if not domain or not preset:
            raise ValueError(
                "oracle preset entries must use non-empty DOMAIN=PRESET, "
                f"got {item!r}"
            )
        if domain in mapping and mapping[domain] != preset:
            raise ValueError(
                f"oracle domain {domain!r} maps to both {mapping[domain]!r} and {preset!r}"
            )
        mapping[domain] = preset
    return mapping


def validate_oracle_preset_map(
    oracle_preset_map: Dict[str, str],
    spans: Sequence[AudioBlockSpan],
) -> None:
    if not oracle_preset_map:
        return
    required_domains = sorted(
        {
            str(span.expected_domain).strip().lower()
            for span in spans
            if int(span.sample_count) > 0 and str(span.expected_domain).strip()
        }
    )
    missing = [domain for domain in required_domains if domain not in oracle_preset_map]
    if missing:
        raise ValueError(
            "oracle preset map is missing playlist domain(s): " + ", ".join(missing)
        )


def iter_oracle_chunk_plan(
    chunks: Iterable[np.ndarray],
    *,
    spans: Sequence[AudioBlockSpan],
    oracle_preset_map: Dict[str, str],
) -> Iterator[OracleChunkPlan]:
    cursor_samples = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        sample_count = int(chunk.shape[0])
        future_cursor_samples = cursor_samples + sample_count
        expected_domain = expected_domain_at(spans, future_cursor_samples).strip().lower()
        preset = str(oracle_preset_map.get(expected_domain) or "")
        if oracle_preset_map and not preset:
            raise RuntimeError(
                f"oracle preset map has no preset for domain {expected_domain!r} "
                f"at future cursor {future_cursor_samples}"
            )
        yield OracleChunkPlan(
            chunk_index=chunk_index,
            start_sample=cursor_samples,
            future_cursor_samples=future_cursor_samples,
            expected_domain=expected_domain,
            glossary_preset=preset,
            chunk=chunk,
        )
        cursor_samples = future_cursor_samples


def switch_session_glossary(
    base_url: str,
    session_id: str,
    glossary_preset: str,
    *,
    language_pair: str,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    body = json.dumps(
        {
            "session_id": session_id,
            "glossary_preset": glossary_preset,
            "language_pair": language_pair,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{_http_base(base_url)}/glossary/build",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_sec))) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            detail = ""
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} failed with "
            f"HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} failed: {exc.reason}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} returned a non-object response"
        )
    if result.get("success") is not True:
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} was rejected: "
            f"{result.get('error') or result.get('detail') or result}"
        )
    if result.get("session_updated") is not True:
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} did not update session {session_id!r}"
        )
    active_preset = str(result.get("active_glossary_preset") or "")
    if active_preset != glossary_preset:
        raise RuntimeError(
            f"oracle glossary switch requested {glossary_preset!r} but server activated "
            f"{active_preset or '<missing>'!r}"
        )
    if result.get("auto_glossary_enabled") is True:
        raise RuntimeError(
            f"oracle glossary switch to {glossary_preset!r} left automatic routing enabled"
        )
    return result


def init_session(base_url: str, language_pair: str, preset: str, latency_multiplier: int) -> Dict[str, Any]:
    q = urllib.parse.urlencode(
        {
            "agent_type": "RASST",
            "language_pair": language_pair,
            "glossary_preset": preset,
            "latency_multiplier": int(latency_multiplier),
        }
    )
    with urllib.request.urlopen(urllib.request.Request(f"{_http_base(base_url)}/init?{q}", data=b""), timeout=30) as response:
        return json.load(response)


def delete_session(base_url: str, session_id: str) -> None:
    q = urllib.parse.urlencode({"session_id": session_id})
    try:
        urllib.request.urlopen(urllib.request.Request(f"{_http_base(base_url)}/delete_session?{q}", data=b""), timeout=15)
    except Exception:  # noqa: BLE001
        pass


class CursorBackpressure:
    """Bound evaluator audio lead using server-reported stream cursors."""

    def __init__(
        self,
        *,
        chunk_samples: int,
        max_unacked_chunks: int = 0,
        stall_timeout_sec: float = 60.0,
    ) -> None:
        if int(chunk_samples) <= 0:
            raise ValueError("chunk_samples must be positive")
        if int(max_unacked_chunks) < 0:
            raise ValueError("max_unacked_chunks must be non-negative")
        if int(max_unacked_chunks) > 0 and float(stall_timeout_sec) <= 0:
            raise ValueError("stall_timeout_sec must be positive when cursor backpressure is enabled")
        self.chunk_samples = int(chunk_samples)
        self.max_unacked_chunks = int(max_unacked_chunks)
        self.max_unacked_samples = self.chunk_samples * self.max_unacked_chunks
        self.stall_timeout_sec = float(stall_timeout_sec)
        self.sent_samples = 0
        self.acknowledged_cursor_samples = 0
        self.max_observed_unacked_samples = 0
        self.wait_count = 0
        self.wait_seconds = 0.0
        self.timeout_release_count = 0
        self.cursor_meta_event_count = 0
        self.cursor_advance_event_count = 0
        self.partial_cursor_event_count = 0
        self.status_cursor_event_count = 0
        self.timeout_release_samples = 0
        self.cursor_barrier_wait_count = 0
        self.cursor_barrier_wait_seconds = 0.0
        self.cursor_barrier_timeout_count = 0
        self._condition = asyncio.Condition()

    @property
    def enabled(self) -> bool:
        return self.max_unacked_chunks > 0

    def _unacked_samples(self) -> int:
        return max(0, self.sent_samples - self.acknowledged_cursor_samples)

    def record_sent(self, sample_count: int, *, timeout_released: bool = False) -> None:
        samples = max(0, int(sample_count))
        self.sent_samples += samples
        if timeout_released:
            self.timeout_release_samples += samples
        self.max_observed_unacked_samples = max(
            self.max_observed_unacked_samples,
            self._unacked_samples(),
        )

    async def wait_to_send(self, sample_count: int) -> bool:
        if not self.enabled:
            return False
        next_samples = max(0, int(sample_count))
        loop = asyncio.get_running_loop()
        wait_started: Optional[float] = None
        timeout_released = False
        deadline = loop.time() + self.stall_timeout_sec
        async with self._condition:
            while self.sent_samples + next_samples - self.acknowledged_cursor_samples > self.max_unacked_samples:
                if wait_started is None:
                    wait_started = loop.time()
                    self.wait_count += 1
                remaining = deadline - loop.time()
                if remaining <= 0:
                    self.timeout_release_count += 1
                    timeout_released = True
                    break
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    self.timeout_release_count += 1
                    timeout_released = True
                    break
        if wait_started is not None:
            self.wait_seconds += max(0.0, loop.time() - wait_started)
        return timeout_released

    async def observe_event(self, event_type: str, meta: Any) -> None:
        normalized_type = str(event_type or "").lower()
        if normalized_type not in {"partial", "status"} or not isinstance(meta, dict):
            return
        if "cursor_samples" not in meta:
            return
        try:
            cursor_samples = int(meta["cursor_samples"])
        except (TypeError, ValueError):
            return
        if cursor_samples < 0:
            return
        async with self._condition:
            self.cursor_meta_event_count += 1
            if normalized_type == "partial":
                self.partial_cursor_event_count += 1
            else:
                self.status_cursor_event_count += 1
            if cursor_samples > self.acknowledged_cursor_samples:
                self.acknowledged_cursor_samples = cursor_samples
                self.cursor_advance_event_count += 1
                self._condition.notify_all()

    async def wait_until_cursor(
        self,
        target_cursor_samples: int,
        *,
        timeout_sec: float,
        reason: str,
    ) -> None:
        target = max(0, int(target_cursor_samples))
        if target <= self.acknowledged_cursor_samples:
            return
        if float(timeout_sec) <= 0:
            raise ValueError("cursor barrier timeout must be positive")
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + float(timeout_sec)
        self.cursor_barrier_wait_count += 1
        async with self._condition:
            while self.acknowledged_cursor_samples < target:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    self.cursor_barrier_timeout_count += 1
                    self.cursor_barrier_wait_seconds += max(0.0, loop.time() - started)
                    raise RuntimeError(
                        f"{reason} timed out waiting for server cursor {target}; "
                        f"latest cursor is {self.acknowledged_cursor_samples}"
                    )
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    self.cursor_barrier_timeout_count += 1
                    self.cursor_barrier_wait_seconds += max(0.0, loop.time() - started)
                    raise RuntimeError(
                        f"{reason} timed out waiting for server cursor {target}; "
                        f"latest cursor is {self.acknowledged_cursor_samples}"
                    ) from exc
        self.cursor_barrier_wait_seconds += max(0.0, loop.time() - started)

    def snapshot(self) -> Dict[str, Any]:
        final_unacked = self._unacked_samples()
        return {
            "enabled": self.enabled,
            "max_unacked_chunks": self.max_unacked_chunks,
            "max_unacked_samples": self.max_unacked_samples,
            "stall_timeout_sec": self.stall_timeout_sec,
            "sent_samples": self.sent_samples,
            "acknowledged_cursor_samples": self.acknowledged_cursor_samples,
            "final_unacked_samples": final_unacked,
            "final_unacked_chunks": round(final_unacked / self.chunk_samples, 4),
            "max_observed_unacked_samples": self.max_observed_unacked_samples,
            "max_observed_unacked_chunks": round(
                self.max_observed_unacked_samples / self.chunk_samples,
                4,
            ),
            "wait_count": self.wait_count,
            "wait_seconds": round(self.wait_seconds, 6),
            "timeout_release_count": self.timeout_release_count,
            "cursor_meta_event_count": self.cursor_meta_event_count,
            "cursor_advance_event_count": self.cursor_advance_event_count,
            "partial_cursor_event_count": self.partial_cursor_event_count,
            "status_cursor_event_count": self.status_cursor_event_count,
            "timeout_release_samples": self.timeout_release_samples,
            "cursor_barrier_wait_count": self.cursor_barrier_wait_count,
            "cursor_barrier_wait_seconds": round(self.cursor_barrier_wait_seconds, 6),
            "cursor_barrier_timeout_count": self.cursor_barrier_timeout_count,
        }


async def run_streaming_eval(
    *,
    base_url: str,
    language_pair: str,
    preset: str,
    blocks: Sequence[AudioBlock],
    spans: Sequence[AudioBlockSpan],
    chunk_samples: int,
    feed_sleep: float,
    latency_multiplier: int,
    max_seconds_per_item: float = 0.0,
    idle_timeout_sec: float = 60.0,
    idle_timeouts_after_eof: int = 2,
    require_router_meta: bool = True,
    max_unacked_chunks: int = 0,
    backpressure_stall_timeout_sec: float = 60.0,
    oracle_preset_map: Optional[Dict[str, str]] = None,
    oracle_switch_timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    import websockets  # noqa: PLC0415 - optional dependency for CLI path only

    oracle_map = {
        str(domain).strip().lower(): str(mapped_preset).strip()
        for domain, mapped_preset in (oracle_preset_map or {}).items()
        if str(domain).strip() and str(mapped_preset).strip()
    }
    validate_oracle_preset_map(oracle_map, spans)
    first_span = next(
        (
            span
            for span in spans
            if int(
                getattr(
                    span,
                    "sample_count",
                    int(span.end_sample) - int(span.start_sample),
                )
            )
            > 0
        ),
        None,
    )
    initial_oracle_domain = str(first_span.expected_domain).strip().lower() if first_span else ""
    effective_initial_preset = oracle_map.get(initial_oracle_domain, preset)
    info = init_session(base_url, language_pair, effective_initial_preset, latency_multiplier)
    session_id = str(info["session_id"])
    records: List[Dict[str, Any]] = []
    events_seen = 0
    oracle_switches: List[Dict[str, Any]] = []
    session_started_s = asyncio.get_running_loop().time()
    pacing = CursorBackpressure(
        chunk_samples=chunk_samples,
        max_unacked_chunks=max_unacked_chunks,
        stall_timeout_sec=backpressure_stall_timeout_sec,
    )
    try:
        if oracle_map:
            active_preset = str(info.get("active_glossary_preset") or "")
            if active_preset != effective_initial_preset:
                raise RuntimeError(
                    f"blockwise oracle requested initial preset {effective_initial_preset!r} "
                    f"but server activated {active_preset or '<missing>'!r}"
                )
            if info.get("auto_glossary_enabled") is True:
                raise RuntimeError(
                    "blockwise oracle initial session unexpectedly left automatic routing enabled"
                )
            oracle_switches.append(
                {
                    "action": "initial",
                    "switch_index": 0,
                    "before_chunk_index": 1,
                    "from_domain": "",
                    "from_preset": "",
                    "to_domain": initial_oracle_domain,
                    "to_preset": effective_initial_preset,
                    "prior_sent_samples": 0,
                    "future_cursor_samples": 0,
                    "acknowledged_cursor_samples": 0,
                    "session_updated": True,
                    "server_active_domain": str(info.get("active_domain") or ""),
                    "server_active_preset": active_preset,
                    "success": True,
                }
            )
        async with websockets.connect(f"{_ws_base(base_url)}/wss/{session_id}?event_format=json", max_size=None) as ws:
            initial_message = await ws.recv()
            try:
                initial_event = json.loads(initial_message)
            except (TypeError, json.JSONDecodeError):
                initial_event = {}
            if isinstance(initial_event, dict):
                await pacing.observe_event(str(initial_event.get("type") or ""), initial_event.get("meta"))

            async def feed() -> None:
                # Server websocket input follows the existing eval_auto_glossary
                # client convention: raw float32 PCM samples in [-1, 1).
                chunks = iter_pcm_chunks(
                    blocks,
                    chunk_samples=chunk_samples,
                    max_seconds_per_item=max_seconds_per_item,
                )
                current_domain = initial_oracle_domain
                current_preset = effective_initial_preset
                for plan in iter_oracle_chunk_plan(
                    chunks,
                    spans=spans,
                    oracle_preset_map=oracle_map,
                ):
                    if oracle_map and (
                        plan.expected_domain != current_domain
                        or plan.glossary_preset != current_preset
                    ):
                        await pacing.wait_until_cursor(
                            plan.start_sample,
                            timeout_sec=oracle_switch_timeout_sec,
                            reason=(
                                f"oracle switch {current_domain or '<none>'} -> "
                                f"{plan.expected_domain or '<none>'} before chunk {plan.chunk_index}"
                            ),
                        )
                        started = asyncio.get_running_loop().time()
                        result = await asyncio.to_thread(
                            switch_session_glossary,
                            base_url,
                            session_id,
                            plan.glossary_preset,
                            language_pair=language_pair,
                            timeout_sec=oracle_switch_timeout_sec,
                        )
                        switch_seconds = max(0.0, asyncio.get_running_loop().time() - started)
                        oracle_switches.append(
                            {
                                "action": "switch",
                                "switch_index": len(oracle_switches),
                                "before_chunk_index": plan.chunk_index,
                                "from_domain": current_domain,
                                "from_preset": current_preset,
                                "to_domain": plan.expected_domain,
                                "to_preset": plan.glossary_preset,
                                "prior_sent_samples": plan.start_sample,
                                "future_cursor_samples": plan.future_cursor_samples,
                                "acknowledged_cursor_samples": pacing.acknowledged_cursor_samples,
                                "session_updated": bool(result.get("session_updated")),
                                "server_active_domain": str(result.get("active_domain") or ""),
                                "server_active_preset": str(result.get("active_glossary_preset") or ""),
                                "switch_seconds": round(switch_seconds, 6),
                                "success": True,
                            }
                        )
                        current_domain = plan.expected_domain
                        current_preset = plan.glossary_preset
                    timeout_released = await pacing.wait_to_send(int(plan.chunk.shape[0]))
                    await ws.send(plan.chunk.tobytes())
                    pacing.record_sent(int(plan.chunk.shape[0]), timeout_released=timeout_released)
                    if float(feed_sleep) > 0:
                        await asyncio.sleep(float(feed_sleep))
                await ws.send("EOF")

            feed_task = asyncio.create_task(feed())
            idle_after_eof = 0
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=max(1.0, float(idle_timeout_sec)))
                except asyncio.TimeoutError:
                    if feed_task.done():
                        idle_after_eof += 1
                        if idle_after_eof >= max(1, int(idle_timeouts_after_eof)):
                            break
                    continue
                event = json.loads(msg)
                event_type = str(event.get("type") or "").lower()
                event_text = str(event.get("text") or "")
                await pacing.observe_event(event_type, event.get("meta"))
                if event_type == "status" and event_text.startswith("PROCESSING_COMPLETE") and feed_task.done():
                    continue
                if event_type != "partial":
                    continue
                idle_after_eof = 0
                events_seen += 1
                records.append(
                    extract_record(
                        event,
                        event_idx=events_seen,
                        spans=spans,
                        require_router_meta=require_router_meta,
                        emitted_wall_s=(
                            asyncio.get_running_loop().time() - session_started_s
                        ),
                    )
                )
            await feed_task
    finally:
        delete_session(base_url, session_id)

    return {
        "session": {
            "session_id": session_id,
            "initial_active_glossary": info.get("active_glossary_preset"),
            "requested_initial_preset": preset,
            "effective_initial_preset": effective_initial_preset,
            "preset_terms": info.get("preset_terms"),
        },
        "oracle": {
            "enabled": bool(oracle_map),
            "preset_map": dict(oracle_map),
            "initial_domain": initial_oracle_domain if oracle_map else "",
            "initial_preset": effective_initial_preset if oracle_map else "",
            "switch_count": max(0, len(oracle_switches) - 1),
            "all_switches_succeeded": all(item.get("success") is True for item in oracle_switches),
            "switches": oracle_switches,
        },
        "pacing": pacing.snapshot(),
        "records": records,
    }


def extract_record(
    event: Dict[str, Any],
    *,
    event_idx: int,
    spans: Sequence[AudioBlockSpan],
    require_router_meta: bool = True,
    emitted_wall_s: Optional[float] = None,
) -> Dict[str, Any]:
    meta = event.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError("server partial event missing dict meta")
    validate_partial_meta_schema(meta, require_router_meta=require_router_meta)
    cursor_samples = int(meta["cursor_samples"])
    expected = expected_domain_at(spans, cursor_samples)
    topic = meta["topic"]
    router = meta.get("topic_router") or {}
    router_evidence = router.get("evidence") or {}
    if not isinstance(router_evidence, dict):
        router_evidence = {}
    slice_selection = router_evidence.get("slice_selection") or {}
    if not isinstance(slice_selection, dict):
        slice_selection = {}
    probe_scores = meta.get("domain_probe_scores") or {}
    target_domain = str(router.get("to_domain") or "")
    if not target_domain:
        target_domain = domain_for_preset(str(router.get("to_preset") or ""))
    return {
        "event_idx": int(event_idx),
        "cursor_samples": cursor_samples,
        "start_sample": int(meta.get("start_sample") or 0),
        "cursor_s": round(cursor_samples / TARGET_SAMPLE_RATE, 3),
        "emitted_wall_s": (
            round(float(emitted_wall_s), 6) if emitted_wall_s is not None else None
        ),
        "expected_domain": expected,
        "active_domain": str(topic["active_domain"]),
        "active_preset": str(topic["active_glossary_preset"]),
        "switch_count": int(topic["switch_count"]),
        "router_action": str(router.get("action") or ""),
        "router_reason": str(router.get("reason") or ""),
        "router_target_domain": target_domain,
        "router_confidence": router.get("confidence"),
        "router_margin": router.get("margin"),
        "selected_slice_presets": [
            str(item)
            for item in (slice_selection.get("selected_slice_presets") or [])
            if str(item)
        ],
        "selected_slice_count": int(slice_selection.get("selected_slice_count") or 0),
        "selected_term_count": int(slice_selection.get("selected_term_count") or 0),
        "domain_probe_top_domain": domain_probe_top_domain(probe_scores),
        "domain_probe_scores": probe_scores,
        "router_text_source": str(meta["router_text_source"]),
        "router_text_chars": int(meta.get("router_text_chars") or 0),
        "prompt_reference_count": int(meta["prompt_reference_count"]),
        "fixed_prompt_k": int(meta["fixed_prompt_k"]),
        "candidate_pool_count": int(meta["candidate_pool_count"]),
        "retrieval_candidate_cost": dict(meta.get("retrieval_candidate_cost") or {}),
        "references": [dict(item) for item in (meta.get("references") or []) if isinstance(item, dict)],
        "retrieve_s": meta.get("retrieve_s"),
        "domain_probe_s": meta.get("domain_probe_s"),
        "text": strip_tags(str(event.get("text") or "")),
        "text_preview": preview(str(event.get("text") or "")),
    }


def validate_partial_meta_schema(meta: Dict[str, Any], *, require_router_meta: bool = True) -> None:
    required = [
        "cursor_samples",
        "topic",
        "router_text_source",
        "prompt_reference_count",
        "fixed_prompt_k",
        "candidate_pool_count",
    ]
    if require_router_meta:
        required.extend(["topic_router", "domain_probe_scores"])
    missing = [key for key in required if key not in meta]
    if missing:
        raise RuntimeError(f"server partial meta missing required key(s): {', '.join(missing)}")
    if not isinstance(meta["topic"], dict):
        raise RuntimeError("server partial meta.topic must be a dict")
    if require_router_meta and not isinstance(meta["topic_router"], dict):
        raise RuntimeError("server partial meta.topic_router must be a dict")
    if require_router_meta and not isinstance(meta["domain_probe_scores"], dict):
        raise RuntimeError("server partial meta.domain_probe_scores must be a dict")
    topic_required = ("active_domain", "active_glossary_preset", "switch_count")
    topic_missing = [key for key in topic_required if key not in meta["topic"]]
    if topic_missing:
        raise RuntimeError(f"server partial meta.topic missing required key(s): {', '.join(topic_missing)}")


def summarize_run(
    *,
    schedule_name: str,
    preset: str,
    spans: Sequence[AudioBlockSpan],
    records: Sequence[Dict[str, Any]],
    chunk_samples: int,
    max_switch_events: int,
    pacing: Optional[Dict[str, Any]] = None,
    oracle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expected_records = [item for item in records if item.get("expected_domain")]
    active_correct = sum(1 for item in expected_records if item.get("active_domain") == item.get("expected_domain"))
    probe_seen = [item for item in expected_records if item.get("domain_probe_top_domain")]
    probe_correct = sum(1 for item in probe_seen if item.get("domain_probe_top_domain") == item.get("expected_domain"))
    transitions = domain_transitions(spans, records, max_switch_events=max_switch_events)
    grace_events = transition_grace_events(transitions, max_switch_events=max_switch_events)
    steady_records = [item for item in expected_records if int(item.get("event_idx") or 0) not in grace_events]
    steady_correct = sum(1 for item in steady_records if item.get("active_domain") == item.get("expected_domain"))
    wrong_switches = [
        item for item in records
        if item.get("router_action") == "switch"
        and item.get("expected_domain")
        and item.get("router_target_domain")
        and item.get("router_target_domain") != item.get("expected_domain")
    ]
    retrieve_values = [float(item["retrieve_s"]) for item in records if item.get("retrieve_s") is not None]
    probe_values = [float(item["domain_probe_s"]) for item in records if item.get("domain_probe_s") is not None]
    max_switch_count = max((int(item.get("switch_count") or 0) for item in records), default=0)
    summary = {
        "schedule": schedule_name,
        "preset": preset,
        "block_count": len(spans),
        "audio_seconds": round((spans[-1].end_sample if spans else 0) / TARGET_SAMPLE_RATE, 3),
        "event_count": len(records),
        "chunk_samples": int(chunk_samples),
        "chunk_seconds": round(int(chunk_samples) / TARGET_SAMPLE_RATE, 3),
        "domain_transition_count": len(transitions),
        "switch_count": max_switch_count,
        "max_switch_events": int(max_switch_events),
        "active_domain_accuracy": round(active_correct / len(expected_records), 4) if expected_records else 0.0,
        "steady_state_active_domain_accuracy": round(steady_correct / len(steady_records), 4) if steady_records else 0.0,
        "steady_state_mismatch_count": sum(1 for item in steady_records if item.get("active_domain") != item.get("expected_domain")),
        "probe_top_accuracy": round(probe_correct / len(probe_seen), 4) if probe_seen else None,
        "probe_seen_events": len(probe_seen),
        "wrong_switch_count": len(wrong_switches),
        "router_text_sources": dict(Counter(str(item.get("router_text_source") or "") for item in records)),
        "active_domains": dict(Counter(str(item.get("active_domain") or "") for item in records)),
        "probe_top_domains": dict(Counter(str(item.get("domain_probe_top_domain") or "") for item in records)),
        "retrieve_p50_ms": round(percentile(retrieve_values, 50) * 1000.0, 2) if retrieve_values else None,
        "retrieve_p95_ms": round(percentile(retrieve_values, 95) * 1000.0, 2) if retrieve_values else None,
        "domain_probe_p50_ms": round(percentile(probe_values, 50) * 1000.0, 2) if probe_values else None,
        "domain_probe_p95_ms": round(percentile(probe_values, 95) * 1000.0, 2) if probe_values else None,
        "transition_pass": all(item["passed"] for item in transitions),
        "regression_pass": bool(
            all(item["passed"] for item in transitions)
            and not wrong_switches
            and all(item.get("active_domain") == item.get("expected_domain") for item in steady_records)
        ),
    }
    if pacing is not None:
        summary["backpressure"] = dict(pacing)
    if oracle is not None:
        summary["oracle"] = {
            "enabled": bool(oracle.get("enabled")),
            "initial_domain": str(oracle.get("initial_domain") or ""),
            "initial_preset": str(oracle.get("initial_preset") or ""),
            "switch_count": int(oracle.get("switch_count") or 0),
            "all_switches_succeeded": bool(oracle.get("all_switches_succeeded")),
        }
    return summary


def domain_transitions(
    spans: Sequence[AudioBlockSpan],
    records: Sequence[Dict[str, Any]],
    *,
    max_switch_events: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    previous: Optional[AudioBlockSpan] = None
    for span in spans:
        if previous is not None and span.expected_domain != previous.expected_domain:
            first_event = None
            for record in records:
                cursor = int(record.get("cursor_samples") or 0)
                if cursor < span.start_sample:
                    continue
                if record.get("active_domain") == span.expected_domain:
                    first_event = record
                    break
            latency_events = None
            latency_s = None
            if first_event is not None:
                first_after = next(
                    (
                        item for item in records
                        if int(item.get("cursor_samples") or 0) >= span.start_sample
                    ),
                    None,
                )
                if first_after is not None:
                    latency_events = int(first_event["event_idx"]) - int(first_after["event_idx"]) + 1
                latency_s = max(0.0, (int(first_event["cursor_samples"]) - span.start_sample) / TARGET_SAMPLE_RATE)
            out.append(
                {
                    "from_block_index": previous.block_index,
                    "to_block_index": span.block_index,
                    "from_item_id": previous.item_id,
                    "to_item_id": span.item_id,
                    "from_domain": previous.expected_domain,
                    "to_domain": span.expected_domain,
                    "boundary_sample": span.start_sample,
                    "boundary_s": round(span.start_sample / TARGET_SAMPLE_RATE, 3),
                    "first_target_active_event": first_event.get("event_idx") if first_event else None,
                    "first_target_active_s": first_event.get("cursor_s") if first_event else None,
                    "latency_events": latency_events,
                    "latency_s": round(latency_s, 3) if latency_s is not None else None,
                    "max_switch_events": int(max_switch_events),
                    "passed": bool(latency_events is not None and latency_events <= int(max_switch_events)),
                }
            )
        previous = span
    return out


def transition_grace_events(transitions: Sequence[Dict[str, Any]], *, max_switch_events: int) -> set[int]:
    out: set[int] = set()
    for transition in transitions:
        first_event = transition.get("first_target_active_event")
        if first_event is None:
            continue
        start = max(1, int(first_event) - max(0, int(max_switch_events)) + 1)
        for event_idx in range(start, int(first_event) + 1):
            out.add(event_idx)
    return out


def expected_domain_at(spans: Sequence[AudioBlockSpan], cursor_samples: int) -> str:
    probe = max(0, int(cursor_samples) - 1)
    for span in spans:
        if span.start_sample <= probe < span.end_sample:
            return span.expected_domain
    return spans[-1].expected_domain if spans else ""


def domain_probe_top_domain(scores: Dict[str, Any]) -> str:
    rows: List[Tuple[float, str]] = []
    for key, value in (scores or {}).items():
        domain = str(key)
        score = 0.0
        if isinstance(value, dict):
            domain = str(value.get("domain") or key)
            try:
                score = max(float(value.get("top_score") or 0.0), float(value.get("mean_topk_score") or 0.0))
            except (TypeError, ValueError):
                score = 0.0
        else:
            try:
                score = float(value or 0.0)
            except (TypeError, ValueError):
                score = 0.0
        rows.append((score, domain))
    rows.sort(reverse=True)
    return rows[0][1] if rows and rows[0][0] > 0.0 else ""


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    idx = min(len(ordered) - 1, max(0, int(round((float(pct) / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def write_markdown(payload: Dict[str, Any], out_path: str) -> None:
    summary = payload["summary"]
    lines = [
        "# Mixed Audio Auto-Glossary Switch Eval",
        "",
        "This streams real ACL/medicine audio through the JSON WS server and scores runtime router metadata.",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "schedule",
        "preset",
        "block_count",
        "audio_seconds",
        "event_count",
        "domain_transition_count",
        "switch_count",
        "active_domain_accuracy",
        "steady_state_active_domain_accuracy",
        "probe_top_accuracy",
        "wrong_switch_count",
        "regression_pass",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    pacing = summary.get("backpressure") or {}
    if isinstance(pacing, dict) and pacing:
        for key in (
            "enabled",
            "max_unacked_chunks",
            "max_observed_unacked_chunks",
            "wait_count",
            "wait_seconds",
            "timeout_release_count",
        ):
            lines.append(f"- backpressure_{key}: `{pacing.get(key)}`")
    oracle = summary.get("oracle") or {}
    if isinstance(oracle, dict) and oracle.get("enabled"):
        for key in (
            "enabled",
            "initial_domain",
            "initial_preset",
            "switch_count",
            "all_switches_succeeded",
        ):
            lines.append(f"- oracle_{key}: `{oracle.get(key)}`")
    lines.extend(["", "## Transitions", "", "| from | to | boundary_s | latency_events | latency_s | pass |", "|---|---|---:|---:|---:|---|"])
    for item in payload.get("domain_transitions", []):
        lines.append(
            "| {from_domain} | {to_domain} | {boundary_s} | {latency_events} | {latency_s} | {passed} |".format(**item)
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preview(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "..."


def strip_tags(text: str) -> str:
    return str(text or "").replace("<t>", "").replace("</t>", "")


def _medicine_audio_sort_key(path: Path) -> Any:
    stem = path.stem[:-3] if path.stem.endswith("_v2") else path.stem
    return (len(stem), stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--acl-root", default=DEFAULT_ACL_ROOT)
    ap.add_argument("--medicine-audio-dir", default=DEFAULT_MEDICINE_AUDIO_DIR)
    ap.add_argument("--acl-items", type=int, default=5)
    ap.add_argument("--medicine-items", type=int, default=5)
    ap.add_argument("--medicine-ids", default="")
    ap.add_argument("--max-acl-segs-per-talk", type=int, default=0)
    ap.add_argument("--max-seconds-per-item", type=float, default=0.0)
    ap.add_argument(
        "--schedule",
        choices=("alternating", "random", "acl_then_medicine", "medicine_then_acl"),
        default="alternating",
    )
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--preset", default="auto_working")
    ap.add_argument(
        "--oracle-preset-map",
        default="",
        help=(
            "enable blockwise Oracle routing with comma-separated DOMAIN=PRESET entries; "
            "the session starts on the first block's mapped preset and switches through "
            "/glossary/build before each cross-domain chunk"
        ),
    )
    ap.add_argument(
        "--oracle-switch-timeout-sec",
        type=float,
        default=120.0,
        help="timeout for draining the prior cursor and applying each Oracle preset switch",
    )
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--latency-multiplier", type=int, default=2)
    ap.add_argument("--base-segment-sec", type=float, default=0.96)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--feed-sleep", type=float, default=0.0)
    ap.add_argument(
        "--max-unacked-chunks",
        type=int,
        default=0,
        help=(
            "pause audio feed before it exceeds this many nominal chunks beyond the latest "
            "partial/status cursor; 0 keeps legacy fixed-rate feeding"
        ),
    )
    ap.add_argument(
        "--backpressure-stall-timeout-sec",
        type=float,
        default=60.0,
        help=(
            "release at most one waiting chunk per timeout window so silent/no-meta streams "
            "make bounded progress without disabling backpressure"
        ),
    )
    ap.add_argument("--idle-timeout-sec", type=float, default=60.0)
    ap.add_argument("--idle-timeouts-after-eof", type=int, default=2)
    ap.add_argument("--max-switch-events", type=int, default=3)
    ap.add_argument("--max-switch-seconds", type=float, default=0.0)
    ap.add_argument("--allow-missing-router-meta", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()
    if args.max_unacked_chunks < 0:
        ap.error("--max-unacked-chunks must be non-negative")
    if args.max_unacked_chunks > 0 and args.backpressure_stall_timeout_sec <= 0:
        ap.error("--backpressure-stall-timeout-sec must be positive when backpressure is enabled")
    if args.oracle_switch_timeout_sec <= 0:
        ap.error("--oracle-switch-timeout-sec must be positive")
    try:
        oracle_preset_map = parse_oracle_preset_map(args.oracle_preset_map)
    except ValueError as exc:
        ap.error(str(exc))

    acl_blocks = read_acl_audio_blocks(
        args.acl_root,
        limit_items=args.acl_items,
        max_segs_per_talk=args.max_acl_segs_per_talk,
    )
    medicine_ids = [item.strip() for item in str(args.medicine_ids or "").split(",") if item.strip()]
    medicine_limit = len(medicine_ids) if medicine_ids else args.medicine_items
    medicine_blocks = read_medicine_audio_blocks(
        args.medicine_audio_dir,
        limit_items=medicine_limit,
        medicine_ids=medicine_ids,
    )
    blocks = build_schedule(acl_blocks, medicine_blocks, schedule=args.schedule, seed=args.seed)
    if not blocks:
        raise SystemExit("no audio blocks found")
    chunk_samples = resolve_chunk_samples(args.chunk, args.base_segment_sec, args.latency_multiplier)
    max_switch_events = resolve_max_switch_events(args.max_switch_events, args.max_switch_seconds, chunk_samples)
    spans = build_spans(blocks, max_seconds_per_item=args.max_seconds_per_item)
    try:
        validate_oracle_preset_map(oracle_preset_map, spans)
    except ValueError as exc:
        ap.error(str(exc))
    first_domain = next(
        (
            str(span.expected_domain).strip().lower()
            for span in spans
            if span.sample_count > 0
        ),
        "",
    )
    effective_initial_preset = oracle_preset_map.get(first_domain, args.preset)
    payload: Dict[str, Any] = {
        "config": {
            "schedule": args.schedule,
            "seed": args.seed,
            "preset": args.preset,
            "effective_initial_preset": effective_initial_preset,
            "oracle_enabled": bool(oracle_preset_map),
            "oracle_preset_map": dict(oracle_preset_map),
            "oracle_switch_timeout_sec": args.oracle_switch_timeout_sec,
            "language_pair": args.language_pair,
            "latency_multiplier": args.latency_multiplier,
            "chunk_samples": chunk_samples,
            "chunk_seconds": round(chunk_samples / TARGET_SAMPLE_RATE, 3),
            "feed_sleep": args.feed_sleep,
            "max_unacked_chunks": args.max_unacked_chunks,
            "max_unacked_samples": args.max_unacked_chunks * chunk_samples,
            "backpressure_enabled": args.max_unacked_chunks > 0,
            "backpressure_stall_timeout_sec": args.backpressure_stall_timeout_sec,
            "idle_timeout_sec": args.idle_timeout_sec,
            "idle_timeouts_after_eof": args.idle_timeouts_after_eof,
            "max_seconds_per_item": args.max_seconds_per_item,
            "max_acl_segs_per_talk": args.max_acl_segs_per_talk,
            "medicine_ids": medicine_ids,
            "base_url": args.base_url,
            "max_switch_events": max_switch_events,
            "max_switch_seconds": args.max_switch_seconds,
        },
        "blocks": [block.__dict__ for block in blocks],
        "block_spans": [span.__dict__ for span in spans],
        "summary": {},
        "domain_transitions": [],
        "oracle": {
            "enabled": bool(oracle_preset_map),
            "preset_map": dict(oracle_preset_map),
            "initial_domain": first_domain if oracle_preset_map else "",
            "initial_preset": effective_initial_preset if oracle_preset_map else "",
            "switch_count": 0,
            "all_switches_succeeded": True,
            "switches": [],
        },
        "records": [],
    }
    if not args.dry_run:
        stream_payload = asyncio.run(
            run_streaming_eval(
                base_url=args.base_url,
                language_pair=args.language_pair,
                preset=args.preset,
                blocks=blocks,
                spans=spans,
                chunk_samples=chunk_samples,
                feed_sleep=args.feed_sleep,
                latency_multiplier=args.latency_multiplier,
                max_seconds_per_item=args.max_seconds_per_item,
                idle_timeout_sec=args.idle_timeout_sec,
                idle_timeouts_after_eof=args.idle_timeouts_after_eof,
                require_router_meta=not args.allow_missing_router_meta,
                max_unacked_chunks=args.max_unacked_chunks,
                backpressure_stall_timeout_sec=args.backpressure_stall_timeout_sec,
                oracle_preset_map=oracle_preset_map,
                oracle_switch_timeout_sec=args.oracle_switch_timeout_sec,
            )
        )
        payload["session"] = stream_payload["session"]
        payload["pacing"] = stream_payload["pacing"]
        payload["oracle"] = stream_payload["oracle"]
        payload["records"] = stream_payload["records"]
        payload["domain_transitions"] = domain_transitions(
            spans,
            payload["records"],
            max_switch_events=max_switch_events,
        )
        payload["summary"] = summarize_run(
            schedule_name=args.schedule,
            preset=args.preset,
            spans=spans,
            records=payload["records"],
            chunk_samples=chunk_samples,
            max_switch_events=max_switch_events,
            pacing=stream_payload["pacing"],
            oracle=stream_payload["oracle"],
        )
    else:
        payload["summary"] = {
            "schedule": args.schedule,
            "preset": args.preset,
            "block_count": len(spans),
            "audio_seconds": round((spans[-1].end_sample if spans else 0) / TARGET_SAMPLE_RATE, 3),
            "dry_run": True,
            "backpressure": {
                "enabled": args.max_unacked_chunks > 0,
                "max_unacked_chunks": args.max_unacked_chunks,
                "max_unacked_samples": args.max_unacked_chunks * chunk_samples,
                "stall_timeout_sec": args.backpressure_stall_timeout_sec,
            },
            "oracle": {
                "enabled": bool(oracle_preset_map),
                "initial_domain": first_domain if oracle_preset_map else "",
                "initial_preset": effective_initial_preset if oracle_preset_map else "",
                "switch_count": 0,
                "all_switches_succeeded": True,
            },
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(payload, args.out_md)
    if not args.no_assert and payload.get("summary") and payload["summary"].get("regression_pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
