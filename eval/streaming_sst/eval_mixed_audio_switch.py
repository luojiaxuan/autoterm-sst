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
) -> List[AudioBlock]:
    if int(limit_items) <= 0:
        return []
    root = Path(audio_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"medicine audio directory not found: {root}")
    paths = sorted(root.glob("sample_*_v2/*_v2.wav"), key=lambda path: _medicine_audio_sort_key(path))
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
) -> Dict[str, Any]:
    import websockets  # noqa: PLC0415 - optional dependency for CLI path only

    info = init_session(base_url, language_pair, preset, latency_multiplier)
    session_id = str(info["session_id"])
    records: List[Dict[str, Any]] = []
    events_seen = 0
    try:
        async with websockets.connect(f"{_ws_base(base_url)}/wss/{session_id}?event_format=json", max_size=None) as ws:
            await ws.recv()

            async def feed() -> None:
                # Server websocket input follows the existing eval_auto_glossary
                # client convention: raw float32 PCM samples in [-1, 1).
                for chunk in iter_pcm_chunks(blocks, chunk_samples=chunk_samples, max_seconds_per_item=max_seconds_per_item):
                    await ws.send(chunk.tobytes())
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
                if event_type == "status" and event_text.startswith("PROCESSING_COMPLETE") and feed_task.done():
                    continue
                if event_type != "partial":
                    continue
                idle_after_eof = 0
                events_seen += 1
                records.append(extract_record(event, event_idx=events_seen, spans=spans, require_router_meta=require_router_meta))
            await feed_task
    finally:
        delete_session(base_url, session_id)

    return {
        "session": {
            "session_id": session_id,
            "initial_active_glossary": info.get("active_glossary_preset"),
            "preset_terms": info.get("preset_terms"),
        },
        "records": records,
    }


def extract_record(
    event: Dict[str, Any],
    *,
    event_idx: int,
    spans: Sequence[AudioBlockSpan],
    require_router_meta: bool = True,
) -> Dict[str, Any]:
    meta = event.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError("server partial event missing dict meta")
    validate_partial_meta_schema(meta, require_router_meta=require_router_meta)
    cursor_samples = int(meta["cursor_samples"])
    expected = expected_domain_at(spans, cursor_samples)
    topic = meta["topic"]
    router = meta.get("topic_router") or {}
    probe_scores = meta.get("domain_probe_scores") or {}
    target_domain = str(router.get("to_domain") or "")
    if not target_domain:
        target_domain = domain_for_preset(str(router.get("to_preset") or ""))
    return {
        "event_idx": int(event_idx),
        "cursor_samples": cursor_samples,
        "cursor_s": round(cursor_samples / TARGET_SAMPLE_RATE, 3),
        "expected_domain": expected,
        "active_domain": str(topic["active_domain"]),
        "active_preset": str(topic["active_glossary_preset"]),
        "switch_count": int(topic["switch_count"]),
        "router_action": str(router.get("action") or ""),
        "router_reason": str(router.get("reason") or ""),
        "router_target_domain": target_domain,
        "router_confidence": router.get("confidence"),
        "router_margin": router.get("margin"),
        "domain_probe_top_domain": domain_probe_top_domain(probe_scores),
        "domain_probe_scores": probe_scores,
        "router_text_source": str(meta["router_text_source"]),
        "router_text_chars": int(meta.get("router_text_chars") or 0),
        "prompt_reference_count": int(meta["prompt_reference_count"]),
        "fixed_prompt_k": int(meta["fixed_prompt_k"]),
        "candidate_pool_count": int(meta["candidate_pool_count"]),
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
    return {
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
    ap.add_argument("--max-acl-segs-per-talk", type=int, default=0)
    ap.add_argument("--max-seconds-per-item", type=float, default=0.0)
    ap.add_argument(
        "--schedule",
        choices=("alternating", "random", "acl_then_medicine", "medicine_then_acl"),
        default="alternating",
    )
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--preset", default="auto_working")
    ap.add_argument("--language-pair", default="English -> Chinese")
    ap.add_argument("--latency-multiplier", type=int, default=2)
    ap.add_argument("--base-segment-sec", type=float, default=0.96)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--feed-sleep", type=float, default=0.0)
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

    acl_blocks = read_acl_audio_blocks(
        args.acl_root,
        limit_items=args.acl_items,
        max_segs_per_talk=args.max_acl_segs_per_talk,
    )
    medicine_blocks = read_medicine_audio_blocks(args.medicine_audio_dir, limit_items=args.medicine_items)
    blocks = build_schedule(acl_blocks, medicine_blocks, schedule=args.schedule, seed=args.seed)
    if not blocks:
        raise SystemExit("no audio blocks found")
    chunk_samples = resolve_chunk_samples(args.chunk, args.base_segment_sec, args.latency_multiplier)
    max_switch_events = resolve_max_switch_events(args.max_switch_events, args.max_switch_seconds, chunk_samples)
    spans = build_spans(blocks, max_seconds_per_item=args.max_seconds_per_item)
    payload: Dict[str, Any] = {
        "config": {
            "schedule": args.schedule,
            "seed": args.seed,
            "preset": args.preset,
            "language_pair": args.language_pair,
            "latency_multiplier": args.latency_multiplier,
            "chunk_samples": chunk_samples,
            "chunk_seconds": round(chunk_samples / TARGET_SAMPLE_RATE, 3),
            "max_seconds_per_item": args.max_seconds_per_item,
            "max_acl_segs_per_talk": args.max_acl_segs_per_talk,
            "base_url": args.base_url,
            "max_switch_events": max_switch_events,
            "max_switch_seconds": args.max_switch_seconds,
        },
        "blocks": [block.__dict__ for block in blocks],
        "block_spans": [span.__dict__ for span in spans],
        "summary": {},
        "domain_transitions": [],
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
            )
        )
        payload["session"] = stream_payload["session"]
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
        )
    else:
        payload["summary"] = {
            "schedule": args.schedule,
            "preset": args.preset,
            "block_count": len(spans),
            "audio_seconds": round((spans[-1].end_sample if spans else 0) / TARGET_SAMPLE_RATE, 3),
            "dry_run": True,
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
