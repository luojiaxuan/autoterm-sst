#!/usr/bin/env python3
"""Mixed ACL/medicine auto-glossary domain-switch benchmark.

This is a target-translation-text router diagnostic. It uses ACL Chinese target
segments and RASST medicine Chinese references as reproducible stand-ins for the
generated target-translation windows available in the E2E demo path. It does not
read source transcripts or ASR text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.topic_router import (  # noqa: E402
    DomainProbeScore,
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


DEFAULT_ACL_ROOT = "/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments"
DEFAULT_MEDICINE_TEXT_DIR = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh"
DEFAULT_MEDICINE_AUDIO_DIR = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/audio/medicine"


@dataclass(frozen=True)
class PlaylistBlock:
    item_id: str
    expected_domain: str
    windows: Sequence[str]
    text_path: str = ""
    audio_path: str = ""
    corpus: str = ""


@dataclass(frozen=True)
class BlockSpan:
    block_index: int
    item_id: str
    corpus: str
    expected_domain: str
    start_window: int
    end_window: int
    window_count: int
    text_path: str
    audio_path: str


def build_router(
    *,
    min_consistent_windows_generated_target: int = 3,
    min_confidence: float = 0.60,
) -> HybridWindowTopicRouter:
    return HybridWindowTopicRouter(
        [
            DomainSlice("nlp_core_10k", "nlp", centroid=[1.0, 0.0], index_path="mock://nlp"),
            DomainSlice("medicine_core_10k", "medicine", centroid=[0.0, 1.0], index_path="mock://medicine"),
        ],
        RouterConfig(
            warmup_sec=0.0,
            update_interval_sec=0.0,
            switch_cooldown_sec=0.0,
            min_confidence=min_confidence,
            min_margin=0.15,
            min_current_margin=0.10,
            min_consistent_windows_with_text=2,
            min_consistent_windows_generated_target=min_consistent_windows_generated_target,
            min_consistent_windows_audio_only=3,
            text_topic_weight=0.60,
            domain_probe_weight=0.25,
            speech_centroid_weight=0.10,
            metadata_prior_weight=0.05,
        ),
    )


def read_acl_blocks(
    acl_root: str,
    *,
    limit_items: int,
    windows_per_item: int,
    text_field: str = "target",
) -> List[PlaylistBlock]:
    root = Path(acl_root)
    text_path = root / ("segments.target" if text_field == "target" else "segments.source")
    meta_path = root / "segments.meta.jsonl"
    if not text_path.is_file():
        raise FileNotFoundError(f"ACL {text_field} file not found: {text_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"ACL segment metadata not found: {meta_path}")

    text_lines = _read_text_lines(text_path)
    by_talk: "OrderedDict[str, List[str]]" = OrderedDict()
    for idx, raw in enumerate(meta_path.read_text(encoding="utf-8").splitlines()):
        if idx >= len(text_lines):
            break
        if not raw.strip():
            continue
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {}
        talk_id = str(meta.get("talk") or meta.get("talk_id") or meta.get("id") or f"acl_{len(by_talk)}")
        window_text = text_lines[idx].strip()
        if window_text:
            by_talk.setdefault(talk_id, []).append(window_text)

    blocks: List[PlaylistBlock] = []
    for talk_id, lines in list(by_talk.items())[: max(0, int(limit_items))]:
        windows = _limit_windows(lines, windows_per_item)
        if windows:
            blocks.append(
                PlaylistBlock(
                    item_id=talk_id,
                    expected_domain="nlp",
                    windows=windows,
                    text_path=str(text_path),
                    corpus="acl",
                )
            )
    return blocks


def read_medicine_blocks(
    text_dir: str,
    *,
    audio_dir: str = "",
    limit_items: int,
    windows_per_item: int,
    text_field: str = "target",
) -> List[PlaylistBlock]:
    root = Path(text_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"medicine text directory not found: {root}")
    pattern = "medicine.ref.zh__medicine_*.txt" if text_field == "target" else "medicine.source_text.en__medicine_*.txt"
    paths = sorted(root.glob(pattern), key=lambda path: _medicine_sort_key(path.name))
    blocks: List[PlaylistBlock] = []
    for path in paths[: max(0, int(limit_items))]:
        medicine_id = _medicine_id(path.name)
        audio_path = _medicine_audio_path(audio_dir, medicine_id)
        windows = _limit_windows(_read_nonempty_lines(path), windows_per_item)
        if windows:
            blocks.append(
                PlaylistBlock(
                    item_id=f"medicine_{medicine_id}",
                    expected_domain="medicine",
                    windows=windows,
                    text_path=str(path),
                    audio_path=audio_path,
                    corpus="medicine",
                )
            )
    return blocks


def build_schedule(
    acl_blocks: Sequence[PlaylistBlock],
    medicine_blocks: Sequence[PlaylistBlock],
    *,
    schedule: str,
    seed: int = 20260707,
) -> List[PlaylistBlock]:
    acl = list(acl_blocks)
    medicine = list(medicine_blocks)
    if schedule == "alternating":
        out: List[PlaylistBlock] = []
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


def initial_state_for(blocks: Sequence[PlaylistBlock], *, initial_domain: str) -> RouterSessionState:
    domain = str(initial_domain or "first").strip().lower()
    if domain == "first":
        domain = blocks[0].expected_domain if blocks else "nlp"
    preset = "medicine_core_10k" if domain == "medicine" else "nlp_core_10k"
    return RouterSessionState(preset, domain, created_s=0.0)


def evaluate_playlist(
    blocks: Sequence[PlaylistBlock],
    *,
    schedule_name: str,
    router_text_source: str = "generated_target",
    probe_mode: str = "expected",
    max_switch_windows: int = 3,
    initial_domain: str = "first",
    min_consistent_windows_generated_target: int = 3,
) -> Dict[str, Any]:
    router = build_router(
        min_consistent_windows_generated_target=min_consistent_windows_generated_target,
    )
    state = initial_state_for(blocks, initial_domain=initial_domain)
    records: List[Dict[str, Any]] = []
    spans: List[BlockSpan] = []
    switch_count = 0
    global_window = 0

    for block_index, block in enumerate(blocks, start=1):
        start_window = global_window + 1
        for window_in_block, text in enumerate(block.windows, start=1):
            global_window += 1
            decision = router.observe(
                state,
                None,
                [],
                now_s=float(global_window),
                router_text=text,
                router_text_source=router_text_source,
                domain_probe_scores=probe_for_domain(block.expected_domain, mode=probe_mode) if probe_mode != "none" else {},
            )
            if decision.action == "switch":
                switch_count += 1
                apply_switch(state, decision.target_preset_id, decision.target_domain_id, float(global_window))
            records.append(
                {
                    "window": global_window,
                    "block_index": block_index,
                    "window_in_block": window_in_block,
                    "item_id": block.item_id,
                    "corpus": block.corpus,
                    "expected_domain": block.expected_domain,
                    "active_domain": state.active_domain_id,
                    "active_preset": state.active_preset_id,
                    "decision_action": decision.action,
                    "decision_target_domain": decision.target_domain_id,
                    "decision_target_preset": decision.target_preset_id,
                    "confidence": decision.confidence,
                    "margin": decision.margin,
                    "reason": decision.reason,
                    "top_domains": decision.scores,
                    "router_text_source": router_text_source,
                    "router_text_preview": _preview(text),
                }
            )
        end_window = global_window
        spans.append(
            BlockSpan(
                block_index=block_index,
                item_id=block.item_id,
                corpus=block.corpus,
                expected_domain=block.expected_domain,
                start_window=start_window,
                end_window=end_window,
                window_count=max(0, end_window - start_window + 1),
                text_path=block.text_path,
                audio_path=block.audio_path,
            )
        )

    transitions = _domain_transitions(spans, records, max_switch_windows=max_switch_windows)
    summary = _summarize(
        schedule_name=schedule_name,
        router_text_source=router_text_source,
        probe_mode=probe_mode,
        initial_domain=initial_domain,
        records=records,
        spans=spans,
        transitions=transitions,
        switch_count=switch_count,
        max_switch_windows=max_switch_windows,
        min_consistent_windows_generated_target=min_consistent_windows_generated_target,
    )
    return {
        "summary": summary,
        "blocks": [span.__dict__ for span in spans],
        "domain_transitions": transitions,
        "records": records,
    }


def probe_for_domain(domain: str, *, mode: str = "expected") -> Dict[str, DomainProbeScore]:
    target = str(domain or "nlp")
    other = "medicine" if target == "nlp" else "nlp"
    target_score = 0.90
    other_score = 0.35
    if mode == "inverted":
        target_score = 0.35
        other_score = 0.90
    elif mode == "contested":
        target_score = 0.30
        other_score = 0.28
    return {
        target: DomainProbeScore(
            domain=target,
            preset_id=f"{target}_core_10k",
            top_score=target_score,
            mean_topk_score=target_score,
            top_terms=(f"{target} probe",),
        ),
        other: DomainProbeScore(
            domain=other,
            preset_id=f"{other}_core_10k",
            top_score=other_score,
            mean_topk_score=other_score,
            top_terms=(f"{other} distractor",),
        ),
    }


def apply_switch(state: RouterSessionState, target_preset: str, target_domain: str, now_s: float) -> None:
    state.active_preset_id = target_preset
    state.active_domain_id = target_domain
    state.last_switch_s = now_s
    state.pending_preset_id = None


def write_markdown(payload: Dict[str, Any], out_path: str) -> None:
    summary = payload["summary"]
    transitions = payload["domain_transitions"]
    lines = [
        "# Mixed ACL/medicine auto-glossary switch benchmark",
        "",
        "这是 target_translation_text-window-router 诊断：窗口文本来自 ACL target 和 medicine zh reference，",
        "不使用 source transcript 或 ASR text。`probe_mode=expected` 表示 speech-domain probe guard 使用可控期望域证据。",
        "",
        "## Summary",
        "",
        f"- schedule: `{summary['schedule']}`",
        f"- router_text_source: `{summary['router_text_source']}`",
        f"- probe_mode: `{summary['probe_mode']}`",
        f"- blocks: `{summary['block_count']}`",
        f"- windows: `{summary['window_count']}`",
        f"- domain transitions: `{summary['domain_transition_count']}`",
        f"- switch count: `{summary['switch_count']}`",
        f"- domain accuracy: `{summary['domain_accuracy']}`",
        f"- steady-state accuracy: `{summary['steady_state_accuracy']}`",
        f"- max switch latency windows: `{summary['max_observed_switch_latency_windows']}`",
        f"- pass: `{summary['regression_pass']}`",
        "",
        "## Transitions",
        "",
        "| from | to | start_window | latency_windows | pass |",
        "|---|---|---:|---:|---|",
    ]
    for item in transitions:
        lines.append(
            "| {from_domain} | {to_domain} | {start_window} | {latency_windows} | {passed} |".format(
                from_domain=item["from_domain"],
                to_domain=item["to_domain"],
                start_window=item["start_window"],
                latency_windows=item["latency_windows"],
                passed=item["passed"],
            )
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(
    *,
    schedule_name: str,
    router_text_source: str,
    probe_mode: str,
    initial_domain: str,
    records: Sequence[Dict[str, Any]],
    spans: Sequence[BlockSpan],
    transitions: Sequence[Dict[str, Any]],
    switch_count: int,
    max_switch_windows: int,
    min_consistent_windows_generated_target: int,
) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for item in records if item["active_domain"] == item["expected_domain"])
    allowed_transition_windows = _allowed_transition_windows(transitions, max_switch_windows)
    steady_records = [item for item in records if int(item["window"]) not in allowed_transition_windows]
    steady_correct = sum(1 for item in steady_records if item["active_domain"] == item["expected_domain"])
    steady_mismatches = [
        {
            "window": item["window"],
            "item_id": item["item_id"],
            "expected_domain": item["expected_domain"],
            "active_domain": item["active_domain"],
            "reason": item["reason"],
        }
        for item in steady_records
        if item["active_domain"] != item["expected_domain"]
    ]
    wrong_switches = [
        item for item in records
        if item["decision_action"] == "switch" and item["decision_target_domain"] != item["expected_domain"]
    ]
    transition_pass = all(item["passed"] for item in transitions)
    regression_pass = bool(transition_pass and not steady_mismatches and not wrong_switches)
    latencies = [int(item["latency_windows"]) for item in transitions if item["latency_windows"] is not None]
    return {
        "schedule": schedule_name,
        "router_text_source": router_text_source,
        "probe_mode": probe_mode,
        "initial_domain": initial_domain,
        "block_count": len(spans),
        "window_count": total,
        "domain_transition_count": len(transitions),
        "switch_count": int(switch_count),
        "max_switch_windows": int(max_switch_windows),
        "min_consistent_windows_generated_target": int(min_consistent_windows_generated_target),
        "domain_accuracy": round(correct / total, 4) if total else 0.0,
        "steady_state_accuracy": round(steady_correct / len(steady_records), 4) if steady_records else 0.0,
        "steady_state_mismatch_count": len(steady_mismatches),
        "wrong_switch_count": len(wrong_switches),
        "max_observed_switch_latency_windows": max(latencies) if latencies else None,
        "transition_pass": transition_pass,
        "regression_pass": regression_pass,
        "steady_state_mismatches": steady_mismatches[:20],
        "wrong_switches": wrong_switches[:20],
    }


def _domain_transitions(
    spans: Sequence[BlockSpan],
    records: Sequence[Dict[str, Any]],
    *,
    max_switch_windows: int,
) -> List[Dict[str, Any]]:
    by_window = {int(item["window"]): item for item in records}
    out: List[Dict[str, Any]] = []
    previous: Optional[BlockSpan] = None
    for span in spans:
        if previous is not None and span.expected_domain != previous.expected_domain:
            first_active = None
            for window in range(span.start_window, span.end_window + 1):
                record = by_window.get(window)
                if record and record["active_domain"] == span.expected_domain:
                    first_active = window
                    break
            latency = None if first_active is None else first_active - span.start_window + 1
            out.append(
                {
                    "from_block_index": previous.block_index,
                    "to_block_index": span.block_index,
                    "from_item_id": previous.item_id,
                    "to_item_id": span.item_id,
                    "from_domain": previous.expected_domain,
                    "to_domain": span.expected_domain,
                    "start_window": span.start_window,
                    "end_window": span.end_window,
                    "first_target_active_window": first_active,
                    "latency_windows": latency,
                    "max_switch_windows": int(max_switch_windows),
                    "passed": bool(latency is not None and latency <= int(max_switch_windows)),
                }
            )
        previous = span
    return out


def _allowed_transition_windows(transitions: Sequence[Dict[str, Any]], max_switch_windows: int) -> set[int]:
    out: set[int] = set()
    for item in transitions:
        start = int(item["start_window"])
        for window in range(start, start + max(0, int(max_switch_windows))):
            out.add(window)
    return out


def _read_nonempty_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_text_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _limit_windows(lines: Sequence[str], windows_per_item: int) -> List[str]:
    if int(windows_per_item) <= 0:
        return list(lines)
    return list(lines[: int(windows_per_item)])


def _medicine_id(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    marker = "__medicine_"
    return stem.split(marker, 1)[1] if marker in stem else stem


def _medicine_sort_key(filename: str) -> Any:
    value = _medicine_id(filename)
    return (len(value), value)


def _medicine_audio_path(audio_dir: str, medicine_id: str) -> str:
    if not audio_dir:
        return ""
    candidate = Path(audio_dir) / f"sample_{medicine_id}_v2" / f"{medicine_id}_v2.wav"
    return str(candidate) if candidate.is_file() else ""


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acl-root", default=DEFAULT_ACL_ROOT)
    ap.add_argument("--medicine-text-dir", default=DEFAULT_MEDICINE_TEXT_DIR)
    ap.add_argument("--medicine-audio-dir", default=DEFAULT_MEDICINE_AUDIO_DIR)
    ap.add_argument("--acl-items", type=int, default=5)
    ap.add_argument("--medicine-items", type=int, default=5)
    ap.add_argument("--windows-per-item", type=int, default=64)
    ap.add_argument("--text-field", choices=("target", "source"), default="target")
    ap.add_argument(
        "--schedule",
        choices=("alternating", "random", "acl_then_medicine", "medicine_then_acl"),
        default="alternating",
    )
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--router-text-source", default="generated_target")
    ap.add_argument("--probe-mode", choices=("expected", "none", "inverted", "contested"), default="expected")
    ap.add_argument("--initial-domain", choices=("first", "nlp", "medicine"), default="first")
    ap.add_argument("--max-switch-windows", type=int, default=3)
    ap.add_argument("--min-consistent-generated-target", type=int, default=3)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    acl_blocks = read_acl_blocks(
        args.acl_root,
        limit_items=args.acl_items,
        windows_per_item=args.windows_per_item,
        text_field=args.text_field,
    )
    medicine_blocks = read_medicine_blocks(
        args.medicine_text_dir,
        audio_dir=args.medicine_audio_dir,
        limit_items=args.medicine_items,
        windows_per_item=args.windows_per_item,
        text_field=args.text_field,
    )
    schedule_blocks = build_schedule(
        acl_blocks,
        medicine_blocks,
        schedule=args.schedule,
        seed=args.seed,
    )
    if not schedule_blocks:
        raise SystemExit("no playlist blocks found")

    payload = evaluate_playlist(
        schedule_blocks,
        schedule_name=args.schedule,
        router_text_source=args.router_text_source,
        probe_mode=args.probe_mode,
        max_switch_windows=args.max_switch_windows,
        initial_domain=args.initial_domain,
        min_consistent_windows_generated_target=args.min_consistent_generated_target,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        write_markdown(payload, args.out_md)
    if not args.no_assert and not payload["summary"]["regression_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
