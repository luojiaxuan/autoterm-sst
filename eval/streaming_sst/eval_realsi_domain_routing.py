#!/usr/bin/env python3
"""Evaluate AutoTerm domain routing on RealSI transcript windows.

The English-to-Chinese reference stream is used as a proxy for accumulated
generated-target context. This isolates the router and does not measure LLM
translation quality, speech probing, or MaxSim retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import DOMAIN_TO_PRESET, WORKING_DOMAINS
from framework.agents.term_memory.topic_router import (
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


DOMAIN_FILES = {
    "nlp": "en2zh-01-tech.json",
    "medicine": "en2zh-02-health.json",
    "education": "en2zh-03-edu.json",
    "finance": "en2zh-04-fin.json",
    "legal": "en2zh-05-law.json",
    "environment": "en2zh-06-env.json",
    "entertainment": "en2zh-07-ent.json",
    "science": "en2zh-08-sci.json",
    "sports": "en2zh-09-sport.json",
    "art": "en2zh-10-art.json",
}


@dataclass(frozen=True)
class RoutingWindow:
    domain: str
    start_segment: int
    end_segment: int
    text: str


def _annotation_dir(root: Path) -> Path:
    direct = root / "en2zh" / "json"
    nested = root / "data" / "en2zh" / "json"
    if direct.is_dir():
        return direct
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"RealSI en2zh annotations not found under {root}")


def load_windows(
    root: Path,
    domain: str,
    *,
    text_field: str,
    window_segments: int,
    step_segments: int,
) -> List[RoutingWindow]:
    path = _annotation_dir(root) / DOMAIN_FILES[domain]
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segment") or []
    texts = [str(item.get(text_field) or "").strip() for item in segments]
    windows: List[RoutingWindow] = []
    for end in range(max(1, step_segments), len(texts) + step_segments, step_segments):
        end = min(end, len(texts))
        start = max(0, end - max(1, window_segments))
        text = " ".join(item for item in texts[start:end] if item).strip()
        if text:
            windows.append(RoutingWindow(domain, start + 1, end, text))
        if end == len(texts):
            break
    return windows


def build_router(domains: Sequence[str]) -> HybridWindowTopicRouter:
    slices = [
        DomainSlice(DOMAIN_TO_PRESET[domain], domain, index_path=f"mock://{domain}")
        for domain in domains
    ]
    return HybridWindowTopicRouter(
        slices,
        RouterConfig(
            warmup_sec=0.0,
            update_interval_sec=0.0,
            switch_cooldown_sec=0.0,
            min_confidence=0.60,
            min_margin=0.15,
            min_current_margin=0.10,
            min_consistent_windows_with_text=2,
            min_consistent_windows_generated_target=3,
            min_consistent_windows_audio_only=3,
            text_topic_weight=0.60,
            domain_probe_weight=0.25,
            speech_centroid_weight=0.10,
            metadata_prior_weight=0.05,
        ),
    )


def _apply_switch(state: RouterSessionState, *, preset: str, domain: str, now_s: float) -> None:
    state.active_preset_id = preset
    state.active_domain_id = domain
    state.last_switch_s = now_s
    state.pending_preset_id = None


def evaluate(
    windows_by_domain: Dict[str, Sequence[RoutingWindow]],
    domains: Sequence[str],
    *,
    grace_windows: int,
    max_switch_windows: int,
    min_classified_accuracy: float,
    min_steady_state_accuracy: float,
) -> Dict[str, Any]:
    router = build_router(domains)
    first_domain = domains[0]
    state = RouterSessionState(
        DOMAIN_TO_PRESET[first_domain],
        first_domain,
        created_s=0.0,
    )
    records: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    now_s = 0.0
    wrong_switches = 0

    for block_index, domain in enumerate(domains):
        block_records: List[Dict[str, Any]] = []
        first_active_window = 0 if state.active_domain_id == domain else None
        for window_index, window in enumerate(windows_by_domain[domain], start=1):
            now_s += 1.0
            decision = router.observe(
                state,
                None,
                [],
                now_s=now_s,
                router_text=window.text,
                router_text_source="generated_target",
            )
            if decision.action == "switch":
                if decision.target_domain_id != domain:
                    wrong_switches += 1
                _apply_switch(
                    state,
                    preset=decision.target_preset_id,
                    domain=decision.target_domain_id,
                    now_s=now_s,
                )
            if first_active_window is None and state.active_domain_id == domain:
                first_active_window = window_index

            top = decision.top_scores[0] if decision.top_scores else None
            has_signal = bool(top and top.evidence.get("has_text_topic_signal"))
            row = {
                "domain_block": block_index + 1,
                "window_in_domain": window_index,
                "expected_domain": domain,
                "active_domain": state.active_domain_id,
                "raw_top_domain": top.domain_id if top else "",
                "raw_top_confidence": top.confidence if top else 0.0,
                "has_topic_signal": has_signal,
                "decision_action": decision.action,
                "decision_target_domain": decision.target_domain_id,
                "reason": decision.reason,
                "start_segment": window.start_segment,
                "end_segment": window.end_segment,
                "text": window.text,
            }
            records.append(row)
            block_records.append(row)

        transition_rows.append(
            {
                "domain": domain,
                "windows": len(block_records),
                "first_active_window": first_active_window,
                "within_limit": bool(
                    first_active_window is not None
                    and first_active_window <= (0 if block_index == 0 else max_switch_windows)
                ),
            }
        )

    classified = [item for item in records if item["has_topic_signal"]]
    classified_correct = sum(
        1 for item in classified if item["raw_top_domain"] == item["expected_domain"]
    )
    steady = [
        item for item in records
        if int(item["window_in_domain"]) > max(0, grace_windows)
    ]
    steady_correct = sum(
        1 for item in steady if item["active_domain"] == item["expected_domain"]
    )
    classified_accuracy = classified_correct / len(classified) if classified else 0.0
    steady_accuracy = steady_correct / len(steady) if steady else 0.0

    per_domain: Dict[str, Dict[str, Any]] = {}
    for domain in domains:
        domain_rows = [item for item in records if item["expected_domain"] == domain]
        domain_classified = [item for item in domain_rows if item["has_topic_signal"]]
        domain_steady = [
            item for item in domain_rows
            if int(item["window_in_domain"]) > max(0, grace_windows)
        ]
        per_domain[domain] = {
            "windows": len(domain_rows),
            "signal_windows": len(domain_classified),
            "signal_coverage": round(
                len(domain_classified) / len(domain_rows) if domain_rows else 0.0,
                4,
            ),
            "classified_top1_accuracy": round(
                sum(
                    1
                    for item in domain_classified
                    if item["raw_top_domain"] == domain
                )
                / len(domain_classified)
                if domain_classified
                else 0.0,
                4,
            ),
            "steady_state_accuracy": round(
                sum(1 for item in domain_steady if item["active_domain"] == domain)
                / len(domain_steady)
                if domain_steady
                else 0.0,
                4,
            ),
        }

    all_domains_have_signal = all(per_domain[domain]["signal_windows"] > 0 for domain in domains)
    transitions_pass = all(item["within_limit"] for item in transition_rows)
    regression_pass = bool(
        all_domains_have_signal
        and classified_accuracy >= min_classified_accuracy
        and steady_accuracy >= min_steady_state_accuracy
        and wrong_switches == 0
        and transitions_pass
    )
    return {
        "domains": list(domains),
        "num_domains": len(domains),
        "windows": len(records),
        "classified_windows": len(classified),
        "classified_top1_accuracy": round(classified_accuracy, 4),
        "steady_state_windows": len(steady),
        "steady_state_accuracy": round(steady_accuracy, 4),
        "wrong_switches": wrong_switches,
        "transitions": transition_rows,
        "per_domain": per_domain,
        "gates": {
            "min_classified_accuracy": min_classified_accuracy,
            "min_steady_state_accuracy": min_steady_state_accuracy,
            "grace_windows": grace_windows,
            "max_switch_windows": max_switch_windows,
            "all_domains_have_signal": all_domains_have_signal,
            "transitions_pass": transitions_pass,
        },
        "regression_pass": regression_pass,
        "records": records,
    }


def _parse_domains(raw: str) -> List[str]:
    domains = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in domains if item not in DOMAIN_FILES]
    if unknown:
        raise ValueError(f"unknown domains: {', '.join(unknown)}")
    if not domains:
        raise ValueError("at least one domain is required")
    return domains


def _summary_lines(result: Dict[str, Any]) -> Iterable[str]:
    yield (
        f"domains={result['num_domains']} windows={result['windows']} "
        f"classified_top1={result['classified_top1_accuracy']:.4f} "
        f"steady_state={result['steady_state_accuracy']:.4f} "
        f"wrong_switches={result['wrong_switches']} pass={result['regression_pass']}"
    )
    for domain, row in result["per_domain"].items():
        transition = next(item for item in result["transitions"] if item["domain"] == domain)
        yield (
            f"{domain:13s} signal={row['signal_windows']:2d}/{row['windows']:2d} "
            f"top1={row['classified_top1_accuracy']:.4f} "
            f"steady={row['steady_state_accuracy']:.4f} "
            f"first_active={transition['first_active_window']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realsi-root", required=True)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--text-field", choices=("src_text", "trg_text"), default="trg_text")
    ap.add_argument("--window-segments", type=int, default=6)
    ap.add_argument("--step-segments", type=int, default=3)
    ap.add_argument("--grace-windows", type=int, default=3)
    ap.add_argument("--max-switch-windows", type=int, default=5)
    ap.add_argument("--min-classified-accuracy", type=float, default=0.85)
    ap.add_argument("--min-steady-state-accuracy", type=float, default=0.90)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    domains = _parse_domains(args.domains)
    root = Path(args.realsi_root).expanduser().resolve()
    windows_by_domain = {
        domain: load_windows(
            root,
            domain,
            text_field=args.text_field,
            window_segments=args.window_segments,
            step_segments=args.step_segments,
        )
        for domain in domains
    }
    result = evaluate(
        windows_by_domain,
        domains,
        grace_windows=args.grace_windows,
        max_switch_windows=args.max_switch_windows,
        min_classified_accuracy=args.min_classified_accuracy,
        min_steady_state_accuracy=args.min_steady_state_accuracy,
    )
    result["source"] = {
        "realsi_root": str(root),
        "text_field": args.text_field,
        "window_segments": args.window_segments,
        "step_segments": args.step_segments,
    }
    for line in _summary_lines(result):
        print(line)
    if args.out_json:
        out = Path(args.out_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if not args.no_assert and not result["regression_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
