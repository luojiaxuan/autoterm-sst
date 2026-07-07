#!/usr/bin/env python3
"""Router-only ACL/NLP <-> medicine auto-glossary switch evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.topic_router import (
    DomainProbeScore,
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


ACL_FIXTURE = (
    "We evaluate BERT and transformer language models on a corpus benchmark for machine translation.",
    "The parser uses dependency parsing, entity recognition, token embeddings, and BLEU evaluation.",
    "Pretraining and fine tuning improve encoder decoder attention on the annotation dataset.",
)

MEDICINE_FIXTURE = (
    "The patient received clinical treatment for diabetes after diagnosis in the hospital.",
    "The oncology trial studies cancer infection symptoms, blood pressure, and heart rate.",
    "Doctors prescribe a tablet dose in mg after surgery and vaccine treatment.",
)


@dataclass(frozen=True)
class TextWindow:
    expected_domain: str
    text: str


def build_router(*, min_confidence: float = 0.60, with_probe: bool = False) -> HybridWindowTopicRouter:
    weight_overrides = {}
    if with_probe:
        weight_overrides = {
            "domain_probe_weight": 0.25,
            "text_topic_weight": 0.60,
            "speech_centroid_weight": 0.10,
            "metadata_prior_weight": 0.05,
        }
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
            min_consistent_windows_audio_only=3,
            **weight_overrides,
        ),
    )


def read_windows(path: str, domain: str, *, max_windows: int, fallback: Sequence[str]) -> List[TextWindow]:
    source = Path(path) if path else None
    lines: List[str] = []
    if source:
        if not source.is_file():
            raise FileNotFoundError(f"text window file not found: {source}")
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        lines = list(fallback)
    return [TextWindow(domain, text) for text in lines[:max(1, int(max_windows))]]


def scenario_windows(
    scenario: str,
    *,
    acl_windows: Sequence[TextWindow],
    medicine_windows: Sequence[TextWindow],
) -> List[TextWindow]:
    if scenario == "acl_only":
        return list(acl_windows)
    if scenario == "medicine_only":
        return list(medicine_windows)
    if scenario == "acl_to_medicine":
        return list(acl_windows) + list(medicine_windows)
    if scenario == "medicine_to_acl":
        return list(medicine_windows) + list(acl_windows)
    raise ValueError(f"unknown scenario: {scenario}")


def initial_state_for(scenario: str) -> RouterSessionState:
    if scenario.startswith("medicine"):
        return RouterSessionState("medicine_core_10k", "medicine", created_s=0.0)
    return RouterSessionState("nlp_core_10k", "nlp", created_s=0.0)


def probe_for(window: TextWindow) -> Dict[str, DomainProbeScore]:
    domain = window.expected_domain
    preset = f"{domain}_core_10k"
    return {
        domain: DomainProbeScore(
            domain=domain,
            preset_id=preset,
            top_score=0.9,
            mean_topk_score=0.8,
            top_terms=(f"{domain} probe term",),
        )
    }


def apply_switch(state: RouterSessionState, target_preset: str, target_domain: str, now_s: float) -> None:
    state.active_preset_id = target_preset
    state.active_domain_id = target_domain
    state.last_switch_s = now_s
    state.pending_preset_id = None


def evaluate_scenario(
    scenario: str,
    windows: Sequence[TextWindow],
    *,
    router_text_source: str = "manifest_source",
    with_probe: bool = False,
    max_switch_windows: int = 2,
) -> Dict[str, Any]:
    router = build_router(with_probe=with_probe)
    state = initial_state_for(scenario)
    initial_domain = state.active_domain_id
    records: List[Dict[str, Any]] = []
    switch_count = 0
    first_boundary: Optional[int] = None
    first_target_active: Optional[int] = None

    for idx, window in enumerate(windows, start=1):
        if first_boundary is None and window.expected_domain != initial_domain:
            first_boundary = idx
        decision = router.observe(
            state,
            None,
            [],
            now_s=float(idx),
            router_text=window.text,
            router_text_source=router_text_source,
            domain_probe_scores=probe_for(window) if with_probe else {},
        )
        if decision.action == "switch":
            switch_count += 1
            apply_switch(state, decision.target_preset_id, decision.target_domain_id, float(idx))
        active_domain = state.active_domain_id
        if first_boundary is not None and first_target_active is None and active_domain == window.expected_domain:
            first_target_active = idx
        records.append(
            {
                "window": idx,
                "expected_domain": window.expected_domain,
                "active_domain": active_domain,
                "decision_action": decision.action,
                "decision_target_domain": decision.target_domain_id,
                "confidence": decision.confidence,
                "margin": decision.margin,
                "reason": decision.reason,
                "text": window.text,
            }
        )

    accuracy = (
        sum(1 for item in records if item["active_domain"] == item["expected_domain"]) / len(records)
        if records
        else 0.0
    )
    acl_records = [item for item in records if item["expected_domain"] == "nlp"]
    medicine_records = [item for item in records if item["expected_domain"] == "medicine"]
    false_medicine_on_acl = sum(1 for item in acl_records if item["active_domain"] == "medicine")
    false_nlp_on_medicine = sum(1 for item in medicine_records if item["active_domain"] == "nlp")
    pre_boundary = [
        item for item in records
        if first_boundary is None or int(item["window"]) < first_boundary
    ]
    false_medicine_on_acl_pre_boundary = sum(
        1 for item in pre_boundary
        if item["expected_domain"] == "nlp" and item["active_domain"] == "medicine"
    )
    false_nlp_on_medicine_pre_boundary = sum(
        1 for item in pre_boundary
        if item["expected_domain"] == "medicine" and item["active_domain"] == "nlp"
    )
    switch_latency_windows = None
    if first_boundary is not None and first_target_active is not None:
        switch_latency_windows = first_target_active - first_boundary + 1
    target_domain = windows[-1].expected_domain if windows else initial_domain
    switch_success = bool(first_boundary is None or state.active_domain_id == target_domain)
    regression_pass = bool(
        false_medicine_on_acl_pre_boundary == 0
        and false_nlp_on_medicine_pre_boundary == 0
        and switch_success
        and (switch_latency_windows is None or switch_latency_windows <= max_switch_windows)
    )
    return {
        "scenario": scenario,
        "router_text_source": router_text_source,
        "with_probe": with_probe,
        "windows": len(records),
        "domain_accuracy": round(accuracy, 4),
        "switch_count": switch_count,
        "switch_success": switch_success,
        "switch_latency_windows": switch_latency_windows,
        "max_switch_windows": max_switch_windows,
        "false_medicine_on_acl": false_medicine_on_acl,
        "false_nlp_on_medicine": false_nlp_on_medicine,
        "false_medicine_on_acl_pre_boundary": false_medicine_on_acl_pre_boundary,
        "false_nlp_on_medicine_pre_boundary": false_nlp_on_medicine_pre_boundary,
        "regression_pass": regression_pass,
        "records": records,
    }


def run_all_scenarios(
    *,
    acl_windows: Sequence[TextWindow],
    medicine_windows: Sequence[TextWindow],
    scenarios: Iterable[str],
    router_text_source: str = "manifest_source",
    with_probe: bool = False,
    max_switch_windows: int = 2,
) -> List[Dict[str, Any]]:
    return [
        evaluate_scenario(
            scenario,
            scenario_windows(scenario, acl_windows=acl_windows, medicine_windows=medicine_windows),
            router_text_source=router_text_source,
            with_probe=with_probe,
            max_switch_windows=max_switch_windows,
        )
        for scenario in scenarios
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acl-text", default="")
    ap.add_argument("--medicine-text", default="")
    ap.add_argument("--max-windows-per-domain", type=int, default=3)
    ap.add_argument(
        "--scenarios",
        default="acl_only,medicine_only,acl_to_medicine,medicine_to_acl",
        help="Comma-separated subset of acl_only,medicine_only,acl_to_medicine,medicine_to_acl",
    )
    ap.add_argument("--router-text-source", default="manifest_source")
    ap.add_argument("--with-probe", action="store_true")
    ap.add_argument("--max-switch-windows", type=int, default=2)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    acl_windows = read_windows(
        args.acl_text,
        "nlp",
        max_windows=args.max_windows_per_domain,
        fallback=ACL_FIXTURE,
    )
    medicine_windows = read_windows(
        args.medicine_text,
        "medicine",
        max_windows=args.max_windows_per_domain,
        fallback=MEDICINE_FIXTURE,
    )
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    rows = run_all_scenarios(
        acl_windows=acl_windows,
        medicine_windows=medicine_windows,
        scenarios=scenarios,
        router_text_source=args.router_text_source,
        with_probe=args.with_probe,
        max_switch_windows=args.max_switch_windows,
    )
    payload = {
        "summary": {
            "passed": all(row["regression_pass"] for row in rows),
            "scenarios": len(rows),
            "router_text_source": args.router_text_source,
            "with_probe": args.with_probe,
        },
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if not args.no_assert and not payload["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
