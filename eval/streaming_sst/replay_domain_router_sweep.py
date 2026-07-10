#!/usr/bin/env python3
"""Replay domain-description scores without re-encoding audio or running translation.

The input is one or more JSON files written by
``eval_domain_description_similarity.py``.  This tool deliberately reads only
the gold domain, the per-domain routing scores, and (optionally) the router
text.  Translation outputs and translation-quality metrics are outside the
selection loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.topic_router import (
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


DEFAULT_TOP_K = (1, 2, 3, 4, 5, 10)
ROUTER_REPLAY_DEFAULTS: Dict[str, Any] = {
    "warmup_sec": 0.0,
    "update_interval_sec": 0.0,
    "switch_cooldown_sec": 0.0,
    "min_confidence": 0.60,
    "min_margin": 0.15,
    "min_current_margin": 0.10,
    "min_consistent_windows_generated_target": 3,
    "context_similarity_weight": 0.60,
    "text_topic_weight": 0.25,
    "domain_probe_weight": 0.10,
    "speech_centroid_weight": 0.03,
    "metadata_prior_weight": 0.02,
    "audio_ema_alpha": 0.80,
    "slice_selection_mode": "hard_top1",
}


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_input(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        label, path_text = raw.split("=", 1)
        label = label.strip()
        path = Path(path_text).expanduser().resolve()
    else:
        path = Path(raw).expanduser().resolve()
        label = path.stem
    if not label:
        raise ValueError(f"input label must not be empty: {raw!r}")
    return label, path


def _ranked_scores(record: Mapping[str, Any]) -> List[Tuple[str, float]]:
    scores = record.get("scores")
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("every record must contain a non-empty 'scores' mapping")
    ranked: List[Tuple[str, float]] = []
    for domain, value in scores.items():
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"non-finite score for domain {domain!r}: {value!r}")
        ranked.append((str(domain), score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def load_dataset(raw_input: str) -> Dict[str, Any]:
    label, path = _parse_input(raw_input)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty records list")

    normalized: List[Dict[str, Any]] = []
    domain_order: List[str] = []
    seen_domains = set()
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: record {index} is not an object")
        gold = str(item.get("expected_domain") or item.get("gold_domain") or "").strip()
        if not gold:
            raise ValueError(f"{path}: record {index} has no expected_domain")
        ranked = _ranked_scores(item)
        score_map = {domain: score for domain, score in ranked}
        if gold not in score_map:
            raise ValueError(f"{path}: record {index} gold domain {gold!r} has no score")
        for domain, _ in ranked:
            if domain not in seen_domains:
                seen_domains.add(domain)
                domain_order.append(domain)
        normalized.append(
            {
                "index": index,
                "gold": gold,
                "scores": score_map,
                "ranked": ranked,
                "text": str(item.get("text") or ""),
                "start_segment": item.get("start_segment"),
                "end_segment": item.get("end_segment"),
            }
        )

    declared_domains = payload.get("domains")
    if isinstance(declared_domains, list) and declared_domains:
        declared = [str(item) for item in declared_domains]
        missing = sorted(set(domain_order) - set(declared))
        if missing:
            raise ValueError(f"{path}: score domains missing from declared domains: {missing}")
        domain_order = declared

    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256(path),
        "domains": domain_order,
        "records": normalized,
        "source_settings": payload.get("settings", {}),
    }


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return _round(ordered[0])
    position = (len(ordered) - 1) * float(percentile)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    value = ordered[low] * (1.0 - fraction) + ordered[high] * fraction
    return _round(value)


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": _round(statistics.fmean(values)),
        "min": _round(min(values)),
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": _round(max(values)),
    }


def score_margins(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    top1_margins: List[float] = []
    gold_margins: List[float] = []
    gold_ranks: List[float] = []
    for record in records:
        ranked = list(record["ranked"])
        top1 = ranked[0][1]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        top1_margins.append(float(top1 - top2))
        gold = str(record["gold"])
        gold_score = float(record["scores"][gold])
        best_other = max(
            (float(score) for domain, score in ranked if domain != gold),
            default=0.0,
        )
        gold_margins.append(gold_score - best_other)
        gold_ranks.append(float(next(i for i, (domain, _) in enumerate(ranked, 1) if domain == gold)))
    return {
        "top1_minus_top2": _distribution(top1_margins),
        "gold_minus_best_non_gold": _distribution(gold_margins),
        "gold_rank": _distribution(gold_ranks),
        "nonpositive_gold_margin": sum(value <= 0.0 for value in gold_margins),
    }


def _transition_summary(
    records: Sequence[Mapping[str, Any]],
    active_sets: Sequence[Iterable[str]],
) -> Dict[str, Any]:
    if len(records) != len(active_sets):
        raise ValueError("records and active_sets must have identical lengths")
    transitions: List[Dict[str, Any]] = []
    block_start = 0
    previous_gold = str(records[0]["gold"])
    for index in range(1, len(records) + 1):
        boundary = index == len(records) or str(records[index]["gold"]) != previous_gold
        if not boundary:
            continue
        if block_start > 0:
            first_offset: Optional[int] = None
            for offset in range(index - block_start):
                if previous_gold in set(active_sets[block_start + offset]):
                    first_offset = offset
                    break
            transitions.append(
                {
                    "from_domain": str(records[block_start - 1]["gold"]),
                    "to_domain": previous_gold,
                    "start_record": block_start,
                    "block_windows": index - block_start,
                    "delay_windows": first_offset,
                    "resolved": first_offset is not None,
                }
            )
        if index < len(records):
            block_start = index
            previous_gold = str(records[index]["gold"])

    resolved_delays = [
        float(item["delay_windows"])
        for item in transitions
        if item["delay_windows"] is not None
    ]
    return {
        "gold_transitions": len(transitions),
        "resolved": len(resolved_delays),
        "unresolved": len(transitions) - len(resolved_delays),
        "delay_windows": _distribution(resolved_delays),
        "details": transitions,
    }


def _selection_churn(active_sets: Sequence[Iterable[str]]) -> Dict[str, Any]:
    sets = [set(items) for items in active_sets]
    if not sets:
        return {
            "selection_changes": 0,
            "churn_per_100_windows": 0.0,
            "mean_symmetric_difference": 0.0,
        }
    selection_changes = 0
    symmetric_differences: List[float] = []
    for previous, current in zip(sets, sets[1:]):
        distance = len(previous.symmetric_difference(current))
        symmetric_differences.append(float(distance))
        selection_changes += int(distance > 0)
    opportunities = max(1, len(sets) - 1)
    return {
        "selection_changes": selection_changes,
        "churn_per_100_windows": _round(100.0 * selection_changes / opportunities),
        "mean_symmetric_difference": _round(
            statistics.fmean(symmetric_differences) if symmetric_differences else 0.0
        ),
    }


def evaluate_top_k(records: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    if k <= 0:
        raise ValueError("top-k values must be positive")
    active_sets: List[set[str]] = []
    correct = 0
    per_domain: Dict[str, Dict[str, int]] = {}
    for record in records:
        selected = {domain for domain, _ in list(record["ranked"])[:k]}
        active_sets.append(selected)
        gold = str(record["gold"])
        included = gold in selected
        correct += int(included)
        row = per_domain.setdefault(gold, {"windows": 0, "gold_included": 0})
        row["windows"] += 1
        row["gold_included"] += int(included)
    total = len(records)
    active_counts = [len(items) for items in active_sets]
    return {
        "policy_type": "top_k",
        "k": int(k),
        "windows": total,
        "gold_included": correct,
        "gold_slice_coverage": _round(correct / total if total else 0.0),
        "mean_active_slices": _round(statistics.fmean(active_counts) if active_counts else 0.0),
        "per_domain": {
            domain: {
                **row,
                "coverage": _round(row["gold_included"] / row["windows"]),
            }
            for domain, row in sorted(per_domain.items())
        },
        "churn": _selection_churn(active_sets),
        "transitions": _transition_summary(records, active_sets),
    }


def load_router_specs(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("configs") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("router config file must be a list or an object with a 'configs' list")
    known_config_keys = {item.name for item in fields(RouterConfig)}
    allowed_spec_keys = {
        "name",
        "router_config",
        "initial_domain",
        "router_text_mode",
        "window_seconds",
    }
    out: List[Dict[str, Any]] = []
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"router config {index} is not an object")
        unknown_spec = sorted(set(row) - allowed_spec_keys)
        if unknown_spec:
            raise ValueError(f"router config {index} has unknown keys: {unknown_spec}")
        name = str(row.get("name") or f"router_{index}")
        if name in names:
            raise ValueError(f"duplicate router config name: {name}")
        names.add(name)
        overrides = row.get("router_config", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"router config {name!r}: router_config must be an object")
        unknown_router = sorted(set(overrides) - known_config_keys)
        if unknown_router:
            raise ValueError(f"router config {name!r} has unknown RouterConfig keys: {unknown_router}")
        resolved = dict(ROUTER_REPLAY_DEFAULTS)
        resolved.update(overrides)
        if str(resolved.get("slice_selection_mode")) != "hard_top1":
            raise ValueError(f"router config {name!r} must use slice_selection_mode='hard_top1'")
        text_mode = str(row.get("router_text_mode") or "scores_only")
        if text_mode not in {"scores_only", "record_text"}:
            raise ValueError(
                f"router config {name!r}: router_text_mode must be scores_only or record_text"
            )
        window_seconds = float(row.get("window_seconds", 1.0))
        if not math.isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError(f"router config {name!r}: window_seconds must be positive")
        out.append(
            {
                "name": name,
                "router_config": resolved,
                "initial_domain": str(row.get("initial_domain") or "first_gold"),
                "router_text_mode": text_mode,
                "window_seconds": window_seconds,
            }
        )
    return out


def replay_hard_top1(
    dataset: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    records = list(dataset["records"])
    domains = list(dataset["domains"])
    if not domains:
        raise ValueError("router replay requires at least one domain")
    slices = [
        DomainSlice(
            preset_id=f"replay::{domain}",
            domain_id=domain,
            index_path=f"mock://{domain}",
        )
        for domain in domains
    ]
    config = RouterConfig(**dict(spec["router_config"]))
    router = HybridWindowTopicRouter(slices, config)
    initial = str(spec["initial_domain"])
    if initial == "first_gold":
        initial = str(records[0]["gold"])
    if initial not in domains:
        raise ValueError(f"router config {spec['name']!r}: initial domain {initial!r} is unavailable")
    state = RouterSessionState(
        active_preset_id=f"replay::{initial}",
        active_domain_id=initial,
        created_s=0.001,
    )

    active_sets: List[set[str]] = []
    decisions: List[Dict[str, Any]] = []
    correct = 0
    switches = 0
    wrong_target_switches = 0
    useful_switches = 0
    window_seconds = float(spec["window_seconds"])
    for index, record in enumerate(records):
        if spec["router_text_mode"] == "record_text":
            router_text = str(record.get("text") or "")
            if not router_text:
                raise ValueError(
                    f"{dataset['label']}: router config {spec['name']!r} requires record text, "
                    f"but record {index} has none"
                )
        else:
            # note (luojiaxuan): The production router requires a text-source marker before it
            # accepts context-similarity evidence.  This neutral sentinel carries no taxonomy
            # keyword signal; the only domain evidence remains the saved similarity score map.
            router_text = "semantic routing evidence"
        previous_active = state.active_domain_id
        decision = router.observe(
            state,
            None,
            [],
            now_s=(index + 1) * window_seconds,
            router_text=router_text,
            router_text_source="generated_target",
            context_similarity_scores=dict(record["scores"]),
        )
        if decision.action == "switch":
            switches += 1
            target_is_gold = decision.target_domain_id == str(record["gold"])
            useful_switches += int(previous_active != str(record["gold"]) and target_is_gold)
            wrong_target_switches += int(not target_is_gold)
            state.active_preset_id = decision.target_preset_id
            state.active_domain_id = decision.target_domain_id
            state.last_switch_s = (index + 1) * window_seconds
            state.pending_preset_id = None
        active = state.active_domain_id
        active_sets.append({active})
        correct += int(active == str(record["gold"]))
        decisions.append(
            {
                "record": index,
                "gold_domain": str(record["gold"]),
                "active_domain": active,
                "action": decision.action,
                "target_domain": decision.target_domain_id,
                "confidence": decision.confidence,
                "margin": decision.margin,
                "reason": decision.reason,
            }
        )

    total = len(records)
    return {
        "policy_type": "hard_top1",
        "name": str(spec["name"]),
        "windows": total,
        "gold_active": correct,
        "gold_slice_coverage": _round(correct / total if total else 0.0),
        "mean_active_slices": 1.0,
        "switches": switches,
        "useful_switches": useful_switches,
        "wrong_target_switches": wrong_target_switches,
        "excess_switches": max(0, switches - useful_switches),
        "churn": _selection_churn(active_sets),
        "transitions": _transition_summary(records, active_sets),
        "initial_domain": initial,
        "router_text_mode": str(spec["router_text_mode"]),
        "window_seconds": window_seconds,
        "router_config": asdict(config),
        "decisions": decisions,
    }


def _aggregate_policy(
    policy_id: str,
    policy_type: str,
    results: Sequence[Tuple[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    total = sum(int(row["windows"]) for _, row in results)
    correct_key = "gold_active" if policy_type == "hard_top1" else "gold_included"
    correct = sum(int(row[correct_key]) for _, row in results)
    dataset_coverages = {
        label: float(row["gold_slice_coverage"])
        for label, row in results
    }
    weighted_active = sum(
        int(row["windows"]) * float(row["mean_active_slices"])
        for _, row in results
    )
    total_changes = sum(int(row["churn"]["selection_changes"]) for _, row in results)
    change_opportunities = sum(max(0, int(row["windows"]) - 1) for _, row in results)
    delays = [
        float(detail["delay_windows"])
        for _, row in results
        for detail in row["transitions"]["details"]
        if detail["delay_windows"] is not None
    ]
    unresolved = sum(int(row["transitions"]["unresolved"]) for _, row in results)
    return {
        "policy_id": policy_id,
        "policy_type": policy_type,
        "windows": total,
        "gold_slice_coverage": _round(correct / total if total else 0.0),
        "minimum_dataset_coverage": _round(min(dataset_coverages.values())),
        "dataset_coverages": dataset_coverages,
        "mean_active_slices": _round(weighted_active / total if total else 0.0),
        "churn_per_100_windows": _round(
            100.0 * total_changes / max(1, change_opportunities)
        ),
        "mean_transition_delay_windows": _round(
            statistics.fmean(delays) if delays else 0.0
        ),
        "unresolved_transitions": unresolved,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        float(left["minimum_dataset_coverage"]) >= float(right["minimum_dataset_coverage"])
        and float(left["mean_active_slices"]) <= float(right["mean_active_slices"])
        and float(left["churn_per_100_windows"]) <= float(right["churn_per_100_windows"])
    )
    strictly_better = (
        float(left["minimum_dataset_coverage"]) > float(right["minimum_dataset_coverage"])
        or float(left["mean_active_slices"]) < float(right["mean_active_slices"])
        or float(left["churn_per_100_windows"]) < float(right["churn_per_100_windows"])
    )
    return no_worse and strictly_better


def choose_pareto(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_gold_coverage: float,
) -> Dict[str, Any]:
    rows = [dict(item) for item in candidates]
    front = [
        item
        for item in rows
        if not any(_dominates(other, item) for other in rows if other is not item)
    ]
    front.sort(
        key=lambda item: (
            -float(item["minimum_dataset_coverage"]),
            float(item["mean_active_slices"]),
            float(item["churn_per_100_windows"]),
            str(item["policy_id"]),
        )
    )
    eligible = [
        item
        for item in rows
        if float(item["minimum_dataset_coverage"]) >= float(min_gold_coverage)
    ]
    if eligible:
        selection_pool = eligible
        fallback_used = False
        selection_reason = "coverage_floor_met_then_minimize_active_slices_churn_delay"
    else:
        best_floor = max(float(item["minimum_dataset_coverage"]) for item in rows)
        selection_pool = [
            item for item in rows if float(item["minimum_dataset_coverage"]) == best_floor
        ]
        fallback_used = True
        selection_reason = "coverage_floor_unmet_choose_maximum_coverage_then_cost"
    recommended = min(
        selection_pool,
        key=lambda item: (
            float(item["mean_active_slices"]),
            float(item["churn_per_100_windows"]),
            int(item["unresolved_transitions"]),
            float(item["mean_transition_delay_windows"]),
            -float(item["gold_slice_coverage"]),
            str(item["policy_id"]),
        ),
    )
    return {
        "selection_policy": {
            "minimum_per_dataset_gold_coverage": float(min_gold_coverage),
            "primary": "meet the predeclared gold-slice coverage floor on every input",
            "secondary": [
                "mean_active_slices (lower is better)",
                "churn_per_100_windows (lower is better)",
                "unresolved and mean transition delay (lower is better)",
            ],
            "translation_metrics_used": False,
        },
        "pareto_front": front,
        "recommended": recommended,
        "fallback_used": fallback_used,
        "selection_reason": selection_reason,
    }


def run_sweep(
    datasets: Sequence[Mapping[str, Any]],
    *,
    top_k_values: Sequence[int],
    router_specs: Sequence[Mapping[str, Any]],
    min_gold_coverage: float,
) -> Dict[str, Any]:
    dataset_results: List[Dict[str, Any]] = []
    by_top_k: Dict[int, List[Tuple[str, Mapping[str, Any]]]] = {
        int(k): [] for k in top_k_values
    }
    by_router: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {
        str(spec["name"]): [] for spec in router_specs
    }
    for dataset in datasets:
        records = list(dataset["records"])
        top_k_rows: Dict[str, Any] = {}
        for k in top_k_values:
            row = evaluate_top_k(records, int(k))
            top_k_rows[str(k)] = row
            by_top_k[int(k)].append((str(dataset["label"]), row))
        router_rows: Dict[str, Any] = {}
        for spec in router_specs:
            row = replay_hard_top1(dataset, spec)
            router_rows[str(spec["name"])] = row
            by_router[str(spec["name"])].append((str(dataset["label"]), row))
        dataset_results.append(
            {
                "label": dataset["label"],
                "path": dataset["path"],
                "sha256": dataset["sha256"],
                "domains": dataset["domains"],
                "windows": len(records),
                "source_settings": dataset["source_settings"],
                "score_margins": score_margins(records),
                "top_k": top_k_rows,
                "hard_top1": router_rows,
            }
        )

    candidates = [
        _aggregate_policy(f"top_k:{k}", "top_k", by_top_k[int(k)])
        for k in top_k_values
    ]
    candidates.extend(
        _aggregate_policy(f"hard_top1:{name}", "hard_top1", rows)
        for name, rows in by_router.items()
    )
    families: Dict[str, Any] = {}
    for family in ("top_k", "hard_top1"):
        family_rows = [item for item in candidates if item["policy_type"] == family]
        if family_rows:
            families[family] = choose_pareto(
                family_rows,
                min_gold_coverage=min_gold_coverage,
            )
    return {
        "tool": "replay_domain_router_sweep",
        "inputs": [
            {
                "label": item["label"],
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for item in datasets
        ],
        "settings": {
            "top_k": [int(k) for k in top_k_values],
            "minimum_per_dataset_gold_coverage": float(min_gold_coverage),
            "router_configs": [dict(spec) for spec in router_specs],
            "selection_uses_translation_metrics": False,
        },
        "datasets": dataset_results,
        "aggregate_candidates": candidates,
        "selection": {
            "all_policies": choose_pareto(
                candidates,
                min_gold_coverage=min_gold_coverage,
            ),
            "by_family": families,
        },
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Domain router replay sweep",
        "",
        "Selection is based only on gold-slice coverage, active slices, churn, and transition delay. "
        "No translation metric is read by this tool.",
        "",
        "## Inputs",
        "",
        "| Label | Windows | SHA-256 |",
        "|---|---:|---|",
    ]
    windows = {str(row["label"]): int(row["windows"]) for row in result["datasets"]}
    for item in result["inputs"]:
        lines.append(
            f"| {item['label']} | {windows[str(item['label'])]} | `{str(item['sha256'])[:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate policies",
            "",
            "| Policy | Min dataset coverage | Micro coverage | Active slices | Churn / 100 windows | Mean delay | Unresolved |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["aggregate_candidates"]:
        lines.append(
            "| {policy_id} | {minimum_dataset_coverage:.4f} | {gold_slice_coverage:.4f} | "
            "{mean_active_slices:.2f} | {churn_per_100_windows:.2f} | "
            "{mean_transition_delay_windows:.2f} | {unresolved_transitions} |".format(**row)
        )
    selection = result["selection"]["all_policies"]
    recommended = selection["recommended"]
    lines.extend(
        [
            "",
            "## Predeclared selection",
            "",
            f"Coverage floor: `{selection['selection_policy']['minimum_per_dataset_gold_coverage']:.4f}` per input.",
            "",
            f"Recommended policy: **{recommended['policy_id']}** "
            f"({selection['selection_reason']}).",
            "",
            "Pareto front: "
            + ", ".join(f"`{row['policy_id']}`" for row in selection["pareto_front"])
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_top_k(raw: str) -> List[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top-k must contain positive comma-separated integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="eval JSON path, optionally prefixed with LABEL=; repeat for multiple inputs",
    )
    parser.add_argument("--top-k", default=",".join(str(k) for k in DEFAULT_TOP_K))
    parser.add_argument(
        "--router-configs",
        type=Path,
        help="optional JSON list of sequential hard-top1 HybridWindowTopicRouter configs",
    )
    parser.add_argument("--min-gold-coverage", type=float, default=0.99)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-markdown", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.min_gold_coverage <= 1.0:
        parser.error("--min-gold-coverage must be in [0, 1]")

    try:
        datasets = [load_dataset(item) for item in args.input]
        labels = [str(item["label"]) for item in datasets]
        if len(labels) != len(set(labels)):
            raise ValueError(f"input labels must be unique: {labels}")
        result = run_sweep(
            datasets,
            top_k_values=_parse_top_k(args.top_k),
            router_specs=load_router_specs(args.router_configs),
            min_gold_coverage=float(args.min_gold_coverage),
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.out_markdown or args.out_json.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    recommended = result["selection"]["all_policies"]["recommended"]
    print(
        f"inputs={len(datasets)} windows={sum(row['windows'] for row in result['datasets'])} "
        f"recommended={recommended['policy_id']} "
        f"min_coverage={recommended['minimum_dataset_coverage']:.4f}"
    )
    print(f"wrote {args.out_json}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
