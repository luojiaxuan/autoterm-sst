#!/usr/bin/env python3
"""Build one fair Oracle/AutoTerm/merged scorecard from mixed-run JSONs.

The headline terminology metric is deliberately narrow: MFA time-aligned,
one-to-one occurrence accuracy over ``raw_plus_medicine``.  This wrapper never
calls the older block-level count-clipping term scorer and never emits the
legacy 419-denominator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst import score_time_aligned_terms as mfa_term_scorer  # noqa: E402
from eval.streaming_sst.score_mixed_audio_terms import (  # noqa: E402
    build_reference_text,
    run_hypothesis_text,
    target_terms_from_occurrences,
)
from eval.streaming_sst.score_prompt_precision import (  # noqa: E402
    playlist_signature,
    score_payload as score_prompt_payload,
    timing_signature,
    timing_signature_sha256,
    validate_same_playlist,
)
from eval.streaming_sst.score_terms import compute_bleu_scores  # noqa: E402
from eval.streaming_sst.score_time_aligned_terms import TimedOccurrence  # noqa: E402
from eval.streaming_sst.selected_window_smoke import (  # noqa: E402
    EXPECTED_RAW_DENOMINATOR as SELECTED_WINDOW_RAW_DENOMINATOR,
    PROTOCOL_ID as SELECTED_WINDOW_PROTOCOL_ID,
    protocol_manifest as selected_window_protocol_manifest,
    validate_payload as validate_selected_window_payload,
)

RUN_ROLES = ("oracle", "autoterm", "merged")
SCHEMA_VERSION = "autoterm_story_scorecard.v1"
HEADLINE_GOLD_KEY = "raw_plus_medicine"
TECHNICAL_GOLD_KEY = "technical_plus_medicine"


def _stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _occurrence_fingerprint(occurrences: Sequence[TimedOccurrence]) -> str:
    rows = [
        {
            "domain": occurrence.domain,
            "block_index": occurrence.block_index,
            "term": occurrence.term,
            "variants": list(occurrence.variants),
            "t_start": occurrence.t_start,
            "t_end": occurrence.t_end,
        }
        for occurrence in occurrences
    ]
    return _stable_sha256(rows)


def _gold_summary(occurrences: Sequence[TimedOccurrence]) -> Dict[str, Any]:
    domains: Dict[str, int] = {}
    for occurrence in occurrences:
        domains[occurrence.domain] = domains.get(occurrence.domain, 0) + 1
    return {
        "gold_occurrences": len(occurrences),
        "by_domain": dict(sorted(domains.items())),
        "sha256": _occurrence_fingerprint(occurrences),
    }


def timing_comparison(payloads: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare persisted decoder windows against Oracle as the reference."""

    missing = [role for role in RUN_ROLES if role not in payloads]
    extra = [role for role in payloads if role not in RUN_ROLES]
    if missing or extra:
        raise ValueError(f"runs must be exactly {RUN_ROLES}; missing={missing}, extra={extra}")
    expected = timing_signature(payloads[RUN_ROLES[0]])
    if not expected:
        raise ValueError("oracle run has no persisted decoder timing windows")
    runs: Dict[str, Any] = {}
    for role in RUN_ROLES:
        signature = timing_signature(payloads[role])
        first_mismatch = None
        for index in range(max(len(signature), len(expected))):
            actual = signature[index] if index < len(signature) else None
            wanted = expected[index] if index < len(expected) else None
            if actual != wanted:
                first_mismatch = {
                    "event_index": index + 1,
                    "actual": list(actual) if actual is not None else None,
                    "oracle": list(wanted) if wanted is not None else None,
                }
                break
        runs[role] = {
            "event_count": len(signature),
            "event_count_delta_vs_oracle": len(signature) - len(expected),
            "exact_match_to_oracle": signature == expected,
            "sha256": timing_signature_sha256(payloads[role]),
            "first_mismatch": first_mismatch,
        }
    return {
        "timing_comparable": all(runs[role]["exact_match_to_oracle"] for role in RUN_ROLES),
        "definition": "exact equality of every persisted (start_sample, cursor_samples) window",
        "oracle_event_count": len(expected),
        "runs": runs,
    }


def validate_timing_compatible(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """Hard-fail non-comparable timing unless the caller explicitly opts out."""

    comparison = timing_comparison(payloads)
    if comparison["timing_comparable"] or allow_mismatch:
        return comparison
    for role in RUN_ROLES:
        row = comparison["runs"][role]
        if row["exact_match_to_oracle"]:
            continue
        mismatch = row["first_mismatch"]
        if row["event_count"] != comparison["oracle_event_count"]:
            raise ValueError(
                f"{role} timing is incompatible: {row['event_count']} events, "
                f"expected {comparison['oracle_event_count']}; first mismatch={mismatch}"
            )
        raise ValueError(
            f"{role} timing is incompatible at event {mismatch['event_index']}: "
            f"{tuple(mismatch['actual'])}, expected {tuple(mismatch['oracle'])}"
        )
    raise AssertionError("unreachable timing comparison state")


def _require_bleu(metrics: Mapping[str, Any], *, role: str, mask_name: str) -> None:
    if metrics.get("bleu_error"):
        raise RuntimeError(f"{role} {mask_name} BLEU failed: {metrics['bleu_error']}")
    missing = [key for key in ("bleu", "masked_terms_bleu") if key not in metrics]
    if missing:
        raise RuntimeError(f"{role} {mask_name} BLEU omitted: {', '.join(missing)}")


def _score_quality(
    payload: Mapping[str, Any],
    *,
    role: str,
    reference_text: str,
    technical_gold: Sequence[TimedOccurrence],
    raw_gold: Sequence[TimedOccurrence],
    sacrebleu_tokenizer: str,
) -> Dict[str, Any]:
    hypothesis = run_hypothesis_text(dict(payload))
    technical = compute_bleu_scores(
        hypothesis=hypothesis,
        reference=reference_text,
        target_terms=target_terms_from_occurrences(technical_gold),
        sacrebleu_tokenizer=sacrebleu_tokenizer,
    )
    raw = compute_bleu_scores(
        hypothesis=hypothesis,
        reference=reference_text,
        target_terms=target_terms_from_occurrences(raw_gold),
        sacrebleu_tokenizer=sacrebleu_tokenizer,
    )
    _require_bleu(technical, role=role, mask_name="technical")
    _require_bleu(raw, role=role, mask_name="raw")
    if technical["bleu"] != raw["bleu"]:
        raise RuntimeError(
            f"{role} produced inconsistent unmasked BLEU across mask sets: "
            f"{technical['bleu']} != {raw['bleu']}"
        )
    return {
        "bleu": raw["bleu"],
        "technical_masked_bleu": technical["masked_terms_bleu"],
        "raw_masked_bleu": raw["masked_terms_bleu"],
        "technical_mask_diagnostics": {
            "target_types": technical.get("masked_terms_types"),
            "hypothesis_mentions_removed": technical.get("masked_terms_hyp_removed"),
            "reference_mentions_removed": technical.get("masked_terms_ref_removed"),
        },
        "raw_mask_diagnostics": {
            "target_types": raw.get("masked_terms_types"),
            "hypothesis_mentions_removed": raw.get("masked_terms_hyp_removed"),
            "reference_mentions_removed": raw.get("masked_terms_ref_removed"),
        },
        "hypothesis_chars": len(hypothesis),
        "hypothesis_sha256": hashlib.sha256(hypothesis.encode("utf-8")).hexdigest(),
    }


def assemble_scorecard(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    run_inputs: Mapping[str, Mapping[str, str]],
    gold_sets: Mapping[str, Sequence[TimedOccurrence]],
    reference_text: str,
    target_lang: str,
    sacrebleu_tokenizer: str,
    post_s: float,
    prompt_lookback_s: float,
    prompt_alignment_tolerance_s: float,
    expected_raw_denominator: int | None = None,
    allow_timing_mismatch: bool = False,
    selected_window_smoke: bool = False,
) -> Dict[str, Any]:
    """Assemble a scorecard from already-loaded and validated run payloads."""

    missing = [role for role in RUN_ROLES if role not in payloads]
    extra = [role for role in payloads if role not in RUN_ROLES]
    if missing or extra:
        raise ValueError(f"runs must be exactly {RUN_ROLES}; missing={missing}, extra={extra}")
    validate_same_playlist([payloads[role] for role in RUN_ROLES])
    timing = validate_timing_compatible(payloads, allow_mismatch=allow_timing_mismatch)
    if selected_window_smoke:
        for role in RUN_ROLES:
            try:
                validate_selected_window_payload(payloads[role])
            except ValueError as exc:
                raise ValueError(f"{role} selected-window protocol is invalid: {exc}") from exc
        if expected_raw_denominator is None:
            expected_raw_denominator = SELECTED_WINDOW_RAW_DENOMINATOR
        elif expected_raw_denominator != SELECTED_WINDOW_RAW_DENOMINATOR:
            raise ValueError(
                f"selected-window protocol requires raw denominator "
                f"{SELECTED_WINDOW_RAW_DENOMINATOR}, got {expected_raw_denominator}"
            )
    if TECHNICAL_GOLD_KEY not in gold_sets or HEADLINE_GOLD_KEY not in gold_sets:
        raise ValueError(
            f"gold_sets must contain {TECHNICAL_GOLD_KEY!r} and {HEADLINE_GOLD_KEY!r}"
        )
    technical_gold = list(gold_sets[TECHNICAL_GOLD_KEY])
    raw_gold = list(gold_sets[HEADLINE_GOLD_KEY])
    if not raw_gold:
        raise ValueError("raw_plus_medicine MFA denominator is empty")
    raw_denominator = len(raw_gold)
    if expected_raw_denominator is not None and raw_denominator != expected_raw_denominator:
        raise ValueError(
            f"raw_plus_medicine MFA denominator is {raw_denominator}, "
            f"expected {expected_raw_denominator}"
        )
    if not reference_text.strip():
        raise ValueError("reference text is empty; BLEU scorecard cannot be built")
    if post_s < 0.0:
        raise ValueError("post_s must be non-negative")

    playlist = playlist_signature(payloads[RUN_ROLES[0]])
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "required_conditions": list(RUN_ROLES),
            "playlist": {
                "identical": True,
                "sha256": _stable_sha256(playlist),
            },
            "timing": {
                **timing,
                "mismatch_explicitly_allowed": bool(allow_timing_mismatch),
                "paired_delta_claims_allowed": bool(timing["timing_comparable"]),
            },
            "headline_term_accuracy": {
                "name": "MFA time-aligned raw_plus_medicine occurrence accuracy",
                "scorer": "score_time_aligned_terms.score_run",
                "matching": "time-local greedy one-to-one output occurrence matching",
                "pre_s": mfa_term_scorer.PRE_S,
                "post_s": float(post_s),
                "fixed_denominator": raw_denominator,
                "gold_sha256": _occurrence_fingerprint(raw_gold),
                "expected_denominator_assertion": expected_raw_denominator,
            },
            "quality": {
                "metrics": ["BLEU", "technical-masked BLEU", "raw-masked BLEU"],
                "sacrebleu_tokenizer": sacrebleu_tokenizer,
                "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            },
            "prompt_precision": {
                "scorer": "score_prompt_precision.score_payload",
                "strict_complete_reference_capture": True,
                "time_local": True,
                "lookback_s": float(prompt_lookback_s),
                "alignment_tolerance_s": float(prompt_alignment_tolerance_s),
            },
            "excluded_metrics": [
                {
                    "name": "legacy 419-denominator TERM_ACC",
                    "status": "not computed",
                    "reason": "denominator is not the fixed raw_plus_medicine MFA occurrence set",
                },
                {
                    "name": "block-level count-clipping TERM_ACC",
                    "status": "not computed",
                    "reason": "headline uses time-local MFA one-to-one matching only",
                },
            ],
        },
        "inputs": {role: dict(run_inputs[role]) for role in RUN_ROLES},
        "gold": {
            TECHNICAL_GOLD_KEY: _gold_summary(technical_gold),
            HEADLINE_GOLD_KEY: _gold_summary(raw_gold),
        },
        "runs": {},
    }
    if selected_window_smoke:
        report["protocol"]["selected_window_smoke"] = selected_window_protocol_manifest()

    original_post_s = mfa_term_scorer.POST_S
    mfa_term_scorer.POST_S = float(post_s)
    try:
        for role in RUN_ROLES:
            payload = payloads[role]
            term_metrics = mfa_term_scorer.score_run(dict(payload), raw_gold, target_lang)
            if term_metrics["gold_occurrences"] != raw_denominator:
                raise RuntimeError(
                    f"{role} TERM_ACC denominator drifted to {term_metrics['gold_occurrences']}; "
                    f"expected {raw_denominator}"
                )
            prompt = score_prompt_payload(
                payload,
                {
                    TECHNICAL_GOLD_KEY: technical_gold,
                    HEADLINE_GOLD_KEY: raw_gold,
                },
                lookback_s=prompt_lookback_s,
                tolerance_s=prompt_alignment_tolerance_s,
                require_complete_reference_capture=True,
            )
            report["runs"][role] = {
                "preset": prompt.get("preset", ""),
                "headline_term_acc": {
                    "protocol": "mfa_time_aligned_raw_plus_medicine",
                    **term_metrics,
                },
                "quality": _score_quality(
                    payload,
                    role=role,
                    reference_text=reference_text,
                    technical_gold=technical_gold,
                    raw_gold=raw_gold,
                    sacrebleu_tokenizer=sacrebleu_tokenizer,
                ),
                "prompt": {
                    "strict_time_local_raw_precision": prompt[HEADLINE_GOLD_KEY]["prompt_precision"],
                    "strict_time_local_technical_precision": prompt[TECHNICAL_GOLD_KEY]["prompt_precision"],
                    "retrieved_references_per_chunk": prompt["retrieved_references_per_chunk"],
                    "prompt_reference_count": prompt["prompt_reference_count"],
                    "chunk_count": prompt["chunk_count"],
                    "reference_capture_ratio": prompt["reference_capture_ratio"],
                    "reference_count_mismatch_chunks": prompt["reference_count_mismatch_chunks"],
                    "raw_relevant_references": prompt[HEADLINE_GOLD_KEY]["relevant_references"],
                    "technical_relevant_references": prompt[TECHNICAL_GOLD_KEY]["relevant_references"],
                },
                "timing_signature_sha256": timing_signature_sha256(payload),
            }
    finally:
        mfa_term_scorer.POST_S = original_post_s
    return report


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    paths = {
        "oracle": Path(args.oracle).expanduser().resolve(),
        "autoterm": Path(args.autoterm).expanduser().resolve(),
        "merged": Path(args.merged).expanduser().resolve(),
    }
    payloads: Dict[str, Mapping[str, Any]] = {}
    run_inputs: Dict[str, Dict[str, str]] = {}
    for role in RUN_ROLES:
        path = paths[role]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{role} input must contain a JSON object: {path}")
        payloads[role] = payload
        run_inputs[role] = {"path": str(path), "sha256": _file_sha256(path)}

    validate_same_playlist([payloads[role] for role in RUN_ROLES])
    validate_timing_compatible(payloads, allow_mismatch=args.allow_timing_mismatch)
    gold_sets = mfa_term_scorer.build_timed_gold(dict(payloads[RUN_ROLES[0]]), args)
    reference_text = build_reference_text(dict(payloads[RUN_ROLES[0]]), args)
    return assemble_scorecard(
        payloads,
        run_inputs=run_inputs,
        gold_sets=gold_sets,
        reference_text=reference_text,
        target_lang=args.target_lang,
        sacrebleu_tokenizer=args.sacrebleu_tokenizer,
        post_s=args.post_s,
        prompt_lookback_s=args.prompt_lookback_s,
        prompt_alignment_tolerance_s=args.prompt_alignment_tolerance_s,
        expected_raw_denominator=(
            args.expected_raw_denominator if args.expected_raw_denominator > 0 else None
        ),
        allow_timing_mismatch=args.allow_timing_mismatch,
        selected_window_smoke=bool(getattr(args, "selected_window_smoke", False)),
    )


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{100.0 * float(value):.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(report: Mapping[str, Any]) -> str:
    term_protocol = report["protocol"]["headline_term_accuracy"]
    timing = report["protocol"]["timing"]
    if timing["timing_comparable"]:
        timing_lines = [
            f"- Playlist and timing checks: identical playlist; "
            f"{timing['oracle_event_count']} exact decoder windows.",
        ]
    else:
        timing_lines = [
            "- **Timing mismatch was explicitly allowed: these are unpaired per-condition scores.**",
            "- `timing_comparable=false`; paired differences, deltas, and superiority claims are forbidden.",
        ]
    lines = [
        "# AutoTerm Story Scorecard",
        "",
        "**Headline TERM_ACC uses only MFA time-aligned `raw_plus_medicine` occurrences.**",
        "",
        f"- Fixed TERM_ACC denominator: `{term_protocol['fixed_denominator']}` "
        f"(gold SHA-256: `{term_protocol['gold_sha256']}`).",
        *timing_lines,
        "- Excluded and not computed: legacy 419-denominator TERM_ACC and block-level count-clipping TERM_ACC.",
        "- Prompt precision is strict and time-local; captured references must exactly match the decoder prompt count.",
        "",
        "| condition | TERM_ACC (%) | hits / fixed gold | BLEU | technical-masked BLEU | raw-masked BLEU | raw prompt precision (%) | technical prompt precision (%) | refs/chunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_names = {"oracle": "Oracle", "autoterm": "AutoTerm", "merged": "Merged glossary"}
    for role in RUN_ROLES:
        row = report["runs"][role]
        term = row["headline_term_acc"]
        quality = row["quality"]
        prompt = row["prompt"]
        lines.append(
            f"| {display_names[role]} | {_fmt(term['term_acc'], percent=True)} | "
            f"{term['hits']} / {term['gold_occurrences']} | {_fmt(quality['bleu'])} | "
            f"{_fmt(quality['technical_masked_bleu'])} | {_fmt(quality['raw_masked_bleu'])} | "
            f"{_fmt(prompt['strict_time_local_raw_precision'], percent=True)} | "
            f"{_fmt(prompt['strict_time_local_technical_precision'], percent=True)} | "
            f"{_fmt(prompt['retrieved_references_per_chunk'])} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion records input hashes, playlist/timing signatures, mask diagnostics, "
            "the fixed gold fingerprint, and all protocol constants needed to reproduce this table.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, help="blockwise-oracle mixed-run JSON")
    parser.add_argument("--autoterm", required=True, help="budgeted AutoTerm mixed-run JSON")
    parser.add_argument("--merged", required=True, help="merged-glossary mixed-run JSON")
    parser.add_argument("--mfa-root", required=True)
    parser.add_argument("--acl-root", required=True, help="ACL segments root with segments.meta.jsonl")
    parser.add_argument("--acl-reference-text", required=True)
    parser.add_argument("--acl-technical-gold", required=True)
    parser.add_argument("--acl-raw-glossary", required=True)
    parser.add_argument("--medicine-oracle-dir", required=True)
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--sacrebleu-tokenizer", default="zh")
    parser.add_argument("--post-s", type=float, default=30.0)
    parser.add_argument("--prompt-lookback-s", type=float, default=1.92)
    parser.add_argument("--prompt-alignment-tolerance-s", type=float, default=0.0)
    parser.add_argument(
        "--expected-raw-denominator",
        type=int,
        default=0,
        help="optional positive assertion for the fixed raw_plus_medicine MFA denominator",
    )
    parser.add_argument(
        "--allow-timing-mismatch",
        action="store_true",
        help=(
            "score nonidentical event windows as unpaired conditions; marks timing_comparable=false "
            "and forbids paired/delta claims"
        ),
    )
    parser.add_argument(
        "--selected-window-smoke",
        action="store_true",
        help=(
            f"require {SELECTED_WINDOW_PROTOCOL_ID} metadata and enforce its fixed "
            f"raw denominator {SELECTED_WINDOW_RAW_DENOMINATOR}"
        ),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_raw_denominator < 0:
        raise ValueError("expected_raw_denominator must be non-negative")
    report = build_report(args)
    json_path = Path(args.out_json).expanduser()
    markdown_path = Path(args.out_md).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
