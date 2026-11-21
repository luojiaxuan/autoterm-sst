#!/usr/bin/env python3
"""Score MT-BLEU from canonical mWER-segmented mixed-session outputs.

Each talk is masked only with the raw glossary for its corpus.  This matches
the RASST release definition for ACL tagged terms and medicine hard/raw terms,
without letting an unrelated domain glossary remove ordinary words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.score_terms import (
    compile_term_mask_patterns,
    load_target_terms_for_masking,
    mask_target_terms,
)


SCHEMA_VERSION = "mixed_mt_bleu.v1"


def parse_domain_glossaries(specs: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"domain glossary must be DOMAIN=PATH, got {spec!r}")
        domain, raw_path = spec.split("=", 1)
        domain = domain.strip()
        path = Path(raw_path)
        if not domain or domain in result:
            raise ValueError(f"empty or duplicate glossary domain: {domain!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[domain] = path
    if not result:
        raise ValueError("at least one domain glossary is required")
    return result


def wav_domains(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in manifest.get("blocks") or []:
        wav = str(block.get("wav") or "")
        domain = str(block.get("corpus") or "")
        if not wav or not domain or wav in result:
            raise ValueError(f"invalid or duplicate block mapping: {block!r}")
        result[wav] = domain
    if not result:
        raise ValueError("bundle manifest has no blocks")
    return result


def score(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for MT-BLEU") from exc

    manifest = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
    by_wav = wav_domains(manifest)
    glossaries = parse_domain_glossaries(args.domain_glossary)
    missing_domains = sorted(set(by_wav.values()) - set(glossaries))
    if missing_domains:
        raise ValueError(f"missing domain glossaries: {missing_domains}")

    patterns: dict[str, Any] = {}
    target_counts: dict[str, int] = {}
    for domain, path in glossaries.items():
        terms = load_target_terms_for_masking(str(path), args.target_lang)
        patterns[domain] = compile_term_mask_patterns(terms)
        target_counts[domain] = len(terms)

    hypotheses: list[str] = []
    references: list[str] = []
    removed_hyp: dict[str, int] = {domain: 0 for domain in glossaries}
    removed_ref: dict[str, int] = {domain: 0 for domain in glossaries}
    segments_by_domain: dict[str, int] = {domain: 0 for domain in glossaries}
    for line_number, raw in enumerate(
        args.segments_jsonl.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        wav = str(row.get("wav") or "")
        domain = by_wav.get(wav)
        if domain is None:
            raise ValueError(f"unknown wav at line {line_number}: {wav!r}")
        hypothesis, hyp_count = mask_target_terms(
            str(row.get("prediction") or ""), patterns[domain]
        )
        reference, ref_count = mask_target_terms(
            str(row.get("reference") or ""), patterns[domain]
        )
        hypotheses.append(hypothesis)
        references.append(reference)
        removed_hyp[domain] += hyp_count
        removed_ref[domain] += ref_count
        segments_by_domain[domain] += 1
    if not hypotheses:
        raise ValueError("resegmented corpus is empty")

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize=args.sacrebleu_tokenizer,
    ).score
    result = {
        "schema_version": SCHEMA_VERSION,
        "metric": "MT-BLEU",
        "definition": (
            "corpus BLEU after mWER resegmentation and target-term masking; "
            "each talk uses its corpus raw glossary"
        ),
        "bleu": float(bleu),
        "segments": len(hypotheses),
        "segments_by_domain": segments_by_domain,
        "target_term_types": target_counts,
        "hypothesis_terms_removed": removed_hyp,
        "reference_terms_removed": removed_ref,
        "sacrebleu_tokenizer": args.sacrebleu_tokenizer,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-jsonl", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument(
        "--domain-glossary",
        action="append",
        required=True,
        help="repeat DOMAIN=PATH; domains must match manifest block corpus values",
    )
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--sacrebleu-tokenizer", default="zh")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = score(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
