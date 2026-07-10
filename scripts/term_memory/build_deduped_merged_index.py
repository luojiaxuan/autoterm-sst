#!/usr/bin/env python3
"""Build a globally source-deduplicated glossary and aligned MaxSim index.

Inputs are ordered glossary/index pairs.  The first occurrence of a normalized
English source term wins, while every later occurrence is retained in a full
duplicate audit.  An optional distractor pair can deterministically top up the
deduplicated union to an exact unique-term target without re-encoding text.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.source_normalization import (  # noqa: E402
    SOURCE_NORMALIZATION_POLICY,
    normalize_english_source,
)
from scripts.term_memory.build_topic_slice_catalog import iter_glossary, source_term  # noqa: E402


@dataclass(frozen=True)
class SourceSpec:
    role: str
    glossary_path: Path
    index_path: Path
    build_report_path: Path | None = None
    kind: str = "base"


@dataclass
class LoadedPair:
    glossary: List[Dict[str, Any]]
    index_terms: List[Dict[str, Any]]
    text_embs: torch.Tensor
    checkpoint_evidence: Dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_fingerprint(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_language(language_pair: str) -> str:
    parts = [part.strip() for part in language_pair.split("-") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"invalid --language-pair {language_pair!r}; expected e.g. en-zh")
    return parts[1]


def _target_translation(entry: Mapping[str, Any], target_lang: str) -> str:
    values = entry.get("target_translations")
    if isinstance(values, Mapping):
        return str(values.get(target_lang) or "").strip()
    if str(entry.get("target_lang") or "").strip().casefold() == target_lang.casefold():
        return str(entry.get("target_label") or entry.get("translation") or "").strip()
    return str(entry.get(target_lang) or entry.get("translation") or "").strip()


def _translation_map(entry: Mapping[str, Any], target_lang: str) -> Dict[str, str]:
    values = entry.get("target_translations")
    if isinstance(values, Mapping):
        return {
            str(language): str(value).strip()
            for language, value in sorted(values.items(), key=lambda item: str(item[0]))
            if str(value).strip()
        }
    target = _target_translation(entry, target_lang)
    return {target_lang: target} if target else {}


def _canonical_target(value: str) -> str:
    return " ".join(str(value).split())


def _index_source_term(entry: Mapping[str, Any]) -> str:
    return str(entry.get("term") or entry.get("source_label") or entry.get("key") or "").strip()


def _checkpoint_evidence(
    payload: Mapping[str, Any],
    spec: SourceSpec,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    metadata = payload.get("build_metadata") or payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    embedded_sha = str(
        metadata.get("embedding_checkpoint_sha256")
        or metadata.get("checkpoint_sha256")
        or ""
    )
    if embedded_sha and embedded_sha != checkpoint_sha256:
        raise ValueError(
            f"{spec.role}: index checkpoint SHA {embedded_sha} != requested {checkpoint_sha256}"
        )

    evidence: Dict[str, Any] = {
        "status": "embedded_sha256_verified" if embedded_sha else "caller_declared_legacy_payload",
        "embedded_checkpoint_sha256": embedded_sha,
        "build_report_path": str(spec.build_report_path) if spec.build_report_path else "",
    }
    if spec.build_report_path:
        report = json.loads(spec.build_report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError(f"{spec.role}: build report must be a JSON object")
        raw_report_model = str(report.get("model_path") or "").strip()
        if not raw_report_model:
            raise ValueError(f"{spec.role}: build report has no model_path")
        report_model = Path(raw_report_model).expanduser()
        try:
            same_checkpoint = report_model.resolve() == checkpoint_path.resolve()
        except OSError:
            same_checkpoint = str(report_model) == str(checkpoint_path)
        if not same_checkpoint:
            raise ValueError(
                f"{spec.role}: build-report model_path {report_model} != {checkpoint_path}"
            )
        evidence.update(
            {
                "status": "build_report_and_embedded_sha256_verified"
                if embedded_sha
                else "build_report_path_verified",
                "build_report_model_path": str(report_model),
                "build_report_sha256": file_sha256(spec.build_report_path),
            }
        )
    return evidence


def load_pair(
    spec: SourceSpec,
    *,
    target_lang: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> LoadedPair:
    glossary = list(iter_glossary(spec.glossary_path))
    try:
        payload = torch.load(
            spec.index_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(spec.index_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{spec.role}: index payload must be a mapping")
    text_embs = payload.get("text_embs")
    index_terms = payload.get("term_list")
    if not isinstance(text_embs, torch.Tensor) or text_embs.ndim < 2:
        raise ValueError(f"{spec.role}: text_embs must be a rank >= 2 tensor")
    if not isinstance(index_terms, list) or not all(isinstance(row, Mapping) for row in index_terms):
        raise ValueError(f"{spec.role}: term_list must be a list of objects")
    if text_embs.shape[0] != len(index_terms) or len(index_terms) != len(glossary):
        raise ValueError(
            f"{spec.role}: row mismatch glossary={len(glossary)}, "
            f"term_list={len(index_terms)}, text_embs={text_embs.shape[0]}"
        )

    clean_index_terms: List[Dict[str, Any]] = []
    for row_index, (glossary_entry, index_entry_raw) in enumerate(zip(glossary, index_terms)):
        index_entry = dict(index_entry_raw)
        glossary_key = normalize_english_source(source_term(glossary_entry))
        index_key = normalize_english_source(_index_source_term(index_entry))
        if not glossary_key or glossary_key != index_key:
            raise ValueError(
                f"{spec.role}: row {row_index} glossary/index term mismatch "
                f"({glossary_key!r} != {index_key!r})"
            )
        glossary_target = _target_translation(glossary_entry, target_lang)
        index_target = _target_translation(index_entry, target_lang)
        if not glossary_target:
            raise ValueError(f"{spec.role}: row {row_index} lacks {target_lang} translation")
        if not index_target or _canonical_target(glossary_target) != _canonical_target(index_target):
            raise ValueError(
                f"{spec.role}: row {row_index} glossary/index {target_lang} translation mismatch"
            )
        clean_index_terms.append(index_entry)

    evidence = _checkpoint_evidence(
        payload,
        spec,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    return LoadedPair(
        glossary=glossary,
        index_terms=clean_index_terms,
        text_embs=text_embs,
        checkpoint_evidence=evidence,
    )


def _occurrence(
    spec: SourceSpec,
    row_index: int,
    glossary_entry: Mapping[str, Any],
    index_entry: Mapping[str, Any],
    *,
    target_lang: str,
) -> Dict[str, Any]:
    return {
        "source_role": spec.role,
        "source_kind": spec.kind,
        "row_index": row_index,
        "source_term": source_term(glossary_entry),
        "target_translation": _target_translation(glossary_entry, target_lang),
        "target_translations": _translation_map(glossary_entry, target_lang),
        "glossary_entry": dict(glossary_entry),
        "index_term_entry": dict(index_entry),
    }


def _record_duplicate(
    duplicate_occurrences: Dict[str, List[Dict[str, Any]]],
    first_occurrence: Mapping[str, Dict[str, Any]],
    key: str,
    occurrence: Dict[str, Any],
) -> None:
    if key not in duplicate_occurrences:
        duplicate_occurrences[key] = [dict(first_occurrence[key])]
    duplicate_occurrences[key].append(occurrence)


def _audit_entries(
    duplicate_occurrences: Mapping[str, Sequence[Mapping[str, Any]]],
    output_winner: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    audit: List[Dict[str, Any]] = []
    for key in sorted(duplicate_occurrences):
        occurrences = list(duplicate_occurrences[key])
        targets = sorted({_canonical_target(row["target_translation"]) for row in occurrences})
        translation_maps = {
            json.dumps(row["target_translations"], ensure_ascii=False, sort_keys=True)
            for row in occurrences
        }
        surfaces = sorted({str(row["source_term"]) for row in occurrences})
        roles = sorted({str(row["source_role"]) for row in occurrences})
        audit.append(
            {
                "normalized_source": key,
                "output_winner": dict(output_winner[key]) if key in output_winner else None,
                "source_roles": roles,
                "surface_forms": surfaces,
                "target_variants": targets,
                "target_variant_conflict": len(targets) > 1,
                "translation_map_conflict": len(translation_maps) > 1,
                "occurrences": occurrences,
            }
        )
    return audit


def _pair_overlap_counts(audit: Sequence[Mapping[str, Any]], *, base_only: bool) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in audit:
        rows = item["occurrences"]
        roles = sorted(
            {
                str(row["source_role"])
                for row in rows
                if not base_only or row["source_kind"] == "base"
            }
        )
        for left, right in itertools.combinations(roles, 2):
            counts[f"{left}|{right}"] += 1
    return dict(sorted(counts.items()))


def build_deduped_merged_index(
    *,
    sources: Sequence[SourceSpec],
    topup_source: SourceSpec | None,
    target_size: int | None,
    out_dir: Path,
    preset_id: str,
    language_pair: str,
    embedding_checkpoint: Path,
    label: str = "",
    description: str = "",
    strict_checkpoint_evidence: bool = False,
) -> Dict[str, Any]:
    if not sources:
        raise ValueError("at least one --source is required")
    roles = [spec.role for spec in sources] + ([topup_source.role] if topup_source else [])
    if any(not role.strip() for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("source roles must be non-empty and globally unique")
    if target_size is not None and target_size <= 0:
        raise ValueError("--target-size must be positive")
    if topup_source is not None and target_size is None:
        raise ValueError("--topup-source requires --target-size")
    target_lang = _target_language(language_pair)
    checkpoint_sha256 = file_sha256(embedding_checkpoint)

    expected_embedding_shape: tuple[int, ...] | None = None
    expected_dtype: torch.dtype | None = None
    output_glossary: List[Dict[str, Any]] = []
    output_index_terms: List[Dict[str, Any]] = []
    output_tensor_chunks: List[torch.Tensor] = []
    output_keys: List[str] = []
    output_winner: Dict[str, Dict[str, Any]] = {}
    first_occurrence: Dict[str, Dict[str, Any]] = {}
    duplicate_occurrences: Dict[str, List[Dict[str, Any]]] = {}
    base_keys: set[str] = set()
    source_reports: List[Dict[str, Any]] = []

    def validate_tensor(spec: SourceSpec, loaded: LoadedPair) -> None:
        nonlocal expected_embedding_shape, expected_dtype
        shape = tuple(int(value) for value in loaded.text_embs.shape[1:])
        if expected_embedding_shape is None:
            expected_embedding_shape = shape
            expected_dtype = loaded.text_embs.dtype
        elif shape != expected_embedding_shape:
            raise ValueError(
                f"{spec.role}: incompatible embedding shape {shape}; expected {expected_embedding_shape}"
            )
        elif loaded.text_embs.dtype != expected_dtype:
            raise ValueError(
                f"{spec.role}: incompatible embedding dtype {loaded.text_embs.dtype}; "
                f"expected {expected_dtype}"
            )
        if (
            strict_checkpoint_evidence
            and loaded.checkpoint_evidence["status"] == "caller_declared_legacy_payload"
        ):
            raise ValueError(f"{spec.role}: index has no independent checkpoint provenance")

    for priority, spec in enumerate(sources):
        loaded = load_pair(
            spec,
            target_lang=target_lang,
            checkpoint_path=embedding_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        validate_tensor(spec, loaded)
        selected_indices: List[int] = []
        within_source_keys: set[str] = set()
        duplicates_before = sum(len(rows) - 1 for rows in duplicate_occurrences.values())
        for row_index, (glossary_entry, index_entry) in enumerate(
            zip(loaded.glossary, loaded.index_terms)
        ):
            key = normalize_english_source(source_term(glossary_entry))
            occurrence = _occurrence(
                spec,
                row_index,
                glossary_entry,
                index_entry,
                target_lang=target_lang,
            )
            if key in first_occurrence:
                _record_duplicate(duplicate_occurrences, first_occurrence, key, occurrence)
            else:
                first_occurrence[key] = occurrence
            if key in base_keys:
                within_source_keys.add(key)
                continue
            base_keys.add(key)
            within_source_keys.add(key)
            selected_indices.append(row_index)
            output_keys.append(key)
            output_glossary.append(glossary_entry)
            output_index_terms.append(index_entry)
            output_winner[key] = {
                "source_role": spec.role,
                "source_kind": spec.kind,
                "row_index": row_index,
                "priority": priority,
            }
        if selected_indices:
            indices = torch.tensor(selected_indices, dtype=torch.long)
            output_tensor_chunks.append(loaded.text_embs.index_select(0, indices).clone())
        duplicate_rows = (
            sum(len(rows) - 1 for rows in duplicate_occurrences.values()) - duplicates_before
        )
        source_reports.append(
            {
                "role": spec.role,
                "kind": spec.kind,
                "priority": priority,
                "glossary_path": str(spec.glossary_path),
                "glossary_sha256": file_sha256(spec.glossary_path),
                "index_path": str(spec.index_path),
                "index_sha256": file_sha256(spec.index_path),
                "input_rows": len(loaded.glossary),
                "input_unique_terms": len(within_source_keys),
                "selected_unique_terms": len(selected_indices),
                "duplicate_rows": duplicate_rows,
                "embedding_shape": list(loaded.text_embs.shape),
                "embedding_dtype": str(loaded.text_embs.dtype),
                "checkpoint_evidence": loaded.checkpoint_evidence,
            }
        )
        del loaded

    base_unique_terms = len(output_keys)
    base_audit = _audit_entries(duplicate_occurrences, output_winner)
    base_audit_summary = {
        "duplicate_term_count": len(base_audit),
        "target_variant_conflict_term_count": sum(
            bool(item["target_variant_conflict"]) for item in base_audit
        ),
        "translation_map_conflict_term_count": sum(
            bool(item["translation_map_conflict"]) for item in base_audit
        ),
        "pair_overlap_term_counts": _pair_overlap_counts(base_audit, base_only=True),
    }
    del base_audit
    if target_size is not None and target_size < base_unique_terms:
        raise ValueError(
            f"--target-size {target_size} is smaller than base deduplicated union {base_unique_terms}; "
            "this builder never truncates base sources"
        )
    needed = (target_size - base_unique_terms) if target_size is not None else 0
    if needed and topup_source is None:
        raise ValueError(f"need {needed} unique top-up terms but no --topup-source was provided")

    if topup_source is not None:
        spec = topup_source
        loaded = load_pair(
            spec,
            target_lang=target_lang,
            checkpoint_path=embedding_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        validate_tensor(spec, loaded)
        topup_seen: set[str] = set()
        selected_indices: List[int] = []
        collision_base = 0
        duplicate_within_topup = 0
        unselected_unique = 0
        for row_index, (glossary_entry, index_entry) in enumerate(
            zip(loaded.glossary, loaded.index_terms)
        ):
            key = normalize_english_source(source_term(glossary_entry))
            occurrence = _occurrence(
                spec,
                row_index,
                glossary_entry,
                index_entry,
                target_lang=target_lang,
            )
            if key in first_occurrence:
                _record_duplicate(duplicate_occurrences, first_occurrence, key, occurrence)
            else:
                first_occurrence[key] = occurrence
            if key in base_keys:
                collision_base += 1
                continue
            if key in topup_seen:
                duplicate_within_topup += 1
                continue
            topup_seen.add(key)
            if len(selected_indices) >= needed:
                unselected_unique += 1
                continue
            selected_indices.append(row_index)
            output_keys.append(key)
            output_glossary.append(glossary_entry)
            output_index_terms.append(index_entry)
            output_winner[key] = {
                "source_role": spec.role,
                "source_kind": spec.kind,
                "row_index": row_index,
                "priority": len(sources),
            }
        if len(selected_indices) != needed:
            raise ValueError(
                f"top-up source has only {len(selected_indices)} eligible unique terms; need {needed}"
            )
        if selected_indices:
            indices = torch.tensor(selected_indices, dtype=torch.long)
            output_tensor_chunks.append(loaded.text_embs.index_select(0, indices).clone())
        source_reports.append(
            {
                "role": spec.role,
                "kind": spec.kind,
                "priority": len(sources),
                "glossary_path": str(spec.glossary_path),
                "glossary_sha256": file_sha256(spec.glossary_path),
                "index_path": str(spec.index_path),
                "index_sha256": file_sha256(spec.index_path),
                "input_rows": len(loaded.glossary),
                "input_unique_terms_excluding_base": len(topup_seen),
                "selected_unique_terms": len(selected_indices),
                "collisions_with_base_rows": collision_base,
                "duplicate_rows_within_topup": duplicate_within_topup,
                "eligible_unique_terms_not_selected": unselected_unique,
                "embedding_shape": list(loaded.text_embs.shape),
                "embedding_dtype": str(loaded.text_embs.dtype),
                "checkpoint_evidence": loaded.checkpoint_evidence,
            }
        )
        del loaded

    expected_size = target_size if target_size is not None else base_unique_terms
    if len(output_keys) != expected_size or len(set(output_keys)) != expected_size:
        raise RuntimeError(
            f"unique output invariant failed: rows={len(output_keys)}, "
            f"unique={len(set(output_keys))}, expected={expected_size}"
        )
    merged_embs = torch.cat(output_tensor_chunks, dim=0)
    if merged_embs.shape[0] != expected_size:
        raise RuntimeError("embedding rows do not match deduplicated output rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    glossary_path = out_dir / "glossary.json"
    index_path = out_dir / "maxsim.pt"
    audit_path = out_dir / "duplicate_audit.json"
    metadata_path = out_dir / "manifest_fragment.json"
    report_path = out_dir / "build_report.json"

    glossary_path.write_text(
        json.dumps(output_glossary, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    index_metadata = {
        "builder": "scripts/term_memory/build_deduped_merged_index.py",
        "preset_id": preset_id,
        "language_pair": language_pair,
        "term_count": expected_size,
        "base_unique_term_count": base_unique_terms,
        "embedding_checkpoint_path": str(embedding_checkpoint),
        "embedding_checkpoint_sha256": checkpoint_sha256,
        "source_normalization": SOURCE_NORMALIZATION_POLICY,
        "source_term_fingerprint_sha256": identity_fingerprint(output_keys),
    }
    torch.save(
        {
            "text_embs": merged_embs,
            "term_list": output_index_terms,
            "build_metadata": index_metadata,
        },
        index_path,
    )

    audit = _audit_entries(duplicate_occurrences, output_winner)
    audit_payload = {
        "conflict_policy": "first base source in CLI order wins; top-up only fills unseen keys",
        "source_normalization": SOURCE_NORMALIZATION_POLICY,
        "duplicate_terms": audit,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_fragment = {
        "scales": {
            preset_id: {
                language_pair: {
                    "terms_path": str(glossary_path.resolve()),
                    "indexes": {"maxsim": str(index_path.resolve())},
                    "num_terms": expected_size,
                }
            }
        },
        "preset_meta": {
            preset_id: {
                "id": preset_id,
                "preset_id": preset_id,
                "label": label or preset_id.replace("_", " ").title(),
                "domain": "merged",
                "domain_id": "merged",
                "description": description or "Globally source-deduplicated merged glossary.",
                "term_count": expected_size,
                "maxsim_index_path": str(index_path.resolve()),
                "source_term_fingerprint_sha256": identity_fingerprint(output_keys),
                "enabled_for_auto_router": False,
            }
        },
    }
    metadata_path.write_text(
        json.dumps(manifest_fragment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_duplicate_rows = sum(
        spec_report.get("duplicate_rows", 0)
        for spec_report in source_reports
        if spec_report["kind"] == "base"
    )
    report: Dict[str, Any] = {
        "status": "complete",
        "preset_id": preset_id,
        "language_pair": language_pair,
        "source_normalization": SOURCE_NORMALIZATION_POLICY,
        "conflict_policy": "first base source in CLI order wins; top-up only fills unseen keys",
        "base_input_rows": sum(
            item["input_rows"] for item in source_reports if item["kind"] == "base"
        ),
        "base_unique_terms": base_unique_terms,
        "base_duplicate_rows": base_duplicate_rows,
        "base_duplicate_term_count": base_audit_summary["duplicate_term_count"],
        "base_target_variant_conflict_term_count": base_audit_summary[
            "target_variant_conflict_term_count"
        ],
        "base_translation_map_conflict_term_count": base_audit_summary[
            "translation_map_conflict_term_count"
        ],
        "output_term_count": expected_size,
        "topup_term_count": expected_size - base_unique_terms,
        "all_duplicate_term_count": len(audit),
        "all_target_variant_conflict_term_count": sum(
            bool(item["target_variant_conflict"]) for item in audit
        ),
        "base_pair_overlap_term_counts": base_audit_summary["pair_overlap_term_counts"],
        "all_pair_overlap_term_counts": _pair_overlap_counts(audit, base_only=False),
        "source_roles": source_reports,
        "embedding": {
            "trailing_shape": list(expected_embedding_shape or ()),
            "dtype": str(expected_dtype),
            "checkpoint_path": str(embedding_checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "fingerprints": {
            "ordered_normalized_source_terms_sha256": identity_fingerprint(output_keys),
            "glossary_sha256": file_sha256(glossary_path),
            "index_sha256": file_sha256(index_path),
            "duplicate_audit_sha256": file_sha256(audit_path),
            "manifest_fragment_sha256": file_sha256(metadata_path),
        },
        "outputs": {
            "glossary": str(glossary_path),
            "index": str(index_path),
            "duplicate_audit": str(audit_path),
            "manifest_fragment": str(metadata_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_report_paths(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role.strip() or not raw_path.strip():
            raise ValueError(f"invalid --build-report {value!r}; expected role=/path/report.json")
        if role in result:
            raise ValueError(f"duplicate --build-report role {role!r}")
        result[role] = Path(raw_path).expanduser()
    return result


def _source_specs(
    values: Sequence[Sequence[str]],
    reports: Mapping[str, Path],
    *,
    kind: str,
) -> List[SourceSpec]:
    return [
        SourceSpec(
            role=role,
            glossary_path=Path(glossary).expanduser(),
            index_path=Path(index).expanduser(),
            build_report_path=reports.get(role),
            kind=kind,
        )
        for role, glossary, index in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("ROLE", "GLOSSARY", "INDEX"),
        default=[],
        help="repeat in deterministic winner-priority order",
    )
    parser.add_argument(
        "--topup-source",
        nargs=3,
        metavar=("ROLE", "GLOSSARY", "INDEX"),
        help="separate distractor pair used only to reach --target-size",
    )
    parser.add_argument(
        "--build-report",
        action="append",
        default=[],
        help="optional role=/path/index_build_report.json checkpoint evidence",
    )
    parser.add_argument("--target-size", type=int)
    parser.add_argument("--embedding-checkpoint", required=True, type=Path)
    parser.add_argument("--strict-checkpoint-evidence", action="store_true")
    parser.add_argument("--language-pair", default="en-zh")
    parser.add_argument("--preset-id", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        reports = _parse_report_paths(args.build_report)
        sources = _source_specs(args.source, reports, kind="base")
        topup = None
        if args.topup_source:
            topup = _source_specs([args.topup_source], reports, kind="topup")[0]
        known_roles = {spec.role for spec in sources} | ({topup.role} if topup else set())
        unknown_reports = sorted(set(reports) - known_roles)
        if unknown_reports:
            raise ValueError(f"--build-report has unknown roles: {unknown_reports}")
        result = build_deduped_merged_index(
            sources=sources,
            topup_source=topup,
            target_size=args.target_size,
            out_dir=args.out_dir.expanduser(),
            preset_id=args.preset_id,
            language_pair=args.language_pair,
            embedding_checkpoint=args.embedding_checkpoint.expanduser(),
            label=args.label,
            description=args.description,
            strict_checkpoint_evidence=args.strict_checkpoint_evidence,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
