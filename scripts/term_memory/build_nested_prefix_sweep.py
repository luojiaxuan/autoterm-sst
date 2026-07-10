#!/usr/bin/env python3
"""Derive aligned nested glossary/index prefixes from existing domain slices.

The builder never re-encodes terms.  For every requested domain it validates
that the source glossary, index ``term_list``, and ``text_embs`` rows are
aligned, then writes deterministic first-N prefixes for each requested size.
It also emits a manifest fragment, artifact hashes/provenance, and an explicit
audit of supplied gold source terms at every prefix size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import (  # noqa: E402
    DOMAIN_TO_PRESET,
    WORKING_DOMAINS,
)
from framework.agents.term_memory.manifest import TermMemoryManifest  # noqa: E402
from framework.agents.term_memory.source_normalization import (  # noqa: E402
    SOURCE_NORMALIZATION_POLICY,
    normalize_english_source,
)
from scripts.term_memory.build_topic_slice_catalog import (  # noqa: E402
    iter_glossary,
    source_term,
)


@dataclass
class AlignedSource:
    domain: str
    preset_id: str
    glossary_path: Path
    index_path: Path
    glossary: List[Dict[str, Any]]
    term_list: List[Dict[str, Any]]
    text_embs: torch.Tensor
    index_build_metadata: Dict[str, Any]
    glossary_sha256: str
    index_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_fingerprint(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_translation(entry: Mapping[str, Any], target_lang: str) -> str:
    translations = entry.get("target_translations")
    if isinstance(translations, Mapping):
        value = translations.get(target_lang)
        if isinstance(value, list):
            return str(value[0]).strip() if value else ""
        return str(value or "").strip()
    if str(entry.get("target_lang") or "").strip().casefold() == target_lang.casefold():
        return str(entry.get("target_label") or entry.get("translation") or "").strip()
    return str(entry.get(target_lang) or entry.get("translation") or "").strip()


def _canonical_target(value: str) -> str:
    return " ".join(str(value).split())


def _index_source_term(entry: Mapping[str, Any]) -> str:
    return str(
        entry.get("term")
        or entry.get("source_label")
        or entry.get("key")
        or ""
    ).strip()


def _torch_load_index(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: index payload must be a mapping")
    return payload


def load_aligned_source(
    manifest: TermMemoryManifest,
    *,
    domain: str,
    target_lang: str,
    expected_source_size: int,
) -> AlignedSource:
    preset_id = DOMAIN_TO_PRESET[domain]
    snapshot = manifest.snapshot_for(preset_id, target_lang)
    if snapshot is None:
        raise ValueError(f"manifest has no en-{target_lang} snapshot for {preset_id}")
    glossary_path = Path(snapshot.terms_path)
    index_path = Path(snapshot.index_path("maxsim"))
    if not glossary_path.is_file():
        raise FileNotFoundError(f"{preset_id}: glossary not found: {glossary_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"{preset_id}: MaxSim index not found: {index_path}")

    glossary = list(iter_glossary(glossary_path))
    payload = _torch_load_index(index_path)
    raw_terms = payload.get("term_list")
    text_embs = payload.get("text_embs")
    if not isinstance(raw_terms, list) or not all(
        isinstance(row, Mapping) for row in raw_terms
    ):
        raise ValueError(f"{preset_id}: term_list must be a list of objects")
    if not isinstance(text_embs, torch.Tensor) or text_embs.ndim != 2:
        raise ValueError(f"{preset_id}: text_embs must be a rank-2 tensor")
    if len(glossary) != len(raw_terms) or len(raw_terms) != int(text_embs.shape[0]):
        raise ValueError(
            f"{preset_id}: row mismatch glossary={len(glossary)}, "
            f"term_list={len(raw_terms)}, text_embs={text_embs.shape[0]}"
        )
    if snapshot.num_terms and int(snapshot.num_terms) != len(glossary):
        raise ValueError(
            f"{preset_id}: manifest num_terms={snapshot.num_terms} != rows={len(glossary)}"
        )
    if expected_source_size > 0 and len(glossary) != expected_source_size:
        raise ValueError(
            f"{preset_id}: expected {expected_source_size} source rows, found {len(glossary)}"
        )

    term_list: List[Dict[str, Any]] = []
    for row_index, (glossary_entry, index_entry_raw) in enumerate(
        zip(glossary, raw_terms)
    ):
        index_entry = dict(index_entry_raw)
        glossary_key = normalize_english_source(source_term(glossary_entry))
        index_key = normalize_english_source(_index_source_term(index_entry))
        if not glossary_key or glossary_key != index_key:
            raise ValueError(
                f"{preset_id}: row {row_index} glossary/index term mismatch "
                f"({glossary_key!r} != {index_key!r})"
            )
        glossary_target = _target_translation(glossary_entry, target_lang)
        index_target = _target_translation(index_entry, target_lang)
        if not glossary_target:
            raise ValueError(
                f"{preset_id}: row {row_index} glossary lacks {target_lang} translation"
            )
        if not index_target or _canonical_target(glossary_target) != _canonical_target(
            index_target
        ):
            raise ValueError(
                f"{preset_id}: row {row_index} glossary/index {target_lang} "
                "translation mismatch"
            )
        term_list.append(index_entry)

    raw_metadata = payload.get("build_metadata") or payload.get("metadata") or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    return AlignedSource(
        domain=domain,
        preset_id=preset_id,
        glossary_path=glossary_path,
        index_path=index_path,
        glossary=glossary,
        term_list=term_list,
        text_embs=text_embs,
        index_build_metadata=metadata,
        glossary_sha256=file_sha256(glossary_path),
        index_sha256=file_sha256(index_path),
    )


def _iter_gold_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield value
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values: Iterable[Any] = payload
    elif isinstance(payload, Mapping):
        if any(key in payload for key in ("term", "en", "source_label")):
            values = [payload]
        else:
            values = payload.values()
    else:
        raise ValueError(f"{path}: gold inventory must be a JSON list/object or JSONL")
    for value in values:
        if isinstance(value, Mapping):
            yield value


def _gold_source_term(entry: Mapping[str, Any]) -> str:
    return str(
        entry.get("term")
        or entry.get("en")
        or entry.get("source_label")
        or entry.get("key")
        or ""
    ).strip()


def load_gold_inventory(paths: Sequence[Path]) -> tuple[set[str], List[Dict[str, Any]]]:
    terms: set[str] = set()
    provenance: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"gold inventory not found: {path}")
        row_count = 0
        path_terms: set[str] = set()
        for row in _iter_gold_rows(path):
            row_count += 1
            key = normalize_english_source(_gold_source_term(row))
            if key:
                terms.add(key)
                path_terms.add(key)
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "input_rows": row_count,
                "unique_source_terms": len(path_terms),
            }
        )
    return terms, provenance


def _size_token(size: int) -> str:
    if size >= 1000 and size % 1000 == 0:
        return f"{size // 1000}k"
    if size >= 1000 and size % 100 == 0:
        value = f"{size / 1000:.3f}".rstrip("0").rstrip(".")
        return f"{value.replace('.', 'p')}k"
    return str(size)


def derived_preset_id(base_preset_id: str, size: int) -> str:
    stem = base_preset_id.removesuffix("_10k")
    return f"{stem}_{_size_token(size)}"


def _gold_audit(
    source: AlignedSource,
    *,
    sizes: Sequence[int],
    gold_terms: set[str],
    gold_provenance: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_keys = [
        normalize_english_source(source_term(entry)) for entry in source.glossary
    ]
    source_positions: Dict[str, int] = {}
    for row_index, key in enumerate(source_keys, start=1):
        source_positions.setdefault(key, row_index)
    present_in_source = gold_terms.intersection(source_positions)
    minimum_full_prefix = max(
        (source_positions[key] for key in present_in_source),
        default=0,
    )
    prefixes: Dict[str, Any] = {}
    for size in sizes:
        prefix_terms = set(source_keys[:size])
        hits = sorted(gold_terms.intersection(prefix_terms))
        missing = sorted(gold_terms.difference(prefix_terms))
        source_gold_missing = sorted(present_in_source.difference(prefix_terms))
        prefixes[str(size)] = {
            "prefix_size": size,
            "gold_hit_count": len(hits),
            "gold_missing_count": len(missing),
            "coverage_of_all_gold": (
                len(hits) / len(gold_terms) if gold_terms else None
            ),
            "coverage_of_gold_present_in_source": (
                len(hits) / len(present_in_source) if present_in_source else None
            ),
            "full_gold_coverage": bool(gold_terms) and not missing,
            "missing_gold_terms": missing,
            "source_gold_missing_from_prefix": source_gold_missing,
        }
    return {
        "domain": source.domain,
        "source_preset_id": source.preset_id,
        "status": "audited" if gold_provenance else "not_provided",
        "gold_files": [dict(item) for item in gold_provenance],
        "gold_unique_term_count": len(gold_terms),
        "gold_present_in_source_count": len(present_in_source),
        "gold_missing_from_source_count": len(gold_terms - present_in_source),
        "gold_missing_from_source": sorted(gold_terms - present_in_source),
        "minimum_prefix_size_for_all_source_gold": minimum_full_prefix or None,
        "prefixes": prefixes,
    }


def _json_safe_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _preflight_output_paths(
    out_root: Path,
    *,
    sources: Sequence[AlignedSource],
    sizes: Sequence[int],
    target_lang: str,
    overwrite: bool,
) -> None:
    paths = [
        out_root / "manifest_fragment.json",
        out_root / "prefix_sweep_report.json",
        out_root / "gold_prefix_coverage.json",
    ]
    for source in sources:
        for size in sizes:
            preset_id = derived_preset_id(source.preset_id, size)
            paths.extend(
                [
                    out_root / preset_id / "glossary.json",
                    out_root / preset_id / f"en-{target_lang}" / "maxsim.pt",
                ]
            )
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output artifacts already exist; pass --overwrite to replace them: "
            + ", ".join(existing[:8])
        )


def build_nested_prefix_sweep(
    *,
    manifest_path: Path,
    out_root: Path,
    domains: Sequence[str],
    sizes: Sequence[int],
    target_lang: str = "zh",
    expected_source_size: int = 10_000,
    gold_paths: Mapping[str, Sequence[Path]] | None = None,
    require_full_gold_prefix_coverage: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    unknown = [domain for domain in domains if domain not in DOMAIN_TO_PRESET]
    if unknown:
        raise ValueError(f"unknown domains: {', '.join(unknown)}")
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("domains must be non-empty and unique")
    clean_sizes = sorted(set(int(size) for size in sizes))
    if not clean_sizes or any(size <= 0 for size in clean_sizes):
        raise ValueError("prefix sizes must be positive")
    if expected_source_size > 0 and clean_sizes[-1] > expected_source_size:
        raise ValueError(
            f"largest prefix {clean_sizes[-1]} exceeds expected source size "
            f"{expected_source_size}"
        )

    manifest = TermMemoryManifest.load(str(manifest_path))
    sources = [
        load_aligned_source(
            manifest,
            domain=domain,
            target_lang=target_lang,
            expected_source_size=expected_source_size,
        )
        for domain in domains
    ]
    for source in sources:
        if clean_sizes[-1] > len(source.glossary):
            raise ValueError(
                f"{source.preset_id}: largest prefix {clean_sizes[-1]} exceeds "
                f"source rows {len(source.glossary)}"
            )

    normalized_gold_paths = {
        domain: [Path(path).expanduser() for path in paths]
        for domain, paths in (gold_paths or {}).items()
    }
    extra_gold_domains = sorted(set(normalized_gold_paths) - set(domains))
    if extra_gold_domains:
        raise ValueError(
            "gold inventory supplied for non-selected domains: "
            + ", ".join(extra_gold_domains)
        )

    gold_provenance_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    audits: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        terms, provenance = load_gold_inventory(
            normalized_gold_paths.get(source.domain, ())
        )
        gold_provenance_by_domain[source.domain] = provenance
        audits[source.domain] = _gold_audit(
            source,
            sizes=clean_sizes,
            gold_terms=terms,
            gold_provenance=provenance,
        )

    if require_full_gold_prefix_coverage:
        failures: List[str] = []
        for domain, audit in audits.items():
            if audit["status"] != "audited":
                continue
            for size, prefix in audit["prefixes"].items():
                if not prefix["full_gold_coverage"]:
                    failures.append(
                        f"{domain}@{size}: {prefix['gold_hit_count']}/"
                        f"{audit['gold_unique_term_count']}"
                    )
        if failures:
            raise ValueError(
                "gold prefix coverage requirement failed: " + ", ".join(failures)
            )

    _preflight_output_paths(
        out_root,
        sources=sources,
        sizes=clean_sizes,
        target_lang=target_lang,
        overwrite=overwrite,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = file_sha256(manifest_path)
    lang_key = f"en-{target_lang}"
    scales: Dict[str, Any] = {}
    preset_meta: Dict[str, Any] = {}
    outputs: List[Dict[str, Any]] = []

    for source in sources:
        base_meta = manifest.meta_for_preset(source.preset_id)
        source_keys = [
            normalize_english_source(source_term(entry)) for entry in source.glossary
        ]
        for size in clean_sizes:
            preset_id = derived_preset_id(source.preset_id, size)
            preset_root = out_root / preset_id
            glossary_path = preset_root / "glossary.json"
            index_path = preset_root / lang_key / "maxsim.pt"
            glossary_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            prefix_glossary = source.glossary[:size]
            prefix_terms = source.term_list[:size]
            prefix_embs = source.text_embs[:size].clone()
            if (
                len(prefix_glossary) != size
                or len(prefix_terms) != size
                or int(prefix_embs.shape[0]) != size
            ):
                raise RuntimeError(f"{preset_id}: synchronized prefix invariant failed")
            prefix_fingerprint = identity_fingerprint(source_keys[:size])
            glossary_path.write_text(
                json.dumps(prefix_glossary, ensure_ascii=False, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )

            build_metadata: Dict[str, Any] = {
                "builder": "scripts/term_memory/build_nested_prefix_sweep.py",
                "derivation": "aligned_first_n_prefix_without_reencoding",
                "preset_id": preset_id,
                "parent_preset_id": source.preset_id,
                "language_pair": lang_key,
                "term_count": size,
                "parent_term_count": len(source.glossary),
                "source_manifest_path": str(manifest_path.resolve()),
                "source_manifest_sha256": manifest_sha256,
                "parent_glossary_path": str(source.glossary_path.resolve()),
                "parent_glossary_sha256": source.glossary_sha256,
                "parent_index_path": str(source.index_path.resolve()),
                "parent_index_sha256": source.index_sha256,
                "source_normalization": SOURCE_NORMALIZATION_POLICY,
                "source_term_fingerprint_sha256": prefix_fingerprint,
            }
            for key in ("embedding_checkpoint_path", "embedding_checkpoint_sha256"):
                if source.index_build_metadata.get(key):
                    build_metadata[key] = source.index_build_metadata[key]
            torch.save(
                {
                    "text_embs": prefix_embs,
                    "term_list": prefix_terms,
                    "build_metadata": build_metadata,
                },
                index_path,
            )

            glossary_sha256 = file_sha256(glossary_path)
            index_sha256 = file_sha256(index_path)
            prefix_audit = audits[source.domain]["prefixes"][str(size)]
            scales[preset_id] = {
                lang_key: {
                    "terms_path": str(glossary_path.resolve()),
                    "indexes": {"maxsim": str(index_path.resolve())},
                    "num_terms": size,
                }
            }
            meta = dict(base_meta)
            base_label = str(meta.get("label") or source.preset_id.replace("_", " "))
            meta.update(
                {
                    "id": preset_id,
                    "preset_id": preset_id,
                    "label": f"{base_label} nested prefix {_size_token(size)}",
                    "domain": str(meta.get("domain") or source.domain),
                    "domain_id": str(meta.get("domain_id") or source.domain),
                    "parent_preset_id": source.preset_id,
                    "term_count": size,
                    "prefix_size": size,
                    "maxsim_index_path": str(index_path.resolve()),
                    "glossary_path": str(glossary_path.resolve()),
                    "source_term_fingerprint_sha256": prefix_fingerprint,
                    "glossary_sha256": glossary_sha256,
                    "maxsim_index_sha256": index_sha256,
                    "gold_unique_term_count": audits[source.domain][
                        "gold_unique_term_count"
                    ],
                    "gold_prefix_hit_count": prefix_audit["gold_hit_count"],
                    "gold_prefix_coverage": prefix_audit["coverage_of_all_gold"],
                    "enabled_for_auto_router": bool(
                        meta.get("enabled_for_auto_router", True)
                    ),
                }
            )
            preset_meta[preset_id] = meta
            outputs.append(
                {
                    "domain": source.domain,
                    "preset_id": preset_id,
                    "parent_preset_id": source.preset_id,
                    "prefix_size": size,
                    "glossary_path": str(glossary_path.resolve()),
                    "glossary_sha256": glossary_sha256,
                    "index_path": str(index_path.resolve()),
                    "index_sha256": index_sha256,
                    "embedding_shape": list(prefix_embs.shape),
                    "embedding_dtype": str(prefix_embs.dtype),
                    "source_term_fingerprint_sha256": prefix_fingerprint,
                    "gold_prefix_coverage": prefix_audit,
                }
            )

    audit_path = out_root / "gold_prefix_coverage.json"
    audit_payload = {
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "target_lang": target_lang,
        "sizes": clean_sizes,
        "domains": audits,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = out_root / "prefix_sweep_report.json"
    fragment_path = out_root / "manifest_fragment.json"
    manifest_fragment = {
        "snapshot_id": f"{manifest.snapshot_id}_nested_prefix_sweep",
        "source": "aligned nested prefixes derived without re-encoding",
        "created_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "snapshot_id": manifest.snapshot_id,
        },
        "build_report_path": str(report_path.resolve()),
        "gold_prefix_coverage_path": str(audit_path.resolve()),
        "scales": scales,
        "preset_meta": preset_meta,
    }
    fragment_path.write_text(
        json.dumps(manifest_fragment, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    fragment_sha256 = file_sha256(fragment_path)
    audit_sha256 = file_sha256(audit_path)

    report: Dict[str, Any] = {
        "status": "complete",
        "builder": "scripts/term_memory/build_nested_prefix_sweep.py",
        "derivation": "aligned_first_n_prefix_without_reencoding",
        "source_normalization": SOURCE_NORMALIZATION_POLICY,
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "source_snapshot_id": manifest.snapshot_id,
        "target_lang": target_lang,
        "language_pair": lang_key,
        "domains": list(domains),
        "prefix_sizes": clean_sizes,
        "expected_source_size": expected_source_size,
        "require_full_gold_prefix_coverage": require_full_gold_prefix_coverage,
        "manifest_fragment_path": str(fragment_path.resolve()),
        "manifest_fragment_sha256": fragment_sha256,
        "gold_prefix_coverage_path": str(audit_path.resolve()),
        "gold_prefix_coverage_sha256": audit_sha256,
        "sources": [
            {
                "domain": source.domain,
                "preset_id": source.preset_id,
                "glossary_path": str(source.glossary_path.resolve()),
                "glossary_sha256": source.glossary_sha256,
                "index_path": str(source.index_path.resolve()),
                "index_sha256": source.index_sha256,
                "term_count": len(source.glossary),
                "embedding_shape": list(source.text_embs.shape),
                "embedding_dtype": str(source.text_embs.dtype),
                "index_build_metadata": _json_safe_mapping(
                    source.index_build_metadata
                ),
                "gold_files": gold_provenance_by_domain[source.domain],
            }
            for source in sources
        ],
        "outputs": outputs,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_domains(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_sizes(raw: str) -> List[int]:
    try:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--sizes must be comma-separated integers") from exc


def _parse_gold_specs(
    values: Sequence[str],
    *,
    domains: Sequence[str],
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {domain: [] for domain in domains}
    for value in values:
        domain, separator, raw_path = value.partition("=")
        domain = domain.strip()
        if not separator or domain not in result or not raw_path.strip():
            raise ValueError(
                f"invalid --gold {value!r}; expected selected-domain=/path/to/gold.json"
            )
        result[domain].append(Path(raw_path).expanduser())
    return {domain: paths for domain, paths in result.items() if paths}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    parser.add_argument("--sizes", default="1000,2500,5000")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--expected-source-size", type=int, default=10_000)
    parser.add_argument(
        "--gold",
        action="append",
        default=[],
        help="repeat selected-domain=/path/to/gold.json; paths are unioned per domain",
    )
    parser.add_argument("--require-full-gold-prefix-coverage", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    domains = _parse_domains(args.domains)
    try:
        sizes = _parse_sizes(args.sizes)
        gold_paths = _parse_gold_specs(args.gold, domains=domains)
        report = build_nested_prefix_sweep(
            manifest_path=args.manifest.expanduser(),
            out_root=args.out_root.expanduser(),
            domains=domains,
            sizes=sizes,
            target_lang=args.target_lang,
            expected_source_size=args.expected_source_size,
            gold_paths=gold_paths,
            require_full_gold_prefix_coverage=args.require_full_gold_prefix_coverage,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": report["status"],
                "domains": report["domains"],
                "prefix_sizes": report["prefix_sizes"],
                "output_count": len(report["outputs"]),
                "manifest_fragment_path": report["manifest_fragment_path"],
                "gold_prefix_coverage_path": report["gold_prefix_coverage_path"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
