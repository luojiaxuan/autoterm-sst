#!/usr/bin/env python3
"""Build and validate provenance-constrained topic glossary slices.

The builder is intended for the ``100 topics x about 10k terms`` AutoTerm
catalog experiment.  Topic membership is defined by an explicit taxonomy over
Wikidata IDs/types and Wikipedia category paths.  It deliberately does not
infer topics from term strings or free-text descriptions.

Input files may be JSON arrays, small dict-shaped JSON glossaries, or JSONL.
Each selected row must have every requested target translation and structured
QID/category provenance.  A normalized English source term is assigned to at
most one slice, so all emitted language views share the same source-term
identity.  Optional ``filler_match`` clauses are evaluated only after primary
assignment and are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_SELECTORS = {
    "category_any",
    "category_path_prefix_any",
    "category_query_any",
    "domain_any",
    "domain_root_qid_any",
    "entity_type_any",
    "rdf_path_any",
    "source_any",
    "wikidata_qid_any",
    "wikidata_type_qid_any",
}
_DEEPCAT_RE = re.compile(r'^deepcat:\s*["\'](.+?)["\']$', re.IGNORECASE)


def normalized_term(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def source_term(row: Mapping[str, Any]) -> str:
    return str(row.get("term") or row.get("source_label") or "").strip()


def _norm_token(value: Any) -> str:
    token = " ".join(str(value or "").strip().casefold().split())
    if token.startswith("category:"):
        token = token[len("category:") :].strip()
    return token


def _as_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _target_translation(row: Mapping[str, Any], language: str) -> str:
    translations = row.get("target_translations")
    if isinstance(translations, Mapping):
        return str(translations.get(language) or "").strip()
    if str(row.get("target_lang") or "").strip().casefold() == language.casefold():
        return str(row.get("target_label") or row.get("translation") or "").strip()
    return ""


def _qid(row: Mapping[str, Any]) -> str:
    value = row.get("wikidata_qid") or row.get("qid") or row.get("term_id") or ""
    value = str(value).strip()
    return value if value.upper().startswith("Q") else ""


def provenance_kind(row: Mapping[str, Any]) -> str:
    source = str(row.get("source") or "").strip()
    if source:
        return source
    if row.get("category_path"):
        return "wikipedia_category"
    if row.get("wikidata_type_qid"):
        return "wikidata_exact_p31"
    if row.get("domain_root_qid"):
        return "wikidata_p31_p279"
    return "unknown"


def has_structured_provenance(row: Mapping[str, Any], *, qid_selector_match: bool = False) -> bool:
    if not _qid(row):
        return False
    if qid_selector_match:
        return True
    return bool(
        row.get("domain_root_qid")
        or row.get("wikidata_type_qid")
        or row.get("entity_types")
        or row.get("category_path")
        or row.get("category_query")
        or row.get("domain_root_categories")
    )


def _iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    opened = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not opened:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"{path}: empty JSON file")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"{path}: streaming parser expected a JSON array")
                opened = True
                position += 1
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"{path}: unterminated JSON array")
                buffer = ""
                position = 0
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"{path}: invalid JSON array") from None
                chunk = handle.read(chunk_size)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
                continue
            position = end
            if not isinstance(value, dict):
                raise ValueError(f"{path}: glossary entries must be JSON objects")
            yield dict(value)


def iter_glossary(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield dict(value)
        return

    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while not first:
            first = handle.read(1)
            if not first:
                raise ValueError(f"{path}: empty JSON file")
            if first.isspace():
                first = ""
    if first == "[":
        yield from _iter_json_array(path)
        return
    if first != "{":
        raise ValueError(f"{path}: expected a JSON array, object, or JSONL")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError(
            f"{path}: dict-shaped JSON exceeds 256 MiB; normalize it to JSONL with "
            "scripts/term_memory/extract_wikidata_terms.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for value in payload.values():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: glossary values must be JSON objects")
        yield dict(value)


@dataclass(frozen=True)
class TopicSpec:
    domain_id: str
    preset_id: str
    domain_description: str
    capacity: int
    priority: int
    ordinal: int
    primary_match: Tuple[Mapping[str, Any], ...]
    filler_match: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TopicTaxonomy:
    taxonomy_id: str
    topics: Tuple[TopicSpec, ...]
    expected_topic_count: int


def _validate_clauses(value: Any, *, location: str) -> Tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{location}: expected a list of match-clause objects")
    clauses: List[Mapping[str, Any]] = []
    for index, clause in enumerate(value):
        unknown = sorted(set(clause) - SUPPORTED_SELECTORS)
        if unknown:
            raise ValueError(f"{location}[{index}]: unsupported selectors: {', '.join(unknown)}")
        if not clause:
            raise ValueError(f"{location}[{index}]: empty clauses match everything and are forbidden")
        for selector, selector_value in clause.items():
            if selector == "category_path_prefix_any":
                valid = isinstance(selector_value, list) and all(
                    isinstance(prefix, list) and prefix for prefix in selector_value
                )
            else:
                valid = isinstance(selector_value, list) and bool(selector_value)
            if not valid:
                raise ValueError(f"{location}[{index}].{selector}: expected a non-empty list")
        clauses.append(dict(clause))
    return tuple(clauses)


def load_taxonomy(path: Path, *, expected_topic_count: int = 0) -> TopicTaxonomy:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("taxonomy must be a JSON object")
    taxonomy_id = str(raw.get("taxonomy_id") or "").strip()
    if not taxonomy_id:
        raise ValueError("taxonomy missing taxonomy_id")
    values = raw.get("topics")
    if not isinstance(values, list) or not values:
        raise ValueError("taxonomy topics must be a non-empty list")
    default_capacity = int(raw.get("slice_capacity") or 10_000)
    configured_expected = int(raw.get("expected_topic_count") or len(values))
    expected = expected_topic_count or configured_expected
    if len(values) != expected:
        raise ValueError(f"taxonomy has {len(values)} topics, expected {expected}")

    topics: List[TopicSpec] = []
    for ordinal, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"topics[{ordinal}]: expected an object")
        domain_id = str(item.get("domain_id") or "").strip()
        preset_id = str(item.get("preset_id") or "").strip()
        description = " ".join(str(item.get("domain_description") or "").split())
        capacity = int(item.get("capacity") or default_capacity)
        if not domain_id or not preset_id or not description:
            raise ValueError(
                f"topics[{ordinal}]: domain_id, preset_id, and domain_description are required"
            )
        if capacity <= 0:
            raise ValueError(f"topics[{ordinal}]: capacity must be positive")
        primary = _validate_clauses(item.get("primary_match"), location=f"topics[{ordinal}].primary_match")
        filler = _validate_clauses(item.get("filler_match"), location=f"topics[{ordinal}].filler_match")
        if not primary:
            raise ValueError(f"topics[{ordinal}]: at least one primary_match clause is required")
        topics.append(
            TopicSpec(
                domain_id=domain_id,
                preset_id=preset_id,
                domain_description=description,
                capacity=capacity,
                priority=int(item.get("priority") or 0),
                ordinal=ordinal,
                primary_match=primary,
                filler_match=filler,
            )
        )

    def _duplicates(values_to_check: Iterable[str]) -> List[str]:
        counts = Counter(values_to_check)
        return sorted(value for value, count in counts.items() if count > 1)

    duplicate_domains = _duplicates(topic.domain_id.casefold() for topic in topics)
    duplicate_presets = _duplicates(topic.preset_id.casefold() for topic in topics)
    duplicate_descriptions = _duplicates(topic.domain_description.casefold() for topic in topics)
    if duplicate_domains or duplicate_presets or duplicate_descriptions:
        raise ValueError(
            "taxonomy values must be unique: "
            f"domain_id={duplicate_domains}, preset_id={duplicate_presets}, "
            f"domain_description={duplicate_descriptions}"
        )
    return TopicTaxonomy(taxonomy_id=taxonomy_id, topics=tuple(topics), expected_topic_count=expected)


def _record_categories(row: Mapping[str, Any]) -> Tuple[str, ...]:
    values: List[str] = []
    values.extend(_as_values(row.get("category_path")))
    values.extend(_as_values(row.get("domain_root_categories")))
    query = str(row.get("category_query") or "").strip()
    match = _DEEPCAT_RE.fullmatch(query)
    if match:
        values.append(match.group(1))
    return tuple(_norm_token(value) for value in values if _norm_token(value))


def _clause_matches(row: Mapping[str, Any], clause: Mapping[str, Any]) -> bool:
    categories = set(_record_categories(row))
    path = tuple(_norm_token(value) for value in _as_values(row.get("category_path")))
    row_values = {
        "category_query_any": {_norm_token(row.get("category_query"))},
        "domain_any": {_norm_token(value) for value in _as_values(row.get("domain") or row.get("domains"))},
        "domain_root_qid_any": {_norm_token(value) for value in _as_values(row.get("domain_root_qid"))},
        "entity_type_any": {_norm_token(value) for value in _as_values(row.get("entity_types"))},
        "rdf_path_any": {_norm_token(value) for value in _as_values(row.get("rdf_path"))},
        "source_any": {_norm_token(value) for value in _as_values(row.get("source"))},
        "wikidata_qid_any": {_norm_token(_qid(row))},
        "wikidata_type_qid_any": {
            _norm_token(value) for value in _as_values(row.get("wikidata_type_qid"))
        },
    }
    for selector, expected_values in clause.items():
        if selector == "category_any":
            expected = {_norm_token(value) for value in expected_values}
            if not categories.intersection(expected):
                return False
        elif selector == "category_path_prefix_any":
            prefixes = [tuple(_norm_token(value) for value in prefix) for prefix in expected_values]
            if not any(path[: len(prefix)] == prefix for prefix in prefixes):
                return False
        else:
            expected = {_norm_token(value) for value in expected_values}
            if not row_values[selector].intersection(expected):
                return False
    return True


def matching_topics(
    row: Mapping[str, Any],
    topics: Sequence[TopicSpec],
    *,
    role: str,
) -> List[Tuple[TopicSpec, int]]:
    matches: List[Tuple[TopicSpec, int]] = []
    for topic in topics:
        clauses = topic.primary_match if role == "primary" else topic.filler_match
        for clause_index, clause in enumerate(clauses):
            if _clause_matches(row, clause):
                matches.append((topic, clause_index))
                break
    matches.sort(key=lambda item: (-item[0].priority, item[0].ordinal))
    return matches


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_fingerprint(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_array_from_jsonl(source: Path, destination: Path) -> Tuple[int, List[str]]:
    count = 0
    keys: List[str] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as out:
        out.write("[")
        first = True
        for line in src:
            row = json.loads(line)
            if not first:
                out.write(",")
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            first = False
            count += 1
            keys.append(normalized_term(source_term(row)))
        out.write("]\n")
    return count, keys


def _parse_inputs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    inputs: List[Tuple[str, Path]] = []
    seen_ids: set[str] = set()
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id.strip() or not raw_path.strip():
            raise ValueError(f"invalid --input {value!r}; expected source_id=/path/to/glossary")
        source_id = source_id.strip()
        path = Path(raw_path).expanduser().resolve()
        if source_id in seen_ids:
            raise ValueError(f"duplicate input source_id: {source_id}")
        if not path.is_file():
            raise ValueError(f"input does not exist: {path}")
        seen_ids.add(source_id)
        inputs.append((source_id, path))
    if not inputs:
        raise ValueError("at least one --input source_id=path is required")
    return inputs


def _write_selected(
    handle: Any,
    row: Mapping[str, Any],
    *,
    taxonomy_id: str,
    topic: TopicSpec,
    role: str,
    clause_index: int,
    source_id: str,
) -> None:
    clauses = topic.primary_match if role == "primary" else topic.filler_match
    selected = dict(row)
    selected["term_key"] = normalized_term(source_term(row))
    selected["catalog_assignment"] = {
        "taxonomy_id": taxonomy_id,
        "domain_id": topic.domain_id,
        "preset_id": topic.preset_id,
        "role": role,
        "matched_clause_index": clause_index,
        "matched_selectors": sorted(clauses[clause_index]),
        "input_source_id": source_id,
        "provenance_kind": provenance_kind(row),
    }
    handle.write(json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def build_catalog(
    *,
    taxonomy: TopicTaxonomy,
    inputs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    snapshot_id: str,
    target_languages: Sequence[str],
    require_full_slices: bool,
    emit_merged: bool,
) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / ".build"
    work_dir.mkdir()
    slices_dir = out_dir / "slices"
    slices_dir.mkdir()

    target_languages = tuple(dict.fromkeys(language.strip().casefold() for language in target_languages if language.strip()))
    if not target_languages:
        raise ValueError("at least one target language is required")
    counts = {topic.domain_id: 0 for topic in taxonomy.topics}
    role_counts = {topic.domain_id: Counter() for topic in taxonomy.topics}
    source_counts = {
        topic.domain_id: {"primary": Counter(), "filler": Counter()} for topic in taxonomy.topics
    }
    provenance_counts = {
        topic.domain_id: {"primary": Counter(), "filler": Counter()} for topic in taxonomy.topics
    }
    selected_terms: set[str] = set()
    sample_limits = 20
    samples: Dict[str, List[Dict[str, Any]]] = {
        "overlap": [],
        "unassigned": [],
        "missing_translation": [],
        "missing_provenance": [],
        "source_term_qid_collision": [],
    }
    stats = Counter()

    state_path = work_dir / "catalog_state.sqlite3"
    state = sqlite3.connect(state_path)
    state.execute("PRAGMA journal_mode=OFF")
    state.execute("PRAGMA synchronous=OFF")
    state.execute("CREATE TABLE valid_terms (term_key TEXT PRIMARY KEY, qid TEXT NOT NULL)")

    spool_paths = {topic.domain_id: work_dir / f"{topic.ordinal:03d}.jsonl" for topic in taxonomy.topics}
    handles = {domain_id: path.open("w", encoding="utf-8") for domain_id, path in spool_paths.items()}

    def eligible(row: Mapping[str, Any], *, count_stats: bool) -> Tuple[bool, str, str]:
        term = source_term(row)
        key = normalized_term(term)
        if count_stats:
            stats["input_rows"] += 1
        if not key:
            if count_stats:
                stats["rejected_missing_source_term"] += 1
            return False, "", ""
        missing = [language for language in target_languages if not _target_translation(row, language)]
        if missing:
            if count_stats:
                stats["rejected_missing_translation"] += 1
                if len(samples["missing_translation"]) < sample_limits:
                    samples["missing_translation"].append({"term": term, "missing": missing})
            return False, key, _qid(row)
        if count_stats:
            stats["translation_complete_rows"] += 1
        return True, key, _qid(row)

    def scan(role: str) -> None:
        for source_id, path in inputs:
            for row in iter_glossary(path):
                ok, key, qid = eligible(row, count_stats=role == "primary")
                if not ok:
                    continue
                matches = matching_topics(row, taxonomy.topics, role=role)
                if not has_structured_provenance(row):
                    matches = [
                        (topic, clause_index)
                        for topic, clause_index in matches
                        if "wikidata_qid_any"
                        in (topic.primary_match if role == "primary" else topic.filler_match)[
                            clause_index
                        ]
                    ]
                if not has_structured_provenance(row) and not matches:
                    if role == "primary":
                        stats["rejected_missing_structured_provenance"] += 1
                        if len(samples["missing_provenance"]) < sample_limits:
                            samples["missing_provenance"].append(
                                {"term": source_term(row), "source": str(row.get("source") or "")}
                            )
                    continue
                if role == "primary":
                    stats["structured_provenance_rows"] += 1
                    inserted = state.execute(
                        "INSERT OR IGNORE INTO valid_terms(term_key, qid) VALUES (?, ?)", (key, qid)
                    ).rowcount
                    if inserted:
                        stats["valid_unique_source_terms"] += 1
                    else:
                        stats["duplicate_input_source_term_rows"] += 1
                        existing = state.execute(
                            "SELECT qid FROM valid_terms WHERE term_key = ?", (key,)
                        ).fetchone()
                        if existing and existing[0] and qid and existing[0] != qid:
                            stats["source_term_qid_collisions"] += 1
                            if len(samples["source_term_qid_collision"]) < sample_limits:
                                samples["source_term_qid_collision"].append(
                                    {"term": source_term(row), "first_qid": existing[0], "later_qid": qid}
                                )
                if key in selected_terms:
                    stats[f"{role}_duplicate_selected_rows"] += 1
                    continue
                if len(matches) > 1:
                    stats[f"{role}_overlap_rows"] += 1
                    if len(samples["overlap"]) < sample_limits:
                        samples["overlap"].append(
                            {
                                "term": source_term(row),
                                "role": role,
                                "matching_domain_ids": [topic.domain_id for topic, _ in matches],
                            }
                        )
                if not matches:
                    stats[f"{role}_unmatched_rows"] += 1
                    if role == "primary" and len(samples["unassigned"]) < sample_limits:
                        samples["unassigned"].append(
                            {"term": source_term(row), "provenance_kind": provenance_kind(row)}
                        )
                    continue
                assigned = False
                for topic, clause_index in matches:
                    if counts[topic.domain_id] >= topic.capacity:
                        continue
                    _write_selected(
                        handles[topic.domain_id],
                        row,
                        taxonomy_id=taxonomy.taxonomy_id,
                        topic=topic,
                        role=role,
                        clause_index=clause_index,
                        source_id=source_id,
                    )
                    selected_terms.add(key)
                    counts[topic.domain_id] += 1
                    role_counts[topic.domain_id][role] += 1
                    source_counts[topic.domain_id][role][source_id] += 1
                    provenance_counts[topic.domain_id][role][provenance_kind(row)] += 1
                    stats[f"selected_{role}"] += 1
                    assigned = True
                    break
                if not assigned:
                    stats[f"{role}_capacity_rejected_rows"] += 1
            state.commit()

    try:
        scan("primary")
        if any(topic.filler_match and counts[topic.domain_id] < topic.capacity for topic in taxonomy.topics):
            scan("filler")
    finally:
        for handle in handles.values():
            handle.close()
        state.commit()
        state.close()

    underfilled = {
        topic.domain_id: {"term_count": counts[topic.domain_id], "capacity": topic.capacity}
        for topic in taxonomy.topics
        if counts[topic.domain_id] != topic.capacity
    }
    if require_full_slices and underfilled:
        partial = {
            "taxonomy_id": taxonomy.taxonomy_id,
            "status": "underfilled",
            "underfilled": underfilled,
            "stats": dict(stats),
        }
        (out_dir / "build_failure_report.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(work_dir)
        raise RuntimeError(f"could not fill every topic slice: {underfilled}")

    slice_keys: Dict[str, List[str]] = {}
    slice_paths: Dict[str, Path] = {}
    for topic in taxonomy.topics:
        path = slices_dir / f"{topic.preset_id}.json"
        count, keys = _write_json_array_from_jsonl(spool_paths[topic.domain_id], path)
        if count != counts[topic.domain_id]:
            raise RuntimeError(f"{topic.domain_id}: spool count changed from {counts[topic.domain_id]} to {count}")
        slice_keys[topic.domain_id] = keys
        slice_paths[topic.domain_id] = path

    merged_path: Path | None = None
    merged_keys: List[str] = []
    if emit_merged:
        merged_path = out_dir / "merged_topics.json"
        with merged_path.open("w", encoding="utf-8") as out:
            out.write("[")
            first = True
            for topic in taxonomy.topics:
                for row in iter_glossary(slice_paths[topic.domain_id]):
                    if not first:
                        out.write(",")
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    first = False
                    merged_keys.append(normalized_term(source_term(row)))
            out.write("]\n")

    input_meta = [
        {
            "source_id": source_id,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for source_id, path in inputs
    ]
    scales: Dict[str, Any] = {}
    preset_meta: Dict[str, Any] = {}
    for topic in taxonomy.topics:
        relative_path = slice_paths[topic.domain_id].relative_to(out_dir).as_posix()
        language_views = {
            f"en-{language}": {
                "terms_path": relative_path,
                "indexes": {},
                "num_terms": counts[topic.domain_id],
            }
            for language in target_languages
        }
        scales[topic.preset_id] = language_views
        fingerprint = _identity_fingerprint(slice_keys[topic.domain_id])
        preset_meta[topic.preset_id] = {
            "id": topic.preset_id,
            "preset_id": topic.preset_id,
            "label": topic.domain_id.replace("_", " ").title(),
            "domain": topic.domain_id,
            "domain_id": topic.domain_id,
            "description": topic.domain_description,
            "domain_description": topic.domain_description,
            "term_count": counts[topic.domain_id],
            "capacity": topic.capacity,
            "primary_term_count": role_counts[topic.domain_id]["primary"],
            "filler_term_count": role_counts[topic.domain_id]["filler"],
            "source_term_fingerprint_sha256": fingerprint,
            "enabled_for_auto_router": True,
            "index_status": "pending",
        }

    if merged_path is not None:
        language_views = {
            f"en-{language}": {
                "terms_path": merged_path.relative_to(out_dir).as_posix(),
                "indexes": {},
                "num_terms": len(merged_keys),
            }
            for language in target_languages
        }
        scales["merged_topics"] = language_views
        preset_meta["merged_topics"] = {
            "id": "merged_topics",
            "preset_id": "merged_topics",
            "label": "Merged topic catalog",
            "domain": "merged",
            "domain_id": "merged",
            "description": "Union of all provenance-constrained topic slices.",
            "domain_description": "Union of all provenance-constrained topic slices.",
            "term_count": len(merged_keys),
            "capacity": sum(topic.capacity for topic in taxonomy.topics),
            "source_term_fingerprint_sha256": _identity_fingerprint(merged_keys),
            "enabled_for_auto_router": False,
            "index_status": "pending",
        }

    global_keys = [key for topic in taxonomy.topics for key in slice_keys[topic.domain_id]]
    manifest = {
        "snapshot_id": snapshot_id,
        "source": "provenance-constrained Wikidata/Wikipedia topic catalog",
        "root": ".",
        "scales": scales,
        "preset_meta": preset_meta,
        "catalog_meta": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "expected_topic_count": taxonomy.expected_topic_count,
            "topic_count": len(taxonomy.topics),
            "target_languages": list(target_languages),
            "source_term_identity": "normalized English source term is globally unique",
            "global_source_term_fingerprint_sha256": _identity_fingerprint(global_keys),
            "global_term_count": len(global_keys),
            "require_full_slices": require_full_slices,
            "index_status": "pending",
        },
    }
    manifest_path = out_dir / "catalog_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    valid_unique_terms = int(stats.get("valid_unique_source_terms", 0))
    report = {
        "status": "complete" if not underfilled else "underfilled_allowed",
        "taxonomy_id": taxonomy.taxonomy_id,
        "topic_count": len(taxonomy.topics),
        "target_languages": list(target_languages),
        "inputs": input_meta,
        "stats": {
            **dict(stats),
            "selected_unique_source_terms": len(selected_terms),
            "unassigned_unique_source_terms": max(0, valid_unique_terms - len(selected_terms)),
        },
        "underfilled": underfilled,
        "per_topic": {
            topic.domain_id: {
                "preset_id": topic.preset_id,
                "domain_description": topic.domain_description,
                "capacity": topic.capacity,
                "term_count": counts[topic.domain_id],
                "primary_term_count": role_counts[topic.domain_id]["primary"],
                "filler_term_count": role_counts[topic.domain_id]["filler"],
                "input_sources_by_role": {
                    role: dict(source_counts[topic.domain_id][role])
                    for role in ("primary", "filler")
                },
                "provenance_by_role": {
                    role: dict(provenance_counts[topic.domain_id][role])
                    for role in ("primary", "filler")
                },
                "source_term_fingerprint_sha256": _identity_fingerprint(slice_keys[topic.domain_id]),
            }
            for topic in taxonomy.topics
        },
        "samples": samples,
        "manifest_path": str(manifest_path),
        "merged_path": str(merged_path) if merged_path is not None else "",
    }
    report_path = out_dir / "catalog_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(work_dir)
    return report


def validate_catalog(
    manifest_path: Path,
    *,
    expected_topic_count: int = 0,
    require_full_slices: bool = False,
) -> Dict[str, Any]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not str(raw.get("snapshot_id") or "").strip():
        raise ValueError("manifest must be an object with snapshot_id")
    scales = raw.get("scales")
    preset_meta = raw.get("preset_meta")
    if not isinstance(scales, dict) or not isinstance(preset_meta, dict):
        raise ValueError("manifest must contain scales and preset_meta objects")
    root = Path(str(raw.get("root") or "."))
    if not root.is_absolute():
        root = manifest_path.resolve().parent / root
    router_presets = [
        preset_id
        for preset_id in scales
        if isinstance(preset_meta.get(preset_id), dict)
        and bool(preset_meta[preset_id].get("enabled_for_auto_router"))
    ]
    expected = expected_topic_count or int((raw.get("catalog_meta") or {}).get("expected_topic_count") or 0)
    if expected and len(router_presets) != expected:
        raise ValueError(f"manifest has {len(router_presets)} router slices, expected {expected}")

    domain_ids: set[str] = set()
    descriptions: set[str] = set()
    global_terms: set[str] = set()
    ordered_global_terms: List[str] = []
    per_topic: Dict[str, Any] = {}
    required_languages = list((raw.get("catalog_meta") or {}).get("target_languages") or [])
    for preset_id in router_presets:
        meta = dict(preset_meta[preset_id])
        domain_id = str(meta.get("domain_id") or "").strip()
        description = " ".join(str(meta.get("domain_description") or meta.get("description") or "").split())
        if not domain_id or domain_id.casefold() in domain_ids:
            raise ValueError(f"{preset_id}: missing or duplicate domain_id {domain_id!r}")
        if not description or description.casefold() in descriptions:
            raise ValueError(f"{preset_id}: missing or duplicate domain_description")
        domain_ids.add(domain_id.casefold())
        descriptions.add(description.casefold())
        snapshots = scales[preset_id]
        if not isinstance(snapshots, dict) or not snapshots:
            raise ValueError(f"{preset_id}: missing language views")
        expected_language_keys = {f"en-{language}" for language in required_languages}
        if expected_language_keys and set(snapshots) != expected_language_keys:
            raise ValueError(
                f"{preset_id}: language views {sorted(snapshots)} != {sorted(expected_language_keys)}"
            )
        paths = {
            str(snapshot.get("terms_path") or "")
            for snapshot in snapshots.values()
            if isinstance(snapshot, dict)
        }
        counts = {
            int(snapshot.get("num_terms") or 0)
            for snapshot in snapshots.values()
            if isinstance(snapshot, dict)
        }
        if len(paths) != 1 or len(counts) != 1:
            raise ValueError(f"{preset_id}: language views do not share source-term identity")
        terms_path = Path(next(iter(paths)))
        if not terms_path.is_absolute():
            terms_path = root / terms_path
        rows = 0
        keys: List[str] = []
        for row in iter_glossary(terms_path):
            rows += 1
            key = normalized_term(source_term(row))
            if not key:
                raise ValueError(f"{preset_id}: row {rows} has no source term")
            if key in global_terms:
                raise ValueError(f"{preset_id}: duplicate global source term {key!r}")
            missing = [language for language in required_languages if not _target_translation(row, language)]
            if missing:
                raise ValueError(f"{preset_id}: {key!r} is missing translations {missing}")
            assignment = row.get("catalog_assignment")
            if not isinstance(assignment, Mapping) or assignment.get("domain_id") != domain_id:
                raise ValueError(f"{preset_id}: {key!r} has invalid catalog_assignment")
            if not has_structured_provenance(
                row,
                qid_selector_match="wikidata_qid_any" in set(assignment.get("matched_selectors") or []),
            ):
                raise ValueError(f"{preset_id}: {key!r} lacks structured provenance")
            global_terms.add(key)
            ordered_global_terms.append(key)
            keys.append(key)
        declared = int(meta["term_count"]) if "term_count" in meta else -1
        capacity = int(meta["capacity"]) if "capacity" in meta else -1
        if rows != declared or rows not in counts:
            raise ValueError(f"{preset_id}: row count {rows} != declared counts {declared}/{counts}")
        if rows > capacity or (require_full_slices and rows != capacity):
            raise ValueError(f"{preset_id}: term_count={rows}, capacity={capacity}")
        fingerprint = _identity_fingerprint(keys)
        if fingerprint != meta.get("source_term_fingerprint_sha256"):
            raise ValueError(f"{preset_id}: source-term fingerprint mismatch")
        per_topic[domain_id] = {"term_count": rows, "capacity": capacity, "fingerprint": fingerprint}

    global_fingerprint = _identity_fingerprint(ordered_global_terms)
    declared_global = str((raw.get("catalog_meta") or {}).get("global_source_term_fingerprint_sha256") or "")
    if declared_global and declared_global != global_fingerprint:
        raise ValueError("global source-term fingerprint mismatch")
    if "merged_topics" in scales:
        merged_meta = preset_meta.get("merged_topics")
        merged_views = scales["merged_topics"]
        if not isinstance(merged_meta, dict) or not isinstance(merged_views, dict) or not merged_views:
            raise ValueError("merged_topics metadata is malformed")
        merged_paths = {
            str(snapshot.get("terms_path") or "")
            for snapshot in merged_views.values()
            if isinstance(snapshot, dict)
        }
        if len(merged_paths) != 1:
            raise ValueError("merged_topics language views do not share one terms file")
        merged_path = Path(next(iter(merged_paths)))
        if not merged_path.is_absolute():
            merged_path = root / merged_path
        merged_keys = [normalized_term(source_term(row)) for row in iter_glossary(merged_path)]
        if merged_keys != ordered_global_terms:
            raise ValueError("merged_topics is not the ordered union of topic slices")
        merged_count = int(merged_meta["term_count"]) if "term_count" in merged_meta else -1
        if merged_count != len(merged_keys):
            raise ValueError("merged_topics term_count mismatch")
        if merged_meta.get("source_term_fingerprint_sha256") != global_fingerprint:
            raise ValueError("merged_topics source-term fingerprint mismatch")
    return {
        "status": "valid",
        "topic_count": len(router_presets),
        "global_term_count": len(global_terms),
        "global_source_term_fingerprint_sha256": global_fingerprint,
        "per_topic": per_topic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, help="topic taxonomy JSON")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="repeat source_id=/path/to/glossary; order is the deterministic tie-break",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--target-languages", default="zh")
    parser.add_argument("--expected-topic-count", type=int, default=0)
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--emit-merged", action="store_true")
    parser.add_argument("--validate-manifest", type=Path)
    args = parser.parse_args()

    try:
        if args.validate_manifest:
            result = validate_catalog(
                args.validate_manifest.expanduser(),
                expected_topic_count=args.expected_topic_count,
                require_full_slices=not args.allow_underfilled,
            )
        else:
            if not args.taxonomy or not args.out_dir or not args.snapshot_id:
                raise ValueError("build mode requires --taxonomy, --out-dir, and --snapshot-id")
            taxonomy = load_taxonomy(
                args.taxonomy.expanduser(), expected_topic_count=args.expected_topic_count
            )
            result = build_catalog(
                taxonomy=taxonomy,
                inputs=_parse_inputs(args.input),
                out_dir=args.out_dir.expanduser(),
                snapshot_id=args.snapshot_id,
                target_languages=[item for item in args.target_languages.split(",") if item.strip()],
                require_full_slices=not args.allow_underfilled,
                emit_merged=args.emit_merged,
            )
            validation = validate_catalog(
                args.out_dir.expanduser() / "catalog_manifest.json",
                expected_topic_count=taxonomy.expected_topic_count,
                require_full_slices=not args.allow_underfilled,
            )
            result["validation"] = validation
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
