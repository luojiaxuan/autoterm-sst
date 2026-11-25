"""Normalized glossary-entry schema for AutoTerm-SST slices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GlossaryEntry:
    """One normalized terminology-memory entry used by slice inventories."""

    entry_id: str
    source: str
    canonical_source: str
    targets: list[str]
    acceptable_targets: list[str]
    preferred_target: str | None
    slice_id: str
    domain: str | None
    entry_type: str | None
    is_acronym: bool = False
    is_name_or_entity: bool = False
    allow_identity_retention: bool = False
    lexical_keys: list[str] = field(default_factory=list)
    genericity_score: float = 0.0
    prior_weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, slice_id: str = "") -> "GlossaryEntry":
        source = str(raw.get("source") or raw.get("term") or raw.get("source_label") or "").strip()
        canonical = str(raw.get("canonical_source") or source.casefold()).strip()
        targets_raw = raw.get("targets") or raw.get("acceptable_targets") or []
        if isinstance(targets_raw, str):
            targets = [targets_raw]
        else:
            targets = [str(item).strip() for item in targets_raw if str(item).strip()]
        preferred = str(raw.get("preferred_target") or raw.get("translation") or "").strip() or None
        acceptable_raw = raw.get("acceptable_targets") or targets or ([preferred] if preferred else [])
        if isinstance(acceptable_raw, str):
            acceptable = [acceptable_raw]
        else:
            acceptable = [str(item).strip() for item in acceptable_raw if str(item).strip()]
        lexical_raw = raw.get("lexical_keys") or raw.get("aliases") or []
        if isinstance(lexical_raw, str):
            lexical = [lexical_raw]
        else:
            lexical = [str(item).strip() for item in lexical_raw if str(item).strip()]
        return cls(
            entry_id=str(raw.get("entry_id") or raw.get("id") or canonical),
            source=source,
            canonical_source=canonical,
            targets=targets,
            acceptable_targets=acceptable,
            preferred_target=preferred,
            slice_id=str(raw.get("slice_id") or slice_id),
            domain=str(raw.get("domain") or "") or None,
            entry_type=str(raw.get("entry_type") or raw.get("type") or "") or None,
            is_acronym=bool(raw.get("is_acronym")),
            is_name_or_entity=bool(raw.get("is_name_or_entity")),
            allow_identity_retention=bool(raw.get("allow_identity_retention")),
            lexical_keys=lexical,
            genericity_score=float(raw.get("genericity_score") or 0.0),
            prior_weight=float(raw.get("prior_weight") or 1.0),
        )
