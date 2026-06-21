"""TermEntry: one terminology record in an open-memory snapshot.

A snapshot stores one ``TermEntry`` per JSONL line (``terms.en-zh.jsonl`` etc.).
Entries are produced offline by the builder (Phase C) from Wikidata/Wikipedia and
consumed at serve time only indirectly: the retriever loads a precomputed index,
and the entry's ``source``/``term_id``/``domain`` ride along into event ``meta``
so the UI can show provenance. ``to_reference`` maps an entry to the lightweight
``TermRef`` dict used by the prompt builder and retrieval plugins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v).strip()]


@dataclass
class TermEntry:
    """A single multilingual terminology record."""

    term_id: str
    source_lang: str
    target_lang: str
    source_label: str
    target_label: str
    source_aliases: List[str] = field(default_factory=list)
    target_aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    entity_types: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    popularity: float = 0.0
    source: str = "wikidata"
    source_url: str = ""
    revision: Optional[str] = None
    updated_at: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TermEntry":
        """Build from a parsed JSON object, tolerant of missing keys."""

        def _f(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return cls(
            term_id=str(raw.get("term_id") or raw.get("qid") or ""),
            source_lang=str(raw.get("source_lang") or "en"),
            target_lang=str(raw.get("target_lang") or ""),
            source_label=str(raw.get("source_label") or raw.get("term") or "").strip(),
            target_label=str(raw.get("target_label") or raw.get("translation") or "").strip(),
            source_aliases=_as_str_list(raw.get("source_aliases")),
            target_aliases=_as_str_list(raw.get("target_aliases")),
            description=(str(raw["description"]) if raw.get("description") else None),
            entity_types=_as_str_list(raw.get("entity_types")),
            domains=_as_str_list(raw.get("domains")),
            popularity=_f(raw.get("popularity")),
            source=str(raw.get("source") or "wikidata"),
            source_url=str(raw.get("source_url") or ""),
            revision=(str(raw["revision"]) if raw.get("revision") else None),
            updated_at=str(raw.get("updated_at") or ""),
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> Optional["TermEntry"]:
        line = (line or "").strip()
        if not line or line.startswith("#"):
            return None
        try:
            raw = json.loads(line)
        except ValueError:
            return None
        if not isinstance(raw, dict):
            return None
        entry = cls.from_dict(raw)
        return entry if entry.is_valid() else None

    def is_valid(self) -> bool:
        return bool(self.source_label and self.target_label)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term_id": self.term_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "source_label": self.source_label,
            "target_label": self.target_label,
            "source_aliases": list(self.source_aliases),
            "target_aliases": list(self.target_aliases),
            "description": self.description,
            "entity_types": list(self.entity_types),
            "domains": list(self.domains),
            "popularity": self.popularity,
            "source": self.source,
            "source_url": self.source_url,
            "revision": self.revision,
            "updated_at": self.updated_at,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_reference(self) -> Dict[str, Any]:
        """Map to the lightweight ``TermRef`` consumed downstream.

        The prompt builder only uses ``term``/``translation``; the extra
        provenance keys are forwarded into event ``meta`` for the evidence UI.
        """

        ref: Dict[str, Any] = {
            "term": self.source_label,
            "translation": self.target_label,
            "source": self.source or "wikidata",
        }
        if self.term_id:
            ref["term_id"] = self.term_id
        if self.domains:
            ref["domain"] = self.domains[0]
        if self.updated_at:
            ref["updated_at"] = self.updated_at
        if self.source_url:
            ref["source_url"] = self.source_url
        return ref
