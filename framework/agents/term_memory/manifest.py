"""TermMemoryManifest: the small JSON that points at a terminology snapshot.

A manifest describes ONE built snapshot (Wikidata/Wikipedia-derived or any
precomputed glossary) and, optionally, several scale variants of it. It maps a
language key (``en-zh``) to the snapshot's term file + precomputed indexes so the
agent can resolve a glossary preset (e.g. ``open_wiki_auto``, ``open_wiki_1m``)
to a concrete ``maxsim`` index path — the same ``.pt`` format the existing
``MaxSimRetrievalPlugin`` already loads.

Manifest JSON shape (paths absolute, or relative to the manifest's directory;
``$VARS`` are expanded)::

    {
      "snapshot_id": "wikidata_20260617",
      "source": "wikidata",
      "created_at": "2026-06-17T02:00:00Z",
      "languages": {
        "en-zh": {
          "terms_path": "snapshots/wikidata_20260617/terms.en-zh.jsonl",
          "indexes": {"maxsim": "indexes/.../en-zh/maxsim.pt"},
          "num_terms": 1000000
        }
      },
      "scales": {                         # OPTIONAL: one entry per selectable preset
        "open_wiki_10k":  {"en-zh": {"terms_path": "...", "indexes": {...}, "num_terms": 10000}},
        "open_wiki_100k": {"en-zh": {...}},
        "open_wiki_1m":   {"en-zh": {...}}
      }
    }

The default (full) ``languages`` map is exposed as the ``open_wiki_auto`` preset.
Each key under ``scales`` is exposed as its own preset.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AUTO_PRESET = "open_wiki_auto"
ENV_MANIFEST = "RASST_TERM_MEMORY_MANIFEST"


def lang_key(lang_code: str) -> str:
    """Map a target language code (``zh``) to a manifest key (``en-zh``).

    Accepts a code (``zh``) or an already-formed key (``en-zh``); English is the
    fixed source language for the current RASST checkpoints.
    """

    code = (lang_code or "").strip().lower()
    if "-" in code:
        return code
    return f"en-{code}"


def _resolve(base_dir: Path, raw_path: str) -> str:
    if not raw_path:
        return ""
    expanded = os.path.expandvars(str(raw_path))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


@dataclass
class LanguageSnapshot:
    """Resolved per-language artifacts for one snapshot/scale."""

    lang_key: str
    terms_path: str = ""
    indexes: Dict[str, str] = field(default_factory=dict)
    num_terms: int = 0

    def index_path(self, kind: str = "maxsim") -> str:
        return str(self.indexes.get(kind) or "")

    def index_ready(self, kind: str = "maxsim") -> bool:
        path = self.index_path(kind)
        return bool(path) and Path(path).is_file()

    @classmethod
    def from_dict(cls, key: str, raw: Dict[str, Any], base_dir: Path) -> "LanguageSnapshot":
        indexes_raw = raw.get("indexes") or {}
        if not isinstance(indexes_raw, dict):
            indexes_raw = {}
        indexes = {str(k): _resolve(base_dir, str(v)) for k, v in indexes_raw.items() if v}
        try:
            num_terms = int(raw.get("num_terms") or 0)
        except (TypeError, ValueError):
            num_terms = 0
        return cls(
            lang_key=key,
            terms_path=_resolve(base_dir, str(raw.get("terms_path") or "")),
            indexes=indexes,
            num_terms=num_terms,
        )


@dataclass
class TermMemoryManifest:
    """A loaded, path-resolved terminology-memory manifest."""

    snapshot_id: str
    source: str = ""
    created_at: str = ""
    languages: Dict[str, LanguageSnapshot] = field(default_factory=dict)
    scales: Dict[str, Dict[str, LanguageSnapshot]] = field(default_factory=dict)
    preset_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    path: str = ""

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, base_dir: Path, path: str = "") -> "TermMemoryManifest":
        if not isinstance(raw, dict):
            raise ValueError("manifest must be a JSON object")
        snapshot_id = str(raw.get("snapshot_id") or "").strip()
        if not snapshot_id:
            raise ValueError("manifest missing 'snapshot_id'")

        # Relative artifact paths resolve against ``root`` if given (the snapshot
        # root, e.g. ``$RASST_DEMO_DATA_ROOT/runtime/term_memory``), else against
        # the manifest file's own directory. Publishers usually write absolute
        # paths, in which case this has no effect.
        root_raw = raw.get("root")
        if root_raw:
            root_path = Path(os.path.expandvars(str(root_raw)))
            base_dir = root_path if root_path.is_absolute() else (base_dir / root_path)

        def _lang_map(node: Any) -> Dict[str, LanguageSnapshot]:
            out: Dict[str, LanguageSnapshot] = {}
            if isinstance(node, dict):
                for key, entry in node.items():
                    if isinstance(entry, dict):
                        out[str(key)] = LanguageSnapshot.from_dict(str(key), entry, base_dir)
            return out

        languages = _lang_map(raw.get("languages"))
        scales: Dict[str, Dict[str, LanguageSnapshot]] = {}
        scales_raw = raw.get("scales")
        if isinstance(scales_raw, dict):
            for preset_id, lang_node in scales_raw.items():
                scales[str(preset_id)] = _lang_map(lang_node)
        preset_meta: Dict[str, Dict[str, Any]] = {}
        for meta_key in ("preset_meta", "slice_meta"):
            meta_raw = raw.get(meta_key)
            if isinstance(meta_raw, dict):
                for preset_id, value in meta_raw.items():
                    if isinstance(value, dict):
                        resolved = dict(value)
                        for path_key in ("centroid_path", "maxsim_index_path", "index_path", "glossary_path"):
                            if resolved.get(path_key):
                                resolved[path_key] = _resolve(base_dir, str(resolved[path_key]))
                        preset_meta[str(preset_id)] = resolved
        if not languages and not scales:
            raise ValueError("manifest has neither 'languages' nor 'scales'")
        return cls(
            snapshot_id=snapshot_id,
            source=str(raw.get("source") or ""),
            created_at=str(raw.get("created_at") or ""),
            languages=languages,
            scales=scales,
            preset_meta=preset_meta,
            path=path,
        )

    @classmethod
    def load(cls, path: str) -> "TermMemoryManifest":
        manifest_path = Path(os.path.expandvars(str(path)))
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw, base_dir=manifest_path.resolve().parent, path=str(manifest_path))

    # ------------------------------------------------------------- resolution
    def preset_ids(self) -> List[str]:
        """Selectable open-memory preset ids this manifest provides."""

        ids: List[str] = []
        if self.languages:
            ids.append(AUTO_PRESET)
        ids.extend(self.scales.keys())
        return ids

    def has_preset(self, preset_id: str) -> bool:
        if preset_id == AUTO_PRESET:
            return bool(self.languages)
        return preset_id in self.scales

    def meta_for_preset(self, preset_id: str) -> Dict[str, Any]:
        return dict(self.preset_meta.get(preset_id) or {})

    def snapshot_for(self, preset_id: Optional[str], lang_code: str) -> Optional[LanguageSnapshot]:
        key = lang_key(lang_code)
        if not preset_id or preset_id == AUTO_PRESET:
            return self.languages.get(key)
        scale = self.scales.get(preset_id)
        if scale is not None:
            return scale.get(key)
        return None

    def maxsim_index(self, preset_id: Optional[str], lang_code: str) -> str:
        snapshot = self.snapshot_for(preset_id, lang_code)
        return snapshot.index_path("maxsim") if snapshot else ""


def load_current_manifest(env_var: str = ENV_MANIFEST) -> Optional[TermMemoryManifest]:
    """Load the manifest pointed at by ``$RASST_TERM_MEMORY_MANIFEST``.

    Retrieval is optional, so any problem (unset env, missing file, malformed
    JSON) degrades gracefully to ``None`` with a log line rather than raising.
    """

    path = os.environ.get(env_var, "").strip()
    if not path:
        return None
    try:
        manifest = TermMemoryManifest.load(path)
        logger.info(
            "loaded term-memory manifest %r (snapshot=%s, presets=%s)",
            path,
            manifest.snapshot_id,
            manifest.preset_ids(),
        )
        return manifest
    except (OSError, ValueError) as exc:
        logger.warning("ignoring term-memory manifest %r: %s", path, exc)
        return None
