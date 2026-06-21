"""Open terminology memory: schema + manifest for large, swappable snapshots.

This package is agent-internal data plumbing (not a framework concern). It lets
the RASST :class:`~framework.agents.omni.OmniAgent` resolve a manifest-described
terminology memory snapshot (Wikidata/Wikipedia-derived, or any precomputed
glossary) into the same ``maxsim`` index format the existing
:class:`~framework.agents.plugins.retrieval.MaxSimRetrievalPlugin` already loads.

The snapshot artifacts (jsonl term files, indexes) live OUTSIDE the repo under a
runtime root (e.g. ``$RASST_DEMO_DATA_ROOT/runtime/term_memory``); only the small
manifest JSON points at them.
"""

from framework.agents.term_memory.manifest import (
    AUTO_PRESET,
    ENV_MANIFEST,
    LanguageSnapshot,
    TermMemoryManifest,
    lang_key,
    load_current_manifest,
)
from framework.agents.term_memory.schema import TermEntry

__all__ = [
    "TermEntry",
    "LanguageSnapshot",
    "TermMemoryManifest",
    "AUTO_PRESET",
    "ENV_MANIFEST",
    "lang_key",
    "load_current_manifest",
]
