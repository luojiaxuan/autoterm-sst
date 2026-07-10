"""RASST language/glossary assets + a small catalog helper.

These are agent-internal data (not framework concerns): the language pairs the
RASST omni model is trained for and the precomputed glossary/index presets the
demo offers. Ported verbatim (paths are env-overridable) from
``serve/rasst_sglang_server.py`` so terminology behavior is unchanged.

:class:`GlossaryCatalog` wraps preset -> path/index resolution per language so
the :class:`~framework.agents.omni.OmniAgent` stays focused on streaming.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from framework.agents.plugins.prompt import merge_references, parse_glossary_text
from framework.agents.term_memory import (
    AUTO_PRESET,
    LanguageSnapshot,
    TermMemoryManifest,
    load_current_manifest,
)
from framework.agents.term_memory.domain_taxonomy import (
    AUTO_WORKING_PRESET,
    WORKING_GLOSSARY_PRESETS,
    WORKING_PRESET_META,
)

RASST_ROOT = Path(os.environ.get("RASST_ROOT", "/mnt/taurus/data2/jiaxuanluo/RASST"))
DEMO_DATA_ROOT = Path(
    os.environ.get("RASST_DEMO_DATA_ROOT", "/mnt/taurus/data2/jiaxuanluo/rasst-demo")
)

INDEX_CACHE_DIR = DEMO_DATA_ROOT / "runtime/glossary_indexes"
GLOSSARY_RUNTIME_DIR = DEMO_DATA_ROOT / "runtime/glossaries"
MAIN_RESULT_INDEX_DIR = RASST_ROOT / "outputs/main_result_eval/20260527T071109Z/index_cache"
ACL_RAW_GLOSSARY = RASST_ROOT / "data/glossaries/acl6060_tagged_gt_raw_min_norm2.json"
MEDICINE_RAW_GLOSSARY = (
    RASST_ROOT / "data/glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"
)
MEDICINE_10K_GLOSSARY = (
    Path("/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only")
    / "medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json"
)

def _default_glossary_preset() -> str:
    """Effective default preset (what a session gets if the client omits one).

    ``RASST_DEFAULT_GLOSSARY_PRESET`` wins; otherwise auto working glossary is
    the default when enabled. ``RASST_OPEN_TERM_MEMORY=1`` remains the explicit
    opt-in for the older manifest-wide open memory default.
    Unresolvable open defaults degrade to ``none`` per-catalog (see
    :meth:`GlossaryCatalog.normalize_preset_id`).
    """

    explicit = os.environ.get("RASST_DEFAULT_GLOSSARY_PRESET", "").strip()
    if explicit:
        return explicit
    auto = os.environ.get("RASST_AUTO_GLOSSARY_ENABLED", "1").strip().lower()
    if auto in {"1", "true", "yes", "on"}:
        return AUTO_WORKING_PRESET
    if os.environ.get("RASST_OPEN_TERM_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return AUTO_PRESET
    return "none"


DEFAULT_GLOSSARY_PRESET = _default_glossary_preset()
RAG_STARTUP_GLOSSARY_PRESET = os.environ.get("RASST_RAG_STARTUP_GLOSSARY_PRESET", "acl_tagged_raw")

# Display labels/domains for manifest-driven open-memory presets. The set of
# *available* open presets is whatever the loaded manifest declares.
OPEN_MEMORY_PRESET_META: Dict[str, Dict[str, str]] = {
    AUTO_PRESET: {"label": "Open terminology memory (auto)", "domain": "open"},
    "open_wiki_10k": {"label": "Open Wiki / Wikidata 10k", "domain": "open"},
    "open_wiki_100k": {"label": "Open Wiki / Wikidata 100k", "domain": "open"},
    "open_wiki_500k": {"label": "Open Wiki / Wikidata 500k", "domain": "open"},
    "open_wiki_1m": {"label": "Open Wiki / Wikidata 1M", "domain": "open"},
    "open_wiki_full": {"label": "Open Wiki / Wikidata (full)", "domain": "open"},
    "open_wiki_medicine": {"label": "Open Wiki / Wikidata medicine", "domain": "open"},
}


def _open_preset_label(preset_id: str) -> str:
    if preset_id in WORKING_PRESET_META:
        return WORKING_PRESET_META[preset_id]["label"]
    meta = OPEN_MEMORY_PRESET_META.get(preset_id)
    return meta["label"] if meta else preset_id


def _open_preset_domain(preset_id: str) -> str:
    if preset_id in WORKING_PRESET_META:
        return WORKING_PRESET_META[preset_id]["domain"]
    meta = OPEN_MEMORY_PRESET_META.get(preset_id)
    return meta["domain"] if meta else "open"


DEFAULT_MAX_IMPORTED_GLOSSARY_TERMS = 10000


LANGUAGE_PAIRS: Dict[str, Dict[str, Any]] = {
    "English -> Chinese": {
        "source_lang": "English",
        "target_lang": "Chinese",
        "lang_code": "zh",
        "model_path": os.environ.get(
            "RASST_MODEL_ZH_CAP16_DENOISE",
            "/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/models/"
            "speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_ZH_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__zh__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> Japanese": {
        "source_lang": "English",
        "target_lang": "Japanese",
        "lang_code": "ja",
        "model_path": os.environ.get(
            "RASST_MODEL_JA_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_JA_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__ja__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> German": {
        "source_lang": "English",
        "target_lang": "German",
        "lang_code": "de",
        "model_path": os.environ.get(
            "RASST_MODEL_DE_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_DE_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__de__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
}


def _main_result_index(domain: str, lang_code: str, latency_multiplier: int, filename: str) -> str:
    return str(MAIN_RESULT_INDEX_DIR / f"{domain}__{lang_code}__lm{latency_multiplier}" / filename)


GLOSSARY_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "id": "none",
        "label": "None",
        "path": "",
        "domain": "none",
        "index_path": "",
    },
    "acl_tagged_raw": {
        "id": "acl_tagged_raw",
        "label": "ACL tagged glossary raw",
        "path": str(ACL_RAW_GLOSSARY),
        "domain": "acl6060",
        "index_paths": {
            # RASST_INDEX_ZH_ACL lets a demo host boot its warmup retriever on
            # a curated slice; eval hosts leave it unset and keep this default.
            "zh": os.environ.get("RASST_INDEX_ZH_ACL") or _main_result_index(
                "acl_tagged_raw", "zh", 2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
            "ja": _main_result_index(
                "acl_tagged_raw", "ja", 2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
            "de": _main_result_index(
                "acl_tagged_raw", "de", 2,
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt",
            ),
        },
    },
    "acl_tagged_1k": {
        "id": "acl_tagged_1k",
        "label": "ACL tagged glossary 1k",
        "path": "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json",
        "domain": "acl6060",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_acl6060_gt_union_gs1000_hn1024_tr128_ta256.pt"),
    },
    "acl_tagged_10k": {
        "id": "acl_tagged_10k",
        "label": "ACL tagged glossary 10k",
        "path": "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json",
        "domain": "acl6060",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_acl6060_gt_union_gs10000_hn1024_tr128_ta256.pt"),
    },
    "medicine_raw": {
        "id": "medicine_raw",
        "label": "Medicine raw glossary",
        "path": str(MEDICINE_RAW_GLOSSARY),
        "domain": "medicine",
        "index_paths": {
            "zh": _main_result_index(
                "medicine_hardraw", "zh", 2,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
            "de": _main_result_index(
                "medicine_hardraw", "de", 1,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
            "ja": _main_result_index(
                "medicine_hardraw", "zh", 2,
                "maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt",
            ),
        },
    },
    "medicine_1k": {
        "id": "medicine_1k",
        "label": "Medicine glossary 1k",
        "path": str(GLOSSARY_RUNTIME_DIR / "medicine_glossary_gt_plus_medicine_wiki_gs1000_translated.json"),
        "domain": "medicine",
        "index_path": str(INDEX_CACHE_DIR / "maxsim_medicine_gt_plus_wiki_gs1000_hn1024_tr128_ta256.pt"),
    },
    "medicine_10k": {
        "id": "medicine_10k",
        "label": "Medicine glossary 10k",
        "path": str(MEDICINE_10K_GLOSSARY),
        "domain": "medicine",
        "index_path": (
            "/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_gs10k_pr_sweep/"
            "maxsim_medicine_glossary_gt_plus_medicine_wiki_gs10000__0d6ee2097a706d9c_tr128_ta256.pt"
        ),
    },
}


class GlossaryCatalog:
    """Per-language preset resolution (path / index / term counting)."""

    _count_cache: Dict[str, int] = {}

    def __init__(
        self,
        language_pair: str,
        max_imported_terms: int = DEFAULT_MAX_IMPORTED_GLOSSARY_TERMS,
        manifest: Optional[TermMemoryManifest] = None,
    ) -> None:
        if language_pair not in LANGUAGE_PAIRS:
            raise ValueError(f"unsupported language pair: {language_pair!r}")
        self.language_pair = language_pair
        self.lang_cfg = LANGUAGE_PAIRS[language_pair]
        self.lang_code = str(self.lang_cfg["lang_code"])
        self.source_lang = str(self.lang_cfg["source_lang"])
        self.target_lang = str(self.lang_cfg["target_lang"])
        self.model_path = str(self.lang_cfg.get("model_path") or "")
        self.default_index = str(self.lang_cfg["index_path"])
        self.max_imported_terms = int(max_imported_terms)
        # Open terminology memory (Phase B): manifest-driven, optional. Falls back
        # to legacy presets only when no manifest is configured/loadable.
        self.manifest = manifest if manifest is not None else load_current_manifest()

    # ------------------------------------------------------------ open memory
    def _is_open_preset(self, preset_id: Optional[str]) -> bool:
        if not preset_id:
            return False
        if str(preset_id).startswith("open_wiki"):
            return True
        if preset_id in WORKING_GLOSSARY_PRESETS:
            return True
        # any preset id the loaded manifest declares (e.g. scale experiments)
        return self.manifest is not None and preset_id in set(self.manifest.preset_ids())

    def _open_snapshot(self, preset_id: Optional[str]) -> Optional[LanguageSnapshot]:
        if not self.manifest or not self._is_open_preset(preset_id):
            return None
        return self.manifest.snapshot_for(preset_id, self.lang_code)

    def open_preset_ids(self) -> List[str]:
        """Open-memory preset ids actually resolvable for THIS language."""
        if not self.manifest:
            return []
        return [pid for pid in self.manifest.preset_ids() if self._open_snapshot(pid) is not None]

    def normalize_preset_id(self, preset_id: Optional[str]) -> str:
        if not preset_id:
            preset_id = DEFAULT_GLOSSARY_PRESET
        if preset_id == AUTO_WORKING_PRESET:
            return AUTO_WORKING_PRESET
        if preset_id in GLOSSARY_PRESETS:
            return preset_id
        if self._is_open_preset(preset_id):
            # Graceful degradation: an open preset with no manifest/snapshot for
            # this language behaves as 'none' rather than failing the session.
            return preset_id if self._open_snapshot(preset_id) is not None else "none"
        raise ValueError(f"Unknown glossary preset: {preset_id}")

    def index_path_for_preset(self, preset_id: Optional[str]) -> str:
        normalized = self.normalize_preset_id(preset_id)
        if normalized == AUTO_WORKING_PRESET:
            normalized = os.environ.get("RASST_AUTO_GLOSSARY_DEFAULT", "nlp_core_10k").strip() or "nlp_core_10k"
        snapshot = self._open_snapshot(normalized)
        if snapshot is not None:
            return snapshot.index_path("maxsim")
        if normalized not in GLOSSARY_PRESETS:
            return ""
        preset = GLOSSARY_PRESETS[normalized]
        if preset["id"] == "none":
            return ""
        index_paths = preset.get("index_paths")
        if isinstance(index_paths, dict) and index_paths.get(self.lang_code):
            return str(index_paths[self.lang_code])
        index_path = str(preset.get("index_path") or "")
        if index_path:
            return index_path
        return self.default_index

    def glossary_path_for_preset(self, preset_id: Optional[str]) -> str:
        normalized = self.normalize_preset_id(preset_id)
        if normalized == AUTO_WORKING_PRESET:
            normalized = os.environ.get("RASST_AUTO_GLOSSARY_DEFAULT", "nlp_core_10k").strip() or "nlp_core_10k"
        snapshot = self._open_snapshot(normalized)
        if snapshot is not None:
            return snapshot.terms_path
        if normalized not in GLOSSARY_PRESETS:
            return ""
        return str(GLOSSARY_PRESETS[normalized].get("path") or "")

    def count_glossary_rows(self, glossary_path: str) -> int:
        if not glossary_path:
            return 0
        if glossary_path in self._count_cache:
            return self._count_cache[glossary_path]
        path = Path(glossary_path)
        if not path.exists():
            self._count_cache[glossary_path] = 0
            return 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            count = len(raw) if isinstance(raw, (dict, list)) else 0
        except (OSError, ValueError):
            count = 0
        self._count_cache[glossary_path] = count
        return count

    def resolve_imported_glossary(self, glossary_text: str) -> List[Dict[str, Any]]:
        merged = merge_references(parse_glossary_text(glossary_text or ""))
        if self.max_imported_terms:
            merged = merged[: self.max_imported_terms]
        return merged

    def _preset_terms(self, normalized: str, glossary_path: str) -> int:
        if normalized == AUTO_WORKING_PRESET:
            normalized = os.environ.get("RASST_AUTO_GLOSSARY_DEFAULT", "nlp_core_10k").strip() or "nlp_core_10k"
        snapshot = self._open_snapshot(normalized)
        if snapshot is not None:
            # Avoid counting a 1M-line JSONL: trust the manifest's num_terms.
            return snapshot.num_terms or self.count_glossary_rows(glossary_path)
        return self.count_glossary_rows(glossary_path)

    def describe_selection(self, preset_id: Optional[str], glossary_text: str) -> Dict[str, Any]:
        normalized = self.normalize_preset_id(preset_id)
        active_normalized = normalized
        if normalized == AUTO_WORKING_PRESET:
            active_normalized = os.environ.get("RASST_AUTO_GLOSSARY_DEFAULT", "nlp_core_10k").strip() or "nlp_core_10k"
            active_normalized = self.normalize_preset_id(active_normalized)
        glossary_path = self.glossary_path_for_preset(active_normalized)
        index_path = self.index_path_for_preset(active_normalized)
        manual_refs = self.resolve_imported_glossary(glossary_text)
        return {
            "glossary_preset": normalized,
            "active_glossary_preset": active_normalized,
            "glossary_path": glossary_path,
            "preset_terms": self._preset_terms(active_normalized, glossary_path),
            "manual_terms": len(manual_refs),
            "manual_refs": manual_refs,
            "index_path": index_path,
            "index_ready": (not index_path) or Path(index_path).is_file(),
        }

    def preset_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        default_auto = os.environ.get("RASST_AUTO_GLOSSARY_DEFAULT", "nlp_core_10k").strip() or "nlp_core_10k"
        auto_index = self.index_path_for_preset(default_auto)
        catalog.append(
            {
                "id": AUTO_WORKING_PRESET,
                "label": WORKING_PRESET_META[AUTO_WORKING_PRESET]["label"],
                "domain": WORKING_PRESET_META[AUTO_WORKING_PRESET]["domain"],
                "preset_terms": self._preset_terms(self.normalize_preset_id(default_auto), self.glossary_path_for_preset(default_auto)),
                "index_path": auto_index,
                "available": True,
                "adaptive": True,
            }
        )
        for preset in GLOSSARY_PRESETS.values():
            index_path = self.index_path_for_preset(preset["id"])
            path = str(preset.get("path") or "")
            available = preset["id"] == "none" or (
                bool(path) and Path(path).exists() and (not index_path or Path(index_path).is_file())
            )
            catalog.append(
                {
                    "id": preset["id"],
                    "label": preset["label"],
                    "domain": preset["domain"],
                    "preset_terms": self.count_glossary_rows(path),
                    "index_path": index_path,
                    "available": bool(available),
                }
            )
        for preset_id in WORKING_GLOSSARY_PRESETS:
            if preset_id in self.open_preset_ids():
                continue
            meta = WORKING_PRESET_META[preset_id]
            catalog.append(
                {
                    "id": preset_id,
                    "label": meta["label"],
                    "domain": meta["domain"],
                    "preset_terms": 0,
                    "index_path": "",
                    "available": False,
                    "adaptive": False,
                }
            )
        # Manifest-driven open-memory presets (Phase B): listed after the legacy
        # presets, available when their precomputed index is on disk.
        for preset_id in self.open_preset_ids():
            snapshot = self._open_snapshot(preset_id)
            if snapshot is None:
                continue
            index_path = snapshot.index_path("maxsim")
            manifest_meta = self.manifest.meta_for_preset(preset_id) if self.manifest else {}
            catalog.append(
                {
                    "id": preset_id,
                    "label": str(manifest_meta.get("label") or _open_preset_label(preset_id)),
                    "domain": str(manifest_meta.get("domain") or _open_preset_domain(preset_id)),
                    "preset_terms": snapshot.num_terms,
                    "index_path": index_path,
                    "available": bool(index_path) and Path(index_path).is_file(),
                    "snapshot_id": self.manifest.snapshot_id if self.manifest else "",
                    "adaptive": preset_id in WORKING_GLOSSARY_PRESETS,
                }
            )
        return catalog
