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

DEFAULT_GLOSSARY_PRESET = "none"
RAG_STARTUP_GLOSSARY_PRESET = "acl_tagged_raw"

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
            "zh": _main_result_index(
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

    def __init__(self, language_pair: str, max_imported_terms: int = DEFAULT_MAX_IMPORTED_GLOSSARY_TERMS) -> None:
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

    def normalize_preset_id(self, preset_id: Optional[str]) -> str:
        if not preset_id:
            return DEFAULT_GLOSSARY_PRESET
        if preset_id not in GLOSSARY_PRESETS:
            raise ValueError(f"Unknown glossary preset: {preset_id}")
        return preset_id

    def index_path_for_preset(self, preset_id: Optional[str]) -> str:
        preset = GLOSSARY_PRESETS[self.normalize_preset_id(preset_id)]
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
        preset = GLOSSARY_PRESETS[self.normalize_preset_id(preset_id)]
        return str(preset.get("path") or "")

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

    def describe_selection(self, preset_id: Optional[str], glossary_text: str) -> Dict[str, Any]:
        normalized = self.normalize_preset_id(preset_id)
        glossary_path = self.glossary_path_for_preset(normalized)
        index_path = self.index_path_for_preset(normalized)
        manual_refs = self.resolve_imported_glossary(glossary_text)
        return {
            "glossary_preset": normalized,
            "glossary_path": glossary_path,
            "preset_terms": self.count_glossary_rows(glossary_path),
            "manual_terms": len(manual_refs),
            "manual_refs": manual_refs,
            "index_path": index_path,
            "index_ready": (not index_path) or Path(index_path).is_file(),
        }

    def preset_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
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
        return catalog
