"""AutoTerm-SST slice roles, retrieval plans, and fixed top-k helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from framework.agents.term_memory.domain_taxonomy import GENERAL_DOMAIN, domain_for_preset

COMMON_TERMS_SLICE_ID = "common_terms"
OPEN_RESCUE_SLICE_ID = "open_wiki_100k"
ORACLE_SLICE_ID = "acl_tagged_raw"
PROMPT_K = 10

_GENERIC_UNIGRAMS = {
    "model",
    "method",
    "data",
    "dataset",
    "result",
    "system",
    "task",
    "approach",
    "baseline",
    "paper",
    "experiment",
    "performance",
    "learning",
    "training",
}
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9.+&/-]{1,}$")

_DEFAULT_COMMON_BACKFILL_TERMS = (
    ("AI", "AI"),
    ("NLP", "自然语言处理"),
    ("machine learning", "机器学习"),
    ("deep learning", "深度学习"),
    ("language model", "语言模型"),
    ("neural network", "神经网络"),
    ("dataset", "数据集"),
    ("benchmark", "基准测试"),
    ("algorithm", "算法"),
    ("speech recognition", "语音识别"),
    ("speech translation", "语音翻译"),
    ("machine translation", "机器翻译"),
    ("named entity recognition", "命名实体识别"),
    ("information retrieval", "信息检索"),
    ("question answering", "问答"),
    ("reinforcement learning", "强化学习"),
    ("Transformer", "Transformer"),
    ("BERT", "BERT"),
    ("evaluation metric", "评测指标"),
    ("training data", "训练数据"),
)


@dataclass(frozen=True)
class RetrievalSlice:
    preset_id: str
    slice_id: str
    role: str
    domain: str
    index_path: str
    term_count: int = 0
    weight: float = 1.0
    eval_only: bool = False

    def to_meta(self) -> Dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "slice_id": self.slice_id,
            "role": self.role,
            "domain": self.domain,
            "index_path": self.index_path,
            "term_count": self.term_count,
            "weight": self.weight,
            "eval_only": self.eval_only,
        }


def slice_id_for_preset(preset_id: str) -> str:
    preset = (preset_id or "").strip()
    if preset in {"common_10k", "common_terms"}:
        return COMMON_TERMS_SLICE_ID
    return preset


def slice_role_for_preset(preset_id: str) -> str:
    preset = (preset_id or "").strip()
    if preset in {"common_10k", COMMON_TERMS_SLICE_ID}:
        return "base"
    if preset.startswith("open_wiki"):
        return "rescue"
    if preset == ORACLE_SLICE_ID:
        return "oracle"
    return "domain"


def slice_weight_for_role(role: str) -> float:
    if role == "base":
        return 1.0
    if role == "domain":
        return 0.8
    if role == "rescue":
        return 0.4
    return 1.0


def canonical_source(ref: Dict[str, Any]) -> str:
    return str(ref.get("canonical_source") or ref.get("key") or ref.get("term") or "").strip().casefold()


def is_acronym(term: str) -> bool:
    return bool(_ACRONYM_RE.fullmatch(str(term or "").strip()))


def genericity_score(term: str) -> float:
    words = re.findall(r"[A-Za-z0-9]+", str(term or "").casefold())
    if not words:
        return 0.0
    if len(words) == 1 and words[0] in _GENERIC_UNIGRAMS:
        return 1.0
    generic_hits = sum(1 for word in words if word in _GENERIC_UNIGRAMS)
    return min(1.0, 0.25 * generic_hits)


def multiword_specificity(term: str) -> float:
    words = re.findall(r"[A-Za-z0-9]+", str(term or ""))
    if len(words) <= 1:
        return 0.0
    return min(1.0, 0.20 * (len(words) - 1))


def dedupe_references(references: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for ref in references or []:
        key = canonical_source(ref)
        if not key:
            continue
        item = dict(ref)
        try:
            score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        old = best.get(key)
        if old is None:
            best[key] = item
            continue
        try:
            old_score = float(old.get("rerank_score", old.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            old_score = 0.0
        if score > old_score:
            best[key] = item
    return list(best.values())


def reference_rerank_score(ref: Dict[str, Any], *, active_domain: str = GENERAL_DOMAIN) -> float:
    term = str(ref.get("term") or "")
    try:
        dense_score = float(ref.get("score") or 0.0)
    except (TypeError, ValueError):
        dense_score = 0.0
    try:
        lexical_match_score = float(ref.get("lexical_match_score") or 0.0)
    except (TypeError, ValueError):
        lexical_match_score = 0.0
    domain = str(ref.get("source_domain") or ref.get("domain") or "")
    role = str(ref.get("source_slice_role") or slice_role_for_preset(str(ref.get("source_preset") or "")))
    domain_prior = 1.0 if active_domain and active_domain != GENERAL_DOMAIN and domain == active_domain else 0.0
    acronym_exact_match = 1.0 if is_acronym(term) else 0.0
    try:
        term_memory_prior = float(ref.get("term_memory_prior") or 0.0)
    except (TypeError, ValueError):
        term_memory_prior = 0.0
    try:
        repeated_window_hits = float(ref.get("repeated_window_hits") or 0.0)
    except (TypeError, ValueError):
        repeated_window_hits = 0.0
    try:
        stale_candidate_penalty = float(ref.get("stale_candidate_penalty") or 0.0)
    except (TypeError, ValueError):
        stale_candidate_penalty = 0.0
    generic_penalty = genericity_score(term)
    specificity = multiword_specificity(term)
    role_weight = slice_weight_for_role(role)
    return (
        1.00 * dense_score
        + 0.90 * lexical_match_score
        + 1.20 * acronym_exact_match
        + 0.40 * domain_prior
        + 0.40 * term_memory_prior
        + 0.30 * repeated_window_hits
        + 0.25 * specificity
        + 0.10 * role_weight
        - 0.50 * generic_penalty
        - 0.30 * stale_candidate_penalty
    )


def rank_references(references: Sequence[Dict[str, Any]], *, active_domain: str = GENERAL_DOMAIN) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for ref in dedupe_references(references):
        item = dict(ref)
        item["dense_score"] = float(item.get("score") or 0.0)
        item["genericity_score"] = genericity_score(str(item.get("term") or ""))
        item["rerank_score"] = reference_rerank_score(item, active_domain=active_domain)
        ranked.append(item)
    return sorted(ranked, key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)


def default_common_backfill_references(k: int = PROMPT_K) -> List[Dict[str, Any]]:
    limit = max(0, int(k))
    refs: List[Dict[str, Any]] = []
    for term, translation in _DEFAULT_COMMON_BACKFILL_TERMS[:limit]:
        refs.append(
            {
                "term": term,
                "translation": translation,
                "target_translations": {"zh": translation},
                "canonical_source": term,
                "source": "fixed_prompt_k_default",
                "source_preset": "common_10k",
                "source_slice_id": COMMON_TERMS_SLICE_ID,
                "source_slice_role": "base",
                "source_domain": GENERAL_DOMAIN,
                "score": -100.0,
                "dense_score": -100.0,
                "rerank_score": -100.0,
                "fallback_reason": "fixed_prompt_k_default",
            }
        )
    return refs


def force_exactly_k_references(
    ranked: Sequence[Dict[str, Any]],
    *,
    k: int = PROMPT_K,
    backfill: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    limit = max(0, int(k))
    if limit == 0:
        return []
    merged = dedupe_references(list(ranked or []) + list(backfill or []))
    merged = sorted(
        merged,
        key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
        reverse=True,
    )
    if len(merged) < limit:
        merged = dedupe_references(merged + default_common_backfill_references(limit))
        merged = sorted(
            merged,
            key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )
    return merged[:limit]


def domain_for_slice_preset(preset_id: str) -> str:
    domain = domain_for_preset(preset_id)
    return domain or GENERAL_DOMAIN
