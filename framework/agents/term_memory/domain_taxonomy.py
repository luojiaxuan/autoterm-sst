"""Small domain taxonomy for adaptive working-glossary defaults.

Runtime routing is manifest/embedding driven by default. The keyword lists here
are fallback labels plus offline working-slice ranking seeds; they are not the
primary online router signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

AUTO_WORKING_PRESET = "auto_working"

GENERAL_DOMAIN = "general"
DOMAIN_TO_PRESET: Dict[str, str] = {
    GENERAL_DOMAIN: "common_10k",
    "nlp": "nlp_core_10k",
    "medicine": "medicine_core_10k",
    "finance": "finance_core_10k",
    "legal": "legal_core_10k",
}

WORKING_GLOSSARY_PRESETS = tuple(DOMAIN_TO_PRESET.values())
WORKING_DOMAINS = tuple(DOMAIN_TO_PRESET.keys())

WORKING_PRESET_META: Dict[str, Dict[str, str]] = {
    AUTO_WORKING_PRESET: {
        "label": "Auto working glossary",
        "domain": "auto",
        "description": "Starts from common_10k and switches to a compact domain slice.",
    },
    "common_10k": {
        "label": "Common working glossary 10k",
        "domain": GENERAL_DOMAIN,
        "description": "General high-precision terms used before a topic is clear.",
    },
    "nlp_core_10k": {
        "label": "NLP core working glossary 10k",
        "domain": "nlp",
        "description": "NLP, speech, translation, ML, dataset, and benchmark terms.",
    },
    "medicine_core_10k": {
        "label": "Medicine core working glossary 10k",
        "domain": "medicine",
        "description": "Clinical, disease, drug, procedure, and biomedical terms.",
    },
    "finance_core_10k": {
        "label": "Finance core working glossary 10k",
        "domain": "finance",
        "description": "Market, instrument, accounting, monetary, and trading terms.",
    },
    "legal_core_10k": {
        "label": "Legal core working glossary 10k",
        "domain": "legal",
        "description": "Law, court, contract, regulation, and liability terms.",
    },
}

DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "nlp": (
        "language model",
        "large language model",
        "natural language processing",
        "machine translation",
        "speech translation",
        "speech recognition",
        "simultaneous interpretation",
        "retrieval augmented generation",
        "rag",
        "token",
        "tokenizer",
        "benchmark",
        "dataset",
        "transformer",
        "encoder",
        "decoder",
        "alignment",
        "bleu",
        "comet",
        "attention",
        "embedding",
        "fine-tuning",
        "pretraining",
        "corpus",
        "treebank",
        "named entity",
        "question answering",
        "summarization",
        "information retrieval",
    ),
    "medicine": (
        "patient",
        "clinical",
        "disease",
        "diagnosis",
        "treatment",
        "medicine",
        "drug",
        "protein",
        "cancer",
        "vaccine",
        "hospital",
        "trial",
        "therapy",
        "symptom",
        "infection",
        "syndrome",
        "tumor",
        "surgery",
        "pharmacology",
        "cardiology",
        "oncology",
        "immunology",
    ),
    "finance": (
        "market",
        "stock",
        "revenue",
        "valuation",
        "equity",
        "bond",
        "interest rate",
        "earnings",
        "etf",
        "inflation",
        "monetary",
        "trading",
        "portfolio",
        "derivative",
        "dividend",
        "asset",
        "liability",
        "cash flow",
        "central bank",
        "treasury",
    ),
    "legal": (
        "law",
        "court",
        "regulation",
        "contract",
        "liability",
        "policy",
        "statute",
        "legal",
        "lawsuit",
        "plaintiff",
        "defendant",
        "jurisdiction",
        "compliance",
        "patent",
        "copyright",
        "tort",
        "appeal",
        "arbitration",
        "clause",
    ),
}

ENTITY_TYPE_DOWNRANK = (
    "human",
    "person",
    "given name",
    "family name",
    "place",
    "country",
    "city",
    "geographic",
    "wikimedia",
    "category",
    "disambiguation",
)


@dataclass(frozen=True)
class DomainScore:
    domain: str
    score: float
    reason: str


def preset_for_domain(domain: str, default_preset: str = "common_10k") -> str:
    return DOMAIN_TO_PRESET.get((domain or "").strip().lower(), default_preset)


def domain_for_preset(preset: str) -> str:
    for domain, candidate in DOMAIN_TO_PRESET.items():
        if candidate == preset:
            return domain
    return WORKING_PRESET_META.get(preset, {}).get("domain", GENERAL_DOMAIN)


def configured_working_presets(raw: str) -> Tuple[str, ...]:
    presets = tuple(p.strip() for p in (raw or "").split(",") if p.strip())
    return presets or WORKING_GLOSSARY_PRESETS


def keyword_hits(text: str, domain: str) -> int:
    blob = (text or "").lower()
    return sum(1 for kw in DOMAIN_KEYWORDS.get(domain, ()) if kw in blob)


def best_keyword_domain(text: str) -> DomainScore:
    scores = [
        DomainScore(domain=domain, score=float(keyword_hits(text, domain)), reason="keyword")
        for domain in DOMAIN_KEYWORDS
    ]
    scores.sort(key=lambda item: item.score, reverse=True)
    return scores[0] if scores else DomainScore(GENERAL_DOMAIN, 0.0, "none")


def entry_domain_score(row: Dict[str, object], domain: str) -> float:
    """Score one glossary row for a domain slice builder.

    This intentionally favors term-like technical phrases and down-ranks generic
    entity rows. It is a lightweight ranking signal, not a taxonomy.
    """

    term = str(row.get("term") or row.get("source_label") or "").strip()
    desc = str(row.get("short_description") or row.get("description") or "").strip()
    blob = f"{term} {desc}".lower()
    score = float(keyword_hits(blob, domain)) * 10.0
    if domain == GENERAL_DOMAIN:
        score = 1.0
    if " " in term.strip():
        score += 2.0
    if len(term) >= 8:
        score += 0.5
    rank = row.get("rank")
    if isinstance(rank, int):
        score += max(0.0, 3.0 - min(float(rank), 1_000_000.0) / 333_333.0)
    types: Iterable[object] = row.get("entity_types") if isinstance(row.get("entity_types"), list) else []
    type_blob = " ".join(str(t).lower() for t in types)
    if any(marker in type_blob or marker in blob for marker in ENTITY_TYPE_DOWNRANK):
        score -= 4.0
    return score
