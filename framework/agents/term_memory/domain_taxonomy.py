"""Small domain taxonomy for adaptive working-glossary defaults.

Runtime routing is window-topic-first by default. The keyword lists here provide
the high-precision source/ASR topic signal, plus offline working-slice ranking
seeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

AUTO_WORKING_PRESET = "auto_working"

GENERAL_DOMAIN = "general"
COMMON_WORKING_PRESET = "common_10k"
DOMAIN_TO_PRESET: Dict[str, str] = {
    "nlp": "nlp_core_10k",
    "medicine": "medicine_core_10k",
    "finance": "finance_core_10k",
    "legal": "legal_core_10k",
}

WORKING_GLOSSARY_PRESETS = (COMMON_WORKING_PRESET, *DOMAIN_TO_PRESET.values())
WORKING_DOMAINS = tuple(DOMAIN_TO_PRESET.keys())

WORKING_PRESET_META: Dict[str, Dict[str, str]] = {
    AUTO_WORKING_PRESET: {
        "label": "Auto working glossary",
        "domain": "auto",
        "description": "Routes directly among domain-specific working glossary slices.",
    },
    "common_10k": {
        "label": "Common working glossary 10k",
        "domain": GENERAL_DOMAIN,
        "description": "Always-on common terms base slice for automatic routing.",
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


@dataclass(frozen=True)
class TopicKeyword:
    pattern: str
    domain: str
    weight: float = 1.0
    case_sensitive: bool = False


DOMAIN_TOPIC_KEYWORDS: Dict[str, Tuple[TopicKeyword, ...]] = {
    "nlp": (
        TopicKeyword(r"\blanguage model(s)?\b", "nlp", 1.2),
        TopicKeyword(r"\blarge language model(s)?\b", "nlp", 1.3),
        TopicKeyword(r"\bnatural language processing\b", "nlp", 1.4),
        TopicKeyword(r"\bNLP\b", "nlp", 1.2, case_sensitive=True),
        TopicKeyword(r"\bBERT\b", "nlp", 1.2, case_sensitive=True),
        TopicKeyword(r"\btransformer(s)?\b", "nlp", 1.0),
        TopicKeyword(r"\bencoder(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bdecoder(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bdataset(s)?\b", "nlp", 0.7),
        TopicKeyword(r"\bbenchmark(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bcorpus|corpora\b", "nlp", 1.0),
        TopicKeyword(r"\bannotation(s)?\b", "nlp", 0.8),
        TopicKeyword(r"\bparser(s)?|parsing\b", "nlp", 1.0),
        TopicKeyword(r"\bmachine translation\b", "nlp", 1.2),
        TopicKeyword(r"\bentity recognition\b", "nlp", 1.1),
        TopicKeyword(r"\bdependency parsing\b", "nlp", 1.1),
        TopicKeyword(r"\bBLEU\b", "nlp", 1.1, case_sensitive=True),
        TopicKeyword(r"\battention\b", "nlp", 0.9),
        TopicKeyword(r"\bembedding(s)?\b", "nlp", 0.9),
        TopicKeyword(r"\btoken(s|ization|izer)?\b", "nlp", 0.9),
        TopicKeyword(r"\bpretraining|pre-trained|pretrained\b", "nlp", 1.0),
        TopicKeyword(r"\bfine[- ]?tuning\b", "nlp", 1.0),
        TopicKeyword(r"\bprompt(s|ing)?\b", "nlp", 0.8),
        TopicKeyword(r"语言模型", "nlp", 1.2),
        TopicKeyword(r"自然语言处理", "nlp", 1.4),
        TopicKeyword(r"机器翻译|语音翻译|同声传译", "nlp", 1.2),
        TopicKeyword(r"数据集|基准测试|语料库", "nlp", 1.0),
        TopicKeyword(r"标注|解析|实体识别|依存句法", "nlp", 1.0),
        TopicKeyword(r"注意力|嵌入|预训练|微调|提示", "nlp", 0.9),
    ),
    "medicine": (
        TopicKeyword(r"\bpatient(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bclinical\b", "medicine", 1.1),
        TopicKeyword(r"\bdiagnos(is|es|tic)\b", "medicine", 1.1),
        TopicKeyword(r"\bsymptom(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdisease(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bfever(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bheadache(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\btablet(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdose(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bmg\b", "medicine", 0.9),
        TopicKeyword(r"\btreatment(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bprescrib(e|ed|es|ing)\b", "medicine", 1.1),
        TopicKeyword(r"\bdrug(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bmedicine(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bdiabetes\b", "medicine", 1.2),
        TopicKeyword(r"\bhypertension\b", "medicine", 1.2),
        TopicKeyword(r"\bcancer(s)?\b", "medicine", 1.2),
        TopicKeyword(r"\btrial(s)?\b", "medicine", 0.9),
        TopicKeyword(r"\binfection(s)?\b", "medicine", 1.0),
        TopicKeyword(r"\bvaccine(s)?\b", "medicine", 1.1),
        TopicKeyword(r"\bMRI\b", "medicine", 1.1, case_sensitive=True),
        TopicKeyword(r"\bCT\b", "medicine", 0.9, case_sensitive=True),
        TopicKeyword(r"\bblood pressure\b", "medicine", 1.2),
        TopicKeyword(r"\bheart rate\b", "medicine", 1.1),
        TopicKeyword(r"\bsurger(y|ies|ical)\b", "medicine", 1.1),
        TopicKeyword(r"\boncolog(y|ical|ist|ists)\b", "medicine", 1.2),
        TopicKeyword(r"\bhospital(s)?\b", "medicine", 1.0),
        TopicKeyword(r"患者|病人", "medicine", 1.0),
        TopicKeyword(r"临床|诊断|症状|疾病", "medicine", 1.1),
        TopicKeyword(r"治疗|处方|药物|医学|医院", "medicine", 1.0),
        TopicKeyword(r"糖尿病|高血压|癌症|肿瘤|感染|疫苗", "medicine", 1.2),
        TopicKeyword(r"试验|手术|血压|心率|剂量|毫克", "medicine", 1.0),
        TopicKeyword(r"核磁|磁共振|CT", "medicine", 0.9, case_sensitive=True),
    ),
    "finance": (
        TopicKeyword(r"\bmarket(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bstock(s)?\b", "finance", 1.0),
        TopicKeyword(r"\brevenue\b", "finance", 1.0),
        TopicKeyword(r"\bvaluation(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bequity\b", "finance", 1.0),
        TopicKeyword(r"\bbond(s)?\b", "finance", 1.0),
        TopicKeyword(r"\binterest rate(s)?\b", "finance", 1.1),
        TopicKeyword(r"\bearnings\b", "finance", 1.0),
        TopicKeyword(r"\binflation\b", "finance", 1.0),
        TopicKeyword(r"\btrading\b", "finance", 1.0),
        TopicKeyword(r"\bportfolio(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bderivative(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bdividend(s)?\b", "finance", 1.0),
        TopicKeyword(r"\bcash flow\b", "finance", 1.1),
        TopicKeyword(r"\bcentral bank(s)?\b", "finance", 1.1),
        TopicKeyword(r"\btreasury\b", "finance", 0.9),
    ),
    "legal": (
        TopicKeyword(r"\blaw(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcourt(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bregulation(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcontract(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bliabilit(y|ies)\b", "legal", 1.0),
        TopicKeyword(r"\bstatute(s)?\b", "legal", 1.0),
        TopicKeyword(r"\blegal\b", "legal", 0.9),
        TopicKeyword(r"\blawsuit(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bplaintiff(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bdefendant(s)?\b", "legal", 1.1),
        TopicKeyword(r"\bjurisdiction(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcompliance\b", "legal", 0.9),
        TopicKeyword(r"\bpatent(s)?\b", "legal", 1.0),
        TopicKeyword(r"\bcopyright(s)?\b", "legal", 1.0),
        TopicKeyword(r"\barbitration\b", "legal", 1.0),
    ),
}


def topic_keyword_scores(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores = {domain: 0.0 for domain in DOMAIN_TOPIC_KEYWORDS}
    hits: Dict[str, List[str]] = {domain: [] for domain in DOMAIN_TOPIC_KEYWORDS}
    if not text:
        return scores, hits
    for domain, keywords in DOMAIN_TOPIC_KEYWORDS.items():
        for keyword in keywords:
            flags = 0 if keyword.case_sensitive else re.IGNORECASE
            if re.search(keyword.pattern, text, flags):
                scores[domain] += float(keyword.weight)
                hits[domain].append(keyword.pattern)
    return scores, hits


def preset_for_domain(domain: str, default_preset: str = "none") -> str:
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
