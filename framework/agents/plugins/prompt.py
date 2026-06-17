"""PromptBuilder: glossary parsing/merging + term_map formatting + system prompt.

Agent-internal. Ported from the prompt-injection logic in
``serve/rasst_sglang_server.py`` so terminology behavior is identical. The
builder turns retrieved/imported terms into a ``term_map`` and assembles the
chat messages an omni model expects. Retrieval is optional: with no terms the
``term_map`` is omitted (or set to ``NONE`` under ``none_block`` policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

TermRef = Dict[str, Any]


def normalize_reference(term: str, translation: str, source: str) -> Optional[TermRef]:
    clean_term = (term or "").replace("\n", " ").strip()
    clean_translation = (translation or "").replace("\n", " ").strip()
    if not clean_term or not clean_translation:
        return None
    return {"term": clean_term, "translation": clean_translation, "source": source}


def parse_glossary_text(text: str) -> List[TermRef]:
    """Parse ``term=translation`` / ``term => translation`` / ``term<TAB>tr`` lines."""

    references: List[TermRef] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            term, translation = line.split("=>", 1)
        elif "\t" in line:
            term, translation = line.split("\t", 1)
        elif "=" in line:
            term, translation = line.split("=", 1)
        else:
            continue
        ref = normalize_reference(term, translation, "manual")
        if ref:
            references.append(ref)
    return references


def merge_references(*groups: Sequence[TermRef]) -> List[TermRef]:
    """Union of term groups, de-duplicated case-insensitively (first wins)."""

    merged: List[TermRef] = []
    seen: Set[str] = set()
    for group in groups:
        for ref in group or []:
            term = str(ref.get("term") or "").strip().lower()
            if not term or term in seen:
                continue
            translation = str(ref.get("translation") or "").strip()
            if not translation:
                continue
            seen.add(term)
            merged.append(ref)
    return merged


def format_term_map(references: Sequence[TermRef], mode: str) -> str:
    lines: List[str] = []
    for ref in references:
        term = str(ref.get("term") or "").replace("\n", " ").strip()
        translation = str(ref.get("translation") or "").replace("\n", " ").strip()
        if not term or not translation:
            continue
        if mode == "xml_tagged":
            lines.append(f"<term>{term} => {translation}</term>")
        elif mode == "tagged":
            lines.append(f"[TERM] {term} => {translation} [/TERM]")
        else:
            lines.append(f"{term}={translation}")
    return "\n".join(lines)


def use_chinese_training_prompt(source_lang: str, target_lang: str) -> bool:
    return source_lang.strip().lower() in {"english", "en"} and target_lang.strip().lower() in {
        "chinese",
        "zh",
        "zh-cn",
        "中文",
    }


def build_system_prompt(
    source_lang: str,
    target_lang: str,
    system_prompt_style: str,
    rag_enabled: bool,
) -> str:
    if use_chinese_training_prompt(source_lang, target_lang):
        return (
            "You are a professional simultaneous interpreter. "
            "Your task is to translate English audio chunks into accurate and fluent "
            "Chinese. Use the 'term_map' as a reference for terminology if provided."
        )
    if system_prompt_style == "given_chunks":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"You will be given chunks of {source_lang} audio and you need to "
            f"translate the audio into {target_lang} text."
        )
    elif system_prompt_style == "translate_task":
        system_text = (
            f"You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}."
        )
    else:
        raise ValueError(f"Unsupported system_prompt_style={system_prompt_style!r}")
    if rag_enabled:
        system_text += " Use the 'term_map' as a reference for terminology if provided."
    return system_text


@dataclass
class PromptBuilder:
    """Stateless helper that builds the chat messages for one audio chunk.

    ``audio_schema`` mirrors the SGLang-Omni server:
    - ``"inline"``     -> audio reference lives in the user message ``content``
    - ``"top_level"``  -> audio path returned separately as ``audios`` payload
    """

    system_prompt_style: str = "given_chunks"
    term_map_format: str = "plain"
    empty_term_map_policy: str = "none_block"
    audio_schema: str = "top_level"

    def system_message(self, source_lang: str, target_lang: str, rag_enabled: bool) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": build_system_prompt(
                        source_lang, target_lang, self.system_prompt_style, rag_enabled
                    ),
                }
            ],
        }

    def term_map(self, imported: Sequence[TermRef], retrieved: Sequence[TermRef]) -> str:
        return format_term_map(merge_references(imported, retrieved), self.term_map_format)

    def user_message(self, wav_path: str, term_map_text: str, rag_enabled: bool):
        """Return ``(user_message, audios_payload)`` for the chosen audio schema."""

        user_text = ""
        if term_map_text:
            user_text = f"\n\nterm_map:\n{term_map_text}"
        elif rag_enabled and self.empty_term_map_policy == "none_block":
            user_text = "\n\nterm_map:\nNONE"

        if self.audio_schema == "inline":
            content: List[Dict[str, Any]] = [{"type": "audio", "audio": str(wav_path)}]
            if user_text:
                content.append({"type": "text", "text": user_text})
            return {"role": "user", "content": content}, []
        return {"role": "user", "content": user_text}, [str(wav_path)]
