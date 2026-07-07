"""Session-level active working-glossary selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from framework.agents.glossary import GlossaryCatalog
from framework.agents.term_memory.domain_taxonomy import (
    AUTO_WORKING_PRESET,
    GENERAL_DOMAIN,
    WORKING_GLOSSARY_PRESETS,
    domain_for_preset,
    preset_for_domain,
)

logger = logging.getLogger(__name__)


@dataclass
class ActiveGlossarySelection:
    requested_preset: str
    active_preset: str
    active_domain: str
    glossary_path: str
    index_path: str
    preset_terms: int
    index_ready: bool
    manual_refs: list
    manual_terms: int
    auto_enabled: bool
    reason: str = ""

    def to_session_meta(self) -> Dict[str, Any]:
        return {
            "glossary_preset": self.requested_preset,
            "active_glossary_preset": self.active_preset,
            "active_domain": self.active_domain,
            "glossary_path": self.glossary_path,
            "preset_terms": self.preset_terms,
            "manual_terms": self.manual_terms,
            "glossary_terms": self.manual_terms,
            "index_path": self.index_path,
            "index_ready": self.index_ready,
            "auto_glossary_enabled": self.auto_enabled,
            "auto_glossary_reason": self.reason,
        }


class ActiveGlossaryManager:
    """Map topic decisions to concrete preset/index selections."""

    def __init__(
        self,
        *,
        default_preset: str = "none",
        allowed_presets: Iterable[str] = WORKING_GLOSSARY_PRESETS,
    ) -> None:
        self.default_preset = default_preset or "none"
        self.allowed_presets = {p for p in allowed_presets if p}

    def is_auto_request(self, preset: Optional[str]) -> bool:
        return (preset or "").strip() == AUTO_WORKING_PRESET

    def initial_selection(
        self,
        catalog: GlossaryCatalog,
        requested_preset: Optional[str],
        glossary_text: str,
        *,
        auto_allowed: bool,
        mock: bool = False,
    ) -> ActiveGlossarySelection:
        requested = (requested_preset or "").strip()
        auto = auto_allowed and (not requested or requested == AUTO_WORKING_PRESET)
        target = self.default_preset if auto else requested
        if not target:
            target = "none"
        selection = self._describe(catalog, target, glossary_text, mock=mock)
        if auto and not selection["index_ready"] and selection["index_path"]:
            logger.warning("auto working default %s is unavailable; falling back to none", target)
            selection = self._describe(catalog, "none", glossary_text, mock=mock)
            target = selection["glossary_preset"]
        active_preset = selection["glossary_preset"]
        return ActiveGlossarySelection(
            requested_preset=AUTO_WORKING_PRESET if auto else active_preset,
            active_preset=active_preset,
            active_domain=domain_for_preset(active_preset),
            glossary_path=selection["glossary_path"],
            index_path=selection["index_path"],
            preset_terms=selection["preset_terms"],
            index_ready=selection["index_ready"],
            manual_refs=selection["manual_refs"],
            manual_terms=selection["manual_terms"],
            auto_enabled=auto,
            reason="initial_auto" if auto else "manual",
        )

    def selection_for_decision(
        self,
        catalog: GlossaryCatalog,
        decision: Any,
        *,
        glossary_text: str = "",
        mock: bool = False,
    ) -> Optional[ActiveGlossarySelection]:
        target = str(getattr(decision, "target_preset_id", "") or "").strip()
        if not target:
            target = preset_for_domain(str(getattr(decision, "primary_domain", "")), self.default_preset)
        if target != "none" and target not in self.allowed_presets:
            return None
        selection = self._describe(catalog, target, glossary_text, mock=mock)
        if not selection["index_ready"] and selection["index_path"]:
            return None
        active_preset = selection["glossary_preset"]
        return ActiveGlossarySelection(
            requested_preset=AUTO_WORKING_PRESET,
            active_preset=active_preset,
            active_domain=decision.primary_domain or domain_for_preset(active_preset),
            glossary_path=selection["glossary_path"],
            index_path=selection["index_path"],
            preset_terms=selection["preset_terms"],
            index_ready=selection["index_ready"],
            manual_refs=selection["manual_refs"],
            manual_terms=selection["manual_terms"],
            auto_enabled=True,
            reason=str(getattr(decision, "reason", "")),
        )

    def _describe(
        self,
        catalog: GlossaryCatalog,
        preset: str,
        glossary_text: str,
        *,
        mock: bool,
    ) -> Dict[str, Any]:
        selection = catalog.describe_selection(preset, glossary_text)
        if mock and preset in WORKING_GLOSSARY_PRESETS and not selection["index_path"]:
            # Mock mode should exercise the full adaptive UI/protocol even when
            # the real runtime snapshot has not been built on this machine.
            selection = dict(selection)
            selection["glossary_preset"] = preset
            selection["index_path"] = f"mock://{preset}"
            selection["index_ready"] = True
            if not selection["preset_terms"]:
                selection["preset_terms"] = 10000
        return selection


def glossary_topic_meta(
    *,
    active_domain: str,
    confidence: float,
    active_glossary_preset: str,
    switch_count: int,
    last_reason: str = "",
) -> Dict[str, Any]:
    return {
        "active_domain": active_domain or GENERAL_DOMAIN,
        "confidence": round(float(confidence or 0.0), 4),
        "active_glossary_preset": active_glossary_preset,
        "switch_count": int(switch_count),
        "last_reason": last_reason,
    }
