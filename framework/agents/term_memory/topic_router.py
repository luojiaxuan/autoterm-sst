"""Active-glossary routing for automatic terminology slices.

The default router is window-topic-first, but the deployable E2E path does not
depend on ASR text. It combines generated target-translation topic text when
available, routing-only speech-window domain probes, weak speech-centroid
evidence, and a small retrieved-reference metadata prior. Source transcript
windows are supported only for controlled diagnostics. The older audio-native
router remains available for explicit compatibility with the previous
embedding/ref-metadata policy.

The old keyword router remains available as ``LegacyKeywordTopicRouter`` for
explicit debug fallback only.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from framework.agents.term_memory.domain_taxonomy import (
    DOMAIN_KEYWORDS,
    GENERAL_DOMAIN,
    WORKING_DOMAINS,
    topic_keyword_scores,
)


VectorLike = Any


@dataclass
class DomainSlice:
    preset_id: str
    domain_id: str
    parent_domain_id: Optional[str] = None
    fallback_preset_id: Optional[str] = None
    centroid: Optional[VectorLike] = None
    enabled: bool = True
    priority: int = 0
    term_count: int = 0
    index_path: str = ""
    description: str = ""


@dataclass
class DomainScore:
    preset_id: str
    domain_id: str
    embedding_score: float
    reference_score: float
    confidence: float
    margin: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "domain_id": self.domain_id,
            "embedding_score": round(float(self.embedding_score), 4),
            "reference_score": round(float(self.reference_score), 4),
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DomainProbeScore:
    domain: str
    preset_id: str
    top_score: float = 0.0
    mean_topk_score: float = 0.0
    top_terms: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class RouterObservation:
    query_embedding: Optional[VectorLike] = None
    references: Sequence[Dict[str, Any]] = field(default_factory=list)
    router_text: str = ""
    router_text_source: Literal["manifest_source", "streaming_asr", "generated_target", "none"] = "none"
    domain_probe_scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    warmup_sec: float = 30.0
    update_interval_sec: float = 45.0
    min_confidence: float = 0.60
    min_margin: float = 0.15
    min_current_margin: float = 0.10
    min_consistent_windows: int = 2
    switch_cooldown_sec: float = 90.0
    candidate_stale_sec: float = 120.0
    embedding_weight: float = 0.65
    reference_weight: float = 0.35
    ema_alpha: float = 0.80
    fallback_preset_id: str = "none"
    ref_score_floor: float = 0.0
    max_recent_refs: int = 80
    consistency_bonus: float = 0.05
    ambiguity_penalty: float = 0.10
    text_topic_weight: float = 0.60
    context_similarity_weight: float = 0.60
    domain_probe_weight: float = 0.25
    speech_centroid_weight: float = 0.10
    metadata_prior_weight: float = 0.05
    min_consistent_windows_with_text: int = 2
    min_consistent_windows_generated_target: int = 3
    min_consistent_windows_audio_only: int = 3
    text_ema_alpha: float = 0.60
    audio_ema_alpha: float = 0.80
    audio_probe_min_top_score: float = 0.50
    audio_probe_min_raw_margin: float = 0.08
    audio_probe_min_positive_domains: int = 2
    generated_target_probe_min_top_score: float = 0.25
    generated_target_probe_min_raw_margin: float = 0.01
    generated_target_probe_min_positive_domains: int = 1
    slice_selection_mode: str = "hard_top1"
    term_budget: int = 100_000
    max_active_slices: int = 0
    unknown_slice_term_count: int = 10_000
    pin_active_slice: bool = True


@dataclass
class RouterSessionState:
    active_preset_id: str
    active_domain_id: str
    created_s: float = 0.0
    ema_query_embedding: Optional[List[float]] = None
    recent_references: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=80))
    last_switch_s: float = 0.0
    last_decision_s: float = 0.0
    pending_preset_id: Optional[str] = None
    candidate_preset_id: Optional[str] = None
    candidate_streak_count: int = 0
    last_candidate_s: float = 0.0
    ema_domain_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RouterDecision:
    action: Literal["stay", "switch", "fallback"]
    target_preset_id: str
    target_domain_id: str
    confidence: float
    margin: float
    reason: str
    top_scores: List[DomainScore]
    from_preset_id: str = ""
    from_domain_id: str = ""
    at_s: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_domain(self) -> str:
        return self.target_domain_id

    @property
    def should_switch(self) -> bool:
        return self.action in {"switch", "fallback"}

    @property
    def scores(self) -> Dict[str, float]:
        return {item.domain_id: round(float(item.confidence), 4) for item in self.top_scores}

    def to_meta(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "from_preset": self.from_preset_id,
            "from_domain": self.from_domain_id,
            "to_preset": self.target_preset_id,
            "to_domain": self.target_domain_id,
            "confidence": round(float(self.confidence), 4),
            "margin": round(float(self.margin), 4),
            "reason": self.reason,
            "top_domains": [
                [score.domain_id, round(float(score.confidence), 4)]
                for score in self.top_scores[:5]
            ],
            "top_scores": [score.to_meta() for score in self.top_scores[:5]],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class BudgetSliceSelection:
    slices: Sequence[DomainSlice] = field(default_factory=tuple)
    scores: Dict[str, float] = field(default_factory=dict)
    term_counts: Dict[str, int] = field(default_factory=dict)
    term_budget: int = 0
    pinned_preset_id: str = ""

    @property
    def preset_ids(self) -> List[str]:
        return [item.preset_id for item in self.slices]

    @property
    def total_terms(self) -> int:
        return sum(int(self.term_counts.get(item.preset_id, 0)) for item in self.slices)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "mode": "budgeted_top_slices",
            "term_budget": int(self.term_budget),
            "selected_term_count": int(self.total_terms),
            "selected_slice_count": len(self.slices),
            "selected_slice_presets": self.preset_ids,
            "pinned_active_preset": self.pinned_preset_id or None,
            "selected_slices": [
                {
                    "preset_id": item.preset_id,
                    "domain_id": item.domain_id,
                    "score": round(float(self.scores.get(item.preset_id, 0.0)), 4),
                    "term_count": int(self.term_counts.get(item.preset_id, 0)),
                }
                for item in self.slices
            ],
        }


# Compatibility shape used by older tests and the legacy keyword router.
@dataclass
class TopicDecision:
    primary_domain: str
    confidence: float
    scores: Dict[str, float]
    should_switch: bool
    reason: str


@dataclass
class TopicContext:
    recent_text: str = ""
    recent_references: Sequence[Dict[str, Any]] = field(default_factory=list)
    manual_glossary_terms: Sequence[Dict[str, Any]] = field(default_factory=list)
    current_domain: str = GENERAL_DOMAIN
    elapsed_s: float = 0.0
    seconds_since_update: float = 0.0


class AudioNativeActiveGlossaryRouter:
    """Select active glossary slices from retrieval-time audio evidence."""

    def __init__(
        self,
        domain_slices: Sequence[DomainSlice],
        config: Optional[RouterConfig] = None,
    ) -> None:
        self.config = config or RouterConfig()
        self.domain_slices: List[DomainSlice] = [item for item in domain_slices if item.enabled]
        self.by_preset = {item.preset_id: item for item in self.domain_slices}
        self.by_domain = {item.domain_id: item for item in self.domain_slices}

    def budgeted_slice_selection_enabled(self) -> bool:
        mode = str(self.config.slice_selection_mode or "hard_top1").strip().lower()
        return mode in {"budgeted", "budgeted_top_slices", "top_slices"}

    def select_budgeted_slices(
        self,
        similarity_scores: Mapping[str, float],
        *,
        term_budget: Optional[int] = None,
        active_preset_id: str = "",
    ) -> BudgetSliceSelection:
        budget = max(
            0,
            int(self.config.term_budget if term_budget is None else term_budget),
        )
        if not self.budgeted_slice_selection_enabled() or budget <= 0:
            return BudgetSliceSelection(term_budget=budget)

        scored: List[tuple[DomainSlice, float]] = []
        for item in self.domain_slices:
            raw_score = similarity_scores.get(item.preset_id)
            if raw_score is None:
                raw_score = similarity_scores.get(item.domain_id)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scored.append((item, score))
        scored.sort(
            key=lambda row: (
                -float(row[1]),
                -int(row[0].priority),
                row[0].preset_id,
            )
        )

        unknown_terms = max(1, int(self.config.unknown_slice_term_count))
        max_slices = max(0, int(self.config.max_active_slices))
        selected: List[DomainSlice] = []
        scores: Dict[str, float] = {}
        term_counts: Dict[str, int] = {}
        used_terms = 0
        pinned_preset_id = ""

        requested_active_preset = str(active_preset_id or "").strip()
        pinned = (
            self.by_preset.get(requested_active_preset)
            if self.config.pin_active_slice and requested_active_preset
            else None
        )
        if pinned is not None and (not max_slices or max_slices >= 1):
            pinned_terms = (
                int(pinned.term_count)
                if int(pinned.term_count) > 0
                else unknown_terms
            )
            if pinned_terms <= budget:
                raw_score = similarity_scores.get(pinned.preset_id)
                if raw_score is None:
                    raw_score = similarity_scores.get(pinned.domain_id)
                try:
                    pinned_score = float(raw_score)
                except (TypeError, ValueError):
                    pinned_score = 0.0
                if not math.isfinite(pinned_score):
                    pinned_score = 0.0
                selected.append(pinned)
                scores[pinned.preset_id] = pinned_score
                term_counts[pinned.preset_id] = pinned_terms
                used_terms = pinned_terms
                pinned_preset_id = pinned.preset_id

        for item, score in scored:
            if max_slices and len(selected) >= max_slices:
                break
            if item.preset_id == pinned_preset_id:
                continue
            item_terms = int(item.term_count) if int(item.term_count) > 0 else unknown_terms
            if used_terms + item_terms > budget:
                continue
            selected.append(item)
            scores[item.preset_id] = float(score)
            term_counts[item.preset_id] = item_terms
            used_terms += item_terms

        return BudgetSliceSelection(
            slices=tuple(selected),
            scores=scores,
            term_counts=term_counts,
            term_budget=budget,
            pinned_preset_id=pinned_preset_id,
        )

    def observe(
        self,
        session_state: RouterSessionState,
        query_embedding: Optional[VectorLike],
        references: Sequence[Dict[str, Any]],
        now_s: float,
    ) -> RouterDecision:
        session_state.recent_references = _bounded_deque(
            session_state.recent_references,
            maxlen=max(1, int(self.config.max_recent_refs)),
        )
        for ref in references or []:
            if isinstance(ref, dict):
                session_state.recent_references.append(dict(ref))

        query_vec = _normalize(_as_vector(query_embedding))
        if query_vec:
            if session_state.ema_query_embedding and len(session_state.ema_query_embedding) == len(query_vec):
                alpha = _clamp(float(self.config.ema_alpha), 0.0, 0.999)
                ema = [
                    alpha * old + (1.0 - alpha) * new
                    for old, new in zip(session_state.ema_query_embedding, query_vec)
                ]
                session_state.ema_query_embedding = _normalize(ema)
            else:
                session_state.ema_query_embedding = query_vec

        top_scores = self._score(session_state)
        top = top_scores[0] if top_scores else None
        second = top_scores[1] if len(top_scores) > 1 else None
        if top is not None:
            top.margin = max(0.0, top.confidence - (second.confidence if second else 0.0))
        margin = top.margin if top is not None else 0.0
        confidence = top.confidence if top is not None else 0.0

        elapsed = max(0.0, float(now_s) - float(session_state.created_s or now_s))
        current_preset = session_state.active_preset_id or self.config.fallback_preset_id
        current_domain = session_state.active_domain_id or GENERAL_DOMAIN
        target_preset = top.preset_id if top is not None else current_preset
        target_domain = top.domain_id if top is not None else current_domain
        current_score = self._score_for_active(top_scores, current_preset, current_domain)
        switch_delta = confidence - current_score if top is not None else 0.0

        reason = "speech_embedding+retrieved_refs"
        action: Literal["stay", "switch", "fallback"] = "stay"
        guard_reason = ""
        evidence = {
            "has_query_embedding": bool(query_vec or session_state.ema_query_embedding),
            "recent_reference_count": len(session_state.recent_references),
            "current_score": round(float(current_score), 4),
            "target_score_delta": round(float(switch_delta), 4),
            "candidate_preset": session_state.candidate_preset_id,
            "candidate_streak": int(session_state.candidate_streak_count),
        }

        unsupported_current = bool(
            current_preset
            and current_preset not in self.by_preset
            and current_preset != self.config.fallback_preset_id
        )
        if unsupported_current:
            action = "fallback"
            target_preset = self.config.fallback_preset_id
            target_domain = self._domain_for_preset(target_preset)
            guard_reason = "unsupported_current_preset"
        elif top is None:
            guard_reason = "no_domain_slices"
        elif not evidence["has_query_embedding"] and not evidence["recent_reference_count"]:
            guard_reason = "no_audio_or_reference_evidence"
        elif elapsed < float(self.config.warmup_sec):
            guard_reason = f"warmup<{float(self.config.warmup_sec):g}s"
        elif (
            session_state.last_decision_s > 0.0
            and float(now_s) - float(session_state.last_decision_s) < float(self.config.update_interval_sec)
        ):
            guard_reason = f"interval<{float(self.config.update_interval_sec):g}s"
        elif target_preset == current_preset or target_domain == current_domain:
            guard_reason = "same_domain"
        elif target_domain in {GENERAL_DOMAIN, "common", "general"}:
            guard_reason = "general_or_common"
        elif not self._slice_available(target_preset):
            guard_reason = "target_unavailable"
        elif confidence < float(self.config.min_confidence):
            guard_reason = f"confidence<{float(self.config.min_confidence):.2f}"
        elif margin < float(self.config.min_margin):
            guard_reason = f"margin<{float(self.config.min_margin):.2f}"
        elif (
            current_preset in self.by_preset
            and switch_delta < float(self.config.min_current_margin)
        ):
            guard_reason = f"current_margin<{float(self.config.min_current_margin):.2f}"
        elif (
            session_state.last_switch_s > 0.0
            and float(now_s) - float(session_state.last_switch_s) < float(self.config.switch_cooldown_sec)
        ):
            guard_reason = f"cooldown<{float(self.config.switch_cooldown_sec):g}s"
        else:
            self._note_candidate(session_state, target_preset, float(now_s))
            required = max(1, int(self.config.min_consistent_windows))
            evidence["candidate_preset"] = session_state.candidate_preset_id
            evidence["candidate_streak"] = int(session_state.candidate_streak_count)
            if session_state.candidate_streak_count < required:
                guard_reason = f"consistent_windows<{required}"
            else:
                action = "switch"

        if guard_reason and action == "stay" and top is not None:
            reason = f"{reason}; {guard_reason}"
        elif guard_reason:
            reason = guard_reason

        if not (
            guard_reason.startswith("warmup<")
            or guard_reason.startswith("interval<")
            or guard_reason == "no_audio_or_reference_evidence"
            or guard_reason.startswith("consistent_windows<")
        ):
            session_state.last_decision_s = float(now_s)
        decision = RouterDecision(
            action=action,
            target_preset_id=target_preset,
            target_domain_id=target_domain,
            confidence=round(float(confidence), 4),
            margin=round(float(margin), 4),
            reason=reason,
            top_scores=top_scores,
            from_preset_id=current_preset,
            from_domain_id=current_domain,
            at_s=round(elapsed, 3),
            evidence=evidence,
        )
        if action == "switch":
            session_state.pending_preset_id = target_preset
        return decision

    def _note_candidate(
        self,
        session_state: RouterSessionState,
        target_preset: str,
        now_s: float,
    ) -> None:
        stale_sec = max(0.0, float(self.config.candidate_stale_sec))
        is_stale = bool(
            session_state.last_candidate_s > 0.0
            and stale_sec > 0.0
            and now_s - float(session_state.last_candidate_s) > stale_sec
        )
        if session_state.candidate_preset_id == target_preset and not is_stale:
            session_state.candidate_streak_count += 1
        else:
            session_state.candidate_preset_id = target_preset
            session_state.candidate_streak_count = 1
        session_state.last_candidate_s = float(now_s)

    def _score_for_active(
        self,
        top_scores: Sequence[DomainScore],
        current_preset: str,
        current_domain: str,
    ) -> float:
        for item in top_scores:
            if item.preset_id == current_preset:
                return float(item.confidence)
        for item in top_scores:
            if item.domain_id == current_domain:
                return float(item.confidence)
        return 0.0

    def _score(self, session_state: RouterSessionState) -> List[DomainScore]:
        ema = session_state.ema_query_embedding
        embedding_raw: Dict[str, Optional[float]] = {}
        for item in self.domain_slices:
            centroid = _normalize(_as_vector(item.centroid))
            if ema and centroid and len(ema) == len(centroid):
                embedding_raw[item.preset_id] = _dot(ema, centroid)
            else:
                embedding_raw[item.preset_id] = None

        embedding_scores = _normalize_score_map(embedding_raw)
        reference_scores = self._reference_scores(session_state.recent_references)
        ref_top = max(reference_scores.values(), default=0.0)

        out: List[DomainScore] = []
        for item in self.domain_slices:
            emb = float(embedding_scores.get(item.preset_id, 0.0))
            ref = float(reference_scores.get(item.preset_id, 0.0))
            conf = (
                float(self.config.embedding_weight) * emb
                + float(self.config.reference_weight) * ref
            )
            if ref > 0.0 and ref >= ref_top and emb > 0.0:
                conf += min(0.05, float(self.config.consistency_bonus))
            out.append(
                DomainScore(
                    preset_id=item.preset_id,
                    domain_id=item.domain_id,
                    embedding_score=round(emb, 4),
                    reference_score=round(ref, 4),
                    confidence=round(_clamp(conf, 0.0, 1.0), 4),
                    evidence={
                        "term_count": item.term_count,
                        "has_centroid": embedding_raw.get(item.preset_id) is not None,
                        "raw_embedding_cos": (
                            round(float(embedding_raw[item.preset_id]), 4)
                            if embedding_raw.get(item.preset_id) is not None
                            else None
                        ),
                    },
                )
            )
        out.sort(key=lambda score: (score.confidence, score.embedding_score, score.reference_score), reverse=True)
        if len(out) >= 2:
            gap = out[0].confidence - out[1].confidence
            if gap < float(self.config.min_margin):
                out[0].confidence = round(
                    max(0.0, out[0].confidence - min(0.10, float(self.config.ambiguity_penalty))),
                    4,
                )
        return out

    def _reference_scores(self, references: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        raw = {item.preset_id: 0.0 for item in self.domain_slices}
        for ref in references or []:
            preset = self._preset_from_reference_metadata(ref)
            if not preset or preset not in raw:
                continue
            try:
                score = float(ref.get("score", 1.0))
            except (TypeError, ValueError):
                score = 1.0
            raw[preset] += max(score - float(self.config.ref_score_floor), 0.0)
        total = sum(max(value, 0.0) for value in raw.values())
        if total <= 0.0:
            return raw
        return {preset: max(value, 0.0) / total for preset, value in raw.items()}

    def _preset_from_reference_metadata(self, ref: Dict[str, Any]) -> str:
        for key in ("active_glossary_preset", "glossary_preset", "preset_id", "source_preset"):
            value = str(ref.get(key) or "").strip()
            if value in self.by_preset:
                return value
        for key in ("domain", "active_domain", "domain_id"):
            value = str(ref.get(key) or "").strip().lower()
            if value in self.by_domain:
                return self.by_domain[value].preset_id
        return ""

    def _domain_for_preset(self, preset_id: str) -> str:
        item = self.by_preset.get(preset_id)
        return item.domain_id if item is not None else GENERAL_DOMAIN

    def _slice_available(self, preset_id: str) -> bool:
        item = self.by_preset.get(preset_id)
        if item is None or not item.enabled:
            return False
        # The router only validates manifest support. Actual cold-load success
        # is checked by the retrieval plugin before the session is switched.
        return bool(item.index_path) or item.preset_id == self.config.fallback_preset_id


class HybridWindowTopicRouter(AudioNativeActiveGlossaryRouter):
    """Route glossary overlays from E2E window topic evidence."""

    def observe(
        self,
        session_state: RouterSessionState,
        query_embedding: Optional[VectorLike],
        references: Sequence[Dict[str, Any]],
        now_s: float,
        *,
        router_text: str = "",
        router_text_source: Literal["manifest_source", "streaming_asr", "generated_target", "none"] = "none",
        domain_probe_scores: Optional[Dict[str, Any]] = None,
        context_similarity_scores: Optional[Dict[str, float]] = None,
    ) -> RouterDecision:
        session_state.recent_references = _bounded_deque(
            session_state.recent_references,
            maxlen=max(1, int(self.config.max_recent_refs)),
        )
        for ref in references or []:
            if isinstance(ref, dict):
                session_state.recent_references.append(dict(ref))

        query_vec = _normalize(_as_vector(query_embedding))
        if query_vec:
            if session_state.ema_query_embedding and len(session_state.ema_query_embedding) == len(query_vec):
                alpha = _clamp(float(self.config.ema_alpha), 0.0, 0.999)
                session_state.ema_query_embedding = _normalize(
                    [
                        alpha * old + (1.0 - alpha) * new
                        for old, new in zip(session_state.ema_query_embedding, query_vec)
                    ]
                )
            else:
                session_state.ema_query_embedding = query_vec

        top_scores = self._score_hybrid(
            session_state,
            router_text=router_text,
            router_text_source=router_text_source,
            domain_probe_scores=domain_probe_scores or {},
            context_similarity_scores=context_similarity_scores or {},
        )
        top = top_scores[0] if top_scores else None
        second = top_scores[1] if len(top_scores) > 1 else None
        if top is not None:
            top.margin = max(0.0, top.confidence - (second.confidence if second else 0.0))
        margin = top.margin if top is not None else 0.0
        confidence = top.confidence if top is not None else 0.0

        elapsed = max(0.0, float(now_s) - float(session_state.created_s or now_s))
        current_preset = session_state.active_preset_id or self.config.fallback_preset_id
        current_domain = session_state.active_domain_id or GENERAL_DOMAIN
        target_preset = top.preset_id if top is not None else current_preset
        target_domain = top.domain_id if top is not None else current_domain
        current_score = self._score_for_active(top_scores, current_preset, current_domain)
        switch_delta = confidence - current_score if top is not None else 0.0

        text_source = str(router_text_source or "none")
        is_generated_target = text_source == "generated_target"
        has_text = bool((router_text or "").strip() and text_source != "none")
        text_raw_for_guard, _ = topic_keyword_scores(router_text)
        has_text_topic_signal = bool(has_text and any(float(value) > 0.0 for value in text_raw_for_guard.values()))
        has_context_similarity = any(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (context_similarity_scores or {}).values()
        )
        has_probe = any(_probe_value(value) > 0.0 for value in (domain_probe_scores or {}).values())
        has_signal = bool(
            has_text
            or has_context_similarity
            or has_probe
            or query_vec
            or session_state.ema_query_embedding
            or session_state.recent_references
        )
        audio_probe_guard = self._audio_probe_guard(
            domain_probe_scores or {},
            target_domain=target_domain,
            target_preset=target_preset,
        )
        generated_target_probe_guard = self._generated_target_probe_guard(
            domain_probe_scores or {},
            target_domain=target_domain,
            target_preset=target_preset,
        )
        reason = "hybrid_window_topic"
        action: Literal["stay", "switch", "fallback"] = "stay"
        guard_reason = ""
        evidence = {
            "router_text_source": text_source,
            "has_router_text": has_text,
            "has_text_topic_signal": has_text_topic_signal,
            "has_context_similarity": has_context_similarity,
            "has_domain_probe": has_probe,
            "has_query_embedding": bool(query_vec or session_state.ema_query_embedding),
            "recent_reference_count": len(session_state.recent_references),
            "current_score": round(float(current_score), 4),
            "target_score_delta": round(float(switch_delta), 4),
            "candidate_preset": session_state.candidate_preset_id,
            "candidate_streak": int(session_state.candidate_streak_count),
            "ema_domain_scores": {
                key: round(float(value), 4) for key, value in session_state.ema_domain_scores.items()
            },
            "audio_probe_guard": audio_probe_guard,
            "generated_target_probe_guard": generated_target_probe_guard,
        }
        unsupported_current = bool(
            current_preset
            and current_preset not in self.by_preset
            and current_preset != self.config.fallback_preset_id
        )
        if unsupported_current:
            action = "fallback"
            target_preset = self.config.fallback_preset_id
            target_domain = self._domain_for_preset(target_preset)
            guard_reason = "unsupported_current_preset"
        elif top is None:
            guard_reason = "no_domain_slices"
        elif not has_signal:
            guard_reason = "no_topic_audio_or_reference_evidence"
        elif elapsed < float(self.config.warmup_sec):
            guard_reason = f"warmup<{float(self.config.warmup_sec):g}s"
        elif (
            session_state.last_decision_s > 0.0
            and float(now_s) - float(session_state.last_decision_s) < float(self.config.update_interval_sec)
        ):
            guard_reason = f"interval<{float(self.config.update_interval_sec):g}s"
        elif target_preset == current_preset or target_domain == current_domain:
            guard_reason = "same_domain"
        elif target_domain in {GENERAL_DOMAIN, "common", "general"}:
            guard_reason = "general_or_common"
        elif not self._slice_available(target_preset):
            guard_reason = "target_unavailable"
        elif (
            is_generated_target
            and not has_text_topic_signal
            and not has_context_similarity
            and not generated_target_probe_guard.get("ok", False)
        ):
            guard_reason = "generated_target_probe_evidence_insufficient"
        elif has_text and not has_text_topic_signal and not has_context_similarity and not has_probe:
            guard_reason = "topic_text_or_probe_required"
        elif (
            has_probe
            and has_text
            and not has_text_topic_signal
            and not has_context_similarity
            and not audio_probe_guard.get("ok", False)
        ):
            guard_reason = "probe_only_evidence_insufficient"
        elif not has_text and not has_probe:
            guard_reason = "audio_probe_required"
        elif not has_text and has_probe and not audio_probe_guard.get("ok", False):
            guard_reason = "audio_probe_evidence_insufficient"
        elif confidence < float(self.config.min_confidence):
            guard_reason = f"confidence<{float(self.config.min_confidence):.2f}"
        elif margin < float(self.config.min_margin):
            guard_reason = f"margin<{float(self.config.min_margin):.2f}"
        elif current_preset in self.by_preset and switch_delta < float(self.config.min_current_margin):
            guard_reason = f"current_margin<{float(self.config.min_current_margin):.2f}"
        elif (
            session_state.last_switch_s > 0.0
            and float(now_s) - float(session_state.last_switch_s) < float(self.config.switch_cooldown_sec)
        ):
            guard_reason = f"cooldown<{float(self.config.switch_cooldown_sec):g}s"
        else:
            self._note_candidate(session_state, target_preset, float(now_s))
            required = max(
                1,
                int(
                    self.config.min_consistent_windows_generated_target
                    if is_generated_target
                    else (
                        self.config.min_consistent_windows_with_text
                        if has_text
                        else self.config.min_consistent_windows_audio_only
                    )
                ),
            )
            evidence["candidate_preset"] = session_state.candidate_preset_id
            evidence["candidate_streak"] = int(session_state.candidate_streak_count)
            evidence["required_consistent_windows"] = required
            if session_state.candidate_streak_count < required:
                guard_reason = f"consistent_windows<{required}"
            else:
                action = "switch"

        if guard_reason and action == "stay" and top is not None:
            reason = f"{reason}; {guard_reason}"
        elif guard_reason:
            reason = guard_reason

        if not (
            guard_reason.startswith("warmup<")
            or guard_reason.startswith("interval<")
            or guard_reason == "no_topic_audio_or_reference_evidence"
            or guard_reason.startswith("consistent_windows<")
        ):
            session_state.last_decision_s = float(now_s)
        if self.budgeted_slice_selection_enabled():
            confirmed_preset = (
                target_preset if action in {"switch", "fallback"} else current_preset
            )
            evidence["slice_selection"] = self.select_budgeted_slices(
                context_similarity_scores or {},
                active_preset_id=confirmed_preset,
            ).to_meta()
        decision = RouterDecision(
            action=action,
            target_preset_id=target_preset,
            target_domain_id=target_domain,
            confidence=round(float(confidence), 4),
            margin=round(float(margin), 4),
            reason=reason,
            top_scores=top_scores,
            from_preset_id=current_preset,
            from_domain_id=current_domain,
            at_s=round(elapsed, 3),
            evidence=evidence,
        )
        if action == "switch":
            session_state.pending_preset_id = target_preset
        return decision

    def _generated_target_probe_guard(
        self,
        domain_probe_scores: Dict[str, Any],
        *,
        target_domain: str,
        target_preset: str,
    ) -> Dict[str, Any]:
        return self._probe_guard(
            domain_probe_scores,
            target_domain=target_domain,
            target_preset=target_preset,
            min_top_score=float(self.config.generated_target_probe_min_top_score),
            min_raw_margin=float(self.config.generated_target_probe_min_raw_margin),
            min_positive_domains=int(self.config.generated_target_probe_min_positive_domains),
        )

    def _audio_probe_guard(
        self,
        domain_probe_scores: Dict[str, Any],
        *,
        target_domain: str,
        target_preset: str,
    ) -> Dict[str, Any]:
        return self._probe_guard(
            domain_probe_scores,
            target_domain=target_domain,
            target_preset=target_preset,
            min_top_score=float(self.config.audio_probe_min_top_score),
            min_raw_margin=float(self.config.audio_probe_min_raw_margin),
            min_positive_domains=int(self.config.audio_probe_min_positive_domains),
        )

    def _probe_guard(
        self,
        domain_probe_scores: Dict[str, Any],
        *,
        target_domain: str,
        target_preset: str,
        min_top_score: float,
        min_raw_margin: float,
        min_positive_domains: int,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for key, value in (domain_probe_scores or {}).items():
            score = _probe_value(value)
            domain = str(getattr(value, "domain", "") or "").strip()
            preset = str(getattr(value, "preset_id", "") or "").strip()
            if isinstance(value, dict):
                domain = str(value.get("domain") or domain).strip()
                preset = str(value.get("preset_id") or preset).strip()
            domain = domain or str(key)
            preset = preset or str(key)
            if not math.isfinite(float(score)):
                score = 0.0
            rows.append({"domain": domain, "preset": preset, "score": max(0.0, float(score))})
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        positive = [item for item in rows if float(item["score"]) > 0.0]
        top = rows[0] if rows else {"domain": "", "preset": "", "score": 0.0}
        second_score = float(rows[1]["score"]) if len(rows) > 1 else 0.0
        raw_margin = max(0.0, float(top["score"]) - second_score)
        target_row = next(
            (
                item for item in rows
                if str(item["domain"]) == str(target_domain) or str(item["preset"]) == str(target_preset)
            ),
            {"domain": target_domain, "preset": target_preset, "score": 0.0},
        )
        top_matches_target = (
            str(top["domain"]) == str(target_domain)
            or str(top["preset"]) == str(target_preset)
        )
        ok = bool(
            len(positive) >= max(1, int(min_positive_domains))
            and top_matches_target
            and float(top["score"]) >= float(min_top_score)
            and raw_margin >= float(min_raw_margin)
        )
        return {
            "ok": ok,
            "positive_domains": len(positive),
            "top_domain": str(top["domain"]),
            "top_preset": str(top["preset"]),
            "top_score": round(float(top["score"]), 4),
            "second_score": round(second_score, 4),
            "raw_margin": round(raw_margin, 4),
            "target_probe_score": round(float(target_row["score"]), 4),
            "min_top_score": round(float(min_top_score), 4),
            "min_raw_margin": round(float(min_raw_margin), 4),
            "min_positive_domains": int(min_positive_domains),
        }

    def _score_hybrid(
        self,
        session_state: RouterSessionState,
        *,
        router_text: str,
        router_text_source: str,
        domain_probe_scores: Dict[str, Any],
        context_similarity_scores: Dict[str, float],
    ) -> List[DomainScore]:
        text_raw, text_hits = topic_keyword_scores(router_text)
        text_by_domain = _normalize_nonnegative_score_map({key: value for key, value in text_raw.items()})
        probe_by_domain = _normalize_nonnegative_score_map(
            {key: _probe_value(value) for key, value in domain_probe_scores.items()}
        )
        context_by_domain = _normalize_nonnegative_score_map(
            {key: value for key, value in context_similarity_scores.items()}
        )
        has_text = bool((router_text or "").strip() and str(router_text_source or "none") != "none")
        has_text_topic_signal = any(float(value) > 0.0 for value in text_by_domain.values())
        has_probe = any(float(value) > 0.0 for value in probe_by_domain.values())
        has_context_similarity = any(float(value) > 0.0 for value in context_by_domain.values())

        ema = session_state.ema_query_embedding
        embedding_raw: Dict[str, Optional[float]] = {}
        for item in self.domain_slices:
            centroid = _normalize(_as_vector(item.centroid))
            if ema and centroid and len(ema) == len(centroid):
                embedding_raw[item.preset_id] = _dot(ema, centroid)
            else:
                embedding_raw[item.preset_id] = None
        embedding_by_preset = _normalize_score_map(embedding_raw)
        reference_by_preset = self._reference_scores(session_state.recent_references)
        has_embedding = any(value is not None for value in embedding_raw.values())
        has_reference = any(float(value) > 0.0 for value in reference_by_preset.values())
        active_weight_sum = (
            (float(self.config.text_topic_weight) if has_text_topic_signal else 0.0)
            + (float(self.config.context_similarity_weight) if has_context_similarity else 0.0)
            + (float(self.config.domain_probe_weight) if has_probe else 0.0)
            + (float(self.config.speech_centroid_weight) if has_embedding else 0.0)
            + (float(self.config.metadata_prior_weight) if has_reference else 0.0)
        )

        out: List[DomainScore] = []
        raw_final_by_preset: Dict[str, float] = {}
        for item in self.domain_slices:
            text_score = float(text_by_domain.get(item.domain_id, 0.0))
            probe_score = float(
                probe_by_domain.get(item.domain_id, probe_by_domain.get(item.preset_id, 0.0))
            )
            context_score = float(context_by_domain.get(item.domain_id, 0.0))
            speech_score = float(embedding_by_preset.get(item.preset_id, 0.0))
            metadata_prior = float(reference_by_preset.get(item.preset_id, 0.0))
            weighted = 0.0
            if has_text_topic_signal:
                weighted += float(self.config.text_topic_weight) * text_score
            if has_context_similarity:
                weighted += float(self.config.context_similarity_weight) * context_score
            if has_probe:
                weighted += float(self.config.domain_probe_weight) * probe_score
            if has_embedding:
                weighted += float(self.config.speech_centroid_weight) * speech_score
            if has_reference:
                weighted += float(self.config.metadata_prior_weight) * metadata_prior
            conf = weighted / active_weight_sum if active_weight_sum > 1e-9 else 0.0
            raw_final_by_preset[item.preset_id] = _clamp(conf, 0.0, 1.0)

        alpha = float(self.config.text_ema_alpha if has_text_topic_signal else self.config.audio_ema_alpha)
        alpha = _clamp(alpha, 0.0, 0.999)
        if raw_final_by_preset:
            if not session_state.ema_domain_scores:
                session_state.ema_domain_scores = dict(raw_final_by_preset)
            else:
                session_state.ema_domain_scores = {
                    preset: alpha * float(session_state.ema_domain_scores.get(preset, value))
                    + (1.0 - alpha) * float(value)
                    for preset, value in raw_final_by_preset.items()
                }

        for item in self.domain_slices:
            text_score = float(text_by_domain.get(item.domain_id, 0.0))
            probe_score = float(
                probe_by_domain.get(item.domain_id, probe_by_domain.get(item.preset_id, 0.0))
            )
            context_score = float(context_by_domain.get(item.domain_id, 0.0))
            speech_score = float(embedding_by_preset.get(item.preset_id, 0.0))
            metadata_prior = float(reference_by_preset.get(item.preset_id, 0.0))
            raw_final = float(raw_final_by_preset.get(item.preset_id, 0.0))
            ema_final = float(session_state.ema_domain_scores.get(item.preset_id, raw_final))
            out.append(
                DomainScore(
                    preset_id=item.preset_id,
                    domain_id=item.domain_id,
                    embedding_score=round(speech_score, 4),
                    reference_score=round(metadata_prior, 4),
                    confidence=round(raw_final, 4),
                    evidence={
                        "term_count": item.term_count,
                        "router_text_source": router_text_source or "none",
                        "has_text_topic_signal": has_text_topic_signal,
                        "text_topic_score": round(text_score, 4),
                        "context_similarity_score": round(context_score, 4),
                        "domain_probe_score": round(probe_score, 4),
                        "speech_centroid_score": round(speech_score, 4),
                        "metadata_prior": round(metadata_prior, 4),
                        "active_weight_sum": round(active_weight_sum, 4),
                        "raw_final_score": round(raw_final, 4),
                        "ema_final_score": round(ema_final, 4),
                        "topic_keyword_hits": text_hits.get(item.domain_id, [])[:8],
                        "has_centroid": embedding_raw.get(item.preset_id) is not None,
                        "raw_embedding_cos": (
                            round(float(embedding_raw[item.preset_id]), 4)
                            if embedding_raw.get(item.preset_id) is not None
                            else None
                        ),
                    },
                )
            )
        out.sort(key=lambda score: (score.confidence, score.embedding_score, score.reference_score), reverse=True)
        if len(out) >= 2:
            gap = out[0].confidence - out[1].confidence
            if gap < float(self.config.min_margin):
                out[0].confidence = round(
                    max(0.0, out[0].confidence - min(0.10, float(self.config.ambiguity_penalty))),
                    4,
                )
        return out


class LegacyKeywordTopicRouter:
    """Old deterministic keyword router, kept for explicit debug fallback."""

    def __init__(
        self,
        *,
        warmup_sec: float = 30.0,
        update_sec: float = 45.0,
        min_confidence: float = 0.60,
        switch_margin: float = 0.15,
    ) -> None:
        self.warmup_sec = float(warmup_sec)
        self.update_sec = float(update_sec)
        self.min_confidence = float(min_confidence)
        self.switch_margin = float(switch_margin)

    def decide(self, context: TopicContext) -> TopicDecision:
        scores = {domain: 0.0 for domain in WORKING_DOMAINS}
        scores[GENERAL_DOMAIN] = 0.25

        text = (context.recent_text or "").lower()
        reasons: List[str] = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = _phrase_hits(text, keywords)
            if hits:
                scores[domain] += hits * 1.0
                reasons.append(f"{domain}:keywords={hits}")

        ref_scores = self._reference_scores(context.recent_references)
        for domain, value in ref_scores.items():
            scores[domain] = scores.get(domain, 0.0) + value
            if value:
                reasons.append(f"{domain}:refs={value:.1f}")

        # Manual glossary terms are intentionally not used by the new router.
        # This legacy router preserves old behavior only when explicitly enabled.
        manual_scores = self._reference_scores(context.manual_glossary_terms, weight=0.5)
        for domain, value in manual_scores.items():
            scores[domain] = scores.get(domain, 0.0) + value
            if value:
                reasons.append(f"{domain}:manual={value:.1f}")

        primary, top_score = max(scores.items(), key=lambda item: item[1])
        total = sum(max(v, 0.0) for v in scores.values()) or 1.0
        confidence = max(0.0, min(1.0, top_score / total))
        current_domain = context.current_domain or GENERAL_DOMAIN
        current_score = scores.get(current_domain, 0.0)

        should_switch = True
        guard_reason = ""
        if context.elapsed_s < self.warmup_sec:
            should_switch = False
            guard_reason = f"warmup<{self.warmup_sec:g}s"
        elif context.seconds_since_update < self.update_sec:
            should_switch = False
            guard_reason = f"interval<{self.update_sec:g}s"
        elif primary == current_domain:
            should_switch = False
            guard_reason = "same_domain"
        elif primary == GENERAL_DOMAIN:
            should_switch = False
            guard_reason = "general"
        elif confidence < self.min_confidence:
            should_switch = False
            guard_reason = f"confidence<{self.min_confidence:.2f}"
        elif confidence - (current_score / total) < self.switch_margin:
            should_switch = False
            guard_reason = f"margin<{self.switch_margin:.2f}"

        reason = "; ".join(reasons) if reasons else "no strong domain signal"
        if guard_reason:
            reason = f"{reason}; {guard_reason}"
        return TopicDecision(
            primary_domain=primary,
            confidence=round(confidence, 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            should_switch=should_switch,
            reason=reason,
        )

    def _reference_scores(
        self,
        references: Sequence[Dict[str, Any]],
        *,
        weight: float = 1.0,
    ) -> Dict[str, float]:
        out = {domain: 0.0 for domain in WORKING_DOMAINS}
        for ref in references or []:
            domain = _legacy_domain_from_ref(ref)
            if domain in out and domain != GENERAL_DOMAIN:
                out[domain] += 1.5 * weight
            blob = " ".join(
                str(ref.get(key) or "")
                for key in ("term", "translation", "source", "domain", "active_glossary_preset")
            ).lower()
            for candidate, keywords in DOMAIN_KEYWORDS.items():
                hits = _phrase_hits(blob, keywords)
                if hits:
                    out[candidate] += 0.35 * hits * weight
        return out


LegacyTopicRouter = LegacyKeywordTopicRouter

# Backward-compatible import alias only. New runtime code should use
# AudioNativeActiveGlossaryRouter; the keyword router is a debug fallback behind
# RASST_ROUTER_MODE=legacy_keywords.
TopicRouter = LegacyKeywordTopicRouter


def _as_vector(value: Optional[VectorLike]) -> List[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().flatten().tolist()
        elif hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[float] = []
    for item in value:
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _normalize(value: Sequence[float]) -> List[float]:
    if not value:
        return []
    norm = math.sqrt(sum(float(item) * float(item) for item in value))
    if norm <= 1e-12:
        return []
    return [float(item) / norm for item in value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _normalize_score_map(raw: Dict[str, Optional[float]]) -> Dict[str, float]:
    values = [float(v) for v in raw.values() if v is not None and math.isfinite(float(v))]
    if not values:
        return {key: 0.0 for key in raw}
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-6:
        return {key: max(0.0, min(1.0, (float(value or 0.0) + 1.0) / 2.0)) for key, value in raw.items()}
    return {
        key: 0.0 if value is None else _clamp((float(value) - lo) / (hi - lo), 0.0, 1.0)
        for key, value in raw.items()
    }


def _normalize_nonnegative_score_map(raw: Dict[str, Optional[float]]) -> Dict[str, float]:
    values = [max(0.0, float(v)) for v in raw.values() if v is not None and math.isfinite(float(v))]
    if not values:
        return {key: 0.0 for key in raw}
    hi = max(values)
    if hi <= 1e-9:
        return {key: 0.0 for key in raw}
    return {
        key: 0.0 if value is None else _clamp(max(0.0, float(value)) / hi, 0.0, 1.0)
        for key, value in raw.items()
    }


def _probe_value(value: Any) -> float:
    if isinstance(value, DomainProbeScore):
        return max(float(value.top_score or 0.0), float(value.mean_topk_score or 0.0))
    if isinstance(value, dict):
        for key in ("top_score", "mean_topk_score", "score"):
            try:
                return float(value.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
        return 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bounded_deque(items: Deque[Dict[str, Any]], *, maxlen: int) -> Deque[Dict[str, Any]]:
    if items.maxlen == maxlen:
        return items
    return deque(list(items)[-maxlen:], maxlen=maxlen)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _phrase_hits(text: str, keywords: Iterable[str]) -> int:
    if not text:
        return 0
    return sum(1 for keyword in keywords if keyword and keyword.lower() in text)


def _legacy_domain_from_ref(ref: Dict[str, Any]) -> str:
    for key in ("domain", "active_domain"):
        value = str(ref.get(key) or "").strip().lower()
        if value in WORKING_DOMAINS:
            return value
    source = str(ref.get("source") or "").strip().lower()
    for domain in WORKING_DOMAINS:
        if domain != GENERAL_DOMAIN and domain in source:
            return domain
    preset = str(ref.get("active_glossary_preset") or "").strip().lower()
    for domain in WORKING_DOMAINS:
        if domain != GENERAL_DOMAIN and domain in preset:
            return domain
    return GENERAL_DOMAIN
