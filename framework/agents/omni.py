"""OmniAgent: streaming omni-model translation agent (the RASST black box).

This agent owns everything below the thin middle layer: per-session audio
buffering, a coalescing micro-batch scheduler, optional MaxSim retrieval, prompt
construction, and generation through a pluggable :class:`ModelBackend`. The
framework only sees ``open_session`` / ``submit_audio`` / ``on_control`` /
``close_session`` and the ``emit`` callback.

Logic is ported from ``serve/rasst_sglang_server.py`` (``RasstSglangRuntime``)
so terminology/streaming behavior matches the original demo. Differences:

* retrieval, the model backend, and prompt building are now agent-internal
  plugins (composition instead of one monolith);
* output goes through ``emit(TranslationEvent)`` instead of per-session queues;
* ``RASST_DEMO_MOCK=1`` runs the whole path with no GPU / no SGLang / no torch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from fastapi import HTTPException

from framework.agent import (
    EVENT_ERROR,
    EVENT_PARTIAL,
    EVENT_STATUS,
    Agent,
    EmitFn,
    SessionInfo,
    TranslationEvent,
)
from framework.agents.glossary import (
    DEFAULT_GLOSSARY_PRESET,
    LANGUAGE_PAIRS,
    RAG_STARTUP_GLOSSARY_PRESET,
    GlossaryCatalog,
)
from framework.agents.plugins.backends import ModelTemplate, Sampling, build_backend, get_template
from framework.agents.plugins.prompt import PromptBuilder
from framework.agents.plugins.retrieval import (
    MaxSimRetrievalPlugin,
    MockRetrieval,
    NullRetrieval,
    RetrievalPlugin,
    RetrievalResult,
)
from framework.agents.term_memory.active_glossary import (
    ActiveGlossaryManager,
    glossary_topic_meta,
)
from framework.agents.term_memory.domain_taxonomy import (
    AUTO_WORKING_PRESET,
    GENERAL_DOMAIN,
    configured_working_presets,
    domain_for_preset,
)
from framework.agents.term_memory.slice_registry import (
    PROMPT_K,
    RetrievalSlice,
    domain_for_slice_preset,
    force_exactly_k_references,
    rank_references as rank_autoterm_references,
    slice_id_for_preset,
    slice_role_for_preset,
    slice_weight_for_role,
)
from framework.agents.term_memory.topic_router import TopicContext
from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainProbeScore,
    DomainSlice,
    HybridWindowTopicRouter,
    LegacyKeywordTopicRouter,
    RouterConfig,
    RouterDecision,
    RouterSessionState,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPTY_GLOSSARY_PRESETS = {"", "none", "no_glossary"}


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_empty_glossary_preset(preset: Any) -> bool:
    return str(preset or "").strip().casefold() in EMPTY_GLOSSARY_PRESETS


def _meta_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _domain_probe_scores_to_meta(scores: Dict[str, DomainProbeScore]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for domain, item in scores.items():
        out[str(domain)] = {
            "domain": item.domain,
            "preset_id": item.preset_id,
            "top_score": round(float(item.top_score or 0.0), 4),
            "mean_topk_score": round(float(item.mean_topk_score or 0.0), 4),
            "top_terms": [str(term) for term in item.top_terms[:5]],
        }
    return out


def _load_autoterm_slice_config() -> Dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "autoterm_slices.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # noqa: WPS433 - optional config parser

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - config defaults should never block startup
        logger.exception("failed to load AutoTerm slice config: %s", path)
        return {}
    auto = data.get("auto_working") if isinstance(data, dict) else None
    return auto if isinstance(auto, dict) else {}


def _preset_for_slice_id(slice_id: str) -> str:
    value = str(slice_id or "").strip()
    if value in {"common_terms", "common_10k"}:
        return "common_10k"
    return value


def _autoterm_base_preset(config: Dict[str, Any], default: str = "none") -> str:
    if "base_slice" in config:
        base = _preset_for_slice_id(str(config.get("base_slice") or ""))
        return base or default
    return default


def _autoterm_initial_preset(config: Dict[str, Any], default: str = "nlp_core_10k") -> str:
    initial = _preset_for_slice_id(str(config.get("initial_slice") or ""))
    if initial:
        return initial
    return default


def _autoterm_working_presets(config: Dict[str, Any], default: str) -> str:
    slices = config.get("slices")
    if not isinstance(slices, dict):
        return default
    presets: List[str] = []
    for slice_id, meta in slices.items():
        if not isinstance(meta, dict):
            continue
        role = str(meta.get("type") or "").strip().lower()
        if role != "domain":
            continue
        preset = _preset_for_slice_id(str(slice_id))
        if preset and preset not in presets:
            presets.append(preset)
    return ",".join(presets) if presets else default


def _autoterm_rescue_preset(config: Dict[str, Any], default: str = "open_wiki_100k") -> str:
    slices = config.get("slices")
    if isinstance(slices, dict):
        for slice_id, meta in slices.items():
            if isinstance(meta, dict) and str(meta.get("type") or "").strip().lower() == "rescue":
                return str(slice_id)
    return default


@dataclass
class OmniConfig:
    """Self-contained runtime config (defaults match the RASST server)."""

    language_pair: str = "English -> Chinese"
    mock: bool = False

    # Default RASST hosting: in-process vLLM (batched). model_path empty => resolve
    # per-language from the GlossaryCatalog.
    vllm_model_path: str = ""
    vllm_tp_size: int = 1
    gpu_memory_utilization: float = 0.86
    max_num_seqs: int = 32
    max_model_len: int = 16384
    enable_prefix_caching: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_audio: int = 16
    disable_custom_all_reduce: bool = False

    # Optional alternative backend: external OpenAI-compatible SGLang/vLLM server.
    sglang_base_url: str = "http://127.0.0.1:8100"
    sglang_timeout_sec: float = 900.0

    segment_sec: float = 1.92
    # latency-multiplier streaming: chunk = base_segment_sec * lm (lm in 1..4 ->
    # 0.96/1.92/2.88/3.84s). The MaxSim retriever adds rag_timeline_lookback_sec
    # (1.92s) of left context, so its encode window is 2.88/3.84/4.8/5.76s,
    # matching the trained variable-context (varctx) form.
    base_segment_sec: float = 0.96
    default_latency_multiplier: int = 2
    scheduler_batch_size: int = 32
    batch_timeout: float = 0.05
    coalesce_sec: float = 0.06
    max_inflight_batches: int = 2
    min_audio_rms: float = 0.001

    max_cache_chunks: int = 16
    keep_cache_chunks: int = 8

    max_new_tokens: int = 40
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50
    seed: int = 998244353

    term_map_format: str = "plain"
    empty_term_map_policy: str = "none_block"
    system_prompt_style: str = "given_chunks"

    rag_enabled: bool = True
    rag_model_path: str = ""
    rag_device: str = "cuda:1"
    rag_top_k: int = 10
    rag_lora_r: int = 128
    rag_text_lora_r: int = 128
    rag_score_threshold: float = 0.78
    rag_timeline_lookback_sec: float = 1.92

    auto_glossary_enabled: bool = True
    auto_glossary_base_preset: str = "none"
    auto_glossary_default_preset: str = "nlp_core_10k"
    auto_glossary_presets: str = "nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k"
    auto_glossary_update_sec: float = 45.0
    auto_glossary_warmup_sec: float = 30.0
    auto_glossary_min_conf: float = 0.60
    auto_glossary_switch_margin: float = 0.15
    auto_glossary_current_margin: float = 0.10
    auto_glossary_min_consistent_windows: int = 2
    auto_glossary_switch_cooldown_sec: float = 90.0
    auto_glossary_candidate_stale_sec: float = 120.0
    auto_glossary_fallback_preset: str = "none"
    auto_glossary_preload: bool = True
    auto_glossary_preload_presets: str = "nlp_core_10k,medicine_core_10k"
    router_mode: str = "hybrid_window_topic"
    router_legacy_keywords: bool = False
    router_embed_weight: float = 0.65
    router_ref_weight: float = 0.35
    router_ema_alpha: float = 0.80
    router_text_topic_weight: float = 0.60
    router_domain_probe_weight: float = 0.25
    router_speech_centroid_weight: float = 0.10
    router_metadata_prior_weight: float = 0.05
    router_domain_probe_top_k: int = 5
    router_min_consistent_windows_with_text: int = 2
    router_min_consistent_windows_generated_target: int = 3
    router_min_consistent_windows_audio_only: int = 3
    router_audio_probe_min_top_score: float = 0.50
    router_audio_probe_min_raw_margin: float = 0.08
    router_audio_probe_min_positive_domains: int = 2
    router_generated_target_probe_min_top_score: float = 0.25
    router_generated_target_probe_min_raw_margin: float = 0.01
    router_generated_target_probe_min_positive_domains: int = 1
    router_generated_target_enabled: bool = True
    router_generated_target_window_chunks: int = 3
    router_generated_target_min_chars: int = 6
    prompt_top_k: int = PROMPT_K
    ui_top_k: int = PROMPT_K
    autoterm_broad_topk_per_slice: int = 50
    autoterm_rescue_preset: str = "open_wiki_100k"
    autoterm_candidate_score_threshold: float = 0.0
    autoterm_enable_open_rescue: bool = True

    max_imported_glossary_terms: int = 10000
    tmp_dir: str = ""

    @classmethod
    def from_env(cls, template: ModelTemplate) -> "OmniConfig":
        autoterm_config = _load_autoterm_slice_config()
        routing_config = autoterm_config.get("routing") if isinstance(autoterm_config.get("routing"), dict) else {}
        retrieval_config = autoterm_config.get("retrieval") if isinstance(autoterm_config.get("retrieval"), dict) else {}
        prompt_k_default = _safe_int(retrieval_config.get("prompt_k", autoterm_config.get("prompt_k")), PROMPT_K)
        broad_topk_default = _safe_int(retrieval_config.get("broad_topk_per_slice"), 50)
        base_preset_default = _autoterm_base_preset(autoterm_config, "none")
        initial_preset_default = _autoterm_initial_preset(autoterm_config, "nlp_core_10k")
        working_presets_default = _autoterm_working_presets(
            autoterm_config,
            "nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k",
        )
        rescue_preset_default = _autoterm_rescue_preset(autoterm_config, "open_wiki_100k")
        mock = _env_bool("RASST_DEMO_MOCK", False)
        rag_enabled = (not mock) and bool(_env_int("RASST_RAG_ENABLED", 1))
        return cls(
            language_pair=_env_str("RASST_DEMO_LANGUAGE_PAIR", "English -> Chinese"),
            mock=mock,
            vllm_model_path=_env_str("RASST_VLLM_MODEL_PATH", ""),
            vllm_tp_size=_env_int("RASST_VLLM_TP_SIZE", 1),
            gpu_memory_utilization=_env_float("RASST_GPU_MEMORY_UTILIZATION", 0.86),
            max_num_seqs=_env_int("RASST_MAX_NUM_SEQS", 32),
            max_model_len=_env_int("RASST_MAX_MODEL_LEN", 16384),
            enable_prefix_caching=bool(_env_int("RASST_ENABLE_PREFIX_CACHING", 1)),
            vllm_enforce_eager=bool(_env_int("RASST_VLLM_ENFORCE_EAGER", 0)),
            vllm_limit_audio=_env_int("RASST_VLLM_LIMIT_AUDIO", 16),
            disable_custom_all_reduce=bool(_env_int("RASST_DISABLE_CUSTOM_ALL_REDUCE", 0)),
            sglang_base_url=_env_str("RASST_SGLANG_BASE_URL", "http://127.0.0.1:8100"),
            sglang_timeout_sec=_env_float("RASST_SGLANG_TIMEOUT_SEC", 900.0),
            segment_sec=_env_float("RASST_VLLM_SEGMENT_SEC", _env_float("RASST_SGLANG_SEGMENT_SEC", 1.92)),
            base_segment_sec=_env_float("RASST_BASE_SEGMENT_SEC", 0.96),
            default_latency_multiplier=_env_int("RASST_DEFAULT_LATENCY_MULTIPLIER", 2),
            scheduler_batch_size=_env_int("RASST_SCHEDULER_BATCH_SIZE", 32),
            batch_timeout=_env_float("RASST_BATCH_TIMEOUT", 0.05),
            coalesce_sec=_env_float("RASST_SGLANG_COALESCE_SEC", 0.06),
            max_inflight_batches=_env_int("RASST_SGLANG_MAX_INFLIGHT_BATCHES", 2),
            min_audio_rms=_env_float("RASST_MIN_AUDIO_RMS", 0.001),
            max_cache_chunks=_env_int("RASST_MAX_CACHE_CHUNKS", 16),
            keep_cache_chunks=_env_int("RASST_KEEP_CACHE_CHUNKS", 8),
            max_new_tokens=_env_int("RASST_MAX_NEW_TOKENS", 40),
            temperature=_env_float("RASST_TEMPERATURE", 0.0),
            top_p=_env_float("RASST_TOP_P", 0.9),
            top_k=_env_int("RASST_TOP_K", 50),
            seed=_env_int("RASST_SEED", 998244353),
            term_map_format=_env_str("RASST_TERM_MAP_FORMAT", "plain"),
            empty_term_map_policy=_env_str("RASST_EMPTY_TERM_MAP_POLICY", "none_block"),
            system_prompt_style=_env_str("RASST_SYSTEM_PROMPT_STYLE", template.system_prompt_style),
            rag_enabled=rag_enabled,
            rag_model_path=_env_str(
                "RASST_HN1024_RETRIEVER",
                str(PROJECT_ROOT / "checkpoints/retriever/rasst-hn1024.pt"),
            ),
            rag_device=_env_str("RASST_RAG_DEVICE", "cuda:1"),
            rag_top_k=_env_int("RASST_RAG_TOP_K", 10),
            rag_lora_r=_env_int("RASST_RAG_LORA_R", 128),
            rag_text_lora_r=_env_int("RASST_RAG_TEXT_LORA_R", 128),
            rag_score_threshold=_env_float("RASST_RAG_SCORE_THRESHOLD", 0.78),
            rag_timeline_lookback_sec=_env_float("RASST_RAG_LOOKBACK_SEC", 1.92),
            auto_glossary_enabled=_env_bool("RASST_AUTO_GLOSSARY_ENABLED", True),
            auto_glossary_base_preset=base_preset_default,
            auto_glossary_default_preset=_env_str("RASST_AUTO_GLOSSARY_DEFAULT", initial_preset_default),
            auto_glossary_presets=_env_str(
                "RASST_AUTO_GLOSSARY_PRESETS",
                working_presets_default,
            ),
            auto_glossary_update_sec=_env_float(
                "RASST_AUTO_GLOSSARY_UPDATE_SEC",
                _safe_float(routing_config.get("production_update_sec"), _safe_float(routing_config.get("update_sec"), 45.0)),
            ),
            auto_glossary_warmup_sec=_env_float(
                "RASST_AUTO_GLOSSARY_WARMUP_SEC",
                _safe_float(routing_config.get("production_warmup_sec"), _safe_float(routing_config.get("warmup_sec"), 30.0)),
            ),
            auto_glossary_min_conf=_env_float("RASST_AUTO_GLOSSARY_MIN_CONF", _safe_float(routing_config.get("domain_activate_threshold"), 0.60)),
            auto_glossary_switch_margin=_env_float(
                "RASST_AUTO_GLOSSARY_MIN_MARGIN",
                _env_float("RASST_AUTO_GLOSSARY_SWITCH_MARGIN", _safe_float(routing_config.get("domain_margin_threshold"), 0.15)),
            ),
            auto_glossary_current_margin=_safe_float(routing_config.get("current_margin_threshold"), 0.10),
            auto_glossary_min_consistent_windows=_env_int(
                "RASST_AUTO_GLOSSARY_MIN_CONSISTENT_WINDOWS",
                _safe_int(routing_config.get("min_consistent_windows"), 2),
            ),
            auto_glossary_switch_cooldown_sec=_safe_float(
                routing_config.get("production_cooldown_sec"),
                _safe_float(routing_config.get("switch_cooldown_sec"), 90.0),
            ),
            auto_glossary_candidate_stale_sec=_safe_float(routing_config.get("candidate_stale_sec"), 120.0),
            auto_glossary_fallback_preset=_env_str("RASST_AUTO_GLOSSARY_FALLBACK", "none"),
            auto_glossary_preload=_env_bool("RASST_AUTO_GLOSSARY_PRELOAD", True),
            auto_glossary_preload_presets=_env_str(
                "RASST_AUTO_GLOSSARY_PRELOAD_PRESETS",
                working_presets_default,
            ),
            router_mode=_env_str("RASST_ROUTER_MODE", str(routing_config.get("mode") or "hybrid_window_topic")),
            router_legacy_keywords=_env_bool("RASST_ROUTER_LEGACY_KEYWORDS", False),
            router_embed_weight=_env_float("RASST_ROUTER_EMBED_WEIGHT", 0.65),
            router_ref_weight=_env_float("RASST_ROUTER_REF_WEIGHT", 0.35),
            router_ema_alpha=_env_float("RASST_ROUTER_EMA_ALPHA", 0.80),
            router_text_topic_weight=_safe_float(routing_config.get("text_topic_weight"), 0.60),
            router_domain_probe_weight=_safe_float(routing_config.get("domain_probe_weight"), 0.25),
            router_speech_centroid_weight=_safe_float(routing_config.get("speech_centroid_weight"), 0.10),
            router_metadata_prior_weight=_safe_float(routing_config.get("metadata_prior_weight"), 0.05),
            router_domain_probe_top_k=_safe_int(routing_config.get("domain_probe_top_k"), 5),
            router_min_consistent_windows_with_text=_safe_int(
                routing_config.get("min_consistent_windows_with_text"),
                2,
            ),
            router_min_consistent_windows_generated_target=_safe_int(
                routing_config.get("min_consistent_windows_generated_target"),
                3,
            ),
            router_min_consistent_windows_audio_only=_safe_int(
                routing_config.get("min_consistent_windows_audio_only"),
                3,
            ),
            router_audio_probe_min_top_score=_safe_float(routing_config.get("audio_probe_min_top_score"), 0.50),
            router_audio_probe_min_raw_margin=_safe_float(routing_config.get("audio_probe_min_raw_margin"), 0.08),
            router_audio_probe_min_positive_domains=_safe_int(routing_config.get("audio_probe_min_positive_domains"), 2),
            router_generated_target_probe_min_top_score=_safe_float(routing_config.get("generated_target_probe_min_top_score"), 0.25),
            router_generated_target_probe_min_raw_margin=_safe_float(routing_config.get("generated_target_probe_min_raw_margin"), 0.01),
            router_generated_target_probe_min_positive_domains=_safe_int(routing_config.get("generated_target_probe_min_positive_domains"), 1),
            router_generated_target_enabled=_meta_bool(routing_config.get("enable_generated_target_text"), True),
            router_generated_target_window_chunks=_safe_int(routing_config.get("generated_target_window_chunks"), 3),
            router_generated_target_min_chars=_safe_int(routing_config.get("generated_target_min_chars"), 6),
            prompt_top_k=_env_int("RASST_PROMPT_TOP_K", prompt_k_default),
            ui_top_k=_env_int("RASST_UI_TOP_K", prompt_k_default),
            autoterm_broad_topk_per_slice=broad_topk_default,
            autoterm_rescue_preset=rescue_preset_default,
            autoterm_enable_open_rescue=_meta_bool(retrieval_config.get("use_open_wiki_rescue"), _meta_bool(routing_config.get("enable_fallback"), True)),
            max_imported_glossary_terms=_env_int("RASST_MAX_IMPORTED_GLOSSARY_TERMS", 10000),
            tmp_dir=_env_str("RASST_TMP_DIR", f"/dev/shm/rasst_omni_{os.getpid()}"),
        )


@dataclass
class OmniSession:
    session_id: str
    language_pair: str
    source_lang: str
    target_lang: str
    lang_code: str
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cursor_samples: int = 0
    last_llm_samples: int = 0
    segment_idx: int = 0
    inflight: bool = False
    flush: bool = False
    final: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    audio_paths: List[Path] = field(default_factory=list)
    imported_glossary: List[Dict[str, Any]] = field(default_factory=list)
    glossary_preset: str = DEFAULT_GLOSSARY_PRESET
    glossary_index_path: str = ""
    requested_glossary_preset: str = DEFAULT_GLOSSARY_PRESET
    active_glossary_preset: str = "none"
    active_domain: str = GENERAL_DOMAIN
    auto_glossary_enabled: bool = False
    topic_confidence: float = 0.0
    last_topic_update_s: float = 0.0
    created_s: float = field(default_factory=time.perf_counter)
    topic_history: List[Dict[str, Any]] = field(default_factory=list)
    glossary_switch_count: int = 0
    topic_update_task: Optional[asyncio.Task] = None
    recent_references: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=64))
    last_topic_reason: str = ""
    router_state: Optional[RouterSessionState] = None
    last_router_decision: Optional[Dict[str, Any]] = None
    last_query_embedding: Any = None
    router_text_window: str = ""
    router_text_source: str = "none"
    active_slice_presets: List[str] = field(default_factory=list)
    active_slice_terms: Dict[str, int] = field(default_factory=dict)
    last_retrieval_plan: List[Dict[str, Any]] = field(default_factory=list)
    last_rescue_triggered: bool = False
    last_candidate_pool_count: int = 0
    last_prompt_reference_count: int = 0
    last_domain_probe_raw_scores: Dict[str, DomainProbeScore] = field(default_factory=dict)
    last_domain_probe_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_domain_probe_slices: List[Dict[str, Any]] = field(default_factory=list)
    last_domain_probe_s: Optional[float] = None
    last_domain_probe_at_s: float = 0.0
    last_domain_probe_cached: bool = False
    active_retrieval_slices: List[RetrievalSlice] = field(default_factory=list)
    pending_since_s: Optional[float] = None
    latency_multiplier: int = 2


class OmniAgent(Agent):
    """Omni-model streaming SST agent (RASST / Qwen3-Omni / MiniCPM-o)."""

    def __init__(self, name: str = "RASST", model_id: str = "qwen3_omni") -> None:
        self.name = name
        self.model_id = model_id
        self.template = get_template(model_id)
        self.config = OmniConfig.from_env(self.template)
        self.prompt = PromptBuilder(
            system_prompt_style=self.config.system_prompt_style,
            term_map_format=self.config.term_map_format,
            empty_term_map_policy=self.config.empty_term_map_policy,
            audio_schema=self.template.audio_schema,
        )

        self._emit: Optional[EmitFn] = None
        self.backend = None
        self.retrieval: RetrievalPlugin = NullRetrieval()
        self.sessions: Dict[str, OmniSession] = {}
        self.pending: Deque[str] = deque()
        self.pending_set: Set[str] = set()

        self._scheduler_task: Optional[asyncio.Task] = None
        self._batch_gate: Optional[asyncio.Semaphore] = None
        self._retriever_lock: Optional[asyncio.Lock] = None
        self.batch_seq = 0
        self.recent_batch_metrics: Deque[Dict[str, Any]] = deque(maxlen=64)
        self._catalogs: Dict[str, GlossaryCatalog] = {}
        self._legacy_topic_router = LegacyKeywordTopicRouter(
            warmup_sec=self.config.auto_glossary_warmup_sec,
            update_sec=self.config.auto_glossary_update_sec,
            min_confidence=self.config.auto_glossary_min_conf,
            switch_margin=self.config.auto_glossary_switch_margin,
        )
        self._topic_routers: Dict[str, AudioNativeActiveGlossaryRouter] = {}
        self.active_glossary = ActiveGlossaryManager(
            default_preset=self.config.auto_glossary_default_preset,
            allowed_presets=configured_working_presets(self.config.auto_glossary_presets),
        )
        self._index_preload_tasks: Dict[str, asyncio.Task] = {}
        self.tmp_dir = Path(self.config.tmp_dir)

    def _catalog(self, language_pair: str) -> GlossaryCatalog:
        catalog = self._catalogs.get(language_pair)
        if catalog is None:
            catalog = GlossaryCatalog(language_pair, self.config.max_imported_glossary_terms)
            self._catalogs[language_pair] = catalog
        return catalog

    def _topic_router_for(self, language_pair: str) -> AudioNativeActiveGlossaryRouter:
        router = self._topic_routers.get(language_pair)
        if router is not None:
            return router
        catalog = self._catalog(language_pair)
        router_config = RouterConfig(
            warmup_sec=self.config.auto_glossary_warmup_sec,
            update_interval_sec=self.config.auto_glossary_update_sec,
            min_confidence=self.config.auto_glossary_min_conf,
            min_margin=self.config.auto_glossary_switch_margin,
            min_current_margin=self.config.auto_glossary_current_margin,
            min_consistent_windows=self.config.auto_glossary_min_consistent_windows,
            switch_cooldown_sec=self.config.auto_glossary_switch_cooldown_sec,
            candidate_stale_sec=self.config.auto_glossary_candidate_stale_sec,
            embedding_weight=self.config.router_embed_weight,
            reference_weight=self.config.router_ref_weight,
            ema_alpha=self.config.router_ema_alpha,
            fallback_preset_id=self.config.auto_glossary_fallback_preset,
            text_topic_weight=self.config.router_text_topic_weight,
            domain_probe_weight=self.config.router_domain_probe_weight,
            speech_centroid_weight=self.config.router_speech_centroid_weight,
            metadata_prior_weight=self.config.router_metadata_prior_weight,
            min_consistent_windows_with_text=self.config.router_min_consistent_windows_with_text,
            min_consistent_windows_generated_target=self.config.router_min_consistent_windows_generated_target,
            min_consistent_windows_audio_only=self.config.router_min_consistent_windows_audio_only,
            audio_probe_min_top_score=self.config.router_audio_probe_min_top_score,
            audio_probe_min_raw_margin=self.config.router_audio_probe_min_raw_margin,
            audio_probe_min_positive_domains=self.config.router_audio_probe_min_positive_domains,
            generated_target_probe_min_top_score=self.config.router_generated_target_probe_min_top_score,
            generated_target_probe_min_raw_margin=self.config.router_generated_target_probe_min_raw_margin,
            generated_target_probe_min_positive_domains=self.config.router_generated_target_probe_min_positive_domains,
        )
        router_cls = (
            HybridWindowTopicRouter
            if (self.config.router_mode or "").strip().lower() == "hybrid_window_topic"
            else AudioNativeActiveGlossaryRouter
        )
        router = router_cls(
            self._domain_slices_for(catalog),
            router_config,
        )
        self._topic_routers[language_pair] = router
        return router

    def _domain_slices_for(self, catalog: GlossaryCatalog) -> List[DomainSlice]:
        allowed = configured_working_presets(self.config.auto_glossary_presets)
        slices: List[DomainSlice] = []
        for preset_id in allowed:
            meta = catalog.manifest.meta_for_preset(preset_id) if catalog.manifest else {}
            domain_id = str(meta.get("domain_id") or meta.get("domain") or domain_for_preset(preset_id)).strip()
            if domain_id == "general":
                domain_id = GENERAL_DOMAIN
            if (
                preset_id == self.config.auto_glossary_fallback_preset
                or domain_id in {GENERAL_DOMAIN, "common", "general"}
            ):
                continue
            enabled = _meta_bool(meta.get("enabled_for_auto_router"), True)
            snapshot = catalog._open_snapshot(preset_id)  # agent-internal catalog helper
            index_path = str(
                meta.get("maxsim_index_path")
                or meta.get("index_path")
                or (snapshot.index_path("maxsim") if snapshot is not None else "")
                or catalog.index_path_for_preset(preset_id)
            )
            if self.config.mock and not index_path and preset_id != "none":
                index_path = f"mock://{preset_id}"
            term_count = 0
            try:
                term_count = int(
                    meta.get("term_count")
                    or meta.get("terms")
                    or (snapshot.num_terms if snapshot is not None else 0)
                    or 0
                )
            except (TypeError, ValueError):
                term_count = 0
            slices.append(
                DomainSlice(
                    preset_id=preset_id,
                    domain_id=domain_id or domain_for_preset(preset_id),
                    parent_domain_id=str(meta.get("parent_domain_id") or "") or None,
                    fallback_preset_id=str(meta.get("fallback_preset_id") or self.config.auto_glossary_fallback_preset or "") or None,
                    centroid=self._load_centroid(meta.get("centroid_path")),
                    enabled=enabled,
                    priority=_safe_int(meta.get("priority"), 0),
                    term_count=term_count,
                    index_path=index_path,
                    description=str(meta.get("domain_description") or meta.get("description") or ""),
                )
            )
        return slices

    def _load_centroid(self, centroid_path: Any) -> Any:
        path = str(centroid_path or "").strip()
        if not path:
            return None
        centroid_file = Path(path)
        if not centroid_file.is_file():
            logger.warning("router centroid missing: %s", centroid_file)
            return None
        try:
            import torch  # noqa: WPS433 - optional real-backend dependency

            data = torch.load(str(centroid_file), map_location="cpu")
            if isinstance(data, dict):
                centroid = data.get("centroid")
                if centroid is None and data.get("prototypes") is not None:
                    prototypes = data["prototypes"]
                    centroid = prototypes[0] if getattr(prototypes, "ndim", 0) >= 2 else prototypes
                return centroid
            return data
        except Exception:  # noqa: BLE001 - missing centroid must not break serving
            logger.exception("failed to load router centroid %s", centroid_file)
            return None

    def _segment_samples(self, session: "OmniSession") -> int:
        lm = max(1, min(4, int(session.latency_multiplier)))
        return int(float(self.config.base_segment_sec) * lm * TARGET_SAMPLE_RATE)

    def _vllm_config(self) -> Dict[str, Any]:
        """Kwargs for VLLMBackend; model_path falls back to the per-language path."""
        catalog = self._catalog(self.config.language_pair)
        return {
            "model_path": self.config.vllm_model_path or catalog.model_path,
            "tp_size": self.config.vllm_tp_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_num_seqs": self.config.max_num_seqs,
            "max_model_len": self.config.max_model_len,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enforce_eager": self.config.vllm_enforce_eager,
            "limit_audio": self.config.vllm_limit_audio,
            "disable_custom_all_reduce": self.config.disable_custom_all_reduce,
            "max_cache_chunks": self.config.max_cache_chunks,
            "keep_cache_chunks": self.config.keep_cache_chunks,
            "empty_term_map_policy": self.config.empty_term_map_policy,
            "rag_enabled": self.config.rag_enabled,
            "default_source_lang": catalog.source_lang,
            "default_target_lang": catalog.target_lang,
            "default_lang_code": catalog.lang_code,
        }

    def _schedule_auto_preloads(self) -> None:
        catalog = self._catalog(self.config.language_pair)
        for preset in configured_working_presets(self.config.auto_glossary_preload_presets):
            try:
                selection = self.active_glossary.initial_selection(
                    catalog,
                    preset,
                    "",
                    auto_allowed=False,
                    mock=self.config.mock,
                )
            except Exception:  # noqa: BLE001 - optional warm path
                logger.exception("failed to resolve auto preload preset %s", preset)
                continue
            if selection.index_path:
                self._schedule_index_preload(selection.index_path)

    def _schedule_index_preload(self, index_path: str) -> None:
        if not index_path or not self.retrieval.enabled or self.retrieval.is_index_ready(index_path):
            return
        existing = self._index_preload_tasks.get(index_path)
        if existing is not None and not existing.done():
            return

        async def _runner() -> None:
            try:
                await self.retrieval.preload_index(index_path)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - cold preload must not sink serving
                logger.exception("failed to preload glossary index %s", index_path)
            finally:
                self._index_preload_tasks.pop(index_path, None)

        self._index_preload_tasks[index_path] = asyncio.create_task(_runner())

    def _slice_for_preset(
        self,
        session: "OmniSession",
        preset_id: str,
        *,
        role: Optional[str] = None,
    ) -> Optional[RetrievalSlice]:
        preset = (preset_id or "").strip()
        if not preset or preset == "none":
            return None
        catalog = self._catalog(session.language_pair)
        try:
            selection = self.active_glossary.initial_selection(
                catalog,
                preset,
                "",
                auto_allowed=False,
                mock=self.config.mock,
            )
        except (ValueError, KeyError):
            return None
        active_preset = selection.active_preset
        index_path = selection.index_path
        if not index_path:
            return None
        resolved_role = role or slice_role_for_preset(active_preset)
        return RetrievalSlice(
            preset_id=active_preset,
            slice_id=slice_id_for_preset(active_preset),
            role=resolved_role,
            domain=domain_for_slice_preset(active_preset),
            index_path=index_path,
            term_count=int(selection.preset_terms or 0),
            weight=slice_weight_for_role(resolved_role),
            eval_only=(active_preset == "acl_tagged_raw"),
        )

    def _active_retrieval_slices(self, session: "OmniSession") -> List[RetrievalSlice]:
        if session.active_retrieval_slices:
            return list(session.active_retrieval_slices)
        if not session.auto_glossary_enabled:
            plan = self._slice_for_preset(
                session,
                session.glossary_preset,
                role=slice_role_for_preset(session.glossary_preset),
            )
            slices = [plan] if plan is not None else []
            self._record_active_slices(session, slices)
            return slices

        active_preset = (session.active_glossary_preset or session.glossary_preset or "").strip()
        presets: List[Tuple[str, str]] = []
        base_preset = (self.config.auto_glossary_base_preset or "").strip()
        if base_preset and base_preset != "none" and base_preset != active_preset:
            presets.append((base_preset, "base"))
        if active_preset and active_preset != "none":
            presets.append((active_preset, "domain"))
        else:
            for preset in configured_working_presets(self.config.auto_glossary_presets):
                domain = domain_for_preset(preset)
                if not preset or preset == "none" or domain in {GENERAL_DOMAIN, "common", "general"}:
                    continue
                presets.append((preset, "domain_probe"))
        if not presets:
            default_preset = (self.config.auto_glossary_default_preset or "").strip()
            if default_preset and default_preset != "none":
                presets.append((default_preset, "domain"))

        slices: List[RetrievalSlice] = []
        seen: Set[str] = set()
        for preset, role in presets:
            plan = self._slice_for_preset(session, preset, role=role)
            if plan is None or plan.preset_id in seen:
                continue
            seen.add(plan.preset_id)
            slices.append(plan)
        self._record_active_slices(session, slices)
        return slices

    def _domain_probe_slices(self, session: "OmniSession") -> List[RetrievalSlice]:
        if not session.auto_glossary_enabled:
            return []
        if float(self.config.router_domain_probe_weight) <= 0.0:
            return []
        slices: List[RetrievalSlice] = []
        seen: Set[str] = set()
        for preset in configured_working_presets(self.config.auto_glossary_presets):
            domain = domain_for_preset(preset)
            if not preset or preset == "none" or domain in {GENERAL_DOMAIN, "common", "general"}:
                continue
            plan = self._slice_for_preset(session, preset, role="domain_probe")
            if plan is None or plan.preset_id in seen:
                continue
            seen.add(plan.preset_id)
            slices.append(plan)
        return slices

    def _domain_probe_request(
        self,
        session: "OmniSession",
        end_sample: int,
        query_embedding: Any = None,
    ) -> Dict[str, Any]:
        end = max(0, min(int(end_sample), int(len(session.audio))))
        current_start = max(0, min(int(session.last_llm_samples), end))
        lookback_samples = max(
            0,
            int(round(float(self.config.rag_timeline_lookback_sec) * TARGET_SAMPLE_RATE)),
        )
        start = max(0, current_start - lookback_samples)
        audio_buffer = session.audio[start:end]
        return {
            "audio_buffer": audio_buffer,
            "current_start_sec": float(current_start - start) / TARGET_SAMPLE_RATE,
            "current_end_sec": float(end - start) / TARGET_SAMPLE_RATE,
            "lookback_sec": float(self.config.rag_timeline_lookback_sec),
            "query_embedding": query_embedding,
        }

    def _clear_domain_probe_meta(self, session: "OmniSession") -> None:
        session.last_domain_probe_raw_scores = {}
        session.last_domain_probe_scores = {}
        session.last_domain_probe_slices = []
        session.last_domain_probe_s = None
        session.last_domain_probe_at_s = 0.0
        session.last_domain_probe_cached = False

    def _cached_domain_probe_scores(
        self,
        session: "OmniSession",
        candidates: Sequence[RetrievalSlice],
    ) -> Dict[str, DomainProbeScore]:
        cached = dict(getattr(session, "last_domain_probe_raw_scores", {}) or {})
        if not cached:
            return {}
        allowed_domains = {str(item.domain) for item in candidates}
        filtered = {domain: score for domain, score in cached.items() if str(domain) in allowed_domains}
        if not filtered:
            return {}
        session.last_domain_probe_slices = [item.to_meta() for item in candidates]
        session.last_domain_probe_scores = _domain_probe_scores_to_meta(filtered)
        session.last_domain_probe_s = 0.0
        session.last_domain_probe_cached = True
        return filtered

    def _domain_probe_refresh_sec(self, session: "OmniSession") -> float:
        source = str(getattr(session, "router_text_source", "none") or "none").strip()
        text = str(getattr(session, "router_text_window", "") or "").strip()
        if text and source != "none":
            return max(0.0, float(self.config.auto_glossary_update_sec))
        try:
            latency_multiplier = int(getattr(session, "latency_multiplier", self.config.default_latency_multiplier))
        except (TypeError, ValueError):
            latency_multiplier = int(self.config.default_latency_multiplier)
        latency_multiplier = max(1, min(4, latency_multiplier))
        return max(0.1, float(self.config.base_segment_sec) * latency_multiplier)

    async def _probe_domain_scores(
        self,
        session: "OmniSession",
        *,
        end_sample: int,
        query_embedding: Any = None,
    ) -> Dict[str, DomainProbeScore]:
        mode = (self.config.router_mode or "embedding_refs").strip().lower()
        if mode != "hybrid_window_topic" or not self.retrieval.enabled:
            self._clear_domain_probe_meta(session)
            return {}
        now = time.perf_counter()
        if now - float(session.created_s) < float(self.config.auto_glossary_warmup_sec):
            self._clear_domain_probe_meta(session)
            return {}
        state = session.router_state

        candidates: List[RetrievalSlice] = []
        for plan in self._domain_probe_slices(session):
            if self.retrieval.is_index_ready(plan.index_path):
                candidates.append(plan)
            else:
                self._schedule_index_preload(plan.index_path)
        session.last_domain_probe_slices = [item.to_meta() for item in candidates]
        if not candidates:
            self._clear_domain_probe_meta(session)
            return {}
        last_probe_at = float(getattr(session, "last_domain_probe_at_s", 0.0) or 0.0)
        refresh_sec = self._domain_probe_refresh_sec(session)
        update_gate = (
            last_probe_at > 0.0
            and now - last_probe_at < refresh_sec
        )
        cooldown_gate = (
            state is not None
            and state.last_switch_s > 0.0
            and now - float(state.last_switch_s) < float(self.config.auto_glossary_switch_cooldown_sec)
        )
        if update_gate or cooldown_gate:
            cached_scores = self._cached_domain_probe_scores(session, candidates)
            if cached_scores:
                return cached_scores
            if update_gate:
                session.last_domain_probe_scores = {}
                session.last_domain_probe_slices = [item.to_meta() for item in candidates]
                session.last_domain_probe_s = 0.0
                session.last_domain_probe_cached = True
                return {}

        t0 = time.perf_counter()
        scores = await self.retrieval.probe_domain_scores(
            self._domain_probe_request(session, end_sample, query_embedding),
            candidate_slices=[item.to_meta() for item in candidates],
            top_k=max(1, int(self.config.router_domain_probe_top_k)),
            lookback_sec=float(self.config.rag_timeline_lookback_sec),
            score_threshold=None,
        )
        session.last_domain_probe_s = time.perf_counter() - t0
        session.last_domain_probe_at_s = now
        session.last_domain_probe_cached = False
        session.last_domain_probe_raw_scores = dict(scores)
        session.last_domain_probe_scores = _domain_probe_scores_to_meta(scores)
        return scores

    def _rescue_retrieval_slice(self, session: "OmniSession") -> Optional[RetrievalSlice]:
        if not self.config.autoterm_enable_open_rescue:
            return None
        return self._slice_for_preset(session, self.config.autoterm_rescue_preset, role="rescue")

    def _record_active_slices(self, session: "OmniSession", slices: Sequence[RetrievalSlice]) -> None:
        session.active_retrieval_slices = list(slices)
        session.active_slice_presets = [item.preset_id for item in slices]
        session.active_slice_terms = {item.preset_id: int(item.term_count or 0) for item in slices}
        session.last_retrieval_plan = [item.to_meta() for item in slices]

    def _retrieval_top_k_for(self, session: "OmniSession") -> int:
        if session.auto_glossary_enabled:
            return max(
                int(self.config.autoterm_broad_topk_per_slice),
                int(self.config.prompt_top_k),
                int(self.config.ui_top_k),
            )
        return max(int(self.config.rag_top_k), int(self.config.prompt_top_k), int(self.config.ui_top_k))

    def _retrieval_score_threshold_for(self, session: "OmniSession") -> Optional[float]:
        if session.auto_glossary_enabled:
            return float(self.config.autoterm_candidate_score_threshold)
        return None

    def _should_rescue_retrieval(
        self,
        session: "OmniSession",
        ranked: Sequence[Dict[str, Any]],
    ) -> bool:
        if not session.auto_glossary_enabled or not self.config.autoterm_enable_open_rescue:
            return False
        decision = session.last_router_decision or {}
        return str(decision.get("action") or "") == "fallback"

    def _select_initial_glossary(
        self,
        catalog: GlossaryCatalog,
        requested_preset: Optional[str],
        glossary_text: str,
    ):
        return self.active_glossary.initial_selection(
            catalog,
            requested_preset,
            glossary_text,
            auto_allowed=self.config.auto_glossary_enabled,
            mock=self.config.mock,
        )

    async def start(self, emit: EmitFn) -> None:
        self._emit = emit
        self._batch_gate = asyncio.Semaphore(max(1, int(self.config.max_inflight_batches)))
        self._retriever_lock = asyncio.Lock()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.backend = build_backend(
            self.template,
            mock=self.config.mock,
            sglang_base_url=self.config.sglang_base_url,
            sglang_timeout_sec=self.config.sglang_timeout_sec,
            vllm_config=self._vllm_config(),
        )
        await self.backend.start()

        if self.config.rag_enabled:
            try:
                catalog = self._catalog(self.config.language_pair)
                startup_index = catalog.index_path_for_preset(RAG_STARTUP_GLOSSARY_PRESET)
                self.retrieval = MaxSimRetrievalPlugin(
                    model_path=self.config.rag_model_path,
                    index_path=startup_index,
                    device=self.config.rag_device,
                    top_k=self.config.rag_top_k,
                    lora_rank=self.config.rag_lora_r,
                    text_lora_rank=self.config.rag_text_lora_r,
                    target_lang=catalog.lang_code,
                    score_threshold=self.config.rag_score_threshold,
                )
                await self.retrieval.start()
            except Exception:  # noqa: BLE001 - retrieval is optional; degrade gracefully
                logger.exception("retrieval failed to load; continuing without RAG")
                self.retrieval = NullRetrieval()
                self.config.rag_enabled = False
        elif self.config.mock:
            # GPU-free dev/demo: deterministic fake retrieval so the evidence
            # UI + JSON protocol can be exercised without a model or retriever.
            self.retrieval = MockRetrieval(
                target_lang=self._catalog(self.config.language_pair).lang_code,
                top_k=max(self.config.rag_top_k, self.config.ui_top_k),
            )
            await self.retrieval.start()
        else:
            self.retrieval = NullRetrieval()

        if self.config.auto_glossary_enabled and self.retrieval.enabled:
            self._schedule_auto_preloads()

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "OmniAgent %r started (model=%s mock=%s rag=%s)",
            self.name,
            self.template.served_model_name,
            self.config.mock,
            self.config.rag_enabled,
        )

    async def shutdown(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            await asyncio.gather(self._scheduler_task, return_exceptions=True)
        if self.backend is not None:
            try:
                await self.backend.stop()
            except Exception:  # noqa: BLE001
                logger.exception("backend stop failed")
        for task in list(self._index_preload_tasks.values()):
            task.cancel()
        if self._index_preload_tasks:
            await asyncio.gather(*self._index_preload_tasks.values(), return_exceptions=True)
        for session in list(self.sessions.values()):
            if session.topic_update_task is not None:
                session.topic_update_task.cancel()
        try:
            await self.retrieval.stop()
        except Exception:  # noqa: BLE001
            logger.exception("retrieval stop failed")
        for path in self.tmp_dir.glob("*.wav"):
            try:
                path.unlink()
            except OSError:
                pass

    async def open_session(self, session_id: str, config: Dict[str, Any]) -> SessionInfo:
        language_pair = str(config.get("language_pair") or self.config.language_pair)
        if language_pair not in LANGUAGE_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported language pair: {language_pair}")
        if not self.config.mock and language_pair != self.config.language_pair:
            raise HTTPException(
                status_code=400,
                detail=f"This agent is loaded for {self.config.language_pair}, not {language_pair}",
            )
        catalog = self._catalog(language_pair)
        try:
            selection = self._select_initial_glossary(
                catalog,
                config.get("glossary_preset", DEFAULT_GLOSSARY_PRESET),
                str(config.get("glossary_text") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if self.config.rag_enabled and selection.index_path and not selection.index_ready:
            raise HTTPException(
                status_code=400, detail=f"RAG text index is not ready: {selection.index_path}"
            )

        try:
            lm = int(config.get("latency_multiplier", self.config.default_latency_multiplier))
        except (TypeError, ValueError):
            lm = self.config.default_latency_multiplier
        lm = max(1, min(4, lm))
        lang_cfg = LANGUAGE_PAIRS[language_pair]
        session = OmniSession(
            session_id=session_id,
            language_pair=language_pair,
            source_lang=str(lang_cfg["source_lang"]),
            target_lang=str(lang_cfg["target_lang"]),
            lang_code=str(lang_cfg["lang_code"]),
            imported_glossary=selection.manual_refs,
            glossary_preset=selection.active_preset,
            glossary_index_path=selection.index_path,
            requested_glossary_preset=selection.requested_preset,
            active_glossary_preset=selection.active_preset,
            active_domain=selection.active_domain,
            auto_glossary_enabled=selection.auto_enabled,
            last_topic_update_s=time.perf_counter(),
            last_topic_reason=selection.reason,
            router_text_window=str(config.get("router_text") or ""),
            router_text_source=str(config.get("router_text_source") or "none"),
            latency_multiplier=lm,
        )
        session.router_state = RouterSessionState(
            active_preset_id=session.active_glossary_preset,
            active_domain_id=session.active_domain,
            created_s=session.created_s,
            last_decision_s=0.0,
            last_switch_s=session.created_s,
        )
        active_slices = self._active_retrieval_slices(session)
        self._record_active_slices(session, active_slices)
        if session.auto_glossary_enabled:
            for item in active_slices:
                self._schedule_index_preload(item.index_path)
        elif session.glossary_index_path:
            self._schedule_index_preload(session.glossary_index_path)
        self.sessions[session_id] = session
        if self.backend is not None and getattr(self.backend, "batched", False):
            await self.backend.open_session(
                session_id,
                {
                    "source_lang": session.source_lang,
                    "target_lang": session.target_lang,
                    "lang_code": session.lang_code,
                },
            )
        meta = {
            "scheduler_based": False,
            "rasst_backend": True,
            "vllm_backend": (not self.config.mock) and self.template.backend_kind == "vllm",
            "sglang_backend": (not self.config.mock) and self.template.backend_kind == "sglang_http",
            "mock_mode": self.config.mock,
            "model": self.name,
            **selection.to_session_meta(),
            "active_slices": list(session.active_slice_presets),
            "active_slice_terms": dict(session.active_slice_terms),
            "fixed_prompt_k": int(self.config.prompt_top_k),
            "topic": self._topic_meta(session),
        }
        return SessionInfo(admitted=True, queued=False, queue_position=0, meta=meta)

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            if session.topic_update_task is not None:
                session.topic_update_task.cancel()
            self._cleanup_audio(session)
        if self.backend is not None and getattr(self.backend, "batched", False):
            try:
                await self.backend.close_session(session_id)
            except Exception:  # noqa: BLE001
                logger.exception("backend close_session failed for %s", session_id)
        self.pending_set.discard(session_id)
        if session_id in self.pending:
            self.pending = deque(item for item in self.pending if item != session_id)

    def _cleanup_audio(self, session: OmniSession) -> None:
        for path in session.audio_paths:
            try:
                path.unlink()
            except OSError:
                pass
        session.audio_paths.clear()

    async def submit_audio(self, session_id: str, pcm: np.ndarray, *, final: bool = False) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        chunk = np.asarray(pcm, dtype=np.float32).flatten()
        if chunk.size:
            session.audio = np.concatenate([session.audio, chunk])
            session.cursor_samples = int(session.audio.shape[0])
        if final:
            session.final = True
            if session.cursor_samples - session.last_llm_samples > 0:
                session.flush = True
                self._mark_pending(session, force=True)
        else:
            self._mark_pending(session, force=False)

    def _mark_pending(self, session: OmniSession, *, force: bool) -> None:
        if session.inflight:
            return
        if not force and (session.cursor_samples - session.last_llm_samples < self._segment_samples(session)):
            return
        if session.session_id not in self.pending_set:
            self.pending.append(session.session_id)
            self.pending_set.add(session.session_id)
            session.pending_since_s = time.perf_counter()

    async def on_control(
        self, session_id: Optional[str], message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        mtype = message.get("type")
        if mtype == "reset":
            result = self._reset(session_id)
            if (
                result.get("success")
                and self.backend is not None
                and getattr(self.backend, "batched", False)
            ):
                try:
                    await self.backend.reset_session(session_id)
                except Exception:  # noqa: BLE001
                    logger.exception("backend reset_session failed for %s", session_id)
            return result
        if mtype == "update_latency":
            session = self.sessions.get(session_id) if session_id else None
            try:
                lm = max(1, min(4, int(message.get("latency_multiplier"))))
            except (TypeError, ValueError):
                lm = None
            if session is not None and lm is not None:
                session.latency_multiplier = lm
            return {"success": True, "latency_multiplier": lm}
        if mtype == "router_text":
            session = self.sessions.get(session_id) if session_id else None
            if session is None:
                return {"success": False, "error": "Invalid session ID"}
            session.router_text_window = str(message.get("router_text") or "")
            session.router_text_source = str(message.get("router_text_source") or "none")
            return {
                "success": True,
                "router_text_source": session.router_text_source,
                "router_text_chars": len(session.router_text_window),
            }
        if mtype == "glossary_build":
            return await self._glossary_build(message)
        return None

    def _reset(self, session_id: Optional[str]) -> Dict[str, Any]:
        session = self.sessions.get(session_id) if session_id else None
        if session is None:
            return {"success": False, "error": "Invalid session ID"}
        self._cleanup_audio(session)
        session.audio = np.zeros(0, dtype=np.float32)
        session.cursor_samples = 0
        session.last_llm_samples = 0
        session.segment_idx = 0
        session.inflight = False
        session.flush = False
        session.final = False
        session.messages.clear()
        session.history.clear()
        session.recent_references.clear()
        session.topic_history.clear()
        session.topic_confidence = 0.0
        session.glossary_switch_count = 0
        session.last_topic_update_s = time.perf_counter()
        session.last_topic_reason = "reset"
        session.last_router_decision = None
        session.last_query_embedding = None
        session.router_text_window = ""
        session.router_text_source = "none"
        self._clear_domain_probe_meta(session)
        if session.auto_glossary_enabled:
            try:
                selection = self.active_glossary.initial_selection(
                    self._catalog(session.language_pair),
                    AUTO_WORKING_PRESET,
                    "",
                    auto_allowed=True,
                    mock=self.config.mock,
                )
                session.glossary_preset = selection.active_preset
                session.active_glossary_preset = selection.active_preset
                session.active_domain = selection.active_domain
                session.glossary_index_path = selection.index_path
                session.router_state = RouterSessionState(
                    active_preset_id=session.active_glossary_preset,
                    active_domain_id=session.active_domain,
                    created_s=time.perf_counter(),
                    last_decision_s=0.0,
                    last_switch_s=time.perf_counter(),
                )
                session.active_retrieval_slices = []
                active_slices = self._active_retrieval_slices(session)
                self._record_active_slices(session, active_slices)
                for item in active_slices:
                    self._schedule_index_preload(item.index_path)
            except Exception:  # noqa: BLE001
                logger.exception("failed to reset auto glossary for %s", session.session_id)
        session.pending_since_s = None
        self.pending_set.discard(session.session_id)
        self.pending = deque(item for item in self.pending if item != session.session_id)
        return {"success": True, "message": "Translation reset successfully.", "session_type": "rasst"}

    async def _glossary_build(self, message: Dict[str, Any]) -> Dict[str, Any]:
        language_pair = str(message.get("language_pair") or self.config.language_pair)
        if language_pair not in LANGUAGE_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported language pair: {language_pair}")
        if not self.config.mock and language_pair != self.config.language_pair:
            raise HTTPException(
                status_code=400,
                detail=f"This agent is loaded for {self.config.language_pair}, not {language_pair}",
            )
        catalog = self._catalog(language_pair)
        try:
            selection = self._select_initial_glossary(
                catalog,
                message.get("glossary_preset", DEFAULT_GLOSSARY_PRESET),
                str(message.get("glossary_text") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if self.config.rag_enabled and selection.index_path and not selection.index_ready:
            raise HTTPException(
                status_code=400, detail=f"RAG text index is not ready: {selection.index_path}"
            )
        if self.retrieval.enabled and selection.index_path:
            try:
                await self.retrieval.preload_index(selection.index_path)
            except Exception:  # noqa: BLE001
                logger.exception("failed to warm glossary index %s", selection.index_path)

        session_updated = False
        session_id = message.get("session_id")
        if session_id:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=400, detail=f"Unknown session: {session_id}")
            if session.topic_update_task is not None and not session.topic_update_task.done():
                session.topic_update_task.cancel()
            session.requested_glossary_preset = selection.requested_preset
            session.glossary_preset = selection.active_preset
            session.active_glossary_preset = selection.active_preset
            session.active_domain = selection.active_domain
            session.auto_glossary_enabled = selection.auto_enabled
            session.topic_confidence = 0.0
            session.last_topic_update_s = time.perf_counter()
            session.last_topic_reason = selection.reason
            session.glossary_index_path = selection.index_path
            session.imported_glossary = selection.manual_refs
            session.last_router_decision = None
            session.last_query_embedding = None
            session.router_state = RouterSessionState(
                active_preset_id=session.active_glossary_preset,
                active_domain_id=session.active_domain,
                created_s=session.created_s,
                last_decision_s=time.perf_counter(),
                last_switch_s=time.perf_counter(),
            )
            session.active_retrieval_slices = []
            active_slices = self._active_retrieval_slices(session)
            self._record_active_slices(session, active_slices)
            for item in active_slices:
                self._schedule_index_preload(item.index_path)
            session_updated = True
        result = {
            "success": True,
            "session_updated": session_updated,
            **selection.to_session_meta(),
            "imported_glossary_terms": selection.manual_terms,
        }
        if session_id and session_id in self.sessions:
            result["topic"] = self._topic_meta(self.sessions[session_id])
        return result

    async def _scheduler_loop(self) -> None:
        assert self._batch_gate is not None
        while True:
            try:
                await asyncio.sleep(float(self.config.batch_timeout))
                await self._batch_gate.acquire()
                batch: List[OmniSession] = []
                deadline = time.perf_counter() + max(0.0, float(self.config.coalesce_sec))
                while True:
                    while self.pending and len(batch) < int(self.config.scheduler_batch_size):
                        session_id = self.pending.popleft()
                        self.pending_set.discard(session_id)
                        session = self.sessions.get(session_id)
                        if session is None or session.inflight:
                            continue
                        ready = session.flush or (
                            session.cursor_samples - session.last_llm_samples >= self._segment_samples(session)
                        )
                        if not ready:
                            continue
                        session.inflight = True
                        batch.append(session)
                    if len(batch) >= int(self.config.scheduler_batch_size):
                        break
                    remaining_s = deadline - time.perf_counter()
                    if not batch or remaining_s <= 0:
                        break
                    await asyncio.sleep(min(float(self.config.batch_timeout), remaining_s))
                if batch:
                    asyncio.create_task(self._process_batch_guarded(batch))
                else:
                    self._batch_gate.release()
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 - scheduler must never die
                logger.exception("scheduler loop error")
                try:
                    self._batch_gate.release()
                except (ValueError, RuntimeError):
                    pass

    async def _process_batch_guarded(self, batch: List[OmniSession]) -> None:
        try:
            await self._process_batch(batch)
        finally:
            if self._batch_gate is not None:
                self._batch_gate.release()

    async def _process_batch(self, batch: List[OmniSession]) -> None:
        batch_t0 = time.perf_counter()
        self.batch_seq += 1
        batch_id = self.batch_seq
        queue_wait_values = [
            batch_t0 - session.pending_since_s
            for session in batch
            if session.pending_since_s is not None
        ]
        for session in batch:
            session.pending_since_s = None
        end_by_session = {session.session_id: session.cursor_samples for session in batch}
        start_by_session = {session.session_id: session.last_llm_samples for session in batch}
        increments = [
            np.asarray(
                session.audio[start_by_session[session.session_id] : end_by_session[session.session_id]],
                dtype=np.float32,
            )
            for session in batch
        ]
        refs_by_session: List[List[Dict[str, Any]]] = [[] for _ in batch]
        results: List[Dict[str, Any]] = []
        retrieve_s = 0.0
        generate_s = 0.0
        batch_error: Optional[str] = None
        try:
            retrieve_t0 = time.perf_counter()
            refs_by_session = await self._retrieve_batch(batch, end_by_session)
            retrieve_s = time.perf_counter() - retrieve_t0
            generate_t0 = time.perf_counter()
            if getattr(self.backend, "batched", False):
                results = await self._generate_batch(
                    batch, increments, refs_by_session, start_by_session, end_by_session,
                    retrieve_s=retrieve_s,
                )
            else:
                tasks = [
                    self._generate_one(
                        session,
                        increment,
                        refs,
                        start_by_session[session.session_id],
                        end_by_session[session.session_id],
                        retrieve_s=retrieve_s,
                    )
                    for session, increment, refs in zip(batch, increments, refs_by_session)
                ]
                results = await asyncio.gather(*tasks)
            generate_s = time.perf_counter() - generate_t0
        except Exception as exc:  # noqa: BLE001
            batch_error = str(exc)
            for session in batch:
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_ERROR,
                        text=batch_error,
                        meta={"error": batch_error},
                    )
                )
        finally:
            for session in batch:
                if session.session_id in self.sessions:
                    session.inflight = False
                    session.flush = False
                    self._mark_pending(session, force=False)

        request_elapsed_values = [
            float(item["elapsed_s"]) for item in results if item.get("elapsed_s") is not None
        ]
        metric = {
            "batch_id": batch_id,
            "batch_size": len(batch),
            "retrieve_s": round(retrieve_s, 4),
            "generate_s": round(generate_s, 4),
            "total_s": round(time.perf_counter() - batch_t0, 4),
            "queue_wait_avg_s": round(sum(queue_wait_values) / len(queue_wait_values), 4)
            if queue_wait_values
            else 0.0,
            "queue_wait_max_s": round(max(queue_wait_values), 4) if queue_wait_values else 0.0,
            "references_avg": round(sum(len(item) for item in refs_by_session) / len(refs_by_session), 4)
            if refs_by_session
            else 0.0,
            "generation_ok": sum(1 for item in results if item.get("ok")),
            "request_elapsed_avg_s": round(sum(request_elapsed_values) / len(request_elapsed_values), 4)
            if request_elapsed_values
            else 0.0,
            "request_elapsed_max_s": round(max(request_elapsed_values), 4)
            if request_elapsed_values
            else 0.0,
        }
        if batch_error:
            metric["error"] = batch_error[:500]
        self.recent_batch_metrics.append(metric)
        print("RASST_BATCH_METRIC " + json.dumps(metric, ensure_ascii=False), flush=True)

    async def _retrieve_batch(
        self,
        batch: Sequence[OmniSession],
        end_by_session: Dict[str, int],
    ) -> List[List[Dict[str, Any]]]:
        if not self.retrieval.enabled or not batch:
            return [[] for _ in batch]
        outputs: List[List[Dict[str, Any]]] = [[] for _ in batch]
        query_embeddings: List[Any] = [None for _ in batch]
        query_window_embeddings: List[Any] = [None for _ in batch]
        grouped: Dict[Tuple[str, int, Optional[float]], List[Tuple[int, OmniSession, RetrievalSlice]]] = {}

        for idx, session in enumerate(batch):
            session.last_rescue_triggered = False
            session.last_candidate_pool_count = 0
            slices = self._active_retrieval_slices(session)
            self._record_active_slices(session, slices)
            for plan in slices:
                if session.auto_glossary_enabled and not self.retrieval.is_index_ready(plan.index_path):
                    self._schedule_index_preload(plan.index_path)
                    continue
                key = (
                    plan.index_path,
                    self._retrieval_top_k_for(session),
                    self._retrieval_score_threshold_for(session),
                )
                grouped.setdefault(key, []).append((idx, session, plan))

        assert self._retriever_lock is not None
        async with self._retriever_lock:
            await self._retrieve_slice_groups(
                grouped,
                end_by_session,
                outputs,
                query_embeddings,
                query_window_embeddings,
            )

            rescue_grouped: Dict[Tuple[str, int, Optional[float]], List[Tuple[int, OmniSession, RetrievalSlice]]] = {}
            for idx, session in enumerate(batch):
                if session.auto_glossary_enabled:
                    ranked = rank_autoterm_references(outputs[idx], active_domain=session.active_domain)
                else:
                    ranked = self._rank_references(outputs[idx])
                outputs[idx] = ranked
                if not self._should_rescue_retrieval(session, ranked):
                    continue
                rescue = self._rescue_retrieval_slice(session)
                if rescue is None or rescue.preset_id in set(session.active_slice_presets):
                    continue
                if session.auto_glossary_enabled and not self.retrieval.is_index_ready(rescue.index_path):
                    self._schedule_index_preload(rescue.index_path)
                    continue
                session.last_rescue_triggered = True
                session.last_retrieval_plan.append(rescue.to_meta())
                key = (
                    rescue.index_path,
                    self._retrieval_top_k_for(session),
                    self._retrieval_score_threshold_for(session),
                )
                rescue_grouped.setdefault(key, []).append((idx, session, rescue))
            await self._retrieve_slice_groups(
                rescue_grouped,
                end_by_session,
                outputs,
                query_embeddings,
                query_window_embeddings,
            )

        for idx, session in enumerate(batch):
            if session.auto_glossary_enabled:
                outputs[idx] = rank_autoterm_references(outputs[idx], active_domain=session.active_domain)
            else:
                outputs[idx] = self._rank_references(outputs[idx])
            session.last_candidate_pool_count = len(outputs[idx])
            if session.auto_glossary_enabled:
                observer_refs = self._prompt_references(session, outputs[idx])
                probe_embedding = (
                    query_window_embeddings[idx]
                    if query_window_embeddings[idx] is not None
                    else query_embeddings[idx]
                )
                domain_probe_scores = await self._probe_domain_scores(
                    session,
                    end_sample=end_by_session[session.session_id],
                    query_embedding=probe_embedding,
                )
                await self._observe_active_glossary(
                    session,
                    RetrievalResult(references=observer_refs, query_embedding=query_embeddings[idx]),
                    domain_probe_scores=domain_probe_scores,
                )
            else:
                session.last_query_embedding = query_embeddings[idx]
        return outputs

    async def _retrieve_slice_groups(
        self,
        grouped: Dict[Tuple[str, int, Optional[float]], List[Tuple[int, OmniSession, RetrievalSlice]]],
        end_by_session: Dict[str, int],
        outputs: List[List[Dict[str, Any]]],
        query_embeddings: List[Any],
        query_window_embeddings: List[Any],
    ) -> None:
        for (index_path, top_k, score_threshold), indexed_sessions in grouped.items():
            await self.retrieval.activate_index(index_path)
            requests = [
                {
                    "audio_buffer": session.audio[: end_by_session[session.session_id]],
                    "current_start_sec": float(session.last_llm_samples) / TARGET_SAMPLE_RATE,
                    "current_end_sec": float(end_by_session[session.session_id]) / TARGET_SAMPLE_RATE,
                    "lookback_sec": float(self.config.rag_timeline_lookback_sec),
                    "return_query_window_embeddings": bool(session.auto_glossary_enabled),
                }
                for _, session, _ in indexed_sessions
            ]
            group_results = await self.retrieval.retrieve_with_metadata(
                requests,
                top_k=top_k,
                lookback_sec=self.config.rag_timeline_lookback_sec,
                score_threshold=score_threshold,
            )
            for (original_idx, session, plan), result in zip(indexed_sessions, group_results):
                if not isinstance(result, RetrievalResult):
                    result = RetrievalResult(references=list(result or []))
                if query_embeddings[original_idx] is None and result.query_embedding is not None:
                    query_embeddings[original_idx] = result.query_embedding
                if (
                    query_window_embeddings[original_idx] is None
                    and result.query_window_embeddings is not None
                ):
                    query_window_embeddings[original_idx] = result.query_window_embeddings
                for ref in result.references:
                    item = dict(ref)
                    item.setdefault("source_preset", plan.preset_id)
                    item["source_slice"] = plan.slice_id
                    item["source_slice_role"] = plan.role
                    item["source_domain"] = plan.domain
                    item["source_slice_weight"] = plan.weight
                    item["candidate_inventory_terms"] = plan.term_count
                    if session.auto_glossary_enabled:
                        item["source"] = f"auto:{plan.slice_id}"
                    outputs[original_idx].append(item)

    def _annotate_references(
        self,
        session: OmniSession,
        references: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        for ref in references or []:
            item = dict(ref)
            item["active_glossary_preset"] = session.active_glossary_preset or session.glossary_preset
            item["active_domain"] = session.active_domain
            item.setdefault("domain", item.get("source_domain") or session.active_domain)
            item.setdefault("source_preset", session.active_glossary_preset or session.glossary_preset)
            if session.auto_glossary_enabled:
                source = str(item.get("source") or "")
                if source in {"", "rag", "wikidata", "glossary"}:
                    item["source"] = f"auto:{item.get('source_slice') or item.get('source_preset') or session.active_glossary_preset}"
            elif not _is_empty_glossary_preset(session.glossary_preset):
                source = str(item.get("source") or "")
                if source in {"", "rag", "wikidata", "glossary"}:
                    item["source"] = f"preset:{session.glossary_preset}"
            annotated.append(item)
        if session.auto_glossary_enabled:
            return rank_autoterm_references(annotated, active_domain=session.active_domain)
        return self._rank_references(annotated)

    def _rank_references(self, references: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _priority(ref: Dict[str, Any]) -> Tuple[int, float]:
            source = str(ref.get("source") or "").lower()
            if source == "manual" or source.startswith("manual"):
                pri = 0
            elif source.startswith("preset:") or source.startswith("curated"):
                pri = 1
            elif source.startswith("auto:"):
                pri = 2
            elif source.startswith("open"):
                pri = 3
            else:
                pri = 4
            try:
                score = float(ref.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            return pri, -score

        return sorted((dict(ref) for ref in references or []), key=_priority)

    def _prompt_references(
        self,
        session: OmniSession,
        references: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        k = max(0, int(self.config.prompt_top_k))
        annotated = self._annotate_references(session, references)
        should_force_fixed_k = bool(
            session.auto_glossary_enabled
            or not _is_empty_glossary_preset(session.glossary_preset)
        )
        if should_force_fixed_k:
            prompt_refs = force_exactly_k_references(
                annotated,
                k=k,
                backfill=list(session.recent_references),
                active_domain=session.active_domain,
            )
        else:
            prompt_refs = annotated[:k]
        session.last_prompt_reference_count = len(prompt_refs)
        return prompt_refs

    def _ui_references(
        self,
        session: OmniSession,
        references: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._annotate_references(session, references)[: max(0, int(self.config.ui_top_k))]

    def _topic_meta(self, session: OmniSession) -> Dict[str, Any]:
        return glossary_topic_meta(
            active_domain=session.active_domain,
            confidence=session.topic_confidence,
            active_glossary_preset=session.active_glossary_preset,
            switch_count=session.glossary_switch_count,
            last_reason=session.last_topic_reason,
        )

    def _event_meta(
        self,
        session: OmniSession,
        *,
        segment_idx: int,
        elapsed_s: Optional[float],
        retrieve_s: Optional[float],
        cursor_samples: int,
        start_sample: int,
        references: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt_refs = self._prompt_references(session, references)
        ui_refs = self._ui_references(session, references)
        prompt_k = max(0, int(self.config.prompt_top_k))
        return {
            "segment_idx": segment_idx,
            "elapsed_s": elapsed_s,
            "retrieve_s": round(retrieve_s, 6) if retrieve_s is not None else None,
            "cursor_samples": cursor_samples,
            "start_sample": start_sample,
            "references": ui_refs,
            "prompt_reference_count": len(prompt_refs),
            "ui_reference_count": len(ui_refs),
            "fixed_prompt_k": prompt_k,
            "prompt_candidate_shortfall": max(0, prompt_k - len(prompt_refs)),
            "candidate_pool_count": int(session.last_candidate_pool_count or len(references or [])),
            "active_slices": list(session.active_slice_presets),
            "active_slice_terms": dict(session.active_slice_terms),
            "retrieval_plan": list(session.last_retrieval_plan),
            "open_wiki_rescue_triggered": bool(session.last_rescue_triggered),
            "domain_probe_s": (
                round(float(session.last_domain_probe_s), 6)
                if session.last_domain_probe_s is not None
                else None
            ),
            "domain_probe_cached": bool(session.last_domain_probe_cached),
            "domain_probe_scores": dict(session.last_domain_probe_scores),
            "domain_probe_slices": list(session.last_domain_probe_slices),
            "router_text_source": session.router_text_source,
            "router_text_chars": len(session.router_text_window or ""),
            "topic": self._topic_meta(session),
            "topic_router": session.last_router_decision,
        }

    def _after_translation_tick(
        self,
        session: OmniSession,
        *,
        text: str,
        references: Sequence[Dict[str, Any]],
    ) -> None:
        del references
        if not session.auto_glossary_enabled:
            return
        if (self.config.router_mode or "").strip().lower() != "hybrid_window_topic":
            return
        if not bool(self.config.router_generated_target_enabled):
            return
        current_source = str(getattr(session, "router_text_source", "none") or "none")
        if current_source not in {"none", "generated_target"}:
            return
        window = max(1, int(self.config.router_generated_target_window_chunks))
        texts = [str(item).strip() for item in session.history[-window:] if str(item).strip()]
        current_text = str(text or "").strip()
        if current_text and (not texts or texts[-1] != current_text):
            texts = (texts + [current_text])[-window:]
        router_text = "\n".join(texts).strip()
        if len(router_text) < max(1, int(self.config.router_generated_target_min_chars)):
            return
        session.router_text_window = router_text
        session.router_text_source = "generated_target"

    async def _observe_active_glossary(
        self,
        session: OmniSession,
        result: RetrievalResult,
        *,
        domain_probe_scores: Optional[Dict[str, DomainProbeScore]] = None,
    ) -> None:
        if not session.auto_glossary_enabled:
            return
        annotated_refs = self._annotate_references(session, result.references)
        for ref in annotated_refs:
            session.recent_references.append(ref)

        mode = (self.config.router_mode or "embedding_refs").strip().lower()
        if self.config.router_legacy_keywords or mode == "legacy_keywords":
            self._schedule_topic_update(session)
            return

        if session.router_state is None:
            session.router_state = RouterSessionState(
                active_preset_id=session.active_glossary_preset,
                active_domain_id=session.active_domain,
                created_s=session.created_s,
                last_decision_s=time.perf_counter(),
                last_switch_s=session.created_s,
            )
        now = time.perf_counter()
        router = self._topic_router_for(session.language_pair)
        if mode == "hybrid_window_topic":
            decision = router.observe(
                session.router_state,
                result.query_embedding,
                annotated_refs,
                now,
                router_text=session.router_text_window,
                router_text_source=session.router_text_source,
                domain_probe_scores=domain_probe_scores or {},
            )
        else:
            decision = router.observe(
                session.router_state,
                result.query_embedding,
                annotated_refs,
                now,
            )
        session.topic_confidence = decision.confidence
        session.last_topic_reason = decision.reason
        session.last_topic_update_s = now
        session.last_router_decision = decision.to_meta()
        session.topic_history.append(
            {
                "t": round(now - session.created_s, 3),
                "router_mode": mode,
                "router_text_source": session.router_text_source,
                "action": decision.action,
                "domain": decision.target_domain_id,
                "preset": decision.target_preset_id,
                "confidence": decision.confidence,
                "margin": decision.margin,
                "scores": decision.scores,
                "reason": decision.reason,
            }
        )
        session.topic_history = session.topic_history[-16:]
        if decision.action not in {"switch", "fallback"}:
            return
        if session.topic_update_task is not None and not session.topic_update_task.done():
            return
        session.topic_update_task = asyncio.create_task(
            self._apply_router_decision_guarded(session.session_id, decision)
        )

    def _schedule_topic_update(self, session: OmniSession) -> None:
        if not session.auto_glossary_enabled:
            return
        if session.topic_update_task is not None and not session.topic_update_task.done():
            return
        now = time.perf_counter()
        if now - session.created_s < float(self.config.auto_glossary_warmup_sec):
            return
        if now - session.last_topic_update_s < float(self.config.auto_glossary_update_sec):
            return
        session.topic_update_task = asyncio.create_task(self._update_active_glossary_guarded(session.session_id))

    async def _apply_router_decision_guarded(
        self,
        session_id: str,
        decision: RouterDecision,
    ) -> None:
        try:
            await self._apply_router_decision(session_id, decision)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - adaptive routing must never fail translation
            logger.exception("auto glossary switch failed for %s", session_id)

    async def _apply_router_decision(
        self,
        session_id: str,
        decision: RouterDecision,
    ) -> None:
        session = self.sessions.get(session_id)
        if session is None or not session.auto_glossary_enabled:
            return
        selection = self.active_glossary.selection_for_decision(
            self._catalog(session.language_pair),
            decision,
            glossary_text="",
            mock=self.config.mock,
        )
        if selection is None:
            session.last_topic_reason = f"{decision.reason}; target_unavailable"
            if session.router_state is not None:
                session.router_state.pending_preset_id = None
            return
        if selection.active_preset == session.active_glossary_preset:
            if session.router_state is not None:
                session.router_state.pending_preset_id = None
            return
        if selection.index_path and self.config.auto_glossary_preload:
            await self.retrieval.preload_index(selection.index_path)
            if not self.retrieval.is_index_ready(selection.index_path):
                session.last_topic_reason = f"{decision.reason}; preload_not_ready"
                if session.router_state is not None:
                    session.router_state.pending_preset_id = None
                return

        old_preset = session.active_glossary_preset
        old_domain = session.active_domain
        session.glossary_preset = selection.active_preset
        session.active_glossary_preset = selection.active_preset
        session.active_domain = selection.active_domain
        session.glossary_index_path = selection.index_path
        session.active_retrieval_slices = []
        active_slices = self._active_retrieval_slices(session)
        self._record_active_slices(session, active_slices)
        for item in active_slices:
            self._schedule_index_preload(item.index_path)
        session.glossary_switch_count += 1
        session.last_topic_reason = selection.reason
        if session.router_state is not None:
            session.router_state.active_preset_id = selection.active_preset
            session.router_state.active_domain_id = selection.active_domain
            session.router_state.last_switch_s = time.perf_counter()
            session.router_state.pending_preset_id = None
        meta = decision.to_meta()
        meta["from_preset"] = old_preset
        meta["from_domain"] = old_domain
        meta["to_preset"] = selection.active_preset
        meta["to_domain"] = selection.active_domain
        session.last_router_decision = meta
        self._emit_event(
            TranslationEvent(
                session_id=session.session_id,
                type=EVENT_STATUS,
                text=f"AUTO_GLOSSARY_SWITCH: {old_preset} -> {selection.active_preset}",
                meta={
                    "topic": self._topic_meta(session),
                    "topic_router": meta,
                    "active_glossary_preset": selection.active_preset,
                    "active_domain": selection.active_domain,
                    "active_terms": selection.preset_terms,
                    "active_slices": list(session.active_slice_presets),
                    "active_slice_terms": dict(session.active_slice_terms),
                },
            )
        )
        logger.info(
            "session %s switched active glossary to %s (domain=%s conf=%.3f)",
            session.session_id,
            selection.active_preset,
            selection.active_domain,
            decision.confidence,
        )

    async def _update_active_glossary_guarded(self, session_id: str) -> None:
        try:
            await self._update_active_glossary(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - adaptive routing must never fail translation
            logger.exception("auto glossary update failed for %s", session_id)

    async def _update_active_glossary(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None or not session.auto_glossary_enabled:
            return
        now = time.perf_counter()
        context = TopicContext(
            recent_text="\n".join(session.history[-int(self.config.keep_cache_chunks):]),
            recent_references=list(session.recent_references),
            manual_glossary_terms=list(session.imported_glossary),
            current_domain=session.active_domain,
            elapsed_s=now - session.created_s,
            seconds_since_update=now - session.last_topic_update_s,
        )
        decision = self._legacy_topic_router.decide(context)
        session.topic_confidence = decision.confidence
        session.last_topic_reason = decision.reason
        session.last_topic_update_s = now
        session.topic_history.append(
            {
                "t": round(now - session.created_s, 3),
                "domain": decision.primary_domain,
                "confidence": decision.confidence,
                "scores": decision.scores,
                "should_switch": decision.should_switch,
                "reason": decision.reason,
            }
        )
        session.topic_history = session.topic_history[-16:]
        if not decision.should_switch:
            return

        selection = self.active_glossary.selection_for_decision(
            self._catalog(session.language_pair),
            decision,
            glossary_text="",
            mock=self.config.mock,
        )
        if selection is None:
            session.last_topic_reason = f"{decision.reason}; target_unavailable"
            return
        if selection.active_preset == session.active_glossary_preset:
            return
        if selection.index_path:
            await self.retrieval.preload_index(selection.index_path)
            if not self.retrieval.is_index_ready(selection.index_path):
                session.last_topic_reason = f"{decision.reason}; preload_not_ready"
                return
        session.glossary_preset = selection.active_preset
        session.active_glossary_preset = selection.active_preset
        session.active_domain = selection.active_domain
        session.glossary_index_path = selection.index_path
        session.active_retrieval_slices = []
        active_slices = self._active_retrieval_slices(session)
        self._record_active_slices(session, active_slices)
        for item in active_slices:
            self._schedule_index_preload(item.index_path)
        session.glossary_switch_count += 1
        session.last_topic_reason = selection.reason
        logger.info(
            "session %s switched active glossary to %s (domain=%s conf=%.3f)",
            session.session_id,
            selection.active_preset,
            selection.active_domain,
            decision.confidence,
        )

    async def _generate_one(
        self,
        session: OmniSession,
        increment: np.ndarray,
        references: Sequence[Dict[str, Any]],
        start_sample: int,
        end_sample: int,
        *,
        retrieve_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        chunk_rms = float(np.sqrt(np.mean(np.square(increment)))) if increment.size else 0.0
        if chunk_rms < float(self.config.min_audio_rms):
            session.last_llm_samples = end_sample
            session.segment_idx += 1
            return {"ok": True, "elapsed_s": 0.0, "skipped": "silence"}

        seg_no = session.segment_idx + 1
        wav_path = self.tmp_dir / f"{session.session_id}_{seg_no:05d}.wav"
        if not self.config.mock:
            import soundfile as sf  # noqa: WPS433 - heavy import only on the real path

            sf.write(str(wav_path), increment, TARGET_SAMPLE_RATE)
            session.audio_paths.append(wav_path)

        rag_enabled_for_prompt = bool(
            self.retrieval.enabled and (session.glossary_index_path or session.imported_glossary)
        )
        if not session.messages:
            session.messages.append(
                self.prompt.system_message(session.source_lang, session.target_lang, rag_enabled_for_prompt)
            )
        prompt_refs = self._prompt_references(session, references)
        term_map_text = self.prompt.term_map(session.imported_glossary, prompt_refs)
        user_message, audios_payload = self.prompt.user_message(
            str(wav_path), term_map_text, rag_enabled_for_prompt
        )
        session.messages.append(user_message)

        sampling = Sampling(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            seed=self.config.seed,
        )
        request_id = f"{session.session_id}-{seg_no}"
        try:
            t0 = time.perf_counter()
            text = await self.backend.generate(
                messages=session.messages,
                audios=audios_payload,
                sampling=sampling,
                request_id=request_id,
            )
            elapsed = time.perf_counter() - t0
            session.last_llm_samples = end_sample
            session.segment_idx += 1
            session.messages.append({"role": "assistant", "content": text})
            self._trim_messages(session)
            if text:
                session.history.append(text)
                session.history = session.history[-int(self.config.keep_cache_chunks) :]
                self._after_translation_tick(session, text=text, references=references)
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_PARTIAL,
                        text=text,
                        meta=self._event_meta(
                            session,
                            segment_idx=session.segment_idx,
                            elapsed_s=round(elapsed, 6),
                            retrieve_s=retrieve_s,
                            cursor_samples=end_sample,
                            start_sample=start_sample,
                            references=references,
                        ),
                    )
                )
            return {"ok": True, "elapsed_s": elapsed}
        except Exception as exc:  # noqa: BLE001
            if session.messages and session.messages[-1] is user_message:
                session.messages.pop()
            self._emit_event(
                TranslationEvent(
                    session_id=session.session_id,
                    type=EVENT_ERROR,
                    text=str(exc),
                    meta={"error": str(exc)},
                )
            )
            return {"ok": False, "error": str(exc), "elapsed_s": None}

    async def _generate_batch(
        self,
        batch: Sequence[OmniSession],
        increments: Sequence[np.ndarray],
        refs_by_session: Sequence[Sequence[Dict[str, Any]]],
        start_by_session: Dict[str, int],
        end_by_session: Dict[str, int],
        *,
        retrieve_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Batched generate for vLLM-style backends: one engine call per tick.

        The backend owns multi-turn chat state + audio, so here we only assemble
        per-session ``{audio, term_map_text, rag_enabled}`` requests, issue a
        single ``generate_batch``, then emit + advance each session.
        """
        requests: List[Dict[str, Any]] = []
        for session, increment, refs in zip(batch, increments, refs_by_session):
            rag_enabled_for_prompt = bool(
                self.retrieval.enabled and (session.glossary_index_path or session.imported_glossary)
            )
            prompt_refs = self._prompt_references(session, refs)
            term_map_text = self.prompt.term_map(session.imported_glossary, prompt_refs)
            requests.append(
                {
                    "session_id": session.session_id,
                    "audio": np.asarray(increment, dtype=np.float32),
                    "term_map_text": term_map_text,
                    "rag_enabled": rag_enabled_for_prompt,
                }
            )
        sampling = Sampling(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            seed=self.config.seed,
        )
        assert self.backend is not None
        try:
            outputs = await self.backend.generate_batch(requests, sampling=sampling)
        except Exception as exc:  # noqa: BLE001 - whole-batch failure
            for session in batch:
                session.last_llm_samples = end_by_session[session.session_id]
                session.segment_idx += 1
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_ERROR,
                        text=str(exc),
                        meta={"error": str(exc)},
                    )
                )
            return [{"ok": False, "error": str(exc), "elapsed_s": None} for _ in batch]

        results: List[Dict[str, Any]] = []
        for session, refs, output in zip(batch, refs_by_session, outputs):
            end_sample = end_by_session[session.session_id]
            start_sample = start_by_session[session.session_id]
            session.last_llm_samples = end_sample
            session.segment_idx += 1
            ok = bool(output.get("ok"))
            text = str(output.get("text") or "")
            elapsed = output.get("elapsed_s")
            if ok and text:
                session.history.append(text)
                session.history = session.history[-int(self.config.keep_cache_chunks):]
                self._after_translation_tick(session, text=text, references=refs)
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_PARTIAL,
                        text=text,
                        meta=self._event_meta(
                            session,
                            segment_idx=session.segment_idx,
                            elapsed_s=elapsed,
                            retrieve_s=retrieve_s,
                            cursor_samples=end_sample,
                            start_sample=start_sample,
                            references=refs,
                        ),
                    )
                )
            elif not ok:
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_ERROR,
                        text=str(output.get("error") or "generation failed"),
                        meta={"error": output.get("error")},
                    )
                )
            results.append({"ok": ok, "elapsed_s": elapsed, "error": output.get("error")})
        return results

    def _trim_messages(self, session: OmniSession) -> None:
        max_cache = int(self.config.max_cache_chunks)
        keep_cache = int(self.config.keep_cache_chunks)
        if len(session.messages) >= 2 * max_cache + 1:
            session.messages = [session.messages[0]] + session.messages[-2 * keep_cache :]

    def _emit_event(self, event: TranslationEvent) -> None:
        if self._emit is not None:
            self._emit(event)

    def describe(self) -> Dict[str, Any]:
        catalog = self._catalog(self.config.language_pair)
        return {
            "models": [
                {
                    "id": self.name,
                    "label": self.name,
                    "default": True,
                    "backend": self.template.served_model_name,
                }
            ],
            "language_pairs": [
                {"id": key, "label": key, "available": self.config.mock or key == self.config.language_pair}
                for key in LANGUAGE_PAIRS
            ],
            "glossary_presets": catalog.preset_catalog(),
            "default_glossary_preset": DEFAULT_GLOSSARY_PRESET,
            "auto_glossary": {
                "enabled": self.config.auto_glossary_enabled,
                "router_mode": self.config.router_mode,
                "base_preset": self.config.auto_glossary_base_preset,
                "default_preset": self.config.auto_glossary_default_preset,
                "fallback_preset": self.config.auto_glossary_fallback_preset,
                "update_sec": self.config.auto_glossary_update_sec,
                "warmup_sec": self.config.auto_glossary_warmup_sec,
                "min_confidence": self.config.auto_glossary_min_conf,
                "min_margin": self.config.auto_glossary_switch_margin,
                "min_consistent_windows": self.config.auto_glossary_min_consistent_windows,
                "min_consistent_windows_with_text": self.config.router_min_consistent_windows_with_text,
                "min_consistent_windows_generated_target": self.config.router_min_consistent_windows_generated_target,
                "min_consistent_windows_audio_only": self.config.router_min_consistent_windows_audio_only,
                "text_topic_weight": self.config.router_text_topic_weight,
                "domain_probe_weight": self.config.router_domain_probe_weight,
                "domain_probe_top_k": self.config.router_domain_probe_top_k,
                "speech_centroid_weight": self.config.router_speech_centroid_weight,
                "metadata_prior_weight": self.config.router_metadata_prior_weight,
                "audio_probe_min_top_score": self.config.router_audio_probe_min_top_score,
                "audio_probe_min_raw_margin": self.config.router_audio_probe_min_raw_margin,
                "audio_probe_min_positive_domains": self.config.router_audio_probe_min_positive_domains,
                "generated_target_probe_min_top_score": self.config.router_generated_target_probe_min_top_score,
                "generated_target_probe_min_raw_margin": self.config.router_generated_target_probe_min_raw_margin,
                "generated_target_probe_min_positive_domains": self.config.router_generated_target_probe_min_positive_domains,
                "generated_target_enabled": self.config.router_generated_target_enabled,
                "generated_target_window_chunks": self.config.router_generated_target_window_chunks,
                "prompt_top_k": self.config.prompt_top_k,
                "ui_top_k": self.config.ui_top_k,
            },
            "loaded_language_pair": self.config.language_pair,
        }

    def _retrieve_latency_ms(self) -> Dict[str, Optional[float]]:
        """p50/p95 of recent per-tick retrieval time (ms). Ignores RAG-off ticks."""
        values = sorted(
            float(m["retrieve_s"])
            for m in self.recent_batch_metrics
            if m.get("retrieve_s")
        )
        if not values:
            return {"p50_retrieve_ms": None, "p95_retrieve_ms": None}

        def _pct(pct: float) -> float:
            idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
            return round(values[idx] * 1000.0, 3)

        return {"p50_retrieve_ms": _pct(50.0), "p95_retrieve_ms": _pct(95.0)}

    def _term_memory_status(self, rag_health: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the active terminology memory for ``/health`` (UI evidence).

        For the current MaxSim retriever this reflects the active glossary index;
        the two-stage open-memory backend (Phase D) populates the same shape with
        a manifest snapshot id and a larger ``active_terms``.
        """
        active_index = rag_health.get("active_index_path")
        manifest = self._catalog(self.config.language_pair).manifest
        snapshot_id = _env_str("RASST_TERM_MEMORY_SNAPSHOT", "")
        if not snapshot_id and manifest is not None:
            snapshot_id = manifest.snapshot_id
        if not snapshot_id and active_index:
            snapshot_id = Path(str(active_index)).stem
        status = {
            "enabled": bool(getattr(self.retrieval, "enabled", False)),
            "backend": rag_health.get("backend") or _env_str("RASST_RETRIEVAL_BACKEND", "maxsim"),
            "default_preset": DEFAULT_GLOSSARY_PRESET,
            "auto_glossary_enabled": self.config.auto_glossary_enabled,
            "router_mode": self.config.router_mode,
            "auto_default_preset": self.config.auto_glossary_default_preset,
            "auto_fallback_preset": self.config.auto_glossary_fallback_preset,
            "open_presets": self._catalog(self.config.language_pair).open_preset_ids(),
            "snapshot_id": snapshot_id,
            "manifest": (manifest.path if manifest is not None else _env_str("RASST_TERM_MEMORY_MANIFEST", "")),
            "active_index_path": active_index,
            "active_terms": rag_health.get("active_terms"),
            "sessions": {
                sid: {
                    "active_domain": sess.active_domain,
                    "confidence": round(sess.topic_confidence, 4),
                    "active_glossary_preset": sess.active_glossary_preset,
                    "switch_count": sess.glossary_switch_count,
                    "router": sess.last_router_decision,
                }
                for sid, sess in list(self.sessions.items())[:16]
            },
        }
        status.update(self._retrieve_latency_ms())
        return status

    def _batch_metric_summary(self) -> Dict[str, Any]:
        metrics = list(self.recent_batch_metrics)
        if not metrics:
            return {"count": 0, "recent": []}
        return {
            "count": len(metrics),
            "recent": metrics[-10:],
            "avg_total_s": round(sum(item["total_s"] for item in metrics) / len(metrics), 4),
            "avg_retrieve_s": round(sum(item["retrieve_s"] for item in metrics) / len(metrics), 4),
            "avg_generate_s": round(sum(item["generate_s"] for item in metrics) / len(metrics), 4),
            "max_total_s": round(max(item["total_s"] for item in metrics), 4),
        }

    async def health(self) -> Dict[str, Any]:
        backend_health = await self.backend.health() if self.backend is not None else {"status": "starting"}
        rag_health = await self.retrieval.health()
        backend_ok = self.config.mock or backend_health.get("status") in {"healthy", "ready"}
        rag_ok = (not self.config.rag_enabled) or rag_health.get("status") in {"ready", "disabled"}
        status = "healthy" if backend_ok and rag_ok else "starting"
        if backend_health.get("status") == "error" or rag_health.get("status") == "error":
            status = "error"
        return {
            "status": status,
            "backend": f"{self.template.backend_kind}:{self.template.served_model_name}",
            "model": self.name,
            "language_pair": self.config.language_pair,
            "supported_languages": list(LANGUAGE_PAIRS.keys()),
            "loaded_language_pair": self.config.language_pair,
            "active_sessions": len(self.sessions),
            "mock_mode": self.config.mock,
            "rag": rag_health,
            "term_memory": self._term_memory_status(rag_health),
            "backend_health": backend_health,
            "scheduler_batch_size": self.config.scheduler_batch_size,
            "batch_timeout": self.config.batch_timeout,
            "coalesce_sec": self.config.coalesce_sec,
            "base_segment_sec": self.config.base_segment_sec,
            "default_latency_multiplier": self.config.default_latency_multiplier,
            "segment_sec": self.config.segment_sec,
            "batch_metrics": self._batch_metric_summary(),
        }
