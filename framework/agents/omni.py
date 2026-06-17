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
from framework.agents.plugins.retrieval import MaxSimRetrievalPlugin, NullRetrieval, RetrievalPlugin

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


@dataclass
class OmniConfig:
    """Self-contained, env-driven config (defaults match the RASST server)."""

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

    max_imported_glossary_terms: int = 10000
    tmp_dir: str = ""

    @classmethod
    def from_env(cls, template: ModelTemplate) -> "OmniConfig":
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
    pending_since_s: Optional[float] = None


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
        self.tmp_dir = Path(self.config.tmp_dir)

    def _catalog(self, language_pair: str) -> GlossaryCatalog:
        catalog = self._catalogs.get(language_pair)
        if catalog is None:
            catalog = GlossaryCatalog(language_pair, self.config.max_imported_glossary_terms)
            self._catalogs[language_pair] = catalog
        return catalog

    def _segment_samples(self) -> int:
        return int(float(self.config.segment_sec) * TARGET_SAMPLE_RATE)

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
        else:
            self.retrieval = NullRetrieval()

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
            selection = catalog.describe_selection(
                config.get("glossary_preset", DEFAULT_GLOSSARY_PRESET),
                str(config.get("glossary_text") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if self.config.rag_enabled and selection["index_path"] and not selection["index_ready"]:
            raise HTTPException(
                status_code=400, detail=f"RAG text index is not ready: {selection['index_path']}"
            )

        lang_cfg = LANGUAGE_PAIRS[language_pair]
        session = OmniSession(
            session_id=session_id,
            language_pair=language_pair,
            source_lang=str(lang_cfg["source_lang"]),
            target_lang=str(lang_cfg["target_lang"]),
            lang_code=str(lang_cfg["lang_code"]),
            imported_glossary=selection["manual_refs"],
            glossary_preset=selection["glossary_preset"],
            glossary_index_path=selection["index_path"],
        )
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
            "glossary_preset": selection["glossary_preset"],
            "glossary_path": selection["glossary_path"],
            "preset_terms": selection["preset_terms"],
            "manual_terms": selection["manual_terms"],
            "glossary_terms": selection["manual_terms"],
            "index_path": selection["index_path"],
            "index_ready": selection["index_ready"],
        }
        return SessionInfo(admitted=True, queued=False, queue_position=0, meta=meta)

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
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
        if not force and (session.cursor_samples - session.last_llm_samples < self._segment_samples()):
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
            return {"success": True, "latency_multiplier": message.get("latency_multiplier")}
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
            selection = catalog.describe_selection(
                message.get("glossary_preset", DEFAULT_GLOSSARY_PRESET),
                str(message.get("glossary_text") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if self.config.rag_enabled and selection["index_path"] and not selection["index_ready"]:
            raise HTTPException(
                status_code=400, detail=f"RAG text index is not ready: {selection['index_path']}"
            )
        if self.retrieval.enabled and selection["index_path"]:
            try:
                await self.retrieval.activate_index(selection["index_path"])
            except Exception:  # noqa: BLE001
                logger.exception("failed to warm glossary index %s", selection["index_path"])

        session_updated = False
        session_id = message.get("session_id")
        if session_id:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=400, detail=f"Unknown session: {session_id}")
            session.glossary_preset = selection["glossary_preset"]
            session.glossary_index_path = selection["index_path"]
            session.imported_glossary = selection["manual_refs"]
            session_updated = True
        return {
            "success": True,
            "session_updated": session_updated,
            "glossary_preset": selection["glossary_preset"],
            "glossary_path": selection["glossary_path"],
            "preset_terms": selection["preset_terms"],
            "manual_terms": selection["manual_terms"],
            "imported_glossary_terms": selection["manual_terms"],
            "index_path": selection["index_path"],
            "index_ready": selection["index_ready"],
        }

    async def _scheduler_loop(self) -> None:
        assert self._batch_gate is not None
        while True:
            try:
                await asyncio.sleep(float(self.config.batch_timeout))
                await self._batch_gate.acquire()
                batch: List[OmniSession] = []
                deadline = time.perf_counter() + max(0.0, float(self.config.coalesce_sec))
                segment_samples = self._segment_samples()
                while True:
                    while self.pending and len(batch) < int(self.config.scheduler_batch_size):
                        session_id = self.pending.popleft()
                        self.pending_set.discard(session_id)
                        session = self.sessions.get(session_id)
                        if session is None or session.inflight:
                            continue
                        ready = session.flush or (
                            session.cursor_samples - session.last_llm_samples >= segment_samples
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
                    batch, increments, refs_by_session, start_by_session, end_by_session
                )
            else:
                tasks = [
                    self._generate_one(
                        session,
                        increment,
                        refs,
                        start_by_session[session.session_id],
                        end_by_session[session.session_id],
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
        grouped: Dict[str, List[Tuple[int, OmniSession]]] = {}
        for idx, session in enumerate(batch):
            index_path = session.glossary_index_path or self._catalog(
                session.language_pair
            ).index_path_for_preset(session.glossary_preset)
            if not index_path:
                continue
            grouped.setdefault(index_path, []).append((idx, session))

        assert self._retriever_lock is not None
        async with self._retriever_lock:
            for index_path, indexed_sessions in grouped.items():
                await self.retrieval.activate_index(index_path)
                requests = [
                    {
                        "audio_buffer": session.audio[: end_by_session[session.session_id]],
                        "current_start_sec": float(session.last_llm_samples) / TARGET_SAMPLE_RATE,
                        "current_end_sec": float(end_by_session[session.session_id]) / TARGET_SAMPLE_RATE,
                        "lookback_sec": float(self.config.rag_timeline_lookback_sec),
                    }
                    for _, session in indexed_sessions
                ]
                group_results = await self.retrieval.retrieve(
                    requests,
                    top_k=self.config.rag_top_k,
                    lookback_sec=self.config.rag_timeline_lookback_sec,
                )
                for (original_idx, _), refs in zip(indexed_sessions, group_results):
                    outputs[original_idx] = refs
        return outputs

    async def _generate_one(
        self,
        session: OmniSession,
        increment: np.ndarray,
        references: Sequence[Dict[str, Any]],
        start_sample: int,
        end_sample: int,
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
            self.config.rag_enabled and (session.glossary_index_path or session.imported_glossary)
        )
        if not session.messages:
            session.messages.append(
                self.prompt.system_message(session.source_lang, session.target_lang, rag_enabled_for_prompt)
            )
        term_map_text = self.prompt.term_map(session.imported_glossary, references)
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
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_PARTIAL,
                        text=text,
                        meta={
                            "segment_idx": session.segment_idx,
                            "elapsed_s": round(elapsed, 6),
                            "cursor_samples": end_sample,
                            "start_sample": start_sample,
                            "references": list(references),
                        },
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
    ) -> List[Dict[str, Any]]:
        """Batched generate for vLLM-style backends: one engine call per tick.

        The backend owns multi-turn chat state + audio, so here we only assemble
        per-session ``{audio, term_map_text, rag_enabled}`` requests, issue a
        single ``generate_batch``, then emit + advance each session.
        """
        requests: List[Dict[str, Any]] = []
        for session, increment, refs in zip(batch, increments, refs_by_session):
            rag_enabled_for_prompt = bool(
                self.config.rag_enabled and (session.glossary_index_path or session.imported_glossary)
            )
            term_map_text = self.prompt.term_map(session.imported_glossary, refs)
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
                self._emit_event(
                    TranslationEvent(
                        session_id=session.session_id,
                        type=EVENT_PARTIAL,
                        text=text,
                        meta={
                            "segment_idx": session.segment_idx,
                            "elapsed_s": elapsed,
                            "cursor_samples": end_sample,
                            "start_sample": start_sample,
                            "references": list(refs),
                        },
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
            "loaded_language_pair": self.config.language_pair,
        }

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
            "backend_health": backend_health,
            "scheduler_batch_size": self.config.scheduler_batch_size,
            "batch_timeout": self.config.batch_timeout,
            "coalesce_sec": self.config.coalesce_sec,
            "segment_sec": self.config.segment_sec,
            "batch_metrics": self._batch_metric_summary(),
        }
