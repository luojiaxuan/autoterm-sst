"""InfiniSSTAgent: legacy paged-attention scheduler wrapped as an Agent.

This adapts the existing ``serve/scheduler.py`` (``LLMScheduler``) +
``serve/inference_engine.py`` (``MultiGPUInferenceEngine`` / ``InfiniSSTFaster``)
stack to the framework's :class:`~framework.agent.Agent` contract. It is the
"systems" black box: prefill/decode batching, KV-cache reuse, paged attention.

Mapping:
* ``submit_audio``  -> ``LLMScheduler.submit_request`` (one incremental chunk)
* ``result_callback`` (scheduler thread) -> ``emit(TranslationEvent)``
* ``on_control`` reset -> ``LLMScheduler.reset_session``
* ``close_session`` -> ``LLMScheduler.cleanup_session``

``RASST_DEMO_MOCK=1`` is honored by the underlying engine, so this agent runs
the full prefill/decode protocol with no GPU or model weights.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from framework.agent import (
    EVENT_ERROR,
    EVENT_PARTIAL,
    Agent,
    EmitFn,
    SessionInfo,
    TranslationEvent,
)

logger = logging.getLogger(__name__)

# Benign scheduler rejections that should not surface to the client as errors.
_BENIGN_SUBMIT_ERRORS = ("No new audio", "too short", "Empty audio")


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class InfiniSSTAgent(Agent):
    """Legacy InfiniSST scheduler/engine behind the Agent interface."""

    def __init__(self, name: str = "InfiniSST") -> None:
        self.name = name
        self.language_pair = os.environ.get("RASST_DEMO_LANGUAGE_PAIR", "English -> Chinese")
        self.gpu_id = int(os.environ.get("RASST_INFINISST_GPU", "0"))
        self.max_batch_size = int(os.environ.get("RASST_SCHEDULER_BATCH_SIZE", "32"))
        self.batch_timeout = float(os.environ.get("RASST_BATCH_TIMEOUT", "0.1"))
        self.mock = _env_bool("RASST_DEMO_MOCK", False)

        self._emit: Optional[EmitFn] = None
        self.scheduler: Any = None
        self.engine: Any = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.supported_languages: List[str] = [self.language_pair]

    # ---------------------------------------------------------------- lifecycle
    async def start(self, emit: EmitFn) -> None:
        self._emit = emit
        # Imported lazily so a missing torch / InfiniSST stack only disables this
        # agent (the router drops it) instead of breaking the whole framework.
        from serve.inference_engine import MultiGPUInferenceEngine
        from serve.scheduler import LLMScheduler

        gpu_language_map = {self.gpu_id: self.language_pair}
        model_args_map = {self.gpu_id: {}}
        self.engine = MultiGPUInferenceEngine(
            gpu_language_map=gpu_language_map,
            model_args_map=model_args_map,
        )
        if not self.engine.load_all_models():
            raise RuntimeError("InfiniSST engine failed to load models (set RASST_DEMO_MOCK=1 to mock)")
        self.engine.start_all()

        class _Args:
            max_batch_size = self.max_batch_size
            batch_timeout = self.batch_timeout
            session_timeout = 3600

        self.scheduler = LLMScheduler(gpu_language_map, _Args())
        self.scheduler.set_inference_engine(self.engine)
        self.scheduler.start()
        try:
            self.supported_languages = list(self.scheduler.get_supported_languages())
        except Exception:  # noqa: BLE001
            self.supported_languages = [self.language_pair]
        logger.info("InfiniSSTAgent %r started (mock=%s langs=%s)", self.name, self.mock, self.supported_languages)

    async def shutdown(self) -> None:
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception:  # noqa: BLE001
                logger.exception("scheduler stop failed")
        if self.engine is not None:
            try:
                self.engine.stop_all()
            except Exception:  # noqa: BLE001
                logger.exception("engine stop failed")

    # ------------------------------------------------------------------ session
    async def open_session(self, session_id: str, config: Dict[str, Any]) -> SessionInfo:
        from fastapi import HTTPException

        language_pair = str(config.get("language_pair") or self.language_pair)
        if language_pair not in self.supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"InfiniSST is not loaded for {language_pair}; supported: {self.supported_languages}",
            )
        latency_multiplier = int(config.get("latency_multiplier", 2) or 2)
        user_id = str(config.get("client_id") or session_id)
        self.sessions[session_id] = {
            "user_id": user_id,
            "language_pair": language_pair,
            "latency_multiplier": latency_multiplier,
        }
        return SessionInfo(
            admitted=True,
            queued=False,
            queue_position=0,
            meta={
                "scheduler_based": True,
                "rasst_backend": False,
                "mock_mode": self.mock,
                "model": self.name,
            },
        )

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None or self.scheduler is None:
            return
        try:
            self.scheduler.cleanup_session(session["user_id"], session["language_pair"])
        except Exception:  # noqa: BLE001
            logger.exception("cleanup_session failed for %s", session_id)

    # -------------------------------------------------------------------- audio
    async def submit_audio(self, session_id: str, pcm: np.ndarray, *, final: bool = False) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        chunk = np.asarray(pcm, dtype=np.float32).flatten()
        if chunk.size == 0:
            return  # EOF / heartbeat: nothing new to schedule
        if self.scheduler is None:
            self._emit_event(self._error_event(session_id, "scheduler not available"))
            return

        from serve.scheduler import RequestStage

        callback = self._make_callback(session_id)
        try:
            self.scheduler.submit_request(
                user_id=session["user_id"],
                language_id=session["language_pair"],
                speech_data=chunk,
                stage=RequestStage.PREFILL,
                is_final=final,
                max_new_tokens=int(session["latency_multiplier"]) * 10,
                result_callback=callback,
            )
        except ValueError as exc:
            message = str(exc)
            if any(token in message for token in _BENIGN_SUBMIT_ERRORS):
                return
            self._emit_event(self._error_event(session_id, message))
        except Exception as exc:  # noqa: BLE001
            self._emit_event(self._error_event(session_id, str(exc)))

    def _make_callback(self, session_id: str):
        def result_callback(result: Dict[str, Any]) -> None:
            try:
                if result.get("success", False):
                    full_translation = result.get("full_translation", "")
                    generated_text = result.get("generated_text", "")
                    text_to_send = full_translation if full_translation else generated_text
                    finished = bool(result.get("finished", False) or result.get("decode_finished", False))
                    if text_to_send or finished:
                        self._emit_event(
                            TranslationEvent(
                                session_id=session_id,
                                type=EVENT_PARTIAL,
                                text=text_to_send,
                                meta={
                                    "finished": finished,
                                    "segment_count": result.get("segment_count"),
                                    "new_segment": result.get("new_segment"),
                                },
                            )
                        )
                else:
                    self._emit_event(self._error_event(session_id, result.get("error", "Unknown error")))
            except Exception as exc:  # noqa: BLE001
                logger.exception("result callback failed for %s", session_id)
                self._emit_event(self._error_event(session_id, f"Callback failed - {exc}"))

        return result_callback

    # ------------------------------------------------------------------ control
    async def on_control(
        self, session_id: Optional[str], message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        mtype = message.get("type")
        if mtype == "reset":
            return self._reset(session_id)
        if mtype == "update_latency":
            session = self.sessions.get(session_id) if session_id else None
            if session is None:
                return {"success": False, "error": "Invalid session ID"}
            try:
                session["latency_multiplier"] = int(message.get("latency_multiplier", session["latency_multiplier"]))
            except (TypeError, ValueError):
                pass
            return {"success": True, "latency_multiplier": session["latency_multiplier"]}
        if mtype == "glossary_build":
            # Terminology/RAG is a RASST-only feature; accept as a no-op.
            return {
                "success": True,
                "session_updated": False,
                "glossary_preset": "none",
                "manual_terms": 0,
                "imported_glossary_terms": 0,
                "index_path": "",
                "index_ready": True,
                "note": "InfiniSST does not support terminology retrieval",
            }
        return None

    def _reset(self, session_id: Optional[str]) -> Dict[str, Any]:
        session = self.sessions.get(session_id) if session_id else None
        if session is None or self.scheduler is None:
            return {"success": False, "error": "Invalid session ID"}
        try:
            ok = bool(self.scheduler.reset_session(session["user_id"], session["language_pair"]))
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": str(exc)}
        return {"success": ok, "message": "Translation reset successfully.", "session_type": "scheduler"}

    # --------------------------------------------------------------- emit utils
    def _emit_event(self, event: TranslationEvent) -> None:
        if self._emit is not None:
            self._emit(event)

    @staticmethod
    def _error_event(session_id: str, message: str) -> TranslationEvent:
        return TranslationEvent(
            session_id=session_id, type=EVENT_ERROR, text=str(message), meta={"error": str(message)}
        )

    # --------------------------------------------------------- introspection
    def describe(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "id": self.name,
                    "label": "InfiniSST Legacy",
                    "default": False,
                    "backend": "legacy_infinisst_faster",
                }
            ],
            "language_pairs": [
                {"id": pair, "label": pair, "available": True} for pair in self.supported_languages
            ],
            "loaded_language_pair": self.language_pair,
        }

    async def health(self) -> Dict[str, Any]:
        running = bool(self.scheduler is not None and getattr(self.scheduler, "is_running", False))
        mock_mode = self.mock
        active = 0
        if self.scheduler is not None:
            try:
                active = int(self.scheduler.stats.get("active_sessions", 0))
                mock_mode = any(
                    getattr(engine, "mock_mode", False) for engine in self.engine.engines.values()
                ) if self.engine is not None else self.mock
            except Exception:  # noqa: BLE001
                pass
        return {
            "status": "healthy" if running else "starting",
            "backend": "legacy_infinisst_faster",
            "model": self.name,
            "scheduler_enabled": running,
            "scheduler_available": self.scheduler is not None,
            "supported_languages": list(self.supported_languages),
            "loaded_language_pair": self.language_pair,
            "active_sessions": active,
            "scheduler_sessions": active,
            "mock_mode": mock_mode,
        }
