"""AgentRouter: the framework's session lifecycle + output routing core.

The router is intentionally small. It:

* loads a set of agents and routes each session to one of them (``agent_type``),
* owns a per-session ``asyncio.Queue`` of :class:`TranslationEvent`,
* exposes a single thread/async-safe ``emit`` that agents call from anywhere,
* sweeps orphaned sessions (no ping within a timeout),
* aggregates ``describe()`` / ``health()`` across agents for ``/config`` and
  ``/health``.

It does not know anything about models, retrieval, batching, or prompting.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from framework.agent import Agent, SessionInfo, TranslationEvent

logger = logging.getLogger(__name__)

DEFAULT_ORPHAN_TIMEOUT_SEC = 300.0
ORPHAN_SWEEP_INTERVAL_SEC = 30.0


@dataclass
class SessionRecord:
    session_id: str
    agent_name: str
    queue: "asyncio.Queue[TranslationEvent]"
    config: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


class AgentRouter:
    def __init__(
        self,
        agents: Dict[str, Agent],
        default_agent: Optional[str] = None,
        orphan_timeout_sec: float = DEFAULT_ORPHAN_TIMEOUT_SEC,
    ) -> None:
        if not agents:
            raise ValueError("AgentRouter requires at least one agent")
        self.agents = agents
        self.default_agent = default_agent or next(iter(agents))
        if self.default_agent not in self.agents:
            raise ValueError(f"default_agent {self.default_agent!r} is not loaded")
        self.orphan_timeout_sec = orphan_timeout_sec

        self.sessions: Dict[str, SessionRecord] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sweep_task: Optional[asyncio.Task] = None
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        started: List[str] = []
        for name, agent in list(self.agents.items()):
            try:
                await agent.start(self._emit)
                started.append(name)
                logger.info("agent %r started", name)
            except Exception:  # noqa: BLE001 - one bad agent must not sink the rest
                logger.exception("agent %r failed to start; dropping it", name)
                self.agents.pop(name, None)
        if not self.agents:
            raise RuntimeError("no agents started successfully")
        if self.default_agent not in self.agents:
            self.default_agent = next(iter(self.agents))
        self._sweep_task = asyncio.create_task(self._orphan_sweep_loop())
        self._started = True
        logger.info("AgentRouter started: agents=%s default=%s", started, self.default_agent)

    async def shutdown(self) -> None:
        if self._sweep_task:
            self._sweep_task.cancel()
            await asyncio.gather(self._sweep_task, return_exceptions=True)
        for name, agent in list(self.agents.items()):
            try:
                await agent.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("agent %r shutdown failed", name)
        self.sessions.clear()
        self._started = False

    # --------------------------------------------------------------------- emit
    def _emit(self, event: TranslationEvent) -> None:
        """Thread/async-safe. Routes an event to its session output queue."""

        loop = self._loop
        if loop is None:
            return
        record = self.sessions.get(event.session_id)
        if record is None:
            return

        def _enqueue() -> None:
            current = self.sessions.get(event.session_id)
            if current is not None:
                current.queue.put_nowait(event)

        try:
            loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            # Event loop is closed/closing; drop the event.
            pass

    # ------------------------------------------------------------------ routing
    def _pick_agent(self, agent_type: Optional[str]) -> Tuple[str, Agent]:
        if agent_type and agent_type in self.agents:
            return agent_type, self.agents[agent_type]
        # Fall back to the default agent (single-agent deployments, or an
        # unknown/blank agent_type from the client).
        return self.default_agent, self.agents[self.default_agent]

    def _new_session_id(self, agent_name: str, config: Dict[str, Any]) -> str:
        client_id = str(config.get("client_id") or "")
        lang = str(config.get("language_pair") or "").replace(" ", "").replace("->", "2")
        suffix = client_id or uuid.uuid4().hex[:8]
        stamp = int(time.time() * 1000)
        token = uuid.uuid4().hex[:6]
        parts = [agent_name]
        if lang:
            parts.append(lang)
        parts.extend([suffix, f"{stamp}{token}"])
        return "_".join(parts)

    async def open_session(self, config: Dict[str, Any]) -> Tuple[str, SessionInfo]:
        agent_type = config.get("agent_type")
        agent_name, agent = self._pick_agent(agent_type)
        session_id = self._new_session_id(agent_name, config)

        record = SessionRecord(
            session_id=session_id,
            agent_name=agent_name,
            queue=asyncio.Queue(),
            config=dict(config),
        )
        self.sessions[session_id] = record
        try:
            info = await agent.open_session(session_id, dict(config))
        except Exception:
            self.sessions.pop(session_id, None)
            raise
        record.meta = dict(info.meta or {})
        return session_id, info

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        return self.sessions.get(session_id)

    def _agent_for(self, session_id: str) -> Optional[Agent]:
        record = self.sessions.get(session_id)
        if record is None:
            return None
        return self.agents.get(record.agent_name)

    async def submit_audio(self, session_id: str, pcm: np.ndarray, *, final: bool = False) -> None:
        record = self.sessions.get(session_id)
        agent = self.agents.get(record.agent_name) if record else None
        if record is None or agent is None:
            raise KeyError(session_id)
        record.last_activity = time.time()
        record.last_ping = time.time()
        await agent.submit_audio(session_id, pcm, final=final)

    async def on_control(self, session_id: Optional[str], message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if session_id:
            record = self.sessions.get(session_id)
            agent = self.agents.get(record.agent_name) if record else None
            if record is None or agent is None:
                raise KeyError(session_id)
            record.last_activity = time.time()
            return await agent.on_control(session_id, message)
        # Agent-global control (no session): use the default agent.
        agent = self.agents[self.default_agent]
        return await agent.on_control(None, message)

    async def close_session(self, session_id: str) -> bool:
        record = self.sessions.pop(session_id, None)
        if record is None:
            return False
        agent = self.agents.get(record.agent_name)
        if agent is not None:
            try:
                await agent.close_session(session_id)
            except Exception:  # noqa: BLE001
                logger.exception("agent %r close_session failed for %s", record.agent_name, session_id)
        return True

    def touch_ping(self, session_id: str) -> bool:
        record = self.sessions.get(session_id)
        if record is None:
            return False
        now = time.time()
        record.last_ping = now
        record.last_activity = now
        return True

    # --------------------------------------------------------- introspection
    def queue_status(self, session_id: str) -> Dict[str, Any]:
        record = self.sessions.get(session_id)
        if record is None:
            return {"session_id": session_id, "status": "not_found", "queued": False, "queue_position": 0}
        status = "queued" if record.meta.get("queued") else "active"
        return {
            "session_id": session_id,
            "status": status,
            "queued": bool(record.meta.get("queued", False)),
            "queue_position": int(record.meta.get("queue_position", 0)),
            "agent": record.agent_name,
            **{k: v for k, v in record.meta.items() if k not in {"queued", "queue_position"}},
        }

    def config(self) -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        language_pairs: List[Dict[str, Any]] = []
        glossary_presets: List[Dict[str, Any]] = []
        seen_models: set = set()
        seen_langs: Dict[str, Dict[str, Any]] = {}
        seen_presets: set = set()
        default_model: Optional[str] = None
        default_glossary_preset = "none"
        loaded_language_pair: Optional[str] = None

        for agent in self.agents.values():
            desc = agent.describe() or {}
            for model in desc.get("models", []) or []:
                mid = model.get("id")
                if mid and mid not in seen_models:
                    seen_models.add(mid)
                    models.append(model)
                    if model.get("default") and default_model is None:
                        default_model = mid
            for lang in desc.get("language_pairs", []) or []:
                lid = lang.get("id")
                if not lid:
                    continue
                if lid in seen_langs:
                    if lang.get("available"):
                        seen_langs[lid]["available"] = True
                else:
                    entry = dict(lang)
                    seen_langs[lid] = entry
                    language_pairs.append(entry)
            for preset in desc.get("glossary_presets", []) or []:
                pid = preset.get("id")
                if pid and pid not in seen_presets:
                    seen_presets.add(pid)
                    glossary_presets.append(preset)
            if desc.get("default_glossary_preset"):
                default_glossary_preset = desc["default_glossary_preset"]
            if desc.get("loaded_language_pair") and loaded_language_pair is None:
                loaded_language_pair = desc["loaded_language_pair"]

        if default_model is None and models:
            default_model = models[0]["id"]
        return {
            "models": models,
            "language_pairs": language_pairs,
            "glossary_presets": glossary_presets,
            "default_model": default_model,
            "default_glossary_preset": default_glossary_preset,
            "loaded_language_pair": loaded_language_pair,
            "agents": list(self.agents.keys()),
        }

    async def health(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "status": "healthy",
            "time": int(time.time()),
            "active_sessions": len(self.sessions),
            "agents": list(self.agents.keys()),
            "default_agent": self.default_agent,
        }
        agent_health: Dict[str, Any] = {}
        for name, agent in self.agents.items():
            try:
                h = await agent.health()
            except Exception as exc:  # noqa: BLE001
                h = {"status": "error", "error": str(exc)}
            agent_health[name] = h
            # Merge top-level keys so legacy fields (scheduler_enabled,
            # supported_languages, mock_mode...) remain available at the root.
            for key, value in h.items():
                if key not in merged:
                    merged[key] = value
        merged["agent_health"] = agent_health
        return merged

    # ---------------------------------------------------------- orphan sweep
    async def _orphan_sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(ORPHAN_SWEEP_INTERVAL_SEC)
                now = time.time()
                stale = [
                    sid
                    for sid, record in self.sessions.items()
                    if now - record.last_ping > self.orphan_timeout_sec
                ]
                for sid in stale:
                    logger.info("closing orphaned session %s", sid)
                    await self.close_session(sid)
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001
                logger.exception("orphan sweep error")
