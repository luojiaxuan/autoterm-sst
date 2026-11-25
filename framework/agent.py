"""The framework <-> agent contract.

The contract is deliberately architecture-agnostic. The framework only knows how
to:

1. ask an agent to ``open_session`` / ``close_session``,
2. push audio (``submit_audio``) and opaque control messages (``on_control``),
3. receive output through a single thread-safe ``emit`` callback.

It knows nothing about models, prompting, batching, KV-cache, or retrieval. An
agent may be a local paged-attention model (InfiniSST) or an HTTP-served omni
model (Qwen3-Omni / MiniCPM-o); both satisfy the same interface.

All lifecycle methods are ``async`` because the transport layer is async. Agents
that are internally threaded (e.g. the InfiniSST scheduler) simply do quick,
non-blocking work inside these coroutines and deliver results later through the
thread-safe ``emit`` callback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


# Event ``type`` values understood by the transport layer.
EVENT_PARTIAL = "partial"  # incremental / cumulative translation text
EVENT_FINAL = "final"      # terminal translation text for a turn
EVENT_STATUS = "status"    # human-readable status string (sent verbatim)
EVENT_ERROR = "error"      # error string (sent to the client as ``ERROR: ...``)


@dataclass
class TranslationEvent:
    """A single output produced by an agent for a session.

    ``meta`` is opaque to the framework. Agents may attach retrieved terms,
    latency, queue position, etc.; the transport layer does not interpret it
    (it is reserved for richer clients without changing the wire protocol).
    """

    session_id: str
    type: str = EVENT_PARTIAL
    text: str = ""
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SessionInfo:
    """Returned by :meth:`Agent.open_session`.

    ``meta`` is merged verbatim into the ``/init`` JSON response, which lets an
    agent surface backend-specific fields (e.g. ``scheduler_based``,
    ``rasst_backend``, glossary metadata) without the framework needing to know
    about them.
    """

    admitted: bool = True
    queued: bool = False
    queue_position: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# A thread/async-safe callback the framework hands to each agent. An agent may
# call it from any thread or coroutine; the framework routes the event to the
# correct session output queue.
EmitFn = Callable[[TranslationEvent], None]


class Agent(ABC):
    """Architecture-agnostic streaming translation agent (the black box)."""

    #: Stable identifier used for ``agent_type`` routing and ``/config``.
    name: str = "agent"

    def describe(self) -> Dict[str, Any]:
        """Static capabilities contributed to ``GET /config``.

        Typical keys: ``models``, ``language_pairs``, ``glossary_presets``.
        """

        return {}

    async def health(self) -> Dict[str, Any]:
        """Dynamic health contributed to ``GET /health`` (merged across agents)."""

        return {}

    @abstractmethod
    async def start(self, emit: EmitFn) -> None:
        """Called once at framework startup. Store ``emit`` and warm up."""

    @abstractmethod
    async def open_session(self, session_id: str, config: Dict[str, Any]) -> SessionInfo:
        """Create per-session state. ``config`` is the raw ``/init`` payload."""

    @abstractmethod
    async def submit_audio(self, session_id: str, pcm: np.ndarray, *, final: bool = False) -> None:
        """Feed a chunk of float32 PCM (mono, 16 kHz). ``final`` marks end-of-input."""

    async def on_control(self, session_id: Optional[str], message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle an opaque control message (latency change, reset, glossary...).

        ``session_id`` may be ``None`` for agent-global control (e.g. building a
        glossary index before any session exists). Returns an optional JSON-able
        result that the framework forwards to the caller.
        """

        return None

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Tear down per-session state and free resources."""

    async def shutdown(self) -> None:
        """Called once at framework shutdown."""

        return None
