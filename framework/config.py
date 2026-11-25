"""Config-driven agent loading.

The framework loads a set of agents named in ``RASST_FRAMEWORK_AGENTS`` (or the
``--agents`` flag) and routes ``/init``'s ``agent_type`` to one of them. Each
agent is constructed lazily so a heavy/optional backend (e.g. the omni agent's
SGLang dependency) cannot prevent the others from loading. Agents that fail to
*start* are dropped by the router at startup.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

from framework.agent import Agent
from framework.router import AgentRouter

logger = logging.getLogger(__name__)

DEFAULT_AGENTS = "InfiniSST,RASST"


def _make_infinisst(name: str) -> Agent:
    from framework.agents.infinisst import InfiniSSTAgent

    return InfiniSSTAgent(name=name)


def _make_omni(name: str, model_id: str) -> Callable[[], Agent]:
    def factory() -> Agent:
        from framework.agents.omni import OmniAgent

        return OmniAgent(name=name, model_id=model_id)

    return factory


# Maps an ``agent_type`` (as sent by the UI / smoke test) to a builder.
AGENT_FACTORIES: Dict[str, Callable[[], Agent]] = {
    "InfiniSST": lambda: _make_infinisst("InfiniSST"),
    "RASST": _make_omni("RASST", model_id="qwen3_omni"),
    # Model-extension entries (omni agent + per-model template). These only
    # appear if explicitly requested via RASST_FRAMEWORK_AGENTS.
    "Qwen3-Omni": _make_omni("Qwen3-Omni", model_id="qwen3_omni"),
    "MiniCPM-o": _make_omni("MiniCPM-o", model_id="minicpm_o"),
}


def _requested_agents(explicit: Optional[str]) -> List[str]:
    raw = explicit if explicit is not None else os.environ.get("RASST_FRAMEWORK_AGENTS", DEFAULT_AGENTS)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or DEFAULT_AGENTS.split(",")


def build_router(
    agents: Optional[str] = None,
    default_agent: Optional[str] = None,
) -> AgentRouter:
    requested = _requested_agents(agents)
    built: Dict[str, Agent] = {}
    for name in requested:
        factory = AGENT_FACTORIES.get(name)
        if factory is None:
            logger.warning("unknown agent %r requested; skipping (known: %s)", name, list(AGENT_FACTORIES))
            continue
        try:
            built[name] = factory()
            logger.info("constructed agent %r", name)
        except Exception:  # noqa: BLE001 - construction must be resilient
            logger.exception("failed to construct agent %r; skipping", name)
    if not built:
        raise RuntimeError(f"no agents could be constructed from {requested!r}")

    chosen_default = default_agent or os.environ.get("RASST_FRAMEWORK_DEFAULT_AGENT")
    if chosen_default not in built:
        # Prefer the first requested-and-built agent for a stable default.
        chosen_default = next((n for n in requested if n in built), next(iter(built)))
    return AgentRouter(built, default_agent=chosen_default)
