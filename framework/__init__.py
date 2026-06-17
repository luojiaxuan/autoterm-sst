"""RASST-Demo streaming speech-translation framework.

A maximally-thin middle layer: it accepts user audio/control over the existing
WS/REST protocol, manages session lifecycle, and routes each session to a
pluggable, architecture-agnostic ``Agent``. Everything else -- model, prompting,
batching, KV-cache, and retrieval/glossary -- lives inside the agent as an
agent-internal concern.

See ``framework/agent.py`` for the contract an agent must implement.
"""

from framework.agent import Agent, EmitFn, SessionInfo, TranslationEvent

__all__ = ["Agent", "EmitFn", "SessionInfo", "TranslationEvent"]
