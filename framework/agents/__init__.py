"""Concrete agents that conform to ``framework.agent.Agent``.

These are the customizable black boxes below the thin middle layer. Each agent
owns its model, prompting, batching, KV-cache, and (optional) retrieval through
the agent-internal plugin conventions in ``framework.agents.plugins``.
"""
