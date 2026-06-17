"""Agent-internal plugin conventions.

These are plain Python interfaces used *inside* agents. The framework does not
know about them -- they exist so that an agent can compose a model backend, an
optional retrieval source, and a prompt builder without re-implementing the
plumbing each time.

* :mod:`framework.agents.plugins.backends`  -- ModelBackend (vLLM, SGLang HTTP, mock)
* :mod:`framework.agents.plugins.retrieval`  -- RetrievalPlugin (MaxSim, null)
* :mod:`framework.agents.plugins.prompt`     -- PromptBuilder (glossary/term_map)
"""

from framework.agents.plugins.backends import (
    MODEL_TEMPLATES,
    HFBackend,
    ModelBackend,
    ModelTemplate,
    MockBackend,
    Sampling,
    SGLangHTTPBackend,
    VLLMBackend,
    build_backend,
    get_template,
)
from framework.agents.plugins.prompt import (
    PromptBuilder,
    format_term_map,
    merge_references,
    parse_glossary_text,
)
from framework.agents.plugins.retrieval import (
    MaxSimRetrievalPlugin,
    NullRetrieval,
    RetrievalPlugin,
)

__all__ = [
    "MODEL_TEMPLATES",
    "HFBackend",
    "ModelBackend",
    "ModelTemplate",
    "MockBackend",
    "Sampling",
    "SGLangHTTPBackend",
    "VLLMBackend",
    "build_backend",
    "get_template",
    "PromptBuilder",
    "format_term_map",
    "merge_references",
    "parse_glossary_text",
    "MaxSimRetrievalPlugin",
    "NullRetrieval",
    "RetrievalPlugin",
]
