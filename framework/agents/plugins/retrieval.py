"""RetrievalPlugin: optional, agent-internal terminology retrieval.

Retrieval is NOT a framework concern. An agent may compose a ``RetrievalPlugin``
to enrich its prompt with a ``term_map`` (this is the RASST behavior), or use
:class:`NullRetrieval` to disable it entirely ("retrieval is optional").

``MaxSimRetrievalPlugin`` wraps the external ``StreamingMaxSimRetriever`` from
the RASST repo, loaded via the existing ``sys.path`` shim. It supports switching
between precomputed glossary indexes (the demo's glossary presets).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

TermRef = Dict[str, Any]

# Mirrors serve/rasst_sglang_server.py defaults so the external retriever resolves.
RASST_ROOT = Path(os.environ.get("RASST_ROOT", "/mnt/taurus/data2/jiaxuanluo/RASST"))
RASST_CODE_ROOT = Path(os.environ.get("RASST_ACTIVE_CODE_ROOT", RASST_ROOT / "code/rasst"))
RASST_EVAL_ROOT = RASST_CODE_ROOT / "eval"


def add_rasst_paths() -> None:
    for path in (RASST_EVAL_ROOT, RASST_CODE_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


class RetrievalPlugin:
    """Interface an agent uses for terminology retrieval."""

    enabled: bool = False

    async def start(self) -> None:  # noqa: D401 - simple lifecycle hook
        return None

    async def activate_index(self, index_path: str) -> None:
        return None

    async def retrieve(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
    ) -> List[List[TermRef]]:
        """Return per-request term lists (same order as ``requests``)."""

        return [[] for _ in requests]

    async def health(self) -> Dict[str, Any]:
        return {"status": "disabled"}

    async def stop(self) -> None:
        return None


class NullRetrieval(RetrievalPlugin):
    """No-op retrieval: the agent runs without a term_map."""

    enabled = False


class MaxSimRetrievalPlugin(RetrievalPlugin):
    """Streaming MaxSim retriever (RASST) loaded from the external repo."""

    enabled = True

    def __init__(
        self,
        *,
        model_path: str,
        index_path: str,
        device: str = "cuda:1",
        top_k: int = 10,
        lora_rank: int = 128,
        text_lora_rank: int = 128,
        target_lang: str = "zh",
        score_threshold: float = 0.78,
    ) -> None:
        self.model_path = model_path
        self.index_path = index_path
        self.device = device
        self.top_k = top_k
        self.lora_rank = lora_rank
        self.text_lora_rank = text_lora_rank
        self.target_lang = target_lang
        self.score_threshold = score_threshold

        self.retriever: Any = None
        self._lock = asyncio.Lock()
        self._active_index_path = ""
        self._text_index_cache: Dict[str, Dict[str, Any]] = {}
        self.status: Dict[str, Any] = {"status": "disabled"}

    async def start(self) -> None:
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        add_rasst_paths()
        from agents.streaming_maxsim_retriever import (  # noqa: WPS433
            MAXSIM_STRIDE,
            MAXSIM_WINDOWS,
            StreamingMaxSimRetriever,
        )

        self.status = {
            "status": "loading",
            "device": self.device,
            "model_path": self.model_path,
            "index_path": self.index_path,
        }
        self.retriever = StreamingMaxSimRetriever(
            model_path=self.model_path,
            index_path=self.index_path,
            device=self.device,
            top_k=int(self.top_k),
            lora_rank=int(self.lora_rank),
            text_lora_rank=int(self.text_lora_rank),
            target_lang=self.target_lang,
            window_sec=0.0,
            score_threshold=float(self.score_threshold),
            maxsim_windows=MAXSIM_WINDOWS,
            maxsim_stride=MAXSIM_STRIDE,
        )
        self._active_index_path = str(Path(self.index_path))
        self._text_index_cache[self._active_index_path] = {
            "text_embs": self.retriever.text_embs,
            "term_list": self.retriever.term_list,
        }
        self.status["active_index_path"] = self._active_index_path
        self.status["active_terms"] = len(self.retriever.term_list)
        self.status["status"] = "ready"

    def _ensure_index(self, index_path: str) -> Dict[str, Any]:
        normalized = str(Path(index_path))
        if normalized in self._text_index_cache:
            return self._text_index_cache[normalized]
        if self.retriever is None:
            raise RuntimeError("retriever not initialized")
        path = Path(normalized)
        if not path.is_file():
            raise FileNotFoundError(f"RAG text index not found: {path}")
        import torch  # noqa: WPS433

        data = torch.load(str(path), map_location="cpu")
        text_embs = data["text_embs"].to(self.retriever.device)
        term_list = data["term_list"]
        if text_embs.shape[0] != len(term_list):
            raise RuntimeError("RAG text index mismatch (text_embs vs term_list)")
        self._text_index_cache[normalized] = {"text_embs": text_embs, "term_list": term_list}
        return self._text_index_cache[normalized]

    def _activate_sync(self, index_path: str) -> None:
        if self.retriever is None:
            return
        normalized = str(Path(index_path))
        if self._active_index_path == normalized:
            return
        data = self._ensure_index(normalized)
        self.retriever.text_embs = data["text_embs"]
        self.retriever.term_list = data["term_list"]
        self._active_index_path = normalized
        self.status["active_index_path"] = normalized
        self.status["active_terms"] = len(data["term_list"])

    async def activate_index(self, index_path: str) -> None:
        if not index_path:
            return
        async with self._lock:
            await asyncio.to_thread(self._activate_sync, index_path)

    async def retrieve(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
    ) -> List[List[TermRef]]:
        if self.retriever is None or not requests:
            return [[] for _ in requests]
        async with self._lock:
            results = await asyncio.to_thread(
                self.retriever.retrieve_timeline_batch,
                list(requests),
                int(top_k),
                float(lookback_sec),
            )
        return list(results)

    async def health(self) -> Dict[str, Any]:
        return dict(self.status)
