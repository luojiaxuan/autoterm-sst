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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from framework.agents.term_memory.topic_router import DomainProbeScore

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

    async def preload_index(self, index_path: str) -> None:
        return None

    def is_index_ready(self, index_path: str) -> bool:
        return False

    async def retrieve(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float] = None,
    ) -> List[List[TermRef]]:
        """Return per-request term lists (same order as ``requests``)."""

        results = await self.retrieve_with_metadata(
            requests,
            top_k=top_k,
            lookback_sec=lookback_sec,
            score_threshold=score_threshold,
        )
        return [item.references for item in results]

    async def retrieve_with_metadata(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float] = None,
    ) -> List["RetrievalResult"]:
        """Return references plus optional speech-side retrieval metadata."""

        return [RetrievalResult(references=[]) for _ in requests]

    async def probe_domain_scores(
        self,
        request: Dict[str, Any],
        *,
        candidate_slices: Sequence[Dict[str, Any]],
        top_k: int = 5,
        lookback_sec: float = 1.92,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, DomainProbeScore]:
        return {}

    async def health(self) -> Dict[str, Any]:
        return {"status": "disabled"}

    async def stop(self) -> None:
        return None


class NullRetrieval(RetrievalPlugin):
    """No-op retrieval: the agent runs without a term_map."""

    enabled = False


@dataclass
class RetrievalResult:
    references: List[TermRef]
    query_embedding: Any = None
    retrieve_s: Optional[float] = None


class MockRetrieval(RetrievalPlugin):
    """Deterministic fake retriever for mock mode (no GPU / no torch).

    Lets the evidence UI and the structured JSON protocol be developed and
    demoed end-to-end without a model or retriever: whenever a glossary index is
    active it returns a small, plausible set of terminology references with
    scores/sources so the evidence panel and ``meta.references`` are non-empty.
    """

    enabled = True

    _SAMPLE = [
        ("retrieval augmented generation", "检索增强生成"),
        ("simultaneous interpretation", "同声传译"),
        ("tensor parallelism", "张量并行"),
        ("knowledge distillation", "知识蒸馏"),
        ("beam search", "束搜索"),
        ("speech translation", "语音翻译"),
    ]

    def __init__(self, *, target_lang: str = "zh", top_k: int = 10) -> None:
        self.target_lang = target_lang
        self.top_k = top_k
        self._active_index_path = ""
        self._active_terms = 0

    async def activate_index(self, index_path: str) -> None:
        if not index_path:
            return
        self._active_index_path = str(index_path)
        text = index_path.lower()
        # surface a believable scale from the preset/index name
        if "10000" in text or "10k" in text:
            self._active_terms = 10000
        elif "1000" in text or "1k" in text:
            self._active_terms = 1000
        else:
            self._active_terms = len(self._SAMPLE)

    async def preload_index(self, index_path: str) -> None:
        await self.activate_index(index_path)

    def is_index_ready(self, index_path: str) -> bool:
        return bool(index_path)

    async def retrieve(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float] = None,
    ) -> List[List[TermRef]]:
        results = await self.retrieve_with_metadata(
            requests,
            top_k=top_k,
            lookback_sec=lookback_sec,
            score_threshold=score_threshold,
        )
        return [item.references for item in results]

    async def retrieve_with_metadata(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        del score_threshold
        if not self._active_index_path:
            return [RetrievalResult(references=[]) for _ in requests]
        per_request = max(1, min(3, int(top_k) if top_k else 3))
        out: List[RetrievalResult] = []
        path_parts = Path(self._active_index_path.replace("mock://", "")).parts
        preset_hint = path_parts[-3] if len(path_parts) >= 3 else (path_parts[-1] if path_parts else self._active_index_path)
        domain = "general"
        if "nlp" in self._active_index_path.lower() or "academic" in self._active_index_path.lower():
            domain = "nlp"
        elif "medicine" in self._active_index_path.lower():
            domain = "medicine"
        elif "finance" in self._active_index_path.lower():
            domain = "finance"
        elif "legal" in self._active_index_path.lower():
            domain = "legal"
        for i, _req in enumerate(requests):
            refs: List[TermRef] = []
            for j in range(per_request):
                term, translation = self._SAMPLE[(i + j) % len(self._SAMPLE)]
                refs.append(
                    {
                        "term": term,
                        "translation": translation,
                        "source": "wikidata",
                        "score": round(0.9 - 0.05 * j, 3),
                        "domain": domain,
                        "source_preset": preset_hint,
                    }
                )
            # Tiny deterministic vector lets mock-mode router tests exercise the
            # metadata path without importing torch or loading a model.
            embedding = {
                "general": [1.0, 0.0, 0.0, 0.0],
                "nlp": [0.0, 1.0, 0.0, 0.0],
                "medicine": [0.0, 0.0, 1.0, 0.0],
                "finance": [0.0, 0.0, 0.0, 1.0],
                "legal": [0.5, 0.0, 0.0, 0.5],
            }.get(domain, [1.0, 0.0, 0.0, 0.0])
            out.append(RetrievalResult(references=refs, query_embedding=embedding, retrieve_s=0.0))
        return out

    async def probe_domain_scores(
        self,
        request: Dict[str, Any],
        *,
        candidate_slices: Sequence[Dict[str, Any]],
        top_k: int = 5,
        lookback_sec: float = 1.92,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, DomainProbeScore]:
        del request, lookback_sec, score_threshold
        out: Dict[str, DomainProbeScore] = {}
        for item in candidate_slices or []:
            domain = str(item.get("domain") or "general")
            preset = str(item.get("preset_id") or domain)
            base = 0.8 if domain in {"nlp", "medicine", "finance", "legal"} else 0.1
            out[domain] = DomainProbeScore(
                domain=domain,
                preset_id=preset,
                top_score=base,
                mean_topk_score=base * 0.9,
                top_terms=tuple(term for term, _translation in self._SAMPLE[: max(1, int(top_k))]),
            )
        return out

    async def health(self) -> Dict[str, Any]:
        return {
            "status": "ready",
            "backend": "mock",
            "active_index_path": self._active_index_path or None,
            "active_terms": self._active_terms,
        }


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
        self._preload_tasks: Dict[str, asyncio.Task] = {}
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

    async def preload_index(self, index_path: str) -> None:
        if not index_path:
            return
        normalized = str(Path(index_path))
        if normalized in self._text_index_cache:
            return
        task = self._preload_tasks.get(normalized)
        if task is None or task.done():
            task = asyncio.create_task(asyncio.to_thread(self._ensure_index, normalized))
            self._preload_tasks[normalized] = task
        try:
            await task
        finally:
            if task.done():
                self._preload_tasks.pop(normalized, None)

    def is_index_ready(self, index_path: str) -> bool:
        if not index_path:
            return False
        return str(Path(index_path)) in self._text_index_cache

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
        score_threshold: Optional[float] = None,
    ) -> List[List[TermRef]]:
        results = await self.retrieve_with_metadata(
            requests,
            top_k=top_k,
            lookback_sec=lookback_sec,
            score_threshold=score_threshold,
        )
        return [item.references for item in results]

    async def retrieve_with_metadata(
        self,
        requests: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        if self.retriever is None or not requests:
            return [RetrievalResult(references=[]) for _ in requests]
        async with self._lock:
            missing = object()
            old_threshold = getattr(self.retriever, "score_threshold", missing)
            if score_threshold is not None:
                self.retriever.score_threshold = float(score_threshold)
            try:
                results = await asyncio.to_thread(
                    self._retrieve_with_query_embeddings_sync,
                    list(requests),
                    int(top_k),
                    float(lookback_sec),
                )
            finally:
                if score_threshold is not None and old_threshold is not missing:
                    self.retriever.score_threshold = old_threshold
        return list(results)

    async def probe_domain_scores(
        self,
        request: Dict[str, Any],
        *,
        candidate_slices: Sequence[Dict[str, Any]],
        top_k: int = 5,
        lookback_sec: float = 1.92,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, DomainProbeScore]:
        if self.retriever is None or not request:
            return {}
        async with self._lock:
            return await asyncio.to_thread(
                self._probe_domain_scores_sync,
                dict(request),
                list(candidate_slices or []),
                max(1, int(top_k)),
                float(lookback_sec),
                score_threshold,
            )

    def _probe_domain_scores_sync(
        self,
        request: Dict[str, Any],
        candidate_slices: Sequence[Dict[str, Any]],
        top_k: int,
        lookback_sec: float,
        score_threshold: Optional[float],
    ) -> Dict[str, DomainProbeScore]:
        out: Dict[str, DomainProbeScore] = {}
        window_embs = self._probe_window_embeddings_from_request(dict(request))
        if window_embs is None:
            window_embs = self._encode_probe_window_sync(dict(request), float(lookback_sec))
        if window_embs is None:
            return out
        for item in candidate_slices or []:
            index_path = str(item.get("index_path") or "").strip()
            domain = str(item.get("domain") or "").strip()
            preset = str(item.get("preset_id") or domain).strip()
            if not index_path or not domain:
                continue
            data = self._ensure_index(index_path)
            text_embs = data["text_embs"].float()
            term_list = data["term_list"]
            if text_embs.numel() == 0:
                continue
            scores_t = window_embs.matmul(text_embs.T).max(dim=0).values
            finite = scores_t.isfinite()
            if score_threshold is not None:
                finite = finite & (scores_t >= float(score_threshold))
            if int(finite.sum().item()) == 0:
                out[domain] = DomainProbeScore(domain=domain, preset_id=preset)
                continue
            n = min(max(1, int(top_k)), int(finite.sum().item()))
            masked_scores = scores_t.masked_fill(~finite, -float("inf"))
            top_scores, top_idx = masked_scores.topk(k=n, largest=True, sorted=True)
            score_values = [float(value) for value in top_scores.detach().cpu().tolist()]
            term_indices = [int(idx) for idx in top_idx.detach().cpu().tolist()]
            terms = [self._term_label(term_list[idx]) for idx in term_indices if 0 <= idx < len(term_list)]
            top_score = max(score_values) if score_values else 0.0
            mean_topk = sum(score_values) / len(score_values) if score_values else 0.0
            out[domain] = DomainProbeScore(
                domain=domain,
                preset_id=preset,
                top_score=top_score,
                mean_topk_score=mean_topk,
                top_terms=tuple(terms),
            )
        return out

    def _probe_window_embeddings_from_request(self, request: Dict[str, Any]) -> Any:
        query_embedding = request.get("query_embedding")
        if query_embedding is None:
            return None
        import torch  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433

        try:
            tensor = torch.as_tensor(
                query_embedding,
                dtype=torch.float32,
                device=self.retriever.device,
            )
        except Exception:  # noqa: BLE001 - malformed optional embedding should fall back to audio
            return None
        if tensor.numel() == 0:
            return None
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        return F.normalize(tensor.float(), p=2, dim=-1)

    def _encode_probe_window_sync(self, request: Dict[str, Any], lookback_sec: float) -> Any:
        from agents.streaming_maxsim_retriever import (  # noqa: WPS433
            EXPECTED_SAMPLE_RATE,
            _build_window_time_ranges,
            _encode_audio_projected_seq_batch,
        )

        import numpy as np  # noqa: WPS433
        import torch  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433

        audio_buffer = np.asarray(request.get("audio_buffer"), dtype=np.float32).flatten()
        if audio_buffer.size == 0:
            return None
        current_start_sec = max(0.0, float(request["current_start_sec"]))
        current_end_sec = max(current_start_sec, float(request["current_end_sec"]))
        cur_lookback = max(0.0, float(request.get("lookback_sec", lookback_sec)))
        buffer_end = min(
            len(audio_buffer),
            int(round(current_end_sec * EXPECTED_SAMPLE_RATE)),
        )
        if buffer_end <= 0:
            return None
        encode_start_sec = max(0.0, current_start_sec - cur_lookback)
        buffer_start = min(
            buffer_end,
            int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)),
        )
        chunk = audio_buffer[buffer_start:buffer_end]
        if len(chunk) == 0:
            return None
        projected_seq, mask = _encode_audio_projected_seq_batch(
            [chunk],
            self.retriever.retriever,
            self.retriever.feat_ext,
            self.retriever.device,
        )
        valid_frames = int(mask[0].sum().item())
        if valid_frames <= 0:
            return None
        use_cuda = getattr(self.retriever.device, "type", "") == "cuda"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            window_embs = self.retriever.retriever._multiscale_pool(
                projected_seq[:1, :valid_frames, :],
                mask[:1, :valid_frames],
            )
            window_embs = F.normalize(window_embs, p=2, dim=-1).float()[0]
        if window_embs.numel() == 0:
            return None
        rel_starts, rel_ends = _build_window_time_ranges(
            self.retriever.retriever.maxsim_windows,
            self.retriever.retriever.maxsim_stride,
            valid_frames,
        )
        if rel_starts.numel() != window_embs.shape[0]:
            return None
        actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
        actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
        actual_duration = max(1e-6, actual_end_sec - actual_start_sec)
        nominal_duration = max(float(rel_ends.max().item()), 1e-6)
        scale = actual_duration / nominal_duration
        rel_starts_d = rel_starts.to(self.retriever.device)
        rel_ends_d = rel_ends.to(self.retriever.device)
        abs_starts = actual_start_sec + rel_starts_d * scale
        abs_ends = actual_start_sec + rel_ends_d * scale
        valid_windows = (abs_ends > current_start_sec) & (abs_starts < current_end_sec)
        if int(valid_windows.sum().item()) == 0:
            return None
        return window_embs[valid_windows].float()

    def _term_label(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("term") or item.get("source") or item.get("src") or item)
        if isinstance(item, (list, tuple)) and item:
            return str(item[0])
        return str(item)

    def _retrieve_with_query_embeddings_sync(
        self,
        requests: Sequence[Dict[str, Any]],
        top_k: int,
        lookback_sec: float,
    ) -> List[RetrievalResult]:
        """Run timeline retrieval and expose pooled audio-window embeddings.

        This mirrors the external RASST ``retrieve_timeline_batch`` implementation
        but keeps the pooled speech representation that is already computed for
        MaxSim scoring. The tensor is returned on CPU so the router can consume it
        without touching the retriever GPU stream.
        """

        from agents.streaming_maxsim_retriever import (  # noqa: WPS433
            EXPECTED_SAMPLE_RATE,
            _build_window_time_ranges,
            _encode_audio_projected_seq_batch,
        )

        import numpy as np  # noqa: WPS433
        import torch  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433

        t0 = time.perf_counter()
        outputs: List[RetrievalResult] = [RetrievalResult(references=[]) for _ in requests]
        if self.retriever is None or not requests:
            return outputs

        k = top_k if top_k is not None else self.top_k
        default_lookback = max(0.0, float(lookback_sec))
        chunks: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for req_idx, req in enumerate(requests):
            audio_buffer = np.asarray(req.get("audio_buffer"), dtype=np.float32).flatten()
            if audio_buffer.size == 0:
                continue
            current_start_sec = max(0.0, float(req["current_start_sec"]))
            current_end_sec = max(current_start_sec, float(req["current_end_sec"]))
            cur_lookback = max(0.0, float(req.get("lookback_sec", default_lookback)))
            buffer_end = min(
                len(audio_buffer),
                int(round(current_end_sec * EXPECTED_SAMPLE_RATE)),
            )
            if buffer_end <= 0:
                continue
            encode_start_sec = max(0.0, current_start_sec - cur_lookback)
            buffer_start = min(
                buffer_end,
                int(round(encode_start_sec * EXPECTED_SAMPLE_RATE)),
            )
            chunk = audio_buffer[buffer_start:buffer_end]
            if len(chunk) == 0:
                continue
            actual_start_sec = float(buffer_start) / EXPECTED_SAMPLE_RATE
            actual_end_sec = float(buffer_end) / EXPECTED_SAMPLE_RATE
            chunks.append(chunk)
            metas.append(
                {
                    "request_idx": req_idx,
                    "current_start_sec": current_start_sec,
                    "current_end_sec": current_end_sec,
                    "actual_start_sec": actual_start_sec,
                    "actual_end_sec": actual_end_sec,
                    "actual_duration": max(1e-6, actual_end_sec - actual_start_sec),
                }
            )

        if not chunks:
            return outputs

        projected_seq, mask = _encode_audio_projected_seq_batch(
            chunks,
            self.retriever.retriever,
            self.retriever.feat_ext,
            self.retriever.device,
        )
        groups_by_frames: Dict[int, List[tuple[int, Dict[str, Any]]]] = {}
        for batch_idx, meta in enumerate(metas):
            valid_frames = int(mask[batch_idx].sum().item())
            if valid_frames > 0:
                groups_by_frames.setdefault(valid_frames, []).append((batch_idx, meta))

        text_bank_t = self.retriever.text_embs.float().T
        use_cuda = getattr(self.retriever.device, "type", "") == "cuda"
        for valid_frames, group_items in groups_by_frames.items():
            group_indices = [batch_idx for batch_idx, _ in group_items]
            seq_g = projected_seq[group_indices, :valid_frames, :]
            mask_g = mask[group_indices, :valid_frames]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                window_embs = self.retriever.retriever._multiscale_pool(seq_g, mask_g)
                window_embs = F.normalize(window_embs, p=2, dim=-1).float()
            if window_embs.numel() == 0:
                continue
            rel_starts, rel_ends = _build_window_time_ranges(
                self.retriever.retriever.maxsim_windows,
                self.retriever.retriever.maxsim_stride,
                valid_frames,
            )
            if rel_starts.numel() != window_embs.shape[1]:
                logger.warning(
                    "Batch timeline retrieval window count mismatch: ranges=%d embs=%d",
                    rel_starts.numel(),
                    window_embs.shape[1],
                )
                continue

            nominal_duration = max(float(rel_ends.max().item()), 1e-6)
            actual_duration = torch.tensor(
                [float(meta["actual_duration"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.retriever.device,
            )
            actual_start = torch.tensor(
                [float(meta["actual_start_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.retriever.device,
            )
            current_start = torch.tensor(
                [float(meta["current_start_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.retriever.device,
            )
            current_end = torch.tensor(
                [float(meta["current_end_sec"]) for _, meta in group_items],
                dtype=torch.float32,
                device=self.retriever.device,
            )
            scale = actual_duration / nominal_duration
            rel_starts_d = rel_starts.to(self.retriever.device)
            rel_ends_d = rel_ends.to(self.retriever.device)
            abs_starts = actual_start.unsqueeze(1) + rel_starts_d.unsqueeze(0) * scale.unsqueeze(1)
            abs_ends = actual_start.unsqueeze(1) + rel_ends_d.unsqueeze(0) * scale.unsqueeze(1)
            valid_windows = (
                (abs_ends > current_start.unsqueeze(1))
                & (abs_starts < current_end.unsqueeze(1))
            )
            if int(valid_windows.sum().item()) == 0:
                continue

            sim_by_window = torch.matmul(window_embs, text_bank_t)
            sim_by_window = sim_by_window.masked_fill(
                ~valid_windows.unsqueeze(2),
                -float("inf"),
            )
            scores, best_window_idx = sim_by_window.max(dim=1)
            for row_idx, (_, meta) in enumerate(group_items):
                request_idx = int(meta["request_idx"])
                row_valid = valid_windows[row_idx]
                if int(row_valid.sum().item()) > 0:
                    pooled = window_embs[row_idx][row_valid].mean(dim=0)
                    pooled = F.normalize(pooled, p=2, dim=-1).detach().cpu().float()
                    outputs[request_idx].query_embedding = pooled

                row_scores = scores[row_idx]
                finite = torch.isfinite(row_scores)
                if int(finite.sum().item()) == 0:
                    continue
                n = min(int(k), int(finite.sum().item()))
                masked_scores = row_scores.masked_fill(~finite, -float("inf"))
                top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
                top_win = best_window_idx[row_idx].gather(0, top_idx)
                top_start = abs_starts[row_idx].gather(0, top_win)
                top_end = abs_ends[row_idx].gather(0, top_win)
                outputs[request_idx].references = self.retriever._build_results(
                    top_idx.detach().cpu().numpy(),
                    top_sco.detach().cpu().numpy(),
                    time_starts=top_start.detach().cpu().numpy(),
                    time_ends=top_end.detach().cpu().numpy(),
                    retrieval_mode="timeline_batch",
                )

        elapsed = time.perf_counter() - t0
        for item in outputs:
            item.retrieve_s = elapsed
        return outputs

    async def health(self) -> Dict[str, Any]:
        return dict(self.status)
