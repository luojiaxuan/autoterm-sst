"""Multilingual accumulated-context similarity for AutoTerm routing."""

from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Sequence

from framework.agents.term_memory.domain_taxonomy import DOMAIN_ROUTER_PROTOTYPES

logger = logging.getLogger(__name__)


class DomainDescriptionSimilarity:
    """Score stream text against stable bilingual domain descriptions."""

    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32,
        prototypes: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        source = DOMAIN_ROUTER_PROTOTYPES if prototypes is None else prototypes
        self.prototypes = {
            str(domain): tuple(str(text).strip() for text in texts if str(text).strip())
            for domain, texts in source.items()
        }
        self.prototypes = {
            domain: texts for domain, texts in self.prototypes.items() if texts
        }
        self.enabled = False
        self.status: Dict[str, object] = {"status": "disabled"}
        self._model = None
        self._tokenizer = None
        self._centroids = None
        self._domains: List[str] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        self.status = {
            "status": "loading",
            "model_id": self.model_id,
            "device": self.device_name,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device_name).eval()
        self._domains = list(self.prototypes)
        flat_texts: List[str] = []
        spans: Dict[str, tuple[int, int]] = {}
        for domain in self._domains:
            start = len(flat_texts)
            flat_texts.extend(self.prototypes[domain])
            spans[domain] = (start, len(flat_texts))
        embeddings = self._encode_sync(flat_texts)
        centroids = []
        for domain in self._domains:
            start, end = spans[domain]
            centroids.append(
                functional.normalize(embeddings[start:end].mean(dim=0), p=2, dim=0)
            )
        self._centroids = torch.stack(centroids, dim=0)
        self.enabled = True
        self.status = {
            "status": "ready",
            "model_id": self.model_id,
            "device": self.device_name,
            "domains": list(self._domains),
        }

    def _encode_sync(self, texts: Sequence[str]):
        import torch
        import torch.nn.functional as functional

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("context similarity model is not loaded")
        batches = []
        device = torch.device(self.device_name)
        for start in range(0, len(texts), self.batch_size):
            tokenized = self._tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            autocast = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                output = self._model(**tokenized)
                embeddings = output.last_hidden_state[:, 0]
            batches.append(functional.normalize(embeddings.float(), p=2, dim=-1))
        return torch.cat(batches, dim=0)

    def _score_batch_sync(
        self,
        texts: Sequence[str],
        allowed_domains: Sequence[str],
    ) -> List[Dict[str, float]]:
        if not self.enabled or self._centroids is None:
            return [{} for _ in texts]
        embeddings = self._encode_sync(texts)
        similarities = embeddings @ self._centroids.T
        allowed = set(allowed_domains)
        output: List[Dict[str, float]] = []
        for row in similarities.detach().cpu().tolist():
            output.append(
                {
                    domain: float(row[index])
                    for index, domain in enumerate(self._domains)
                    if domain in allowed
                }
            )
        return output

    async def score_batch(
        self,
        texts: Sequence[str],
        *,
        allowed_domains: Sequence[str],
    ) -> List[Dict[str, float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(
                self._score_batch_sync,
                list(texts),
                list(allowed_domains),
            )

    async def health(self) -> Dict[str, object]:
        return dict(self.status)

    async def stop(self) -> None:
        async with self._lock:
            self.enabled = False
            self._model = None
            self._tokenizer = None
            self._centroids = None
            self.status = {"status": "stopped"}
