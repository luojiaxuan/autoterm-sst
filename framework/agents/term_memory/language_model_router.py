"""Language-model scoring of topic slices for AutoTerm routing.

A small instruction-tuned language model reads the recent translation and a
lettered list of slice descriptions. A slice's score is the probability the model
assigns to its letter as the first answer token, averaged over the listed order
and its reverse so that position in the list does not matter. A sticky hidden
Markov filter turns these per-window probabilities into a posterior over the
current topic.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT = (
    "A live speech translator is translating a talk from English into {language}. "
    "Below is its most recent {language} output, followed by terminology glossaries, "
    "one per topic. Which glossary best matches the topic of the talk at this point?\n\n"
    "Recent translation:\n{context}\n\nGlossaries:\n{options}\n\n"
    "Answer with the letter of the best glossary only."
)


def filter_step(
    posterior: Mapping[str, float],
    probabilities: Mapping[str, float],
    *,
    switch_rate: float,
    evidence_weight: float,
    floor: float = 1e-6,
) -> Dict[str, float]:
    """One forward step of a sticky topic HMM.

    The topic stays with probability 1 - switch_rate per chunk and otherwise moves
    to one of the other slices uniformly; the window probabilities enter with
    exponent ``evidence_weight``.
    """
    domains = list(probabilities)
    k = len(domains)
    if k == 0:
        return {}
    if k == 1:
        return {domains[0]: 1.0}
    uniform = 1.0 / k
    move = switch_rate / (k - 1)
    log_post = {}
    for d in domains:
        prev = float(posterior.get(d, uniform))
        prior = (1.0 - switch_rate) * prev + move * (1.0 - prev)
        log_post[d] = math.log(max(prior, floor)) + evidence_weight * math.log(max(float(probabilities[d]), floor))
    top = max(log_post.values())
    weights = {d: math.exp(v - top) for d, v in log_post.items()}
    total = sum(weights.values())
    return {d: w / total for d, w in weights.items()}


class LanguageModelSliceScorer:
    """Score stream text against slice descriptions with a small language model."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 16,
        prototypes: Mapping[str, Sequence[str]],
        target_language: str = "Chinese",
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self.target_language = target_language
        self.prototypes = {
            str(domain): tuple(str(t).strip() for t in texts if str(t).strip())
            for domain, texts in prototypes.items()
        }
        # The prompt is English, so a slice is described by its English prototype texts.
        self.descriptions = {
            domain: " ".join(t for t in texts if t.isascii()) or " ".join(texts)
            for domain, texts in self.prototypes.items()
            if texts
        }
        self.enabled = False
        self.status: Dict[str, object] = {"status": "disabled"}
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.status = {"status": "loading", "model_id": self.model_id, "device": self.device_name}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._tokenizer.padding_side = "left"
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16)
            .to(self.device_name)
            .eval()
        )
        self.enabled = True
        self.status = {
            "status": "ready",
            "model_id": self.model_id,
            "device": self.device_name,
            "domains": list(self.descriptions),
            "descriptions": dict(self.descriptions),
        }

    def _letter_ids(self, k: int) -> List[int]:
        ids = []
        for letter in LETTERS[:k]:
            pieces = self._tokenizer.encode(letter, add_special_tokens=False)
            if len(pieces) != 1:
                raise ValueError(f"letter {letter!r} is not a single token")
            ids.append(pieces[0])
        return ids

    def _score_batch_sync(self, texts: Sequence[str], allowed_domains: Sequence[str]) -> List[Dict[str, float]]:
        import torch

        domains = [d for d in allowed_domains if d in self.descriptions]
        if not self.enabled or not domains:
            return [{} for _ in texts]
        k = len(domains)
        ids = self._letter_ids(k)
        totals = [[0.0] * k for _ in texts]
        # Both option orders of every text go through the model in the same batches.
        jobs = []
        for order in (list(range(k)), list(range(k))[::-1]):
            block = "\n".join(f"{LETTERS[j]}. {self.descriptions[domains[i]]}" for j, i in enumerate(order))
            for t_idx, t in enumerate(texts):
                prompt = self._tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": PROMPT.format(language=self.target_language, context=t, options=block),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                jobs.append((t_idx, order, prompt))
        for start in range(0, len(jobs), self.batch_size):
            part = jobs[start : start + self.batch_size]
            enc = self._tokenizer([prompt for _, _, prompt in part], return_tensors="pt", padding=True).to(
                self.device_name
            )
            with torch.inference_mode():
                logits = self._model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits[:, ids], dim=-1).cpu().tolist()
            for (t_idx, order, _), p in zip(part, probs):
                for j, src in enumerate(order):
                    totals[t_idx][src] += p[j] / 2.0
        return [{domains[i]: row[i] for i in range(k)} for row in totals]

    async def score_batch(self, texts: Sequence[str], *, allowed_domains: Sequence[str]) -> List[Dict[str, float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._score_batch_sync, list(texts), list(allowed_domains))

    async def health(self) -> Dict[str, object]:
        return dict(self.status)

    async def stop(self) -> None:
        async with self._lock:
            self.enabled = False
            self._model = None
            self._tokenizer = None
            self.status = {"status": "stopped"}
