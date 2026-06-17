"""ModelBackend: agent-internal abstraction over a generation engine.

A backend turns assembled chat messages (+ audio) into translated text. This is
where "support more omni models" lives: adding a model is a new
:class:`ModelTemplate` (served name, audio schema, prompt style) plus, if the
serving stack differs, a new ``ModelBackend`` implementation.

Implementations:
* :class:`MockBackend`       -- deterministic text, no GPU (smoke tests / UI dev)
* :class:`VLLMBackend`       -- in-process vLLM (``vllm.LLM``) with **batched**
  generation; one ``llm.generate(batch)`` per scheduler tick drives vLLM
  continuous batching (32+ concurrent sessions/GPU). This is the RASST default.
* :class:`SGLangHTTPBackend` -- OpenAI-compatible ``/v1/chat/completions``
  client to an external SGLang/vLLM server (per-request; optional alternative).
* :class:`HFBackend`         -- in-process Transformers fallback (stub; see notes)

Batched backends (``batched = True``) own per-session chat state and expose
``generate_batch`` + ``open_session`` / ``reset_session`` / ``close_session``;
per-request backends keep ``batched = False`` and use ``generate``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# vLLM omni models expect at least ~0.96s of audio per turn; pad short tails.
_MIN_AUDIO_SAMPLES = 15360


@dataclass
class Sampling:
    max_tokens: int = 40
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50
    seed: int = 998244353


@dataclass
class ModelTemplate:
    """Per-model adapter: how to address and prompt a served omni model."""

    model_id: str
    served_model_name: str
    audio_schema: str = "top_level"          # "top_level" | "inline"
    system_prompt_style: str = "given_chunks"  # "given_chunks" | "translate_task"
    backend_kind: str = "sglang_http"        # "sglang_http" | "vllm" | "hf"
    sampling: Sampling = field(default_factory=Sampling)
    notes: str = ""


# Model-extension registry. Add an entry here to support a new omni model.
MODEL_TEMPLATES: Dict[str, ModelTemplate] = {
    "qwen3_omni": ModelTemplate(
        model_id="qwen3_omni",
        served_model_name="rasst-qwen3-omni",
        audio_schema="top_level",
        system_prompt_style="given_chunks",
        backend_kind="vllm",
        notes=(
            "Reference RASST model. Hosted in-process by vLLM (batched generate "
            "for 32+ concurrent sessions/GPU). Set backend_kind='sglang_http' to "
            "instead use an external OpenAI-compatible SGLang/vLLM server."
        ),
    ),
    "minicpm_o": ModelTemplate(
        model_id="minicpm_o",
        served_model_name="minicpm-o",
        audio_schema="inline",
        system_prompt_style="translate_task",
        backend_kind="sglang_http",
        notes=(
            "MiniCPM-o 4.5. Uses inline audio-in-content schema. Verify the "
            "serving engine exposes an OpenAI-compatible audio chat endpoint; "
            "otherwise switch backend_kind to 'hf' (HFBackend)."
        ),
    ),
}


def get_template(model_id: str) -> ModelTemplate:
    if model_id not in MODEL_TEMPLATES:
        raise KeyError(f"unknown model_id {model_id!r}; known: {list(MODEL_TEMPLATES)}")
    return MODEL_TEMPLATES[model_id]


class ModelBackend:
    """Interface an agent uses to generate text from messages + audio.

    Two flavors:
    * ``batched = False`` (default): the agent builds full chat ``messages`` and
      calls :meth:`generate` once per session (server-side batching, e.g. HTTP).
    * ``batched = True``: the backend owns per-session chat state and exposes
      :meth:`generate_batch` (one engine call over the whole batch) plus the
      :meth:`open_session` / :meth:`reset_session` / :meth:`close_session` hooks.
    """

    #: Whether the agent should drive this backend via :meth:`generate_batch`.
    batched: bool = False

    async def start(self) -> None:
        return None

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        audios: Sequence[str],
        sampling: Sampling,
        request_id: str,
    ) -> str:
        raise NotImplementedError

    async def generate_batch(
        self, requests: List[Dict[str, Any]], *, sampling: Sampling
    ) -> List[Dict[str, Any]]:
        """Generate for a whole batch at once (batched backends only).

        Each request: ``{session_id, audio: np.ndarray, term_map_text: str,
        rag_enabled: bool}``. Returns one ``{ok, text, elapsed_s, error}`` per
        request, in order.
        """
        raise NotImplementedError

    # Per-session chat state lifecycle (no-ops for stateless backends).
    async def open_session(self, session_id: str, meta: Dict[str, Any]) -> None:
        return None

    async def reset_session(self, session_id: str) -> None:
        return None

    async def close_session(self, session_id: str) -> None:
        return None

    async def health(self) -> Dict[str, Any]:
        return {"status": "ready"}

    async def stop(self) -> None:
        return None


class MockBackend(ModelBackend):
    """Deterministic, dependency-free backend for protocol/UI testing."""

    def __init__(self, template: ModelTemplate) -> None:
        self.template = template

    @staticmethod
    def _term_aware(messages: Sequence[Dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            if "term_map:" in text and "term_map:\nNONE" not in text:
                return True
        return False

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        audios: Sequence[str],
        sampling: Sampling,
        request_id: str,
    ) -> str:
        match = re.search(r"(\d+)$", request_id or "")
        seg = match.group(1) if match else "1"
        suffix = " (term-aware)" if self._term_aware(messages) else ""
        return f"[mock:{self.template.model_id}] streaming translation segment {seg}{suffix}"

    async def health(self) -> Dict[str, Any]:
        return {"status": "ready", "mock": True, "model": self.template.served_model_name}


class SGLangHTTPBackend(ModelBackend):
    """OpenAI-compatible chat-completions backend (SGLang-Omni / vLLM)."""

    def __init__(self, template: ModelTemplate, base_url: str, timeout_sec: float = 900.0) -> None:
        self.template = template
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session: Any = None

    async def start(self) -> None:
        import aiohttp  # noqa: WPS433

        timeout = aiohttp.ClientTimeout(total=float(self.timeout_sec))
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        audios: Sequence[str],
        sampling: Sampling,
        request_id: str,
    ) -> str:
        if self._session is None:
            raise RuntimeError("SGLangHTTPBackend not started")
        payload: Dict[str, Any] = {
            "model": self.template.served_model_name,
            "request_id": request_id,
            "messages": messages,
            "audios": list(audios),
            "audio_target_sr": 16000,
            "modalities": ["text"],
            "max_tokens": int(sampling.max_tokens),
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "seed": int(sampling.seed),
            "stream": False,
        }
        async with self._session.post(
            f"{self.base_url}/v1/chat/completions", json=payload
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"SGLang status={resp.status} body={data}")
        return str(
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        ).strip()

    async def health(self) -> Dict[str, Any]:
        if self._session is None:
            return {"status": "starting"}
        try:
            async with self._session.get(f"{self.base_url}/health") as resp:
                data = await resp.json()
                data["http_status"] = resp.status
                return data
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


@dataclass
class _VLLMSessionState:
    """Per-session multi-turn chat state owned by the vLLM backend."""

    session_id: str
    source_lang: str
    target_lang: str
    lang_code: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    segment_idx: int = 0


class VLLMBackend(ModelBackend):
    """In-process vLLM omni backend with batched generation.

    Mirrors ``serve/rasst_server.py``'s worker: loads ``vllm.LLM`` +
    ``Qwen3OmniMoeProcessor`` once and serves a whole scheduler batch with a
    single ``llm.generate(prepared_batch)`` call so vLLM continuous-batches all
    active sessions (32+/GPU). The heavy engine runs on a dedicated single
    worker thread; every state-touching op is serialized onto it (same model as
    the reference worker loop), so the asyncio event loop never blocks.

    The model is loaded into the framework process, so launch it on a GPU host
    with ``CUDA_VISIBLE_DEVICES`` / ``tensor_parallel_size`` set appropriately.
    """

    batched = True

    def __init__(
        self,
        template: ModelTemplate,
        *,
        model_path: str,
        tp_size: int = 1,
        gpu_memory_utilization: float = 0.86,
        max_num_seqs: int = 32,
        max_model_len: int = 16384,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        limit_audio: int = 16,
        disable_custom_all_reduce: bool = False,
        max_cache_chunks: int = 16,
        keep_cache_chunks: int = 8,
        empty_term_map_policy: str = "none_block",
        rag_enabled: bool = True,
        default_source_lang: str = "English",
        default_target_lang: str = "Chinese",
        default_lang_code: str = "zh",
    ) -> None:
        if not model_path:
            raise ValueError("VLLMBackend requires a model_path")
        self.template = template
        self.model_path = model_path
        self.tp_size = int(tp_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.enforce_eager = bool(enforce_eager)
        self.limit_audio = int(limit_audio)
        self.disable_custom_all_reduce = bool(disable_custom_all_reduce)
        self.max_cache_chunks = int(max_cache_chunks)
        self.keep_cache_chunks = int(keep_cache_chunks)
        self.empty_term_map_policy = empty_term_map_policy
        self.rag_enabled = bool(rag_enabled)
        self.default_source_lang = default_source_lang
        self.default_target_lang = default_target_lang
        self.default_lang_code = default_lang_code

        self._exec: Optional[ThreadPoolExecutor] = None
        self._llm: Any = None
        self._processor: Any = None
        self._process_mm_info: Any = None
        self._sampling_cls: Any = None
        self._states: Dict[str, _VLLMSessionState] = {}
        self._ready = False
        self._load_error: Optional[str] = None

    async def start(self) -> None:
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-omni")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._exec, self._load_model)

    def _load_model(self) -> None:
        try:
            os.environ.setdefault("VLLM_USE_V1", "0")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from qwen_omni_utils import process_mm_info  # noqa: WPS433
            from transformers import Qwen3OmniMoeProcessor  # noqa: WPS433

            self._patch_vllm_transformers_register_conflict()
            from vllm import LLM, SamplingParams  # noqa: WPS433

            self._sampling_cls = SamplingParams
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
            self._process_mm_info = process_mm_info
            llm_kwargs: Dict[str, Any] = {
                "model": self.model_path,
                "trust_remote_code": True,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tp_size,
                "max_num_seqs": self.max_num_seqs,
                "max_model_len": self.max_model_len,
                "enable_prefix_caching": self.enable_prefix_caching,
                "enforce_eager": self.enforce_eager,
            }
            if self.limit_audio > 0:
                llm_kwargs["limit_mm_per_prompt"] = {"audio": self.limit_audio}
            if self.disable_custom_all_reduce:
                llm_kwargs["disable_custom_all_reduce"] = True
            logger.info(
                "loading vLLM omni model %s from %s (tp=%s max_num_seqs=%s)",
                self.template.served_model_name,
                self.model_path,
                self.tp_size,
                self.max_num_seqs,
            )
            self._llm = LLM(**llm_kwargs)
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            self._load_error = repr(exc)
            logger.exception("vLLM model load failed")
            raise

    @staticmethod
    def _patch_vllm_transformers_register_conflict() -> None:
        """vLLM 0.9.x re-registers aimv2, which newer Transformers already ships."""
        from transformers import AutoConfig  # noqa: WPS433

        original_register = AutoConfig.register

        def safe_register(model_type, config, exist_ok=False):  # noqa: ANN001, ANN202
            try:
                return original_register(model_type, config, exist_ok=exist_ok)
            except ValueError as exc:
                if model_type == "aimv2" and "already used" in str(exc):
                    return None
                raise

        AutoConfig.register = safe_register

    async def open_session(self, session_id: str, meta: Dict[str, Any]) -> None:
        self._states[session_id] = _VLLMSessionState(
            session_id=session_id,
            source_lang=str(meta.get("source_lang") or self.default_source_lang),
            target_lang=str(meta.get("target_lang") or self.default_target_lang),
            lang_code=str(meta.get("lang_code") or self.default_lang_code),
        )

    async def reset_session(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is not None:
            state.messages = []
            state.segment_idx = 0

    async def close_session(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    async def generate_batch(
        self, requests: List[Dict[str, Any]], *, sampling: Sampling
    ) -> List[Dict[str, Any]]:
        if self._exec is None or self._llm is None:
            raise RuntimeError("VLLMBackend not started")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._exec, self._run_batch, requests, sampling)

    def _build_sampling(self, sampling: Sampling) -> Any:
        return self._sampling_cls(
            temperature=float(sampling.temperature),
            top_p=float(sampling.top_p),
            top_k=int(sampling.top_k),
            max_tokens=int(sampling.max_tokens),
            seed=int(sampling.seed),
        )

    def _system_message(self, state: _VLLMSessionState) -> Dict[str, Any]:
        from framework.agents.plugins.prompt import build_system_prompt  # noqa: WPS433

        text = build_system_prompt(
            state.source_lang, state.target_lang, self.template.system_prompt_style, self.rag_enabled
        )
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def _prepare_input(
        self,
        state: _VLLMSessionState,
        increment: "np.ndarray",
        term_map_text: str,
        rag_enabled: bool,
    ) -> Dict[str, Any]:
        if not state.messages:
            state.messages.append(self._system_message(state))
        if self.limit_audio > 0:
            keep_pairs = max(0, self.limit_audio - 1)
            body = state.messages[1:]
            state.messages = [state.messages[0]] + body[-2 * keep_pairs:]

        user_content: List[Dict[str, Any]] = [{"type": "audio", "audio": increment}]
        if term_map_text:
            user_content.append({"type": "text", "text": f"\n\nterm_map:\n{term_map_text}"})
        elif rag_enabled and self.empty_term_map_policy == "none_block":
            user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})
        state.messages.append({"role": "user", "content": user_content})

        prompt = self._processor.apply_chat_template(
            state.messages, add_generation_prompt=True, tokenize=False
        )
        audios, _images, _videos = self._process_mm_info(state.messages, use_audio_in_video=False)
        return {
            "prompt": prompt,
            "multi_modal_data": {"audio": audios},
            "mm_processor_kwargs": {"use_audio_in_video": False},
        }

    def _trim(self, state: _VLLMSessionState) -> None:
        body = state.messages[1:]
        if len(body) > 2 * self.max_cache_chunks:
            state.messages = [state.messages[0]] + body[-2 * self.keep_cache_chunks:]

    def _run_batch(self, requests: List[Dict[str, Any]], sampling: Sampling) -> List[Dict[str, Any]]:
        order: List[_VLLMSessionState] = []
        prepared: List[Dict[str, Any]] = []
        for req in requests:
            session_id = req["session_id"]
            state = self._states.get(session_id)
            if state is None:
                state = _VLLMSessionState(
                    session_id=session_id,
                    source_lang=self.default_source_lang,
                    target_lang=self.default_target_lang,
                    lang_code=self.default_lang_code,
                )
                self._states[session_id] = state
            increment = np.asarray(req["audio"], dtype=np.float32).flatten()
            if increment.shape[0] < _MIN_AUDIO_SAMPLES:
                increment = np.pad(increment, (0, _MIN_AUDIO_SAMPLES - increment.shape[0]))
            prepared.append(
                self._prepare_input(
                    state, increment, str(req.get("term_map_text") or ""), bool(req.get("rag_enabled", False))
                )
            )
            order.append(state)

        t0 = time.perf_counter()
        try:
            outputs = self._llm.generate(
                prepared, sampling_params=self._build_sampling(sampling), use_tqdm=False
            )
        except Exception as exc:  # noqa: BLE001
            for state in order:  # roll back the user turn so a retry is clean
                if state.messages and state.messages[-1].get("role") == "user":
                    state.messages.pop()
            return [{"ok": False, "text": "", "elapsed_s": None, "error": str(exc)} for _ in order]

        elapsed = round(time.perf_counter() - t0, 6)
        results: List[Dict[str, Any]] = []
        for state, output in zip(order, outputs):
            text = output.outputs[0].text if output.outputs else ""
            state.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            self._trim(state)
            state.segment_idx += 1
            results.append({"ok": True, "text": text, "elapsed_s": elapsed})
        return results

    async def health(self) -> Dict[str, Any]:
        if self._load_error:
            return {"status": "error", "error": self._load_error, "model": self.template.served_model_name}
        if not self._ready:
            return {"status": "loading", "model": self.template.served_model_name}
        return {
            "status": "ready",
            "engine": "vllm",
            "model": self.template.served_model_name,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tp_size,
            "max_num_seqs": self.max_num_seqs,
            "active_sessions": len(self._states),
        }

    async def stop(self) -> None:
        if self._exec is not None:
            self._exec.shutdown(wait=False, cancel_futures=True)
            self._exec = None
        self._llm = None
        self._states.clear()
        self._ready = False


class HFBackend(ModelBackend):
    """In-process Transformers fallback.

    Placeholder for omni models without an OpenAI-compatible serving path (e.g.
    if MiniCPM-o 4.5 audio chat is not yet exposed by SGLang/vLLM). Wire the
    model's processor + ``generate`` here when needed.
    """

    def __init__(self, template: ModelTemplate, model_path: str, device: str = "cuda:0") -> None:
        self.template = template
        self.model_path = model_path
        self.device = device

    async def start(self) -> None:
        raise NotImplementedError(
            "HFBackend is a stub. Implement processor + generate for "
            f"{self.template.model_id!r} ({self.model_path})."
        )

    async def generate(self, **kwargs: Any) -> str:  # noqa: D401
        raise NotImplementedError("HFBackend.generate not implemented")


def build_backend(
    template: ModelTemplate,
    *,
    mock: bool = False,
    sglang_base_url: str = "http://127.0.0.1:8100",
    sglang_timeout_sec: float = 900.0,
    vllm_config: Optional[Dict[str, Any]] = None,
    hf_model_path: str = "",
    hf_device: str = "cuda:0",
) -> ModelBackend:
    if mock:
        return MockBackend(template)
    if template.backend_kind == "vllm":
        return VLLMBackend(template, **(vllm_config or {}))
    if template.backend_kind == "sglang_http":
        return SGLangHTTPBackend(template, base_url=sglang_base_url, timeout_sec=sglang_timeout_sec)
    if template.backend_kind == "hf":
        return HFBackend(template, model_path=hf_model_path, device=hf_device)
    raise ValueError(f"unsupported backend_kind={template.backend_kind!r}")
