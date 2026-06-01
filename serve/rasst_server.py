#!/usr/bin/env python3
"""RASST serving path for the demo UI.

The server uses two independent TP=1 worker replicas by default. Each worker is
started with one visible GPU, and the MaxSim RAG retriever is loaded on the same
logical ``cuda:0`` device as that worker's vLLM instance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import site
import sys
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Set

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


TARGET_SAMPLE_RATE = 16000
DEFAULT_VLLM_SEGMENT_SEC = 1.92
DEFAULT_RAG_LOOKBACK_SEC = 1.92
DEFAULT_MAX_CACHE_CHUNKS = 16
DEFAULT_KEEP_CACHE_CHUNKS = 8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RASST_ROOT = Path(os.environ.get("RASST_ROOT", "/mnt/taurus/data2/jiaxuanluo/RASST"))
RASST_CODE_ROOT = Path(os.environ.get("RASST_ACTIVE_CODE_ROOT", RASST_ROOT / "code/rasst"))
RASST_EVAL_ROOT = RASST_CODE_ROOT / "eval"


def _drop_user_site_packages() -> None:
    """Keep vLLM/Transformers imports inside the selected conda environment."""
    candidates = {site.getusersitepackages()}
    candidates.update(site.getsitepackages() if hasattr(site, "getsitepackages") else [])
    user_home_local = str(Path.home() / ".local")
    cleaned = []
    for item in sys.path:
        if not item:
            cleaned.append(item)
            continue
        if item.startswith(user_home_local):
            continue
        cleaned.append(item)
    sys.path[:] = cleaned


_drop_user_site_packages()

LANGUAGE_PAIRS = {
    "English -> Chinese": {
        "source_lang": "English",
        "target_lang": "Chinese",
        "lang_code": "zh",
        "model_path": os.environ.get(
            "RASST_MODEL_ZH_CAP16_DENOISE",
            "/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/models/"
            "speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_ZH_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__zh__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> Japanese": {
        "source_lang": "English",
        "target_lang": "Japanese",
        "lang_code": "ja",
        "model_path": os.environ.get(
            "RASST_MODEL_JA_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_JA_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__ja__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
    "English -> German": {
        "source_lang": "English",
        "target_lang": "German",
        "lang_code": "de",
        "model_path": os.environ.get(
            "RASST_MODEL_DE_CAP16_DENOISE",
            "/mnt/taurus/data1/jiaxuanluo/slm_local_cache/"
            "de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf",
        ),
        "index_path": os.environ.get(
            "RASST_INDEX_DE_ACL",
            str(
                RASST_ROOT
                / "outputs/main_result_eval/20260527T071109Z/index_cache/"
                "acl_tagged_raw__de__lm2/"
                "maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt"
            ),
        ),
    },
}

MODEL_PROFILES = [
    {
        "id": "RASST",
        "label": "RASST",
        "default": True,
        "backend": "qwen3_omni_vllm_maxsim_rag",
    },
    {
        "id": "InfiniSST",
        "label": "InfiniSST Legacy",
        "default": False,
        "backend": "legacy_infinisst_faster",
    },
]


@dataclass
class StreamState:
    session_id: str
    language_pair: str
    source_lang: str
    target_lang: str
    lang_code: str
    samplerate: int = TARGET_SAMPLE_RATE
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cursor_samples: int = 0
    last_vllm_samples: int = 0
    segment_idx: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_worker_gpu_groups(value: str, tp_size: int) -> List[str]:
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    tokens = _split_csv(value)
    if tp_size <= 1:
        return tokens
    if len(tokens) % tp_size != 0:
        raise RuntimeError(
            f"worker GPU count ({len(tokens)}) must be divisible by vLLM TP size ({tp_size})"
        )
    return [",".join(tokens[idx : idx + tp_size]) for idx in range(0, len(tokens), tp_size)]


def _format_term_map(references: Sequence[Dict[str, Any]], mode: str) -> str:
    lines: List[str] = []
    for ref in references:
        term = str(ref.get("term") or "").replace("\n", " ").strip()
        translation = str(ref.get("translation") or "").replace("\n", " ").strip()
        if not term or not translation:
            continue
        if mode == "xml_tagged":
            lines.append(f"<term>{term} => {translation}</term>")
        elif mode == "tagged":
            lines.append(f"[TERM] {term} => {translation} [/TERM]")
        else:
            lines.append(f"{term}={translation}")
    return "\n".join(lines)


def _system_prompt(source_lang: str, target_lang: str, lang_code: str, rag_enabled: bool) -> str:
    if source_lang == "English" and lang_code == "zh":
        text = (
            "You are a professional simultaneous interpreter. "
            "Your task is to translate English audio chunks into accurate and fluent "
            "Chinese."
        )
    else:
        text = (
            "You are a professional simultaneous interpreter. "
            f"Your task is to translate {source_lang} audio chunks into accurate and fluent "
            f"{target_lang}."
        )
    if rag_enabled:
        text += " Use the 'term_map' as a reference for terminology if provided."
    return text


def _trim_messages(state: StreamState, max_cache_chunks: int, keep_cache_chunks: int) -> None:
    body = state.messages[1:]
    if len(body) > 2 * max_cache_chunks:
        state.messages = [state.messages[0]] + body[-2 * keep_cache_chunks :]


def _prepare_vllm_input(
    *,
    state: StreamState,
    processor: Any,
    process_mm_info: Any,
    increment: np.ndarray,
    references: Sequence[Dict[str, Any]],
    rag_enabled: bool,
    term_map_format: str,
    empty_term_map_policy: str,
    vllm_prompt_audio_limit: int,
) -> Dict[str, Any]:
    if not state.messages:
        state.messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": _system_prompt(
                            state.source_lang,
                            state.target_lang,
                            state.lang_code,
                            rag_enabled=rag_enabled,
                        ),
                    }
                ],
            }
        )

    if vllm_prompt_audio_limit > 0:
        keep_pairs = max(0, vllm_prompt_audio_limit - 1)
        body = state.messages[1:]
        state.messages = [state.messages[0]] + body[-2 * keep_pairs :]

    user_content: List[Dict[str, Any]] = [{"type": "audio", "audio": increment}]
    kv = _format_term_map(references, term_map_format)
    if kv:
        user_content.append({"type": "text", "text": f"\n\nterm_map:\n{kv}"})
    elif rag_enabled and empty_term_map_policy == "none_block":
        user_content.append({"type": "text", "text": "\n\nterm_map:\nNONE"})
    state.messages.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(
        state.messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, _images, _videos = process_mm_info(state.messages, use_audio_in_video=False)
    return {
        "prompt": prompt,
        "multi_modal_data": {"audio": audios},
        "mm_processor_kwargs": {"use_audio_in_video": False},
    }


def _add_rasst_paths() -> None:
    for path in (RASST_EVAL_ROOT, RASST_CODE_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _patch_vllm_transformers_register_conflict() -> None:
    """vLLM 0.9.2 re-registers aimv2, which newer Transformers already ships."""
    from transformers import AutoConfig

    original_register = AutoConfig.register

    def safe_register(model_type, config, exist_ok=False):
        try:
            return original_register(model_type, config, exist_ok=exist_ok)
        except ValueError as exc:
            if model_type == "aimv2" and "already used" in str(exc):
                return None
            raise

    AutoConfig.register = safe_register


def _worker_main(worker_idx: int, gpu_token: str, command_queue: mp.Queue, result_queue: mp.Queue, config: Dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RASST_ACTIVE_CODE_ROOT", str(RASST_CODE_ROOT))
    _add_rasst_paths()

    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor
    _patch_vllm_transformers_register_conflict()
    from vllm import LLM, SamplingParams
    from agents.streaming_maxsim_retriever import (  # noqa: WPS433
        MAXSIM_STRIDE,
        MAXSIM_WINDOWS,
        StreamingMaxSimRetriever,
    )

    language_pair = config["language_pair"]
    lang_cfg = LANGUAGE_PAIRS[language_pair]
    worker_info = {
        "worker_idx": worker_idx,
        "gpu_token": str(gpu_token),
        "language_pair": language_pair,
        "model_path": lang_cfg["model_path"],
        "rag_index_path": lang_cfg["index_path"],
    }
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            worker_info["preload_free_gib"] = round(free / 1024**3, 3)
            worker_info["preload_total_gib"] = round(total / 1024**3, 3)

        processor = Qwen3OmniMoeProcessor.from_pretrained(lang_cfg["model_path"])
        llm_kwargs = {
            "model": lang_cfg["model_path"],
            "trust_remote_code": True,
            "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
            "tensor_parallel_size": int(config["vllm_tp_size"]),
            "max_num_seqs": int(config["max_num_seqs"]),
            "max_model_len": int(config["max_model_len"]),
            "enable_prefix_caching": bool(config["enable_prefix_caching"]),
            "enforce_eager": bool(config["vllm_enforce_eager"]),
        }
        if int(config["vllm_limit_audio"]) > 0:
            llm_kwargs["limit_mm_per_prompt"] = {"audio": int(config["vllm_limit_audio"])}
        if config["disable_custom_all_reduce"]:
            llm_kwargs["disable_custom_all_reduce"] = True
        result_queue.put({"type": "worker_loading_model", **worker_info, "llm_kwargs": llm_kwargs})
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            top_k=int(config["top_k"]),
            max_tokens=int(config["max_new_tokens"]),
            seed=int(config["seed"]),
        )

        retriever = None
        if bool(config["rag_enabled"]):
            result_queue.put({"type": "worker_loading_retriever", **worker_info})
            retriever = StreamingMaxSimRetriever(
                model_path=config["rag_model_path"],
                index_path=lang_cfg["index_path"],
                device=config["rag_device"],
                top_k=int(config["rag_top_k"]),
                lora_rank=int(config["rag_lora_r"]),
                text_lora_rank=int(config["rag_text_lora_r"]),
                target_lang=lang_cfg["lang_code"],
                window_sec=0.0,
                score_threshold=float(config["rag_score_threshold"]),
                maxsim_windows=MAXSIM_WINDOWS,
                maxsim_stride=MAXSIM_STRIDE,
            )

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            worker_info["loaded_free_gib"] = round(free / 1024**3, 3)
            worker_info["loaded_total_gib"] = round(total / 1024**3, 3)
        result_queue.put({"type": "worker_ready", **worker_info})
    except Exception as exc:
        result_queue.put({"type": "worker_error", **worker_info, "error": repr(exc)})
        raise

    states: Dict[str, StreamState] = {}
    pending: Deque[str] = deque()
    pending_set: Set[str] = set()
    segment_samples = int(float(config["vllm_segment_sec"]) * TARGET_SAMPLE_RATE)
    max_batch_size = int(config["scheduler_batch_size"])
    batch_timeout = float(config["batch_timeout"])

    def mark_pending(session_id: str) -> None:
        if session_id not in pending_set:
            pending.append(session_id)
            pending_set.add(session_id)

    def process_ready_batch() -> None:
        batch_session_ids: List[str] = []
        while pending and len(batch_session_ids) < max_batch_size:
            session_id = pending.popleft()
            pending_set.discard(session_id)
            state = states.get(session_id)
            if state is None:
                continue
            if state.cursor_samples - state.last_vllm_samples < segment_samples:
                continue
            batch_session_ids.append(session_id)

        if not batch_session_ids:
            return

        states_batch = [states[session_id] for session_id in batch_session_ids]
        increments: List[np.ndarray] = []
        for state in states_batch:
            increment = state.audio[state.last_vllm_samples : state.cursor_samples]
            if len(increment) < 15360:
                increment = np.pad(increment, (0, 15360 - len(increment)))
            increments.append(np.asarray(increment, dtype=np.float32))

        refs_by_state: List[List[Dict[str, Any]]]
        if retriever is None:
            refs_by_state = [[] for _ in states_batch]
        else:
            requests = []
            for state in states_batch:
                requests.append(
                    {
                        "audio_buffer": state.audio[: state.cursor_samples],
                        "current_start_sec": float(state.last_vllm_samples) / TARGET_SAMPLE_RATE,
                        "current_end_sec": float(state.cursor_samples) / TARGET_SAMPLE_RATE,
                        "lookback_sec": float(config["rag_timeline_lookback_sec"]),
                    }
                )
            t_rag = time.perf_counter()
            refs_by_state = retriever.retrieve_timeline_batch(
                requests,
                top_k=int(config["rag_top_k"]),
                lookback_sec=float(config["rag_timeline_lookback_sec"]),
            )
            result_queue.put(
                {
                    "type": "batch_rag_done",
                    "worker_idx": worker_idx,
                    "batch_size": len(states_batch),
                    "elapsed_s": round(time.perf_counter() - t_rag, 6),
                }
            )

        prepared: List[Dict[str, Any]] = []
        for state, increment, refs in zip(states_batch, increments, refs_by_state):
            prepared.append(
                _prepare_vllm_input(
                    state=state,
                    processor=processor,
                    process_mm_info=process_mm_info,
                    increment=increment,
                    references=refs,
                    rag_enabled=retriever is not None,
                    term_map_format=config["term_map_format"],
                    empty_term_map_policy=config["empty_term_map_policy"],
                    vllm_prompt_audio_limit=int(config["vllm_limit_audio"]),
                )
            )

        t_gen = time.perf_counter()
        outputs = llm.generate(prepared, sampling_params=sampling_params, use_tqdm=False)
        gen_elapsed = time.perf_counter() - t_gen
        result_queue.put(
            {
                "type": "batch_llm_done",
                "worker_idx": worker_idx,
                "batch_size": len(states_batch),
                "elapsed_s": round(gen_elapsed, 6),
            }
        )

        for state, output in zip(states_batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            state.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            _trim_messages(
                state,
                max_cache_chunks=int(config["max_cache_chunks"]),
                keep_cache_chunks=int(config["keep_cache_chunks"]),
            )
            state.last_vllm_samples = state.cursor_samples
            state.segment_idx += 1
            result_queue.put(
                {
                    "type": "translation",
                    "session_id": state.session_id,
                    "worker_idx": worker_idx,
                    "text": text,
                    "segment_idx": state.segment_idx,
                    "cursor_samples": state.cursor_samples,
                    "batch_size": len(states_batch),
                    "llm_elapsed_s": round(gen_elapsed, 6),
                }
            )
            if state.cursor_samples - state.last_vllm_samples >= segment_samples:
                mark_pending(state.session_id)

    while True:
        try:
            command = command_queue.get(timeout=batch_timeout)
        except queue.Empty:
            process_ready_batch()
            continue

        command_type = command.get("type")
        if command_type == "stop":
            result_queue.put({"type": "worker_stopped", "worker_idx": worker_idx})
            return
        if command_type == "init":
            session_id = command["session_id"]
            states[session_id] = StreamState(
                session_id=session_id,
                language_pair=language_pair,
                source_lang=lang_cfg["source_lang"],
                target_lang=lang_cfg["target_lang"],
                lang_code=lang_cfg["lang_code"],
            )
            result_queue.put({"type": "session_ready", "session_id": session_id, "worker_idx": worker_idx})
            continue
        if command_type == "delete":
            states.pop(command["session_id"], None)
            continue
        if command_type == "audio":
            session_id = command["session_id"]
            state = states.get(session_id)
            if state is None:
                result_queue.put(
                    {
                        "type": "translation_error",
                        "session_id": session_id,
                        "worker_idx": worker_idx,
                        "error": "session not found in RASST worker",
                    }
                )
                continue
            chunk = np.asarray(command["audio"], dtype=np.float32).flatten()
            if chunk.size == 0:
                continue
            state.audio = np.concatenate([state.audio, chunk])
            state.cursor_samples = int(state.audio.shape[0])
            if state.cursor_samples - state.last_vllm_samples >= segment_samples:
                mark_pending(session_id)
            process_ready_batch()


class RasstRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.command_queues: List[mp.Queue] = []
        self.processes: List[mp.Process] = []
        self.worker_status: Dict[int, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_queues: Dict[str, asyncio.Queue] = {}
        self.next_worker = 0
        self.router_task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return bool(self.worker_status) and all(
            item.get("status") == "ready" for item in self.worker_status.values()
        )

    def _worker_config(self) -> Dict[str, Any]:
        return {
            "language_pair": self.args.language_pair,
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
            "vllm_tp_size": self.args.vllm_tp_size,
            "max_num_seqs": self.args.max_num_seqs,
            "max_model_len": self.args.max_model_len,
            "vllm_limit_audio": self.args.vllm_limit_audio,
            "enable_prefix_caching": self.args.enable_prefix_caching,
            "vllm_enforce_eager": self.args.vllm_enforce_eager,
            "safetensors_load_strategy": self.args.safetensors_load_strategy,
            "disable_custom_all_reduce": self.args.disable_custom_all_reduce,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_new_tokens": self.args.max_new_tokens,
            "seed": self.args.seed,
            "rag_enabled": self.args.rag_enabled,
            "rag_model_path": self.args.rag_model_path,
            "rag_device": self.args.rag_device,
            "rag_top_k": self.args.rag_top_k,
            "rag_lora_r": self.args.rag_lora_r,
            "rag_text_lora_r": self.args.rag_text_lora_r,
            "rag_score_threshold": self.args.rag_score_threshold,
            "rag_timeline_lookback_sec": self.args.rag_timeline_lookback_sec,
            "term_map_format": self.args.term_map_format,
            "empty_term_map_policy": self.args.empty_term_map_policy,
            "vllm_segment_sec": self.args.vllm_segment_sec,
            "scheduler_batch_size": self.args.scheduler_batch_size,
            "batch_timeout": self.args.batch_timeout,
            "max_cache_chunks": self.args.max_cache_chunks,
            "keep_cache_chunks": self.args.keep_cache_chunks,
        }

    async def start(self) -> None:
        gpu_tokens = _split_worker_gpu_groups(self.args.worker_gpus, int(self.args.vllm_tp_size))
        if not gpu_tokens:
            raise RuntimeError("No RASST worker GPUs configured")
        for idx, gpu_token in enumerate(gpu_tokens):
            command_queue = self.ctx.Queue()
            proc = self.ctx.Process(
                target=_worker_main,
                args=(idx, gpu_token, command_queue, self.result_queue, self._worker_config()),
                daemon=False,
            )
            proc.start()
            self.command_queues.append(command_queue)
            self.processes.append(proc)
            self.worker_status[idx] = {"status": "starting", "gpu_token": gpu_token, "pid": proc.pid}
        self.router_task = asyncio.create_task(self._result_router())

    async def stop(self) -> None:
        for command_queue in self.command_queues:
            command_queue.put({"type": "stop"})
        for proc in self.processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        if self.router_task:
            self.router_task.cancel()

    async def _result_router(self) -> None:
        while True:
            item = await asyncio.to_thread(self.result_queue.get)
            item_type = item.get("type")
            worker_idx = item.get("worker_idx")
            if item_type == "worker_ready":
                self.worker_status[int(worker_idx)] = {"status": "ready", **item}
            elif item_type == "worker_error":
                self.worker_status[int(worker_idx)] = {"status": "error", **item}
            elif item_type in {"worker_loading_model", "worker_loading_retriever", "batch_rag_done", "batch_llm_done"}:
                if worker_idx is not None:
                    current = self.worker_status.get(int(worker_idx), {})
                    current.update({"status": current.get("status", "starting"), "last_event": item})
                    self.worker_status[int(worker_idx)] = current
                print(json.dumps(item, ensure_ascii=False), flush=True)
            elif item_type in {"translation", "translation_error"}:
                session_id = item["session_id"]
                q = self.session_queues.get(session_id)
                if q:
                    await q.put(item)
            elif item_type == "session_ready":
                pass

    def init_session(self, agent_type: str, language_pair: str, client_id: Optional[str], latency_multiplier: int) -> str:
        if agent_type != "RASST":
            raise HTTPException(status_code=400, detail="This server only serves the RASST model")
        if language_pair != self.args.language_pair:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This RASST process is loaded for {self.args.language_pair}; "
                    f"restart with --language-pair {language_pair!r} to serve that direction."
                ),
            )
        if not self.ready:
            raise HTTPException(status_code=503, detail="RASST workers are still loading")

        worker_idx = self.next_worker % len(self.command_queues)
        self.next_worker += 1
        timestamp = int(time.time() * 1000)
        suffix = client_id or str(timestamp)
        safe_pair = language_pair.replace(" ", "").replace("->", "2")
        session_id = f"RASST_{safe_pair}_{suffix}_{timestamp}"
        self.active_sessions[session_id] = {
            "worker_idx": worker_idx,
            "language_pair": language_pair,
            "created_at": time.time(),
            "latency_multiplier": latency_multiplier,
        }
        self.session_queues[session_id] = asyncio.Queue()
        self.command_queues[worker_idx].put({"type": "init", "session_id": session_id})
        return session_id

    def delete_session(self, session_id: str) -> bool:
        session = self.active_sessions.pop(session_id, None)
        self.session_queues.pop(session_id, None)
        if not session:
            return False
        self.command_queues[int(session["worker_idx"])].put({"type": "delete", "session_id": session_id})
        return True

    def submit_audio(self, session_id: str, audio: np.ndarray) -> None:
        session = self.active_sessions[session_id]
        self.command_queues[int(session["worker_idx"])].put(
            {"type": "audio", "session_id": session_id, "audio": np.asarray(audio, dtype=np.float32)}
        )

    def health(self) -> Dict[str, Any]:
        status = "healthy" if self.ready else "starting"
        if any(item.get("status") == "error" for item in self.worker_status.values()):
            status = "error"
        return {
            "status": status,
            "backend": "rasst_qwen3_omni_vllm_maxsim_rag",
            "model": "RASST",
            "language_pair": self.args.language_pair,
            "supported_languages": list(LANGUAGE_PAIRS.keys()),
            "loaded_language_pair": self.args.language_pair,
            "worker_count": len(self.worker_status),
            "workers": self.worker_status,
            "active_sessions": len(self.active_sessions),
            "mock_mode": False,
            "tp_size_per_worker": self.args.vllm_tp_size,
            "rag_device_per_worker": self.args.rag_device,
        }


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
runtime: Optional[RasstRuntime] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    print(f"RASST server exception: {exc}", flush=True)
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.on_event("startup")
async def startup_event():
    if runtime is None:
        raise RuntimeError("RASST runtime was not configured")
    await runtime.start()


@app.on_event("shutdown")
async def shutdown_event():
    if runtime is not None:
        await runtime.stop()


@app.get("/config")
async def get_config():
    return {
        "models": MODEL_PROFILES,
        "language_pairs": [
            {"id": key, "label": key.replace("->", "→"), "available": runtime.args.language_pair == key}
            for key in LANGUAGE_PAIRS
        ],
        "default_model": "RASST",
        "loaded_language_pair": runtime.args.language_pair if runtime else None,
    }


@app.get("/health")
async def health_check():
    if runtime is None:
        return {"status": "error", "error": "runtime not configured"}
    return runtime.health()


@app.post("/init")
async def initialize_translation(
    agent_type: str,
    language_pair: str,
    latency_multiplier: int = 2,
    client_id: Optional[str] = None,
):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not configured")
    session_id = runtime.init_session(agent_type, language_pair, client_id, latency_multiplier)
    return {
        "session_id": session_id,
        "queued": False,
        "queue_position": 0,
        "scheduler_based": False,
        "rasst_backend": True,
    }


@app.post("/delete_session")
async def delete_session(session_id: str):
    if runtime is None:
        return {"success": False, "error": "runtime not configured"}
    return {"success": runtime.delete_session(session_id)}


@app.post("/ping")
async def ping_session(session_id: str):
    if runtime is None or session_id not in runtime.active_sessions:
        return {"success": False, "error": "Invalid session ID"}
    return {"success": True}


@app.websocket("/wss/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if runtime is None or session_id not in runtime.active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
    result_queue = runtime.session_queues[session_id]
    await websocket.send_text("READY: RASST workers ready")

    async def sender():
        while True:
            item = await result_queue.get()
            if item.get("type") == "translation_error":
                await websocket.send_text(f"ERROR: {item.get('error')}")
            else:
                await websocket.send_text(str(item.get("text", "")))

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                runtime.submit_audio(session_id, audio)
            elif "text" in message:
                text = message["text"]
                if text == "EOF":
                    await websocket.send_text("PROCESSING_COMPLETE: File processing finished")
    except Exception:
        pass
    finally:
        sender_task.cancel()


@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parser = argparse.ArgumentParser(description="RASST demo FastAPI server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--worker-gpus", default=os.environ.get("RASST_WORKER_GPUS", visible or "0,1"))
    parser.add_argument("--vllm-tp-size", type=int, default=int(os.environ.get("RASST_VLLM_TP_SIZE", "1")))
    parser.add_argument("--language-pair", default=os.environ.get("RASST_DEMO_LANGUAGE_PAIR", "English -> Chinese"))
    parser.add_argument(
        "--rag-model-path",
        default=os.environ.get("RASST_HN1024_RETRIEVER", str(PROJECT_ROOT / "checkpoints/retriever/rasst-hn1024.pt")),
    )
    parser.add_argument("--rag-enabled", type=int, default=int(os.environ.get("RASST_RAG_ENABLED", "1")))
    parser.add_argument("--rag-device", default=os.environ.get("RASST_RAG_DEVICE", "cuda:0"))
    parser.add_argument("--rag-top-k", type=int, default=int(os.environ.get("RASST_RAG_TOP_K", "10")))
    parser.add_argument("--rag-score-threshold", type=float, default=float(os.environ.get("RASST_RAG_SCORE_THRESHOLD", "0.78")))
    parser.add_argument("--rag-lora-r", type=int, default=128)
    parser.add_argument("--rag-text-lora-r", type=int, default=128)
    parser.add_argument("--rag-timeline-lookback-sec", type=float, default=DEFAULT_RAG_LOOKBACK_SEC)
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("RASST_GPU_MEMORY_UTILIZATION", "0.86")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("RASST_MAX_NUM_SEQS", "16")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("RASST_MAX_MODEL_LEN", "16384")))
    parser.add_argument("--vllm-limit-audio", type=int, default=int(os.environ.get("RASST_VLLM_LIMIT_AUDIO", "16")))
    parser.add_argument("--enable-prefix-caching", type=int, default=int(os.environ.get("RASST_ENABLE_PREFIX_CACHING", "1")))
    parser.add_argument("--vllm-enforce-eager", type=int, default=int(os.environ.get("RASST_VLLM_ENFORCE_EAGER", "0")))
    parser.add_argument("--safetensors-load-strategy", default=os.environ.get("RASST_SAFETENSORS_LOAD_STRATEGY", "lazy"))
    parser.add_argument("--disable-custom-all-reduce", type=int, default=int(os.environ.get("RASST_DISABLE_CUSTOM_ALL_REDUCE", "0")))
    parser.add_argument("--vllm-segment-sec", type=float, default=float(os.environ.get("RASST_VLLM_SEGMENT_SEC", str(DEFAULT_VLLM_SEGMENT_SEC))))
    parser.add_argument("--scheduler-batch-size", type=int, default=int(os.environ.get("RASST_SCHEDULER_BATCH_SIZE", "16")))
    parser.add_argument("--batch-timeout", type=float, default=float(os.environ.get("RASST_BATCH_TIMEOUT", "0.05")))
    parser.add_argument("--max-cache-chunks", type=int, default=int(os.environ.get("RASST_MAX_CACHE_CHUNKS", str(DEFAULT_MAX_CACHE_CHUNKS))))
    parser.add_argument("--keep-cache-chunks", type=int, default=int(os.environ.get("RASST_KEEP_CACHE_CHUNKS", str(DEFAULT_KEEP_CACHE_CHUNKS))))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("RASST_MAX_NEW_TOKENS", "40")))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=998244353)
    parser.add_argument("--term-map-format", default=os.environ.get("RASST_TERM_MAP_FORMAT", "tagged"))
    parser.add_argument("--empty-term-map-policy", default=os.environ.get("RASST_EMPTY_TERM_MAP_POLICY", "none_block"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global runtime
    args = parse_args(argv)
    if args.language_pair not in LANGUAGE_PAIRS:
        raise SystemExit(f"Unsupported language pair: {args.language_pair}")
    if bool(args.rag_enabled) and not Path(args.rag_model_path).is_file():
        raise SystemExit(f"RAG checkpoint not found: {args.rag_model_path}")
    index_path = Path(LANGUAGE_PAIRS[args.language_pair]["index_path"])
    if bool(args.rag_enabled) and not index_path.is_file():
        raise SystemExit(f"RAG index not found: {index_path}")
    runtime = RasstRuntime(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
