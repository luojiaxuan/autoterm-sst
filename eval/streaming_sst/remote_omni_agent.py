#!/usr/bin/env python3
"""SimulEval-native streaming SST agent backed by a *remote* omni engine.

This mirrors the RASST demo streaming policy (``framework/agents/omni.py``)
exactly, except generation is delegated to an external OpenAI-style server so we
can A/B the engine (vLLM vs sglang-omni) with one agent:

* every ``source-segment-size`` ms of new audio -> one WRITE: the new increment
  is sent to the engine and its translation is emitted;
* multi-turn chat carries prior *translations as text* (the audio increment is
  not resent), matching the demo's ``given_chunks`` streaming behavior;
* no-RAG: no term_map is attached (pure Qwen3-Omni-30B-A3B).

SimulEval records per-write source-time delays and computation-aware timing, so
the resulting ``instances.log`` is scored for BLEU / StreamLAAL / StreamLAAL_CA
by the canonical ``offline_streamlaal_eval.py`` (FBK ``stream_laal_term.py``).

Run (one stream):

  simuleval --agent eval/streaming_sst/remote_omni_agent.py \
    --source seg/segments.source --target seg/segments.target \
    --output OUT --source-segment-size 1920 \
    --quality-metrics BLEU --eval-latency-unit char --sacrebleu-tokenizer zh \
    --remote-engine sglang --remote-base-url http://127.0.0.1:8100 \
    --remote-model-name qwen3-omni --source-lang English --target-lang Chinese
"""

# NOTE: do NOT add `from __future__ import annotations`. SimulEval loads this
# file via spec_from_file_location("agents", ...) WITHOUT registering it in
# sys.modules; with stringized annotations, @dataclass's KW_ONLY probe does
# sys.modules["agents"].__dict__ and crashes. Real type objects avoid that path.

import base64
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint

TARGET_SR = 16000
# Qwen3-Omni expects >= ~0.96 s of audio per turn; pad short tails.
MIN_AUDIO_SAMPLES = 15360


def build_system_prompt(source_lang: str, target_lang: str) -> str:
    """Inlined from framework/agents/plugins/prompt.py (en->zh training prompt)."""
    sl, tl = source_lang.strip().lower(), target_lang.strip().lower()
    if sl in {"english", "en"} and tl in {"chinese", "zh", "zh-cn", "中文"}:
        return (
            "You are a professional simultaneous interpreter. "
            "Your task is to translate English audio chunks into accurate and fluent "
            "Chinese. Use the 'term_map' as a reference for terminology if provided."
        )
    return (
        f"You are a professional simultaneous interpreter. "
        f"You will be given chunks of {source_lang} audio and you need to "
        f"translate the audio into {target_lang} text."
    )


def _wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    import soundfile as sf  # noqa: WPS433

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class RemoteOmniStates(AgentStates):
    consumed: int = 0
    seg_idx: int = 0
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user_text, assistant_text)

    def reset(self) -> None:
        super().reset()
        self.consumed = 0
        self.seg_idx = 0
        self.history = []


@entrypoint
class RemoteOmniAgent(SpeechToTextAgent):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.engine = args.remote_engine
        self.base_url = args.remote_base_url.rstrip("/")
        self.model_name = args.remote_model_name
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        self.min_start_sec = float(args.min_start_sec)
        self.max_new_tokens = int(args.max_new_tokens)
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.top_k = int(args.top_k)
        self.seed = int(args.seed)
        self.keep_cache_chunks = int(args.keep_cache_chunks)
        self.request_timeout = float(args.request_timeout)
        self.system_prompt = build_system_prompt(self.source_lang, self.target_lang)
        self.tmp_dir = args.segment_tmp_dir or f"/dev/shm/remote_omni_{os.getpid()}"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._worker = getattr(args, "worker_id", 0)
        self._req = 0
        # Optional per-turn latency capture for profiling (env-gated: no effect
        # unless REMOTE_OMNI_LAT_DIR is set). One append-mode file per worker.
        self._lat_fh = None
        _lat_dir = os.environ.get("REMOTE_OMNI_LAT_DIR")
        if _lat_dir:
            os.makedirs(_lat_dir, exist_ok=True)
            self._lat_fh = open(os.path.join(_lat_dir, f"w{self._worker}.lat"), "a")

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--remote-engine", choices=["vllm", "sglang"], required=True)
        parser.add_argument("--remote-base-url", required=True)
        parser.add_argument("--remote-model-name", default="qwen3-omni")
        parser.add_argument("--source-lang", default="English")
        parser.add_argument("--target-lang", default="Chinese")
        parser.add_argument("--min-start-sec", type=float, default=0.96)
        parser.add_argument("--max-new-tokens", type=int, default=40)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top-p", type=float, default=0.9)
        parser.add_argument("--top-k", type=int, default=50)
        parser.add_argument("--seed", type=int, default=998244353)
        parser.add_argument("--keep-cache-chunks", type=int, default=8)
        parser.add_argument("--request-timeout", type=float, default=120.0)
        parser.add_argument("--segment-tmp-dir", default="")
        parser.add_argument("--worker-id", type=int, default=0)

    def build_states(self) -> RemoteOmniStates:
        return RemoteOmniStates()

    def policy(self, states: Optional[RemoteOmniStates] = None) -> Any:
        if states is None:
            states = self.states
        sr = states.source_sample_rate or TARGET_SR
        n = len(states.source)
        length_s = n / sr if sr else 0.0

        if not states.source_finished and length_s < self.min_start_sec:
            return ReadAction()
        new_samples = n - states.consumed
        if not states.source_finished and new_samples <= 0:
            return ReadAction()
        if states.source_finished and new_samples <= 0:
            return WriteAction(content="", finished=True)

        increment = np.asarray(states.source[states.consumed:n], dtype=np.float32)
        states.consumed = n
        states.seg_idx += 1

        text = ""
        try:
            text = self._translate(states, increment)
        except Exception as exc:  # noqa: BLE001 - never kill the whole eval run
            print(f"[remote_omni] worker={self._worker} generate error: {exc!r}", flush=True)
        return WriteAction(content=text, finished=bool(states.source_finished))

    def _translate(self, states: RemoteOmniStates, increment: np.ndarray) -> str:
        if increment.shape[0] < MIN_AUDIO_SAMPLES:
            increment = np.pad(increment, (0, MIN_AUDIO_SAMPLES - increment.shape[0]))
        self._req += 1
        if self.engine == "sglang":
            text = self._call_sglang(states, increment)
        else:
            text = self._call_vllm(states, increment)
        text = (text or "").strip()
        states.history.append(("", text))
        if len(states.history) > self.keep_cache_chunks:
            states.history = states.history[-self.keep_cache_chunks:]
        return text

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        import time as _time  # noqa: WPS433
        _t0 = _time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if self._lat_fh is not None:
            self._lat_fh.write(f"{(_time.perf_counter() - _t0) * 1000.0:.1f}\n")
            self._lat_fh.flush()
        return body

    @staticmethod
    def _content_text(body: Dict[str, Any]) -> str:
        choices = body.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return str(content or "")

    def _history_text_turns(self, states: RemoteOmniStates) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []
        for user_text, asst_text in states.history:
            turns.append({"role": "user", "content": user_text})
            turns.append({"role": "assistant", "content": asst_text})
        return turns

    def _call_sglang(self, states: RemoteOmniStates, increment: np.ndarray) -> str:
        wav_path = os.path.join(
            self.tmp_dir, f"w{self._worker}_{states.seg_idx:05d}_{self._req}.wav"
        )
        import soundfile as sf  # noqa: WPS433

        sf.write(wav_path, increment, TARGET_SR)
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        messages += self._history_text_turns(states)
        messages.append({"role": "user", "content": ""})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "audios": [wav_path],
            "audio_target_sr": TARGET_SR,
            "modalities": ["text"],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "stream": False,
        }
        try:
            return self._content_text(self._post(payload))
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _call_vllm(self, states: RemoteOmniStates, increment: np.ndarray) -> str:
        b64 = base64.b64encode(_wav_bytes(increment)).decode("ascii")
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        messages += self._history_text_turns(states)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
                ],
            }
        )
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "stream": False,
        }
        return self._content_text(self._post(payload))
