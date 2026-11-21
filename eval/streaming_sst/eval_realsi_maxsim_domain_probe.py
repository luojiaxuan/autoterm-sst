#!/usr/bin/env python3
"""Evaluate ten-way speech-window MaxSim domain probes on RealSI audio."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import DOMAIN_TO_PRESET, WORKING_DOMAINS
from framework.agents.term_memory.manifest import TermMemoryManifest


DOMAIN_AUDIO = {
    "nlp": "en2zh-01-tech.wav",
    "medicine": "en2zh-02-health.wav",
    "education": "en2zh-03-edu.wav",
    "finance": "en2zh-04-fin.wav",
    "legal": "en2zh-05-law.wav",
    "environment": "en2zh-06-env.wav",
    "entertainment": "en2zh-07-ent.wav",
    "science": "en2zh-08-sci.wav",
    "sports": "en2zh-09-sport.wav",
    "art": "en2zh-10-art.wav",
}


def _audio_dir(root: Path) -> Path:
    for candidate in (root / "en2zh" / "wav", root / "data" / "en2zh" / "wav"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"RealSI en2zh audio not found under {root}")


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sample width {sample_width}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float,
    step_sec: float,
    max_windows: int,
) -> List[Tuple[float, float, np.ndarray]]:
    window_samples = max(1, int(round(window_sec * sample_rate)))
    step_samples = max(1, int(round(step_sec * sample_rate)))
    result: List[Tuple[float, float, np.ndarray]] = []
    for end in range(window_samples, len(audio) + step_samples, step_samples):
        end = min(end, len(audio))
        start = max(0, end - window_samples)
        result.append((start / sample_rate, end / sample_rate, audio[start:end]))
        if len(result) >= max_windows or end == len(audio):
            break
    return result


def _load_runtime(retriever_dir: Path):
    eval_root = retriever_dir.parent / "eval"
    for path in (eval_root, retriever_dir.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from agents.streaming_maxsim_retriever import (  # type: ignore
        MAXSIM_STRIDE,
        MAXSIM_WINDOWS,
        StreamingMaxSimRetriever,
        _encode_audio,
    )

    return StreamingMaxSimRetriever, _encode_audio, MAXSIM_WINDOWS, MAXSIM_STRIDE


def _index_paths(manifest: TermMemoryManifest, domains: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for domain in domains:
        path = manifest.maxsim_index(DOMAIN_TO_PRESET[domain], "zh")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"missing index for {domain}: {path}")
        result[domain] = path
    return result


def evaluate(
    *,
    realsi_root: Path,
    manifest_path: Path,
    retriever_dir: Path,
    model_path: Path,
    device_name: str,
    domains: Sequence[str],
    window_sec: float,
    step_sec: float,
    max_windows: int,
    top_k: int,
) -> Dict[str, Any]:
    manifest = TermMemoryManifest.load(str(manifest_path))
    indexes = _index_paths(manifest, domains)
    runtime_cls, encode_audio, maxsim_windows, maxsim_stride = _load_runtime(retriever_dir)
    first_domain = domains[0]
    runtime = runtime_cls(
        model_path=str(model_path),
        index_path=indexes[first_domain],
        device=device_name,
        top_k=top_k,
        lora_rank=128,
        text_lora_rank=128,
        target_lang="zh",
        window_sec=0.0,
        score_threshold=0.0,
        maxsim_windows=maxsim_windows,
        maxsim_stride=maxsim_stride,
    )
    device = torch.device(device_name)
    text_indexes: Dict[str, Dict[str, Any]] = {}
    for domain, path in indexes.items():
        payload = torch.load(path, map_location="cpu")
        text_indexes[domain] = {
            "text_embs": F.normalize(payload["text_embs"].to(device).float(), p=2, dim=-1),
            "term_list": payload["term_list"],
        }

    records: List[Dict[str, Any]] = []
    audio_dir = _audio_dir(realsi_root)
    for expected in domains:
        audio, sample_rate = load_wav(audio_dir / DOMAIN_AUDIO[expected])
        if sample_rate != 16000:
            raise ValueError(f"{DOMAIN_AUDIO[expected]}: expected 16 kHz, got {sample_rate}")
        for window_index, (start_s, end_s, chunk) in enumerate(
            audio_windows(
                audio,
                sample_rate,
                window_sec=window_sec,
                step_sec=step_sec,
                max_windows=max_windows,
            ),
            start=1,
        ):
            speech = encode_audio(chunk, runtime.retriever, runtime.feat_ext, device)
            speech = F.normalize(speech.reshape(-1, speech.shape[-1]).float(), p=2, dim=-1)
            scores: Dict[str, Dict[str, Any]] = {}
            for domain, payload in text_indexes.items():
                per_term = speech.matmul(payload["text_embs"].T).max(dim=0).values
                n = min(max(1, top_k), int(per_term.numel()))
                values, indices = per_term.topk(n, largest=True, sorted=True)
                term_list = payload["term_list"]
                terms = [str(term_list[int(index)].get("term") or "") for index in indices]
                scores[domain] = {
                    "top_score": float(values[0]),
                    "mean_topk_score": float(values.mean()),
                    "top_terms": terms,
                }
            ranked = sorted(
                scores,
                key=lambda domain: (scores[domain]["top_score"], scores[domain]["mean_topk_score"]),
                reverse=True,
            )
            records.append(
                {
                    "expected_domain": expected,
                    "window": window_index,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "predicted_domain": ranked[0],
                    "margin": round(
                        scores[ranked[0]]["top_score"] - scores[ranked[1]]["top_score"],
                        6,
                    ),
                    "ranking": ranked,
                    "scores": scores,
                }
            )

    correct = sum(1 for item in records if item["predicted_domain"] == item["expected_domain"])
    per_domain = {
        domain: {
            "windows": sum(1 for item in records if item["expected_domain"] == domain),
            "correct": sum(
                1
                for item in records
                if item["expected_domain"] == domain and item["predicted_domain"] == domain
            ),
        }
        for domain in domains
    }
    for row in per_domain.values():
        row["accuracy"] = round(row["correct"] / row["windows"] if row["windows"] else 0.0, 4)
    return {
        "domains": list(domains),
        "windows": len(records),
        "accuracy": round(correct / len(records) if records else 0.0, 4),
        "per_domain": per_domain,
        "records": records,
        "settings": {
            "window_sec": window_sec,
            "step_sec": step_sec,
            "max_windows": max_windows,
            "top_k": top_k,
            "device": device_name,
            "manifest": str(manifest_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realsi-root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--retriever-dir", required=True, type=Path)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--device", required=True)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--step-sec", type=float, default=30.0)
    ap.add_argument("--max-windows", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    result = evaluate(
        realsi_root=args.realsi_root,
        manifest_path=args.manifest,
        retriever_dir=args.retriever_dir,
        model_path=args.model_path,
        device_name=args.device,
        domains=domains,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        max_windows=args.max_windows,
        top_k=args.top_k,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"windows={result['windows']} accuracy={result['accuracy']:.4f}")
    for domain, row in result["per_domain"].items():
        print(f"{domain:13s} {row['correct']}/{row['windows']} accuracy={row['accuracy']:.4f}")
    print(f"wrote {args.out_json}")
    if not args.no_assert and result["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
