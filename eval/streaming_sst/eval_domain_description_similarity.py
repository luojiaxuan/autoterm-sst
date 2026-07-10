#!/usr/bin/env python3
"""Evaluate BGE-M3 similarity between stream context and domain descriptions."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.eval_realsi_domain_routing import load_windows
from framework.agents.term_memory.domain_taxonomy import (
    DOMAIN_TO_PRESET,
    DOMAIN_ROUTER_PROTOTYPES,
    WORKING_DOMAINS,
)
from framework.agents.term_memory.topic_router import (
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


def _load_builder(retriever_dir: Path):
    sys.path.insert(0, str(retriever_dir))
    return importlib.import_module("build_maxsim_index")


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    *,
    encoder: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    encoder_mode: str,
) -> torch.Tensor:
    batches: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if encoder_mode == "base":
                outputs = encoder(**tokenized)
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                embeddings = encoder(tokenized.input_ids, tokenized.attention_mask)
        batches.append(F.normalize(embeddings.float(), p=2, dim=-1))
    return torch.cat(batches, dim=0)


def evaluate(
    *,
    realsi_root: Path,
    retriever_dir: Path,
    model_path: Path,
    device_name: str,
    domains: Sequence[str],
    window_segments: int,
    step_segments: int,
    batch_size: int,
    encoder_mode: str,
) -> Dict[str, Any]:
    device = torch.device(device_name)
    if encoder_mode == "base":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        encoder = AutoModel.from_pretrained("BAAI/bge-m3").to(device).eval()
    else:
        builder = _load_builder(retriever_dir)
        encoder, tokenizer = builder.build_text_encoder(
            device,
            builder.TEXT_LORA_RANK,
            builder.TEXT_LORA_ALPHA,
        )
        builder.load_text_checkpoint(encoder, str(model_path), device)

    prototype_texts = [text for domain in domains for text in DOMAIN_ROUTER_PROTOTYPES[domain]]
    prototype_embeddings = encode_texts(
        prototype_texts,
        encoder=encoder,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        encoder_mode=encoder_mode,
    )
    prototypes_per_domain = len(DOMAIN_ROUTER_PROTOTYPES[domains[0]])
    centroids = prototype_embeddings.reshape(len(domains), prototypes_per_domain, -1).mean(dim=1)
    centroids = F.normalize(centroids, p=2, dim=-1)

    records: List[Dict[str, Any]] = []
    for expected in domains:
        windows = load_windows(
            realsi_root,
            expected,
            text_field="trg_text",
            window_segments=window_segments,
            step_segments=step_segments,
        )
        query_embeddings = encode_texts(
            [item.text for item in windows],
            encoder=encoder,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            encoder_mode=encoder_mode,
        )
        similarities = query_embeddings @ centroids.T
        for window, scores in zip(windows, similarities):
            values, indices = scores.topk(k=min(2, len(domains)), largest=True, sorted=True)
            predicted = domains[int(indices[0])]
            records.append(
                {
                    "expected_domain": expected,
                    "predicted_domain": predicted,
                    "start_segment": window.start_segment,
                    "end_segment": window.end_segment,
                    "text": window.text,
                    "margin": round(float(values[0] - values[1]), 6),
                    "scores": {
                        domain: round(float(scores[index]), 6)
                        for index, domain in enumerate(domains)
                    },
                }
            )

    correct = sum(1 for item in records if item["expected_domain"] == item["predicted_domain"])
    per_domain = {}
    for domain in domains:
        selected = [item for item in records if item["expected_domain"] == domain]
        domain_correct = sum(1 for item in selected if item["predicted_domain"] == domain)
        per_domain[domain] = {
            "windows": len(selected),
            "correct": domain_correct,
            "accuracy": round(domain_correct / len(selected) if selected else 0.0, 4),
        }
    result = {
        "domains": list(domains),
        "windows": len(records),
        "accuracy": round(correct / len(records) if records else 0.0, 4),
        "per_domain": per_domain,
        "records": records,
        "settings": {
            "model_path": str(model_path),
            "device": device_name,
            "window_segments": window_segments,
            "step_segments": step_segments,
            "prototype_sentences_per_domain": prototypes_per_domain,
            "encoder_mode": encoder_mode,
        },
    }
    result["routing"] = evaluate_routing(records, domains)
    return result


def evaluate_routing(records: Sequence[Dict[str, Any]], domains: Sequence[str]) -> Dict[str, Any]:
    router = HybridWindowTopicRouter(
        [
            DomainSlice(DOMAIN_TO_PRESET[domain], domain, index_path=f"mock://{domain}")
            for domain in domains
        ],
        RouterConfig(
            warmup_sec=0.0,
            update_interval_sec=0.0,
            switch_cooldown_sec=0.0,
            min_confidence=0.60,
            min_margin=0.15,
            min_current_margin=0.10,
            min_consistent_windows_generated_target=3,
            context_similarity_weight=0.60,
            text_topic_weight=0.25,
            domain_probe_weight=0.10,
            speech_centroid_weight=0.03,
            metadata_prior_weight=0.02,
        ),
    )
    first_domain = domains[0]
    state = RouterSessionState(DOMAIN_TO_PRESET[first_domain], first_domain, created_s=0.0)
    now_s = 0.0
    routed: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    wrong_switches = 0
    for block_index, domain in enumerate(domains):
        selected = [item for item in records if item["expected_domain"] == domain]
        first_active = 0 if state.active_domain_id == domain else None
        for window_index, item in enumerate(selected, start=1):
            now_s += 1.0
            decision = router.observe(
                state,
                None,
                [],
                now_s,
                router_text=str(item["text"]),
                router_text_source="generated_target",
                context_similarity_scores=dict(item["scores"]),
            )
            if decision.action == "switch":
                if decision.target_domain_id != domain:
                    wrong_switches += 1
                state.active_preset_id = decision.target_preset_id
                state.active_domain_id = decision.target_domain_id
                state.last_switch_s = now_s
                state.pending_preset_id = None
            if first_active is None and state.active_domain_id == domain:
                first_active = window_index
            routed.append(
                {
                    "expected_domain": domain,
                    "window": window_index,
                    "active_domain": state.active_domain_id,
                    "decision_action": decision.action,
                    "decision_target_domain": decision.target_domain_id,
                    "confidence": decision.confidence,
                    "margin": decision.margin,
                    "reason": decision.reason,
                }
            )
        transitions.append(
            {
                "domain": domain,
                "first_active_window": first_active,
                "within_limit": bool(
                    first_active is not None and first_active <= (0 if block_index == 0 else 5)
                ),
            }
        )
    steady = [item for item in routed if int(item["window"]) > 3]
    steady_correct = sum(
        1 for item in steady if item["active_domain"] == item["expected_domain"]
    )
    steady_accuracy = steady_correct / len(steady) if steady else 0.0
    regression_pass = bool(
        steady_accuracy >= 0.90
        and wrong_switches == 0
        and all(item["within_limit"] for item in transitions)
    )
    return {
        "steady_state_accuracy": round(steady_accuracy, 4),
        "wrong_switches": wrong_switches,
        "transitions": transitions,
        "regression_pass": regression_pass,
        "records": routed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realsi-root", required=True, type=Path)
    ap.add_argument("--retriever-dir", required=True, type=Path)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--device", required=True)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--window-segments", type=int, default=6)
    ap.add_argument("--step-segments", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--encoder-mode", choices=("base", "retriever"), default="base")
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    result = evaluate(
        realsi_root=args.realsi_root,
        retriever_dir=args.retriever_dir,
        model_path=args.model_path,
        device_name=args.device,
        domains=domains,
        window_segments=args.window_segments,
        step_segments=args.step_segments,
        batch_size=args.batch_size,
        encoder_mode=args.encoder_mode,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"windows={result['windows']} accuracy={result['accuracy']:.4f}")
    for domain, row in result["per_domain"].items():
        print(f"{domain:13s} {row['correct']}/{row['windows']} accuracy={row['accuracy']:.4f}")
    print(f"wrote {args.out_json}")
    print(
        f"routing_steady={result['routing']['steady_state_accuracy']:.4f} "
        f"wrong_switches={result['routing']['wrong_switches']} "
        f"pass={result['routing']['regression_pass']}"
    )
    if not args.no_assert and (
        result["accuracy"] < args.min_accuracy or not result["routing"]["regression_pass"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
