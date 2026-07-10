#!/usr/bin/env python3
"""Serve fixed-glossary capacity sweeps with explicit runtime parameters."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.omni import OmniAgent, OmniConfig  # noqa: E402
from framework.agents.plugins.backends import get_template  # noqa: E402
from framework.agents.term_memory.manifest import TermMemoryManifest  # noqa: E402
from framework.app import create_app  # noqa: E402
from framework.router import AgentRouter  # noqa: E402


DEFAULT_PRESETS = "acl_tagged_gs10k,acl_tagged_gs100k,acl_tagged_gs500k,acl_tagged_gs1m"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rag-model-path", type=Path, required=True)
    parser.add_argument("--rag-device", default="cuda:0")
    parser.add_argument("--required-presets", default=DEFAULT_PRESETS)
    parser.add_argument("--vllm-tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--vllm-limit-audio", type=int, default=16)
    parser.add_argument("--vllm-enforce-eager", type=int, choices=(0, 1), default=1)
    parser.add_argument("--enable-prefix-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument("--disable-custom-all-reduce", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vllm-use-v1", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vllm-enable-v1-multiprocessing", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vllm-worker-multiproc-method", choices=("spawn", "fork", "forkserver"), default="spawn")
    parser.add_argument("--vllm-moe-use-deep-gemm", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vllm-use-fused-moe-grouped-topk", type=int, choices=(0, 1), default=0)
    parser.add_argument("--nccl-p2p-disable", type=int, choices=(0, 1), default=1)
    parser.add_argument("--nccl-ib-disable", type=int, choices=(0, 1), default=1)
    parser.add_argument("--torch-nccl-enable-monitoring", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vllm-compat-dir", type=Path, default=PROJECT_ROOT / "serve" / "vllm_compat")
    parser.add_argument(
        "--extra-python-path",
        action="append",
        type=Path,
        default=[],
        help="Additional import root required by an external retriever implementation.",
    )
    parser.add_argument("--scheduler-batch-size", type=int, default=8)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-score-threshold", type=float, default=0.78)
    parser.add_argument("--retrieval-candidate-budget", type=int, default=0)
    parser.add_argument("--term-map-format", choices=("plain", "tagged", "xml_tagged"), default="tagged")
    parser.add_argument("--empty-term-map-policy", default="none_block")
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def _required_presets(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def configure_vllm_runtime(args: argparse.Namespace) -> None:
    extra_paths = [str(path) for path in getattr(args, "extra_python_path", ())]
    python_paths = [str(args.vllm_compat_dir), str(PROJECT_ROOT), *extra_paths]
    for path in reversed(extra_paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    values = {
        "VLLM_USE_V1": args.vllm_use_v1,
        "VLLM_ENABLE_V1_MULTIPROCESSING": args.vllm_enable_v1_multiprocessing,
        "VLLM_WORKER_MULTIPROC_METHOD": args.vllm_worker_multiproc_method,
        "VLLM_MOE_USE_DEEP_GEMM": args.vllm_moe_use_deep_gemm,
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK": args.vllm_use_fused_moe_grouped_topk,
        "NCCL_P2P_DISABLE": args.nccl_p2p_disable,
        "NCCL_IB_DISABLE": args.nccl_ib_disable,
        "TORCH_NCCL_ENABLE_MONITORING": args.torch_nccl_enable_monitoring,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(python_paths),
    }
    for key, value in values.items():
        os.environ[key] = str(value)


def validate_inputs(args: argparse.Namespace, manifest: TermMemoryManifest) -> list[str]:
    if args.retrieval_candidate_budget < 0:
        raise ValueError("--retrieval-candidate-budget must be non-negative")
    for path in (args.model_path, args.rag_model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.vllm_compat_dir, *args.extra_python_path):
        if not path.is_dir():
            raise NotADirectoryError(path)
    presets = _required_presets(args.required_presets)
    for preset in presets:
        snapshot = manifest.snapshot_for(preset, "zh")
        if snapshot is None:
            raise ValueError(f"manifest has no en-zh snapshot for {preset}")
        for path in (snapshot.terms_path, snapshot.index_path("maxsim")):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"{preset}: {path}")
    return presets


def build_agent(
    args: argparse.Namespace,
    manifest: TermMemoryManifest,
    presets: Sequence[str],
) -> OmniAgent:
    template = get_template("qwen3_omni")
    config = OmniConfig(
        language_pair="English -> Chinese",
        vllm_model_path=str(args.model_path),
        vllm_tp_size=args.vllm_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enable_prefix_caching=bool(args.enable_prefix_caching),
        vllm_enforce_eager=bool(args.vllm_enforce_eager),
        vllm_limit_audio=args.vllm_limit_audio,
        disable_custom_all_reduce=bool(args.disable_custom_all_reduce),
        scheduler_batch_size=args.scheduler_batch_size,
        max_inflight_batches=args.max_inflight_batches,
        max_new_tokens=args.max_new_tokens,
        term_map_format=args.term_map_format,
        empty_term_map_policy=args.empty_term_map_policy,
        system_prompt_style=template.system_prompt_style,
        rag_enabled=True,
        rag_model_path=str(args.rag_model_path),
        rag_device=args.rag_device,
        rag_top_k=args.rag_top_k,
        rag_score_threshold=args.rag_score_threshold,
        rag_startup_glossary_preset=presets[0],
        retrieval_candidate_budget=max(0, int(args.retrieval_candidate_budget)),
        auto_glossary_enabled=False,
        auto_glossary_preload=False,
        router_context_similarity_enabled=False,
        tmp_dir=str(args.tmp_dir),
    )
    return OmniAgent(name="RASST", model_id="qwen3_omni", config=config, manifest=manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_vllm_runtime(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = TermMemoryManifest.load(str(args.manifest))
    presets = validate_inputs(args, manifest)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "capacity server model=%s manifest=%s presets=%s rag_device=%s tp=%d",
        args.model_path,
        args.manifest,
        presets,
        args.rag_device,
        args.vllm_tp_size,
    )
    agent = build_agent(args, manifest, presets)
    router = AgentRouter({"RASST": agent}, default_agent="RASST")
    uvicorn.run(create_app(router), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
