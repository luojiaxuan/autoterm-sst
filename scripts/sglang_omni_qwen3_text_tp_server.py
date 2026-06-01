#!/usr/bin/env python3
"""Launch a Qwen3-Omni text-output SGLang-Omni server with thinker TP.

This is intentionally small and repo-local because the upstream text-only
example does not expose thinker tensor parallelism on the command line.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from typing import Any, Sequence


def _set_stage_gpu(config: Any, stage_name: str, gpu_id: int | list[int]) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu_id
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_process(config: Any, stage_name: str, process_name: str) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.process = process_name
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_tp_size(config: Any, stage_name: str, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.tp_size = tp_size
            stage.parallelism.tp = tp_size
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return
    raise ValueError(f"stage {stage_name!r} not found")


def _set_stage_memory_fraction(config: Any, stage_name: str, fraction: float) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.runtime.resources.total_gpu_memory_fraction = fraction
            return
    raise ValueError(f"stage {stage_name!r} not found")


def _parse_gpu_list(raw: str, tp_size: int) -> list[int]:
    gpu_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(gpu_ids) != tp_size:
        raise ValueError(
            f"--thinker-gpus needs exactly {tp_size} entries, got {gpu_ids}"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--thinker-gpus must list distinct ids, got {gpu_ids}")
    return gpu_ids


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model-name", default="rasst-qwen3-omni")
    parser.add_argument("--pipeline-name", default="rasst")
    parser.add_argument("--ipc-base-path", default="/tmp/r")
    parser.add_argument("--thinker-tp-size", type=int, default=2)
    parser.add_argument("--thinker-gpus", default="0,1")
    parser.add_argument("--encoder-gpu", type=int, default=1)
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--encoder-memory-fraction", type=float, default=0.025)
    parser.add_argument("--thinker-memory-fraction", type=float, default=0.75)
    parser.add_argument("--relay-backend", default="shm", choices=["shm", "nccl", "nixl"])
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    mp.set_start_method("spawn", force=True)
    args = parse_args(argv)
    if args.thinker_tp_size < 1:
        raise SystemExit("--thinker-tp-size must be >= 1")

    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.serve import launch_server

    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    config.name = args.pipeline_name
    config.endpoints.base_path = args.ipc_base_path

    thinker_gpus = _parse_gpu_list(args.thinker_gpus, args.thinker_tp_size)
    _set_stage_process(config, "thinker", "thinker")
    _set_stage_tp_size(config, "thinker", args.thinker_tp_size)
    _set_stage_gpu(config, "thinker", thinker_gpus)
    _set_stage_gpu(config, "image_encoder", args.encoder_gpu)
    _set_stage_gpu(config, "audio_encoder", args.encoder_gpu)
    _set_stage_memory_fraction(config, "image_encoder", float(args.encoder_memory_fraction))
    _set_stage_memory_fraction(config, "audio_encoder", float(args.encoder_memory_fraction))
    _set_stage_memory_fraction(config, "thinker", float(args.thinker_memory_fraction))

    stage_updates = {"thinker_max_seq_len": int(args.thinker_max_seq_len)}
    server_arg_updates: dict[str, object] = {
        "disable_custom_all_reduce": True,
        "max_running_requests": int(args.max_running_requests),
        "max_prefill_tokens": int(args.max_prefill_tokens),
    }
    if args.mem_fraction_static is not None:
        server_arg_updates["mem_fraction_static"] = float(args.mem_fraction_static)

    _apply_stage_factory_updates(
        config,
        stage_name="thinker",
        updates=stage_updates,
        server_arg_updates=server_arg_updates,
    )
    _apply_stage_factory_updates(
        config,
        stage_name="preprocessing",
        updates=stage_updates,
    )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
