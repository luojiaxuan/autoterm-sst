#!/usr/bin/env python3
"""Run a resumable, one-server-at-a-time glossary-capacity sweep."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_presets(raw: str) -> list[str]:
    presets = [item.strip() for item in raw.split(",") if item.strip()]
    if not presets:
        raise ValueError("--presets must contain at least one preset")
    if len(presets) != len(set(presets)):
        raise ValueError("--presets contains duplicates")
    safe_names = [safe_preset_name(item) for item in presets]
    if len(safe_names) != len(set(safe_names)):
        raise ValueError("--presets collide after filename sanitization")
    return presets


def safe_preset_name(preset: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", preset).strip("._")
    if not value:
        raise ValueError(f"preset cannot be converted to a safe filename: {preset!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class ValidatedOutput:
    payload: Mapping[str, Any]
    event_count: int
    audio_seconds: float
    last_cursor_samples: int
    tail_gap_samples: int


def validate_completed_output(
    path: Path,
    *,
    preset: str,
    chunk_samples: int,
    latency_multiplier: int,
    acl_items: int,
) -> ValidatedOutput | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        config = payload["config"]
        summary = payload["summary"]
        blocks = payload["blocks"]
        spans = payload["block_spans"]
        records = payload["records"]
        if not all(isinstance(item, dict) for item in (config, summary)):
            return None
        if str(config.get("preset")) != preset or str(summary.get("preset")) != preset:
            return None
        if str(config.get("schedule")) != "acl_then_medicine":
            return None
        if int(config.get("chunk_samples") or 0) != int(chunk_samples):
            return None
        if int(config.get("latency_multiplier") or 0) != int(latency_multiplier):
            return None
        if not isinstance(blocks, list) or len(blocks) != int(acl_items):
            return None
        if any(not isinstance(item, dict) or item.get("corpus") != "acl" for item in blocks):
            return None
        if not isinstance(spans, list) or len(spans) != len(blocks):
            return None
        if not isinstance(records, list) or not records:
            return None
        event_count = int(summary.get("event_count") or 0)
        if event_count != len(records):
            return None
        cursors: list[int] = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                return None
            if not isinstance(record.get("references"), list):
                return None
            prompt_reference_count = int(record.get("prompt_reference_count"))
            if prompt_reference_count < 0 or len(record["references"]) != prompt_reference_count:
                return None
            int(record["start_sample"])
            cursors.append(int(record["cursor_samples"]))
        if cursors != sorted(cursors):
            return None
        expected_end = int(spans[-1].get("end_sample") or 0)
        tail_tolerance = 4 * int(chunk_samples)
        tail_gap_samples = expected_end - cursors[-1]
        if expected_end <= 0 or tail_gap_samples < 0 or tail_gap_samples > tail_tolerance:
            return None
        audio_seconds = float(summary.get("audio_seconds") or 0.0)
        if audio_seconds <= 0.0:
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return ValidatedOutput(
        payload=payload,
        event_count=event_count,
        audio_seconds=audio_seconds,
        last_cursor_samples=cursors[-1],
        tail_gap_samples=tail_gap_samples,
    )


class TerminationRequested(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = int(signum)


class ProcessSupervisor:
    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self._popen_factory = popen_factory
        self._killpg = killpg
        self._active: dict[int, subprocess.Popen[Any]] = {}

    def spawn(self, command: Sequence[str], log_path: Path) -> subprocess.Popen[Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        try:
            process = self._popen_factory(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        self._active[int(process.pid)] = process
        return process

    def wait(self, process: subprocess.Popen[Any]) -> int:
        return_code = int(process.wait())
        self._active.pop(int(process.pid), None)
        return return_code

    def stop(self, process: subprocess.Popen[Any], *, timeout_sec: float) -> None:
        pid = int(process.pid)
        if process.poll() is None:
            try:
                self._killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(0.1, float(timeout_sec)))
            except subprocess.TimeoutExpired:
                try:
                    self._killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        self._active.pop(pid, None)

    def stop_all(self, *, timeout_sec: float) -> None:
        for process in list(self._active.values())[::-1]:
            self.stop(process, timeout_sec=timeout_sec)


@contextlib.contextmanager
def termination_signals() -> Iterator[None]:
    previous: dict[int, Any] = {}
    triggered = False

    def handler(signum: int, _frame: Any) -> None:
        nonlocal triggered
        if triggered:
            return
        triggered = True
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(handled_signal, signal.SIG_IGN)
        raise TerminationRequested(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old_handler in previous.items():
            signal.signal(signum, old_handler)


def server_command(args: argparse.Namespace, *, preset: str, tmp_dir: Path) -> list[str]:
    command = [
        str(args.python_bin),
        str(args.server_script),
        "--host",
        args.server_host,
        "--port",
        str(args.port),
        "--model-path",
        str(args.model_path),
        "--manifest",
        str(args.term_memory_manifest),
        "--rag-model-path",
        str(args.rag_model_path),
        "--rag-device",
        args.rag_device,
        "--required-presets",
        preset,
        "--vllm-tp-size",
        str(args.vllm_tp_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--vllm-limit-audio",
        str(args.vllm_limit_audio),
        "--vllm-enforce-eager",
        str(args.vllm_enforce_eager),
        "--enable-prefix-caching",
        str(args.enable_prefix_caching),
        "--disable-custom-all-reduce",
        str(args.disable_custom_all_reduce),
        "--vllm-use-v1",
        str(args.vllm_use_v1),
        "--vllm-enable-v1-multiprocessing",
        str(args.vllm_enable_v1_multiprocessing),
        "--vllm-worker-multiproc-method",
        args.vllm_worker_multiproc_method,
        "--vllm-moe-use-deep-gemm",
        str(args.vllm_moe_use_deep_gemm),
        "--vllm-use-fused-moe-grouped-topk",
        str(args.vllm_use_fused_moe_grouped_topk),
        "--nccl-p2p-disable",
        str(args.nccl_p2p_disable),
        "--nccl-ib-disable",
        str(args.nccl_ib_disable),
        "--torch-nccl-enable-monitoring",
        str(args.torch_nccl_enable_monitoring),
        "--vllm-compat-dir",
        str(args.vllm_compat_dir),
        "--scheduler-batch-size",
        str(args.scheduler_batch_size),
        "--max-inflight-batches",
        str(args.max_inflight_batches),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--rag-top-k",
        str(args.rag_top_k),
        "--rag-score-threshold",
        str(args.rag_score_threshold),
        "--term-map-format",
        args.term_map_format,
        "--empty-term-map-policy",
        args.empty_term_map_policy,
        "--tmp-dir",
        str(tmp_dir),
        "--log-level",
        args.log_level,
    ]
    for path in args.extra_python_path:
        command.extend(("--extra-python-path", str(path)))
    return command


def eval_command(args: argparse.Namespace, *, preset: str, out_json: Path) -> list[str]:
    return [
        str(args.python_bin),
        str(args.eval_script),
        "--base-url",
        f"http://{args.connect_host}:{args.port}",
        "--acl-root",
        str(args.acl_root),
        "--medicine-audio-dir",
        str(args.medicine_audio_dir),
        "--acl-items",
        str(args.acl_items),
        "--medicine-items",
        "0",
        "--max-acl-segs-per-talk",
        str(args.max_acl_segs_per_talk),
        "--max-seconds-per-item",
        str(args.max_seconds_per_item),
        "--schedule",
        "acl_then_medicine",
        "--preset",
        preset,
        "--language-pair",
        args.language_pair,
        "--latency-multiplier",
        str(args.latency_multiplier),
        "--chunk",
        str(args.chunk_samples),
        "--feed-sleep",
        str(args.feed_sleep),
        "--idle-timeout-sec",
        str(args.idle_timeout_sec),
        "--idle-timeouts-after-eof",
        str(args.idle_timeouts_after_eof),
        "--allow-missing-router-meta",
        "--no-assert",
        "--out-json",
        str(out_json),
    ]


def rag_health_error(payload: Any, expected_rag_terms: int) -> str:
    rag = payload.get("rag") if isinstance(payload, dict) else None
    if not isinstance(rag, dict) or rag.get("status") != "ready":
        return f"RAG is not ready: {rag!r}"
    active_terms = int(rag.get("active_terms") or 0)
    if active_terms != int(expected_rag_terms):
        return f"RAG active_terms={active_terms}, expected={expected_rag_terms}"
    return ""


def wait_for_health(
    base_url: str,
    server: subprocess.Popen[Any],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    expected_rag_terms: int,
) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    health_url = base_url.rstrip("/") + "/health"
    last_error = "health endpoint did not respond"
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited before health check: returncode={return_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                if 200 <= int(response.status) < 300:
                    payload = json.loads(response.read().decode("utf-8"))
                    last_error = rag_health_error(payload, expected_rag_terms)
                    if not last_error:
                        return
                else:
                    last_error = f"HTTP {response.status}"
        except (OSError, TypeError, ValueError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(f"server health timeout after {timeout_sec}s: {last_error}")


def expected_term_count(manifest_path: Path, preset: str, language_pair: str) -> int:
    target = str(language_pair).split("->")[-1].strip().casefold()
    language = {"chinese": "zh", "japanese": "ja", "german": "de"}.get(target)
    if not language:
        raise ValueError(f"unsupported capacity target language: {language_pair}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = ((payload.get("scales") or {}).get(preset) or {}).get(f"en-{language}")
    if not isinstance(snapshot, dict):
        raise ValueError(f"manifest has no en-{language} snapshot for {preset}")
    count = int(snapshot.get("num_terms") or 0)
    if count <= 0:
        raise ValueError(f"manifest has invalid num_terms for {preset}: {count}")
    return count


def normalized_run_row(
    *,
    preset: str,
    output_path: Path,
    validated: ValidatedOutput,
) -> dict[str, Any]:
    records = validated.payload["records"]
    return {
        "preset": preset,
        "streaming_chunk_samples": int(validated.payload["config"]["chunk_samples"]),
        "output_json": str(output_path),
        "output_sha256": sha256_file(output_path),
        "event_count": validated.event_count,
        "audio_seconds": validated.audio_seconds,
        "last_cursor_samples": validated.last_cursor_samples,
        "tail_gap_samples": validated.tail_gap_samples,
        "output_events": [
            {
                "cursor_samples": int(record["cursor_samples"]),
                "start_sample": int(record.get("start_sample") or 0),
                "text": str(record["text"]),
            }
            for record in records
        ],
    }


def _manifest_template(args: argparse.Namespace, presets: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "config": {
            "presets": list(presets),
            "python_bin": str(args.python_bin),
            "server_script": str(args.server_script),
            "eval_script": str(args.eval_script),
            "model_path": str(args.model_path),
            "term_memory_manifest": str(args.term_memory_manifest),
            "rag_model_path": str(args.rag_model_path),
            "rag_device": args.rag_device,
            "vllm_compat_dir": str(args.vllm_compat_dir),
            "extra_python_path": [str(path) for path in args.extra_python_path],
            "server_host": args.server_host,
            "connect_host": args.connect_host,
            "port": args.port,
            "vllm_tp_size": args.vllm_tp_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "vllm_limit_audio": args.vllm_limit_audio,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "enable_prefix_caching": args.enable_prefix_caching,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "vllm_use_v1": args.vllm_use_v1,
            "vllm_enable_v1_multiprocessing": args.vllm_enable_v1_multiprocessing,
            "vllm_worker_multiproc_method": args.vllm_worker_multiproc_method,
            "vllm_moe_use_deep_gemm": args.vllm_moe_use_deep_gemm,
            "vllm_use_fused_moe_grouped_topk": args.vllm_use_fused_moe_grouped_topk,
            "nccl_p2p_disable": args.nccl_p2p_disable,
            "nccl_ib_disable": args.nccl_ib_disable,
            "torch_nccl_enable_monitoring": args.torch_nccl_enable_monitoring,
            "scheduler_batch_size": args.scheduler_batch_size,
            "max_inflight_batches": args.max_inflight_batches,
            "max_new_tokens": args.max_new_tokens,
            "rag_top_k": args.rag_top_k,
            "rag_score_threshold": args.rag_score_threshold,
            "term_map_format": args.term_map_format,
            "empty_term_map_policy": args.empty_term_map_policy,
            "acl_root": str(args.acl_root),
            "acl_items": args.acl_items,
            "max_acl_segs_per_talk": args.max_acl_segs_per_talk,
            "max_seconds_per_item": args.max_seconds_per_item,
            "language_pair": args.language_pair,
            "chunk_samples": args.chunk_samples,
            "latency_multiplier": args.latency_multiplier,
            "feed_sleep": args.feed_sleep,
            "idle_timeout_sec": args.idle_timeout_sec,
            "idle_timeouts_after_eof": args.idle_timeouts_after_eof,
        },
        "runs": [],
    }


def load_or_create_manifest(
    args: argparse.Namespace, presets: Sequence[str]
) -> dict[str, Any]:
    if args.resume and args.run_manifest.is_file():
        payload = json.loads(args.run_manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise ValueError(f"invalid run manifest: {args.run_manifest}")
        previous_presets = list((payload.get("config") or {}).get("presets") or [])
        if previous_presets and previous_presets != list(presets):
            raise ValueError("resume manifest preset order does not match --presets")
        expected_config = _manifest_template(args, presets)["config"]
        previous_config = payload.get("config") or {}
        mismatched = [
            key
            for key, expected in expected_config.items()
            if previous_config.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "resume manifest runtime configuration changed: "
                + ", ".join(mismatched)
            )
        payload["status"] = "running"
        payload["updated_at"] = utc_now()
        return payload
    return _manifest_template(args, presets)


def upsert_run(manifest: dict[str, Any], row: Mapping[str, Any]) -> None:
    runs = manifest["runs"]
    for index, existing in enumerate(runs):
        if existing.get("preset") == row.get("preset"):
            runs[index] = dict(row)
            return
    runs.append(dict(row))


def persist_state(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    presets: Sequence[str],
) -> None:
    manifest["updated_at"] = utc_now()
    atomic_write_json(args.run_manifest, manifest)
    rows: list[dict[str, Any]] = []
    completed = {
        str(row.get("preset"))
        for row in manifest.get("runs") or []
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    for preset in presets:
        if preset not in completed:
            continue
        path = args.output_dir / "runs" / f"{safe_preset_name(preset)}.json"
        validated = validate_completed_output(
            path,
            preset=preset,
            chunk_samples=args.chunk_samples,
            latency_multiplier=args.latency_multiplier,
            acl_items=args.acl_items,
        )
        if validated is not None:
            rows.append(
                normalized_run_row(
                    preset=preset,
                    output_path=path,
                    validated=validated,
                )
            )
    atomic_write_json(args.runs_json, rows)


def run_controller(
    args: argparse.Namespace,
    *,
    supervisor: ProcessSupervisor | None = None,
    health_checker: Callable[..., None] = wait_for_health,
) -> dict[str, Any]:
    presets = parse_presets(args.presets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supervisor = supervisor or ProcessSupervisor()
    manifest = load_or_create_manifest(args, presets)
    persist_state(args, manifest, presets)
    try:
        for preset in presets:
            safe_name = safe_preset_name(preset)
            output_path = args.output_dir / "runs" / f"{safe_name}.json"
            server_log = args.output_dir / "logs" / f"{safe_name}.server.log"
            eval_log = args.output_dir / "logs" / f"{safe_name}.eval.log"
            tmp_dir = args.tmp_root / safe_name
            expected_rag_terms = expected_term_count(
                args.term_memory_manifest,
                preset,
                args.language_pair,
            )
            validated = validate_completed_output(
                output_path,
                preset=preset,
                chunk_samples=args.chunk_samples,
                latency_multiplier=args.latency_multiplier,
                acl_items=args.acl_items,
            )
            if args.resume and validated is not None:
                upsert_run(
                    manifest,
                    {
                        "preset": preset,
                        "status": "completed",
                        "resume_skipped": True,
                        "output_json": str(output_path),
                        "output_sha256": sha256_file(output_path),
                        "event_count": validated.event_count,
                        "audio_seconds": validated.audio_seconds,
                        "last_cursor_samples": validated.last_cursor_samples,
                        "tail_gap_samples": validated.tail_gap_samples,
                        "server_log": str(server_log),
                        "eval_log": str(eval_log),
                    },
                )
                persist_state(args, manifest, presets)
                continue

            server_cmd = server_command(args, preset=preset, tmp_dir=tmp_dir)
            client_cmd = eval_command(args, preset=preset, out_json=output_path)
            run_row: dict[str, Any] = {
                "preset": preset,
                "status": "running",
                "resume_skipped": False,
                "started_at": utc_now(),
                "output_json": str(output_path),
                "server_log": str(server_log),
                "eval_log": str(eval_log),
                "server_command": server_cmd,
                "eval_command": client_cmd,
                "expected_rag_terms": expected_rag_terms,
            }
            upsert_run(manifest, run_row)
            persist_state(args, manifest, presets)
            output_path.unlink(missing_ok=True)
            server_process: subprocess.Popen[Any] | None = None
            try:
                server_process = supervisor.spawn(server_cmd, server_log)
                health_checker(
                    f"http://{args.connect_host}:{args.port}",
                    server_process,
                    timeout_sec=args.health_timeout_sec,
                    poll_interval_sec=args.health_poll_interval_sec,
                    expected_rag_terms=expected_rag_terms,
                )
                eval_process = supervisor.spawn(client_cmd, eval_log)
                eval_returncode = supervisor.wait(eval_process)
                if eval_returncode != 0:
                    raise RuntimeError(
                        f"evaluation failed for {preset}: returncode={eval_returncode}"
                    )
                validated = validate_completed_output(
                    output_path,
                    preset=preset,
                    chunk_samples=args.chunk_samples,
                    latency_multiplier=args.latency_multiplier,
                    acl_items=args.acl_items,
                )
                if validated is None:
                    raise RuntimeError(f"evaluation output failed validation: {output_path}")
                run_row.update(
                    {
                        "status": "completed",
                        "completed_at": utc_now(),
                        "output_sha256": sha256_file(output_path),
                        "event_count": validated.event_count,
                        "audio_seconds": validated.audio_seconds,
                        "last_cursor_samples": validated.last_cursor_samples,
                        "tail_gap_samples": validated.tail_gap_samples,
                    }
                )
                upsert_run(manifest, run_row)
                persist_state(args, manifest, presets)
            except BaseException as exc:
                run_row.update(
                    {
                        "status": "failed",
                        "failed_at": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                upsert_run(manifest, run_row)
                manifest["status"] = "failed"
                persist_state(args, manifest, presets)
                raise
            finally:
                if server_process is not None:
                    supervisor.stop(
                        server_process, timeout_sec=args.server_stop_timeout_sec
                    )

        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        persist_state(args, manifest, presets)
        return manifest
    finally:
        supervisor.stop_all(timeout_sec=args.server_stop_timeout_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--server-script", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--term-memory-manifest", type=Path, required=True)
    parser.add_argument("--rag-model-path", type=Path, required=True)
    parser.add_argument("--vllm-compat-dir", type=Path, required=True)
    parser.add_argument("--extra-python-path", action="append", type=Path, default=[])
    parser.add_argument("--acl-root", type=Path, required=True)
    parser.add_argument("--medicine-audio-dir", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--presets", required=True)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--connect-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vllm-tp-size", type=int, required=True)
    parser.add_argument("--rag-device", required=True)
    parser.add_argument("--feed-sleep", type=float, required=True)
    parser.add_argument("--chunk-samples", type=int, required=True)
    parser.add_argument("--latency-multiplier", type=int, required=True)
    parser.add_argument("--acl-items", type=int, default=5)
    parser.add_argument("--max-acl-segs-per-talk", type=int, default=0)
    parser.add_argument("--max-seconds-per-item", type=float, default=0.0)
    parser.add_argument("--language-pair", default="English -> Chinese")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--vllm-limit-audio", type=int, default=16)
    parser.add_argument("--vllm-enforce-eager", type=int, choices=(0, 1), default=1)
    parser.add_argument("--enable-prefix-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument("--disable-custom-all-reduce", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vllm-use-v1", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vllm-enable-v1-multiprocessing", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--vllm-worker-multiproc-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
    )
    parser.add_argument("--vllm-moe-use-deep-gemm", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--vllm-use-fused-moe-grouped-topk", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--nccl-p2p-disable", type=int, choices=(0, 1), default=1)
    parser.add_argument("--nccl-ib-disable", type=int, choices=(0, 1), default=1)
    parser.add_argument("--torch-nccl-enable-monitoring", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scheduler-batch-size", type=int, default=8)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--rag-top-k", type=int, default=10)
    parser.add_argument("--rag-score-threshold", type=float, default=0.78)
    parser.add_argument(
        "--term-map-format", choices=("plain", "tagged", "xml_tagged"), default="tagged"
    )
    parser.add_argument("--empty-term-map-policy", default="none_block")
    parser.add_argument("--health-timeout-sec", type=float, default=600.0)
    parser.add_argument("--health-poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--server-stop-timeout-sec", type=float, default=30.0)
    parser.add_argument("--idle-timeout-sec", type=float, default=60.0)
    parser.add_argument("--idle-timeouts-after-eof", type=int, default=2)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    required_paths = (
        args.python_bin,
        args.server_script,
        args.eval_script,
        args.model_path,
        args.term_memory_manifest,
        args.rag_model_path,
        args.vllm_compat_dir,
        args.acl_root,
        *args.extra_python_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("required runtime path(s) missing: " + ", ".join(missing))
    if args.port <= 0 or args.vllm_tp_size <= 0:
        raise ValueError("--port and --vllm-tp-size must be positive")
    if args.chunk_samples <= 0 or args.latency_multiplier <= 0 or args.acl_items <= 0:
        raise ValueError("chunk, latency multiplier, and ACL item count must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        with termination_signals():
            run_controller(args)
    except TerminationRequested as exc:
        print(str(exc), file=sys.stderr)
        return 128 + exc.signum
    except Exception as exc:  # noqa: BLE001
        print(f"capacity sweep failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
