from __future__ import annotations

import argparse
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from eval.streaming_sst.run_glossary_capacity_sweep import (
    ProcessSupervisor,
    TerminationRequested,
    eval_command,
    run_controller,
    server_command,
    validate_completed_output,
)


def _args(root: Path, *, presets: str = "10k,1m", resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        python_bin=root / "python",
        server_script=root / "server.py",
        eval_script=root / "eval.py",
        model_path=root / "model",
        term_memory_manifest=root / "term-memory.json",
        rag_model_path=root / "retriever.pt",
        vllm_compat_dir=root / "compat",
        acl_root=root / "acl",
        medicine_audio_dir=root / "medicine",
        tmp_root=root / "tmp",
        output_dir=root / "output",
        run_manifest=root / "output" / "manifest.json",
        runs_json=root / "output" / "runs.json",
        presets=presets,
        server_host="127.0.0.1",
        connect_host="127.0.0.1",
        port=8123,
        vllm_tp_size=2,
        rag_device="cuda:0",
        feed_sleep=1.6,
        chunk_samples=30720,
        latency_multiplier=2,
        acl_items=1,
        max_acl_segs_per_talk=0,
        max_seconds_per_item=0.0,
        language_pair="English -> Chinese",
        gpu_memory_utilization=0.58,
        max_num_seqs=8,
        max_model_len=16384,
        vllm_limit_audio=16,
        vllm_enforce_eager=1,
        enable_prefix_caching=1,
        disable_custom_all_reduce=1,
        vllm_use_v1=1,
        vllm_enable_v1_multiprocessing=1,
        vllm_worker_multiproc_method="spawn",
        vllm_moe_use_deep_gemm=0,
        vllm_use_fused_moe_grouped_topk=0,
        nccl_p2p_disable=1,
        nccl_ib_disable=1,
        torch_nccl_enable_monitoring=0,
        scheduler_batch_size=8,
        max_inflight_batches=2,
        max_new_tokens=40,
        rag_top_k=10,
        rag_score_threshold=0.78,
        term_map_format="tagged",
        empty_term_map_policy="none_block",
        health_timeout_sec=30.0,
        health_poll_interval_sec=0.1,
        server_stop_timeout_sec=1.0,
        idle_timeout_sec=60.0,
        idle_timeouts_after_eof=2,
        log_level="info",
        resume=resume,
    )


def _valid_payload(preset: str, *, chunk_samples: int = 30720) -> dict:
    return {
        "config": {
            "schedule": "acl_then_medicine",
            "preset": preset,
            "chunk_samples": chunk_samples,
            "latency_multiplier": 2,
        },
        "blocks": [{"corpus": "acl", "item_id": "talk-1"}],
        "block_spans": [
            {"block_index": 1, "start_sample": 0, "end_sample": 61440}
        ],
        "summary": {
            "preset": preset,
            "event_count": 2,
            "audio_seconds": 3.84,
        },
        "records": [
            {
                "cursor_samples": 30720,
                "start_sample": 0,
                "text": "甲",
                "prompt_reference_count": 1,
                "references": [{"term": "algorithm"}],
            },
            {
                "cursor_samples": 61440,
                "start_sample": 30720,
                "text": "乙",
                "prompt_reference_count": 0,
                "references": [],
            },
        ],
    }


class FakeProcess:
    _next_pid = 9000

    def __init__(self, returncode: int | None = None) -> None:
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.returncode = returncode
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.returncode is None and timeout is not None:
            self.returncode = 0
        return int(self.returncode or 0)


class RecordingSupervisor(ProcessSupervisor):
    def __init__(self, root: Path, *, fail_eval: bool = False) -> None:
        super().__init__()
        self.root = root
        self.fail_eval = fail_eval
        self.spawned: list[list[str]] = []
        self.stopped: list[int] = []

    def spawn(self, command, log_path):
        command = list(command)
        self.spawned.append(command)
        is_eval = "--out-json" in command
        process = FakeProcess(returncode=(1 if self.fail_eval and is_eval else None))
        self._active[process.pid] = process
        if is_eval and not self.fail_eval:
            output = Path(command[command.index("--out-json") + 1])
            preset = command[command.index("--preset") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(_valid_payload(preset)), encoding="utf-8")
            process.returncode = 0
        return process

    def stop(self, process, *, timeout_sec):
        self.stopped.append(process.pid)
        process.returncode = 0
        self._active.pop(process.pid, None)


class GlossaryCapacitySweepControllerTest(unittest.TestCase):
    def test_commands_forward_explicit_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            server = server_command(args, preset="1m", tmp_dir=Path(tmp) / "tmp" / "1m")
            client = eval_command(args, preset="1m", out_json=Path(tmp) / "1m.json")

        self.assertEqual(server[0:2], [str(args.python_bin), str(args.server_script)])
        self.assertEqual(server[server.index("--required-presets") + 1], "1m")
        self.assertEqual(server[server.index("--vllm-tp-size") + 1], "2")
        self.assertEqual(server[server.index("--port") + 1], "8123")
        self.assertEqual(client[client.index("--medicine-items") + 1], "0")
        self.assertEqual(client[client.index("--chunk") + 1], "30720")
        self.assertEqual(client[client.index("--feed-sleep") + 1], "1.6")
        self.assertNotIn("env", server)

    def test_output_validation_rejects_truncated_and_mismatched_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(_valid_payload("10k")), encoding="utf-8")
            valid = validate_completed_output(
                path,
                preset="10k",
                chunk_samples=30720,
                latency_multiplier=2,
                acl_items=1,
            )
            self.assertIsNotNone(valid)
            payload = _valid_payload("10k")
            payload["block_spans"][-1]["end_sample"] = 500000
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                validate_completed_output(
                    path,
                    preset="10k",
                    chunk_samples=30720,
                    latency_multiplier=2,
                    acl_items=1,
                )
            )

            payload = _valid_payload("10k")
            payload["records"][0]["references"].append({"term": "UI-only reference"})
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                validate_completed_output(
                    path,
                    preset="10k",
                    chunk_samples=30720,
                    latency_multiplier=2,
                    acl_items=1,
                )
            )

    def test_output_validation_accepts_bounded_silent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            payload = _valid_payload("10k")
            payload["block_spans"][-1]["end_sample"] += 3 * 30720
            path.write_text(json.dumps(payload), encoding="utf-8")

            valid = validate_completed_output(
                path,
                preset="10k",
                chunk_samples=30720,
                latency_multiplier=2,
                acl_items=1,
            )

        self.assertIsNotNone(valid)
        self.assertEqual(valid.tail_gap_samples, 3 * 30720)

    def test_resume_skips_only_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root, resume=True)
            run_path = args.output_dir / "runs" / "10k.json"
            run_path.parent.mkdir(parents=True)
            run_path.write_text(json.dumps(_valid_payload("10k")), encoding="utf-8")
            supervisor = RecordingSupervisor(root)

            manifest = run_controller(
                args,
                supervisor=supervisor,
                health_checker=lambda *_args, **_kwargs: None,
            )
            normalized_rows = json.loads(args.runs_json.read_text(encoding="utf-8"))

        self.assertEqual(len(supervisor.spawned), 2)
        self.assertEqual(supervisor.spawned[0][supervisor.spawned[0].index("--required-presets") + 1], "1m")
        self.assertEqual(manifest["status"], "completed")
        rows = {row["preset"]: row for row in manifest["runs"]}
        self.assertTrue(rows["10k"]["resume_skipped"])
        self.assertFalse(rows["1m"]["resume_skipped"])
        self.assertEqual([row["preset"] for row in normalized_rows], ["10k", "1m"])
        self.assertEqual(normalized_rows[0]["output_events"][0]["text"], "甲")

    def test_stops_server_between_scales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root)
            supervisor = RecordingSupervisor(root)

            run_controller(
                args,
                supervisor=supervisor,
                health_checker=lambda *_args, **_kwargs: None,
            )

        self.assertEqual(len(supervisor.spawned), 4)
        self.assertEqual(len(supervisor.stopped), 2)
        self.assertEqual(supervisor.stopped[1], supervisor.stopped[0] + 2)

    def test_failure_marks_manifest_and_cleans_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root, presets="10k")
            supervisor = RecordingSupervisor(root, fail_eval=True)

            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                run_controller(
                    args,
                    supervisor=supervisor,
                    health_checker=lambda *_args, **_kwargs: None,
                )

            payload = json.loads(args.run_manifest.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["runs"][0]["status"], "failed")
        self.assertEqual(len(supervisor.stopped), 1)

    def test_sigterm_request_cleans_server_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _args(root, presets="10k")
            supervisor = RecordingSupervisor(root)

            with self.assertRaises(TerminationRequested):
                run_controller(
                    args,
                    supervisor=supervisor,
                    health_checker=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        TerminationRequested(signal.SIGTERM)
                    ),
                )

        self.assertEqual(len(supervisor.stopped), 1)

    def test_process_supervisor_escalates_to_kill(self) -> None:
        process = Mock()
        process.pid = 12345
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 0.1), 0]
        killpg = Mock()
        supervisor = ProcessSupervisor(killpg=killpg)
        supervisor._active[process.pid] = process

        supervisor.stop(process, timeout_sec=0.1)

        self.assertEqual(
            killpg.call_args_list,
            [unittest.mock.call(12345, signal.SIGTERM), unittest.mock.call(12345, signal.SIGKILL)],
        )


if __name__ == "__main__":
    unittest.main()
