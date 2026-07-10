from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.streaming_sst.serve_glossary_capacity_sweep import build_agent, configure_vllm_runtime
from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.term_memory.manifest import TermMemoryManifest


class OmniExplicitConfigTest(unittest.TestCase):
    def test_capacity_server_configures_vllm_runtime_explicitly(self) -> None:
        args = SimpleNamespace(
            vllm_use_v1=1,
            vllm_enable_v1_multiprocessing=1,
            vllm_worker_multiproc_method="spawn",
            vllm_moe_use_deep_gemm=0,
            vllm_use_fused_moe_grouped_topk=0,
            nccl_p2p_disable=1,
            nccl_ib_disable=1,
            torch_nccl_enable_monitoring=0,
            vllm_compat_dir=Path("/tmp/vllm-compat"),
            extra_python_path=[Path("/tmp/external-retriever")],
        )
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "path", list(sys.path)):
            configure_vllm_runtime(args)
            self.assertEqual(os.environ["VLLM_USE_V1"], "1")
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "1")
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            self.assertEqual(os.environ["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(os.environ["NCCL_IB_DISABLE"], "1")
            self.assertEqual(
                os.environ["PYTHONPATH"],
                os.pathsep.join(
                    [
                        "/tmp/vllm-compat",
                        str(Path(__file__).resolve().parent),
                        "/tmp/external-retriever",
                    ]
                ),
            )
            self.assertEqual(sys.path[0], "/tmp/external-retriever")

    def test_agent_uses_injected_config_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = TermMemoryManifest.from_dict(
                {
                    "snapshot_id": "explicit-test",
                    "scales": {
                        "capacity_10k": {
                            "en-zh": {
                                "terms_path": "terms.json",
                                "indexes": {"maxsim": "maxsim.pt"},
                                "num_terms": 10_000,
                            }
                        }
                    },
                },
                base_dir=root,
                path=str(root / "manifest.json"),
            )
            config = OmniConfig(
                mock=True,
                rag_enabled=False,
                auto_glossary_enabled=False,
                tmp_dir=str(root / "runtime"),
            )

            agent = OmniAgent(config=config, manifest=manifest)

            self.assertIs(agent.config, config)
            self.assertIs(agent._catalog("English -> Chinese").manifest, manifest)
            self.assertEqual(agent._catalog("English -> Chinese").open_preset_ids(), ["capacity_10k"])

    def test_capacity_agent_starts_rag_from_required_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = TermMemoryManifest.from_dict(
                {
                    "snapshot_id": "capacity-startup",
                    "scales": {
                        "capacity_1m": {
                            "en-zh": {
                                "terms_path": "terms.json",
                                "indexes": {"maxsim": "maxsim.pt"},
                                "num_terms": 1_000_000,
                            }
                        }
                    },
                },
                base_dir=root,
                path=str(root / "manifest.json"),
            )
            args = SimpleNamespace(
                model_path=root / "model",
                rag_model_path=root / "retriever.pt",
                rag_device="cuda:0",
                vllm_tp_size=1,
                gpu_memory_utilization=0.6,
                max_num_seqs=8,
                max_model_len=16384,
                enable_prefix_caching=1,
                vllm_enforce_eager=1,
                vllm_limit_audio=16,
                disable_custom_all_reduce=1,
                scheduler_batch_size=8,
                max_inflight_batches=2,
                max_new_tokens=40,
                term_map_format="tagged",
                empty_term_map_policy="none_block",
                rag_top_k=10,
                rag_score_threshold=0.78,
                tmp_dir=root / "tmp",
            )

            agent = build_agent(args, manifest, ["capacity_1m"])

            self.assertEqual(agent.config.rag_startup_glossary_preset, "capacity_1m")


if __name__ == "__main__":
    unittest.main()
