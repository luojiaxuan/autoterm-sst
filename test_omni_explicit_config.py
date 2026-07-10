from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.streaming_sst.serve_glossary_capacity_sweep import configure_vllm_runtime
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
        )
        with patch.dict(os.environ, {}, clear=True):
            configure_vllm_runtime(args)
            self.assertEqual(os.environ["VLLM_USE_V1"], "1")
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "1")
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            self.assertEqual(os.environ["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(os.environ["NCCL_IB_DISABLE"], "1")
            self.assertEqual(
                os.environ["PYTHONPATH"],
                os.pathsep.join(["/tmp/vllm-compat", str(Path(__file__).resolve().parent)]),
            )

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


if __name__ == "__main__":
    unittest.main()
