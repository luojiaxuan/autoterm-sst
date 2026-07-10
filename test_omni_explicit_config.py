from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.term_memory.manifest import TermMemoryManifest


class OmniExplicitConfigTest(unittest.TestCase):
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
