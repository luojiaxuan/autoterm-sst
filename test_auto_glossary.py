from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.agents.glossary import GlossaryCatalog
from framework.agents.term_memory.active_glossary import ActiveGlossaryManager
from framework.agents.term_memory.domain_taxonomy import AUTO_WORKING_PRESET
from framework.agents.term_memory.manifest import TermMemoryManifest
from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainSlice,
    RouterConfig,
    RouterSessionState,
)


class AutoGlossaryTests(unittest.TestCase):
    def _manifest(self, root: Path) -> TermMemoryManifest:
        for preset in ("common_10k", "nlp_core_10k"):
            index = root / "indexes" / preset / "en-zh" / "maxsim.pt"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_bytes(b"stub")
        raw = {
            "snapshot_id": "unit_working",
            "source": "unit",
            "root": str(root),
            "scales": {
                "common_10k": {
                    "en-zh": {
                        "terms_path": "snapshots/unit/common.en-zh.jsonl",
                        "indexes": {"maxsim": "indexes/common_10k/en-zh/maxsim.pt"},
                        "num_terms": 10000,
                    }
                },
                "nlp_core_10k": {
                    "en-zh": {
                        "terms_path": "snapshots/unit/nlp.en-zh.jsonl",
                        "indexes": {"maxsim": "indexes/nlp_core_10k/en-zh/maxsim.pt"},
                        "num_terms": 10000,
                    }
                },
            },
            "preset_meta": {
                "common_10k": {"label": "Common", "domain": "general"},
                "nlp_core_10k": {"label": "NLP", "domain": "nlp"},
            },
        }
        return TermMemoryManifest.from_dict(raw, base_dir=root, path=str(root / "manifest.json"))

    def test_topic_router_switches_on_clear_nlp_embedding(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [
                DomainSlice("common_10k", "general", centroid=[1.0, 0.0], index_path="mock://common"),
                DomainSlice("nlp_core_10k", "nlp", centroid=[0.0, 1.0], index_path="mock://nlp"),
            ],
            RouterConfig(warmup_sec=0, update_interval_sec=0, min_confidence=0.5, min_margin=0.1, min_consistent_windows=1),
        )
        state = RouterSessionState("common_10k", "general", created_s=0.0)
        decision = router.observe(state, [0.0, 1.0], [], now_s=60.0)
        self.assertEqual(decision.target_domain_id, "nlp")
        self.assertEqual(decision.action, "switch")
        self.assertGreaterEqual(decision.confidence, 0.5)

    def test_auto_initial_selection_uses_common_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(Path(tmp))
            catalog = GlossaryCatalog("English -> Chinese", manifest=manifest)
            manager = ActiveGlossaryManager(default_preset="common_10k")
            selection = manager.initial_selection(
                catalog,
                AUTO_WORKING_PRESET,
                "",
                auto_allowed=True,
                mock=False,
            )
            self.assertTrue(selection.auto_enabled)
            self.assertEqual(selection.requested_preset, AUTO_WORKING_PRESET)
            self.assertEqual(selection.active_preset, "common_10k")
            self.assertEqual(selection.preset_terms, 10000)
            self.assertTrue(selection.index_ready)

    def test_manager_selects_nlp_slice_for_nlp_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(Path(tmp))
            catalog = GlossaryCatalog("English -> Chinese", manifest=manifest)
            manager = ActiveGlossaryManager(default_preset="common_10k")
            router = AudioNativeActiveGlossaryRouter(
                [
                    DomainSlice("common_10k", "general", centroid=[1.0, 0.0], index_path="mock://common"),
                    DomainSlice("nlp_core_10k", "nlp", centroid=[0.0, 1.0], index_path="mock://nlp"),
                ],
                RouterConfig(warmup_sec=0, update_interval_sec=0, min_confidence=0.5, min_margin=0.1, min_consistent_windows=1),
            )
            decision = router.observe(RouterSessionState("common_10k", "general"), [0.0, 1.0], [], now_s=60.0)
            selection = manager.selection_for_decision(catalog, decision, mock=False)
            self.assertIsNotNone(selection)
            assert selection is not None
            self.assertEqual(selection.active_preset, "nlp_core_10k")
            self.assertEqual(selection.active_domain, "nlp")


if __name__ == "__main__":
    unittest.main()
