from __future__ import annotations

import unittest

from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainSlice,
    RouterConfig,
    RouterSessionState,
)


class TopicRouterFallbackTests(unittest.TestCase):
    def test_unsupported_current_preset_falls_back_to_none(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [DomainSlice("nlp_core_10k", "nlp", centroid=[1.0, 0.0], index_path="mock://nlp")],
            RouterConfig(warmup_sec=0, update_interval_sec=0, fallback_preset_id="none"),
        )
        decision = router.observe(
            RouterSessionState("missing_core_10k", "missing", created_s=0.0),
            None,
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "fallback")
        self.assertEqual(decision.target_preset_id, "none")

    def test_common_domain_never_triggers_narrow_switch(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [
                DomainSlice("common_10k", "general", centroid=[1.0, 0.0], index_path="mock://common"),
                DomainSlice("nlp_core_10k", "nlp", centroid=[0.0, 1.0], index_path="mock://nlp"),
            ],
            RouterConfig(warmup_sec=0, update_interval_sec=0, min_confidence=0.5, min_margin=0.1),
        )
        decision = router.observe(
            RouterSessionState("nlp_core_10k", "nlp", created_s=0.0),
            [1.0, 0.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "stay")
        self.assertEqual(decision.target_preset_id, "common_10k")
        self.assertIn("general_or_common", decision.reason)

    def test_missing_target_index_stays(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [
                DomainSlice("common_10k", "general", centroid=[1.0, 0.0], index_path="mock://common"),
                DomainSlice("nlp_core_10k", "nlp", centroid=[0.0, 1.0], index_path=""),
            ],
            RouterConfig(warmup_sec=0, update_interval_sec=0, min_confidence=0.5, min_margin=0.1),
        )
        decision = router.observe(
            RouterSessionState("none", "general", created_s=0.0),
            [0.0, 1.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "stay")
        self.assertIn("target_unavailable", decision.reason)


if __name__ == "__main__":
    unittest.main()
