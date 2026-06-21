from __future__ import annotations

import unittest

from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainSlice,
    RouterConfig,
    RouterSessionState,
)


def _router(**overrides):
    values = {
        "warmup_sec": 0,
        "update_interval_sec": 0,
        "min_confidence": 0.55,
        "min_margin": 0.15,
        "min_consistent_windows": 1,
    }
    values.update(overrides)
    cfg = RouterConfig(**values)
    return AudioNativeActiveGlossaryRouter(
        [
            DomainSlice("common_10k", "general", centroid=[1.0, 0.0, 0.0], index_path="mock://common"),
            DomainSlice("nlp_core_10k", "nlp", centroid=[0.0, 1.0, 0.0], index_path="mock://nlp"),
            DomainSlice("medicine_core_10k", "medicine", centroid=[0.0, 0.0, 1.0], index_path="mock://medicine"),
        ],
        cfg,
    )


class TopicRouterEmbeddingRefsTests(unittest.TestCase):
    def test_strong_embedding_switches_to_nlp(self) -> None:
        decision = _router().observe(
            RouterSessionState("common_10k", "general", created_s=0.0),
            [0.0, 1.0, 0.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_preset_id, "nlp_core_10k")

    def test_ambiguous_embedding_stays(self) -> None:
        decision = _router(min_margin=0.40).observe(
            RouterSessionState("common_10k", "general", created_s=0.0),
            [0.0, 0.7, 0.7],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "stay")
        self.assertIn("margin<", decision.reason)

    def test_refs_only_can_switch_when_metadata_is_consistent(self) -> None:
        decision = _router(embedding_weight=0.0, reference_weight=1.0).observe(
            RouterSessionState("common_10k", "general", created_s=0.0),
            None,
            [
                {"term": "x", "translation": "y", "score": 0.9, "active_glossary_preset": "nlp_core_10k"},
                {"term": "a", "translation": "b", "score": 0.8, "domain": "nlp"},
            ],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_domain_id, "nlp")

    def test_warmup_and_interval_guard_switching(self) -> None:
        warmup = _router(warmup_sec=30).observe(
            RouterSessionState("common_10k", "general", created_s=50.0),
            [0.0, 1.0, 0.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(warmup.action, "stay")
        self.assertIn("warmup<", warmup.reason)

        state = RouterSessionState("common_10k", "general", created_s=0.0, last_decision_s=50.0)
        interval = _router(update_interval_sec=45).observe(state, [0.0, 1.0, 0.0], [], now_s=60.0)
        self.assertEqual(interval.action, "stay")
        self.assertIn("interval<", interval.reason)

    def test_manual_terms_have_no_router_path(self) -> None:
        state = RouterSessionState("common_10k", "general", created_s=0.0)
        decision = _router().observe(state, None, [], now_s=60.0)
        self.assertEqual(decision.action, "stay")
        self.assertIn("no_audio_or_reference_evidence", decision.reason)


if __name__ == "__main__":
    unittest.main()
