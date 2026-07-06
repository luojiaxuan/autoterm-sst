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
            DomainSlice("nlp_core_10k", "nlp", centroid=[1.0, 0.0], index_path="mock://nlp"),
            DomainSlice("medicine_core_10k", "medicine", centroid=[0.0, 1.0], index_path="mock://medicine"),
        ],
        cfg,
    )


class TopicRouterEmbeddingRefsTests(unittest.TestCase):
    def test_strong_embedding_switches_to_nlp(self) -> None:
        decision = _router().observe(
            RouterSessionState("none", "general", created_s=0.0),
            [1.0, 0.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_preset_id, "nlp_core_10k")

    def test_ambiguous_embedding_stays(self) -> None:
        decision = _router(min_confidence=0.0, min_margin=0.40).observe(
            RouterSessionState("none", "general", created_s=0.0),
            [0.7, 0.7],
            [],
            now_s=60.0,
        )
        self.assertEqual(decision.action, "stay")
        self.assertIn("margin<", decision.reason)

    def test_refs_only_can_switch_when_metadata_is_consistent(self) -> None:
        decision = _router(embedding_weight=0.0, reference_weight=1.0).observe(
            RouterSessionState("none", "general", created_s=0.0),
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
            RouterSessionState("none", "general", created_s=50.0),
            [1.0, 0.0],
            [],
            now_s=60.0,
        )
        self.assertEqual(warmup.action, "stay")
        self.assertIn("warmup<", warmup.reason)

        state = RouterSessionState("none", "general", created_s=0.0, last_decision_s=50.0)
        interval = _router(update_interval_sec=45).observe(state, [1.0, 0.0], [], now_s=60.0)
        self.assertEqual(interval.action, "stay")
        self.assertIn("interval<", interval.reason)

    def test_manual_terms_have_no_router_path(self) -> None:
        state = RouterSessionState("none", "general", created_s=0.0)
        decision = _router().observe(state, None, [], now_s=60.0)
        self.assertEqual(decision.action, "stay")
        self.assertIn("no_audio_or_reference_evidence", decision.reason)

    def test_switch_cooldown_blocks_immediate_ping_pong(self) -> None:
        state = RouterSessionState(
            "nlp_core_10k",
            "nlp",
            created_s=0.0,
            last_switch_s=100.0,
        )
        router = _router(switch_cooldown_sec=90.0)

        blocked = router.observe(state, [0.0, 1.0], [], now_s=120.0)
        allowed = router.observe(state, [0.0, 1.0], [], now_s=191.0)

        self.assertEqual(blocked.action, "stay")
        self.assertIn("cooldown<90", blocked.reason)
        self.assertEqual(blocked.evidence["candidate_streak"], 0)
        self.assertEqual(allowed.action, "switch")
        self.assertEqual(allowed.target_preset_id, "medicine_core_10k")

    def test_cooldown_tick_preserves_interval_throttle(self) -> None:
        state = RouterSessionState(
            "nlp_core_10k",
            "nlp",
            created_s=0.0,
            last_switch_s=100.0,
        )
        router = _router(update_interval_sec=45.0, switch_cooldown_sec=90.0)

        cooldown = router.observe(state, [0.0, 1.0], [], now_s=120.0)
        interval = router.observe(state, [0.0, 1.0], [], now_s=121.0)

        self.assertEqual(cooldown.action, "stay")
        self.assertIn("cooldown<90", cooldown.reason)
        self.assertEqual(state.last_decision_s, 120.0)
        self.assertEqual(interval.action, "stay")
        self.assertIn("interval<45", interval.reason)
        self.assertEqual(state.last_decision_s, 120.0)

    def test_switch_requires_target_to_beat_current_slice(self) -> None:
        router = _router(
            embedding_weight=0.0,
            reference_weight=1.0,
            min_margin=0.0,
            min_current_margin=0.20,
        )
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=0.0)

        decision = router.observe(
            state,
            None,
            [
                {"score": 0.55, "active_glossary_preset": "medicine_core_10k"},
                {"score": 0.45, "active_glossary_preset": "nlp_core_10k"},
            ],
            now_s=60.0,
        )

        self.assertEqual(decision.action, "stay")
        self.assertIn("current_margin<0.20", decision.reason)
        self.assertEqual(decision.evidence["current_score"], 0.45)
        self.assertEqual(decision.evidence["target_score_delta"], 0.1)

    def test_candidate_streak_resets_when_target_changes(self) -> None:
        router = _router(min_consistent_windows=2, ema_alpha=0.0)
        state = RouterSessionState("none", "general", created_s=0.0)

        first = router.observe(state, [1.0, 0.0], [], now_s=60.0)
        second = router.observe(state, [0.0, 1.0], [], now_s=61.0)
        third = router.observe(state, [0.0, 1.0], [], now_s=62.0)

        self.assertEqual(first.action, "stay")
        self.assertEqual(first.evidence["candidate_preset"], "nlp_core_10k")
        self.assertEqual(second.action, "stay")
        self.assertEqual(second.evidence["candidate_preset"], "medicine_core_10k")
        self.assertEqual(second.evidence["candidate_streak"], 1)
        self.assertEqual(third.action, "switch")
        self.assertEqual(third.evidence["candidate_streak"], 2)

    def test_interval_gates_candidates_after_same_domain_tick(self) -> None:
        router = _router(
            update_interval_sec=45.0,
            min_consistent_windows=1,
            ema_alpha=0.0,
        )
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=0.0)

        steady = router.observe(state, [1.0, 0.0], [], now_s=60.0)
        blocked = router.observe(state, [0.0, 1.0], [], now_s=61.0)
        allowed = router.observe(state, [0.0, 1.0], [], now_s=106.0)

        self.assertEqual(steady.action, "stay")
        self.assertIn("same_domain", steady.reason)
        self.assertEqual(state.last_decision_s, 106.0)
        self.assertEqual(blocked.action, "stay")
        self.assertIn("interval<45", blocked.reason)
        self.assertEqual(blocked.evidence["candidate_streak"], 0)
        self.assertEqual(allowed.action, "switch")
        self.assertEqual(allowed.target_preset_id, "medicine_core_10k")

    def test_candidate_streak_resets_after_stale_gap(self) -> None:
        router = _router(min_consistent_windows=2, candidate_stale_sec=10.0)
        state = RouterSessionState("none", "general", created_s=0.0)

        first = router.observe(state, [1.0, 0.0], [], now_s=60.0)
        stale = router.observe(state, [1.0, 0.0], [], now_s=75.0)
        fresh = router.observe(state, [1.0, 0.0], [], now_s=76.0)

        self.assertEqual(first.evidence["candidate_streak"], 1)
        self.assertEqual(stale.action, "stay")
        self.assertEqual(stale.evidence["candidate_streak"], 1)
        self.assertEqual(fresh.action, "switch")
        self.assertEqual(fresh.evidence["candidate_streak"], 2)


if __name__ == "__main__":
    unittest.main()
