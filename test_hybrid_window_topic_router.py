from __future__ import annotations

import unittest

from framework.agents.term_memory.topic_router import (
    DomainProbeScore,
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


def _router(**overrides):
    values = {
        "warmup_sec": 0,
        "update_interval_sec": 0,
        "switch_cooldown_sec": 0,
        "min_confidence": 0.60,
        "min_margin": 0.15,
        "min_current_margin": 0.10,
        "min_consistent_windows_with_text": 2,
        "min_consistent_windows_audio_only": 3,
        "text_topic_weight": 0.60,
        "domain_probe_weight": 0.25,
        "speech_centroid_weight": 0.10,
        "metadata_prior_weight": 0.05,
    }
    values.update(overrides)
    return HybridWindowTopicRouter(
        [
            DomainSlice("nlp_core_10k", "nlp", centroid=[1.0, 0.0], index_path="mock://nlp"),
            DomainSlice("medicine_core_10k", "medicine", centroid=[0.0, 1.0], index_path="mock://medicine"),
        ],
        RouterConfig(**values),
    )


class HybridWindowTopicRouterTests(unittest.TestCase):
    def test_acl_window_topic_stays_on_nlp_without_medicine_false_switch(self) -> None:
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        decision = _router().observe(
            state,
            [0.2, 0.8],
            [],
            now_s=10.0,
            router_text="We evaluate BERT language models on a corpus benchmark for machine translation.",
            router_text_source="manifest_source",
        )

        self.assertEqual(decision.action, "stay")
        self.assertEqual(decision.target_domain_id, "nlp")
        self.assertIn("same_domain", decision.reason)
        self.assertLess(decision.scores["medicine"], decision.scores["nlp"])

    def test_acl_to_medicine_switches_after_two_text_windows(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        medicine_text = (
            "The patient received clinical treatment for cancer after oncological "
            "surgery at the hospital."
        )

        first = router.observe(
            state,
            [1.0, 0.0],
            [{"active_glossary_preset": "nlp_core_10k", "score": 0.9}],
            now_s=10.0,
            router_text=medicine_text,
            router_text_source="manifest_source",
        )
        second = router.observe(
            state,
            [1.0, 0.0],
            [{"active_glossary_preset": "nlp_core_10k", "score": 0.9}],
            now_s=11.0,
            router_text=medicine_text,
            router_text_source="manifest_source",
        )

        self.assertEqual(first.action, "stay")
        self.assertIn("consistent_windows<2", first.reason)
        self.assertEqual(second.action, "switch")
        self.assertEqual(second.target_preset_id, "medicine_core_10k")
        self.assertGreaterEqual(second.confidence, 0.60)

    def test_audio_only_probe_requires_three_consistent_windows(self) -> None:
        router = _router(min_confidence=0.30)
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        probe = {
            "medicine": DomainProbeScore(
                "medicine",
                "medicine_core_10k",
                top_score=0.9,
                mean_topk_score=0.8,
                top_terms=("clinical trial",),
            )
        }

        first = router.observe(state, [0.0, 1.0], [], now_s=10.0, domain_probe_scores=probe)
        second = router.observe(state, [0.0, 1.0], [], now_s=11.0, domain_probe_scores=probe)
        third = router.observe(state, [0.0, 1.0], [], now_s=12.0, domain_probe_scores=probe)

        self.assertEqual(first.action, "stay")
        self.assertIn("consistent_windows<3", first.reason)
        self.assertEqual(second.action, "stay")
        self.assertIn("consistent_windows<3", second.reason)
        self.assertEqual(third.action, "switch")
        self.assertEqual(third.target_domain_id, "medicine")

    def test_metadata_prior_does_not_veto_high_confidence_text_topic(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        refs = [{"active_glossary_preset": "nlp_core_10k", "score": 0.99}]
        text = "Diagnosis and treatment of diabetes patients in a clinical trial."

        router.observe(state, [1.0, 0.0], refs, now_s=10.0, router_text=text, router_text_source="streaming_asr")
        decision = router.observe(
            state,
            [1.0, 0.0],
            refs,
            now_s=11.0,
            router_text=text,
            router_text_source="streaming_asr",
        )

        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_domain_id, "medicine")
        self.assertLessEqual(decision.top_scores[1].evidence["metadata_prior"], 1.0)


if __name__ == "__main__":
    unittest.main()
