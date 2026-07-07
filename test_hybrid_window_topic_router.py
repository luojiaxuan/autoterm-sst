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


def _router_all_domains(**overrides):
    values = {
        "warmup_sec": 0,
        "update_interval_sec": 0,
        "switch_cooldown_sec": 0,
        "min_confidence": 0.60,
        "min_margin": 0.15,
        "min_current_margin": 0.10,
        "min_consistent_windows_with_text": 2,
        "min_consistent_windows_generated_target": 3,
        "min_consistent_windows_audio_only": 3,
        "text_topic_weight": 0.60,
        "domain_probe_weight": 0.25,
        "speech_centroid_weight": 0.10,
        "metadata_prior_weight": 0.05,
    }
    values.update(overrides)
    return HybridWindowTopicRouter(
        [
            DomainSlice("nlp_core_10k", "nlp", index_path="mock://nlp"),
            DomainSlice("medicine_core_10k", "medicine", index_path="mock://medicine"),
            DomainSlice("finance_core_10k", "finance", index_path="mock://finance"),
            DomainSlice("legal_core_10k", "legal", index_path="mock://legal"),
        ],
        RouterConfig(**values),
    )


def _probe_for(target: str) -> dict[str, DomainProbeScore]:
    scores = {
        "nlp": 0.35,
        "medicine": 0.35,
        "finance": 0.35,
        "legal": 0.35,
    }
    scores[target] = 0.90
    return {
        domain: DomainProbeScore(
            domain,
            f"{domain}_core_10k",
            top_score=score,
            mean_topk_score=score,
            top_terms=(domain,),
        )
        for domain, score in scores.items()
    }


def _contested_probe(target: str, other: str, *, target_score: float, other_score: float) -> dict[str, DomainProbeScore]:
    return {
        target: DomainProbeScore(
            target,
            f"{target}_core_10k",
            top_score=target_score,
            mean_topk_score=target_score,
            top_terms=(target,),
        ),
        other: DomainProbeScore(
            other,
            f"{other}_core_10k",
            top_score=other_score,
            mean_topk_score=other_score,
            top_terms=(other,),
        ),
    }


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

    def test_generated_target_chinese_topic_switches_after_three_windows(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        target_text = "患者接受临床治疗，医生根据诊断和症状调整药物剂量。"

        first = router.observe(
            state,
            [1.0, 0.0],
            [],
            now_s=10.0,
            router_text=target_text,
            router_text_source="generated_target",
            domain_probe_scores=_probe_for("medicine"),
        )
        second = router.observe(
            state,
            [1.0, 0.0],
            [],
            now_s=11.0,
            router_text=target_text,
            router_text_source="generated_target",
            domain_probe_scores=_probe_for("medicine"),
        )
        third = router.observe(
            state,
            [1.0, 0.0],
            [],
            now_s=12.0,
            router_text=target_text,
            router_text_source="generated_target",
            domain_probe_scores=_probe_for("medicine"),
        )

        self.assertEqual(first.action, "stay")
        self.assertIn("consistent_windows<3", first.reason)
        self.assertEqual(second.action, "stay")
        self.assertIn("consistent_windows<3", second.reason)
        self.assertEqual(third.action, "switch")
        self.assertEqual(third.target_domain_id, "medicine")

    def test_generated_target_chinese_finance_topic_switches_after_three_windows(self) -> None:
        router = _router_all_domains()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        target_text = "市场利率和债券收益率上升，投资组合中的股票估值承压。"

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=target_text,
                router_text_source="generated_target",
                domain_probe_scores=_probe_for("finance"),
            )
            for step in (10, 11, 12)
        ]

        self.assertEqual(decisions[0].action, "stay")
        self.assertEqual(decisions[1].action, "stay")
        self.assertEqual(decisions[2].action, "switch")
        self.assertEqual(decisions[2].target_domain_id, "finance")

    def test_generated_target_chinese_legal_topic_switches_after_three_windows(self) -> None:
        router = _router_all_domains()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        target_text = "法院根据合同条款和管辖权作出判决，原告要求赔偿。"

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=target_text,
                router_text_source="generated_target",
                domain_probe_scores=_probe_for("legal"),
            )
            for step in (10, 11, 12)
        ]

        self.assertEqual(decisions[0].action, "stay")
        self.assertEqual(decisions[1].action, "stay")
        self.assertEqual(decisions[2].action, "switch")
        self.assertEqual(decisions[2].target_domain_id, "legal")

    def test_generated_target_topic_text_can_switch_without_probe_floor(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        target_text = "患者接受临床治疗，医生根据诊断和症状调整药物剂量。"

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=target_text,
                router_text_source="generated_target",
            )
            for step in (10, 11, 12, 13)
        ]

        self.assertEqual(decisions[0].action, "stay")
        self.assertEqual(decisions[1].action, "stay")
        self.assertEqual(decisions[2].action, "switch")
        self.assertEqual(decisions[2].target_domain_id, "medicine")

    def test_generated_target_generic_text_does_not_dilute_strong_probe(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        generic_target_text = "这个部分主要介绍相关背景和实验设置。"

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=generic_target_text,
                router_text_source="generated_target",
                domain_probe_scores=_probe_for("medicine"),
            )
            for step in (10, 11, 12)
        ]

        self.assertEqual(decisions[0].action, "stay")
        self.assertEqual(decisions[1].action, "stay")
        self.assertEqual(decisions[2].action, "switch")
        self.assertEqual(decisions[2].target_domain_id, "medicine")

    def test_generic_manifest_text_with_contested_probe_does_not_false_switch(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        generic_text = "This part describes the background and experimental setup."
        weak_probe = _contested_probe(
            "medicine",
            "nlp",
            target_score=0.30,
            other_score=0.28,
        )

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=generic_text,
                router_text_source="manifest_source",
                domain_probe_scores=weak_probe,
            )
            for step in (10, 11, 12)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all("probe_only_evidence_insufficient" in decision.reason for decision in decisions))

    def test_generic_manifest_text_with_centroid_and_no_probe_does_not_false_switch(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        generic_text = "This part describes the background and experimental setup."

        decisions = [
            router.observe(
                state,
                [0.0, 1.0],
                [],
                now_s=float(step),
                router_text=generic_text,
                router_text_source="manifest_source",
                domain_probe_scores={},
            )
            for step in (10, 11, 12)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all("topic_text_or_probe_required" in decision.reason for decision in decisions))

    def test_generic_generated_target_with_contested_probe_does_not_false_switch(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        generic_text = "这个部分主要介绍相关背景和实验设置。"
        weak_probe = _contested_probe(
            "medicine",
            "nlp",
            target_score=0.30,
            other_score=0.28,
        )

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=generic_text,
                router_text_source="generated_target",
                domain_probe_scores=weak_probe,
            )
            for step in (10, 11, 12)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all("probe_only_evidence_insufficient" in decision.reason for decision in decisions))

    def test_generated_target_noisy_generic_chinese_does_not_switch_to_nlp(self) -> None:
        router = _router_all_domains()
        state = RouterSessionState("medicine_core_10k", "medicine", created_s=1.0)
        noisy_text = "结果提示需要进一步解析该案例的证据。"

        decisions = [
            router.observe(
                state,
                None,
                [],
                now_s=float(step),
                router_text=noisy_text,
                router_text_source="generated_target",
            )
            for step in (10, 11, 12, 13)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all(decision.confidence < 0.60 for decision in decisions))

    def test_audio_only_probe_requires_three_consistent_windows(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        probe = {
            "nlp": DomainProbeScore(
                "nlp",
                "nlp_core_10k",
                top_score=0.4,
                mean_topk_score=0.35,
                top_terms=("language model",),
            ),
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
        self.assertGreaterEqual(third.confidence, 0.60)

    def test_audio_only_contested_probe_does_not_false_switch_on_small_raw_margin(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        probe = {
            "nlp": DomainProbeScore(
                "nlp",
                "nlp_core_10k",
                top_score=0.50,
                mean_topk_score=0.48,
                top_terms=("language model",),
            ),
            "medicine": DomainProbeScore(
                "medicine",
                "medicine_core_10k",
                top_score=0.53,
                mean_topk_score=0.50,
                top_terms=("clinical trial",),
            ),
        }

        decisions = [
            router.observe(state, None, [], now_s=float(step), domain_probe_scores=probe)
            for step in (10, 11, 12, 13)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all("audio_probe_evidence_insufficient" in decision.reason for decision in decisions))
        self.assertEqual(state.active_domain_id, "nlp")

    def test_audio_only_centroid_without_probe_does_not_switch(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)

        decisions = [
            router.observe(state, [0.0, 1.0], [], now_s=float(step), domain_probe_scores={})
            for step in (10, 11, 12, 13)
        ]

        self.assertTrue(all(decision.action == "stay" for decision in decisions))
        self.assertTrue(all("audio_probe_required" in decision.reason for decision in decisions))
        self.assertEqual(state.active_domain_id, "nlp")

    def test_metadata_prior_does_not_veto_high_confidence_text_topic(self) -> None:
        router = _router()
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=1.0)
        refs = [{"active_glossary_preset": "nlp_core_10k", "score": 0.99}]
        text = "Diagnosis and treatment of diabetes patients in a clinical trial."

        router.observe(state, [1.0, 0.0], refs, now_s=10.0, router_text=text, router_text_source="manifest_source")
        decision = router.observe(
            state,
            [1.0, 0.0],
            refs,
            now_s=11.0,
            router_text=text,
            router_text_source="manifest_source",
        )

        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_domain_id, "medicine")
        self.assertLessEqual(decision.top_scores[1].evidence["metadata_prior"], 1.0)


if __name__ == "__main__":
    unittest.main()
