from __future__ import annotations

import unittest

from eval.streaming_sst.score_terms import allowed_identity_retention_source, score
from framework.agents.term_memory.slice_registry import (
    force_exactly_k_references,
    rank_references,
    slice_id_for_preset,
    slice_role_for_preset,
)
from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainSlice,
    RouterConfig,
    RouterSessionState,
)


class AutoWorkingFixedTop10Tests(unittest.TestCase):
    def test_common_preset_maps_to_common_terms_slice(self) -> None:
        self.assertEqual(slice_id_for_preset("common_10k"), "common_terms")
        self.assertEqual(slice_role_for_preset("common_10k"), "base")
        self.assertEqual(slice_role_for_preset("nlp_core_10k"), "domain")
        self.assertEqual(slice_role_for_preset("open_wiki_100k"), "rescue")

    def test_rank_and_force_exactly_k_dedupes_then_backfills(self) -> None:
        candidates = [
            {"term": "model", "translation": "模型", "score": 0.99, "source_slice_role": "domain"},
            {"term": "BERT", "translation": "BERT", "score": 0.80, "source_slice_role": "base"},
            {"term": "BERT", "translation": "伯特", "score": 0.70, "source_slice_role": "domain"},
            {"term": "neural machine translation", "translation": "神经机器翻译", "score": 0.65, "source_slice_role": "domain"},
        ]
        ranked = rank_references(candidates, active_domain="nlp")
        prompt = force_exactly_k_references(
            ranked,
            k=3,
            backfill=[{"term": "fallback", "translation": "回填", "score": 0.1}],
        )
        self.assertEqual(len(prompt), 3)
        self.assertEqual(len({item["term"].lower() for item in prompt}), 3)
        self.assertIn("BERT", {item["term"] for item in prompt})

    def test_identity_retention_metric_allows_acronyms_not_lowercase_phrases(self) -> None:
        gold = [("AI", ["AI"]), ("machine learning", ["machine learning"]), ("syntax", ["句法"])]
        row = score("AI and machine learning improve 句法。", gold, surfaced_terms={"ai", "syntax"})
        self.assertTrue(allowed_identity_retention_source("AI"))
        self.assertFalse(allowed_identity_retention_source("machine learning"))
        self.assertEqual(row["gold"], 3)
        self.assertEqual(row["term_recall"], 0.667)
        self.assertEqual(row["identity_retention_recall"], 0.333)
        self.assertEqual(row["translation_term_recall"], 0.333)
        self.assertEqual(row["term_recall_surfaced"], 1.0)
        self.assertEqual(row["term_recall_not_surfaced"], 0.0)

    def test_router_can_route_from_common_base_to_domain_slice(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [
                DomainSlice(
                    preset_id="nlp_core_10k",
                    domain_id="nlp",
                    centroid=[1.0, 0.0],
                    index_path="mock://nlp",
                )
            ],
            RouterConfig(
                warmup_sec=0.0,
                update_interval_sec=0.0,
                min_confidence=0.5,
                min_margin=0.0,
                min_consistent_windows=1,
                fallback_preset_id="common_10k",
            ),
        )
        state = RouterSessionState(
            active_preset_id="common_10k",
            active_domain_id="general",
            created_s=0.0,
            last_decision_s=0.0,
        )
        decision = router.observe(state, [1.0, 0.0], [], now_s=1.0)

        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.target_preset_id, "nlp_core_10k")
        self.assertEqual(decision.target_domain_id, "nlp")

    def test_router_consistency_pending_does_not_start_interval_gate(self) -> None:
        router = AudioNativeActiveGlossaryRouter(
            [
                DomainSlice(
                    preset_id="nlp_core_10k",
                    domain_id="nlp",
                    centroid=[1.0, 0.0],
                    index_path="mock://nlp",
                )
            ],
            RouterConfig(
                warmup_sec=0.0,
                update_interval_sec=45.0,
                min_confidence=0.5,
                min_margin=0.0,
                min_consistent_windows=2,
                fallback_preset_id="common_10k",
            ),
        )
        state = RouterSessionState(
            active_preset_id="common_10k",
            active_domain_id="general",
            created_s=0.0,
            last_decision_s=0.0,
        )

        first = router.observe(state, [1.0, 0.0], [], now_s=1.0)
        second = router.observe(state, [1.0, 0.0], [], now_s=2.0)

        self.assertEqual(first.action, "stay")
        self.assertIn("consistent_windows<2", first.reason)
        self.assertEqual(state.last_decision_s, 2.0)
        self.assertEqual(second.action, "switch")
        self.assertEqual(second.target_preset_id, "nlp_core_10k")


if __name__ == "__main__":
    unittest.main()
