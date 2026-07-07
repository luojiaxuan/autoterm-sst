from __future__ import annotations

import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eval.streaming_sst.score_terms import allowed_identity_retention_source, score
from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.plugins.backends import get_template
from framework.agents.plugins.retrieval import MockRetrieval
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
    def test_autoterm_yaml_hysteresis_defaults_reach_omni_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = OmniConfig.from_env(get_template("qwen3_omni"))

        self.assertEqual(config.auto_glossary_min_conf, 0.60)
        self.assertEqual(config.auto_glossary_switch_margin, 0.15)
        self.assertEqual(config.auto_glossary_current_margin, 0.10)
        self.assertEqual(config.auto_glossary_min_consistent_windows, 2)
        self.assertEqual(config.auto_glossary_base_preset, "common_10k")
        self.assertEqual(config.auto_glossary_default_preset, "nlp_core_10k")
        self.assertEqual(config.router_mode, "hybrid_window_topic")
        self.assertEqual(config.router_domain_probe_top_k, 5)
        self.assertEqual(config.router_min_consistent_windows_with_text, 2)
        self.assertEqual(config.router_min_consistent_windows_audio_only, 3)
        self.assertEqual(config.auto_glossary_switch_cooldown_sec, 90.0)
        self.assertEqual(config.auto_glossary_candidate_stale_sec, 120.0)

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

    def test_force_exactly_k_uses_nlp_domain_defaults_when_pool_is_short(self) -> None:
        ranked = rank_references([{"term": "BERT", "translation": "BERT", "score": 0.9}])
        prompt = force_exactly_k_references(ranked, k=10, backfill=[], active_domain="nlp")

        self.assertEqual(len(prompt), 10)
        self.assertEqual(len({item["term"].lower() for item in prompt}), 10)
        self.assertTrue(all(item.get("term") and item.get("translation") for item in prompt))
        self.assertEqual(prompt[0]["term"], "BERT")
        self.assertTrue(all(item.get("source_slice_role") != "base" for item in prompt[1:]))
        self.assertTrue(all(item.get("source_domain") == "nlp" for item in prompt[1:]))

    def test_force_exactly_k_uses_neutral_defaults_outside_nlp(self) -> None:
        prompt = force_exactly_k_references([], k=10, backfill=[], active_domain="medicine")
        terms = {item["term"] for item in prompt}

        self.assertEqual(len(prompt), 10)
        self.assertNotIn("BERT", terms)
        self.assertNotIn("Transformer", terms)
        self.assertNotIn("named entity recognition", terms)
        self.assertTrue(all(item.get("term") and item.get("translation") for item in prompt))
        self.assertTrue(all(item.get("source_domain") == "medicine" for item in prompt))
        self.assertTrue(all(item.get("fallback_reason") == "fixed_prompt_k_domain_neutral_default" for item in prompt))
        self.assertTrue(all(item.get("source_preset") != "common_10k" for item in prompt))

    def test_rescue_requires_router_fallback_not_prompt_shortfall(self) -> None:
        agent = OmniAgent()
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            last_router_decision={},
        )
        agent.config.autoterm_enable_open_rescue = True
        agent.config.prompt_top_k = 10

        self.assertFalse(agent._should_rescue_retrieval(session, [{"term": "BERT"}]))

        session.last_router_decision = {"action": "fallback"}
        self.assertTrue(agent._should_rescue_retrieval(session, [{"term": "BERT"}]))

    def test_domain_probe_slices_are_domain_only_debug_inventory(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
        )

        slices = agent._domain_probe_slices(session)
        domains = {item.domain for item in slices}
        roles = {item.role for item in slices}
        presets = {item.preset_id for item in slices}

        self.assertIn("nlp", domains)
        self.assertIn("medicine", domains)
        self.assertNotIn("general", domains)
        self.assertEqual(roles, {"domain_probe"})
        self.assertNotIn("common_10k", presets)

    def test_domain_probe_populates_metadata_without_changing_active_inventory(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        agent.config.auto_glossary_warmup_sec = 0.0
        agent.config.auto_glossary_update_sec = 30.0
        agent.config.auto_glossary_switch_cooldown_sec = 0.0
        agent.retrieval = MockRetrieval(target_lang="zh", top_k=10)
        now = time.perf_counter()
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
            audio=[0.0] * 16000,
            last_llm_samples=0,
            router_text_window="",
            router_text_source="none",
            created_s=now - 100.0,
            router_state=RouterSessionState("nlp_core_10k", "nlp", created_s=now - 100.0),
            last_domain_probe_scores={},
            last_domain_probe_slices=[],
            last_domain_probe_s=None,
            active_slice_presets=["common_10k", "nlp_core_10k"],
        )

        before = list(session.active_slice_presets)
        scores = asyncio.run(agent._probe_domain_scores(session, end_sample=16000))

        self.assertIn("nlp", scores)
        self.assertIn("medicine", scores)
        self.assertEqual(session.active_slice_presets, before)
        self.assertTrue(session.last_domain_probe_scores)
        self.assertTrue(session.last_domain_probe_slices)
        self.assertTrue(all(item["role"] == "domain_probe" for item in session.last_domain_probe_slices))

    def test_domain_probe_respects_router_update_gate(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        agent.config.auto_glossary_warmup_sec = 0.0
        agent.config.auto_glossary_update_sec = 30.0
        agent.config.auto_glossary_switch_cooldown_sec = 0.0
        agent.retrieval = MockRetrieval(target_lang="zh", top_k=10)
        now = time.perf_counter()
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
            audio=[0.0] * 16000,
            last_llm_samples=0,
            router_text_window="",
            router_text_source="none",
            created_s=now - 100.0,
            router_state=RouterSessionState(
                "nlp_core_10k",
                "nlp",
                created_s=now - 100.0,
                last_decision_s=now,
            ),
            last_domain_probe_scores={"old": {}},
            last_domain_probe_slices=[{"preset_id": "old"}],
            last_domain_probe_s=1.0,
        )

        scores = asyncio.run(agent._probe_domain_scores(session, end_sample=16000))

        self.assertEqual(scores, {})
        self.assertEqual(session.last_domain_probe_scores, {})
        self.assertEqual(session.last_domain_probe_slices, [])
        self.assertIsNone(session.last_domain_probe_s)

    def test_identity_retention_metric_allows_acronyms_not_lowercase_phrases(self) -> None:
        gold = [("AI", ["AI"]), ("machine learning", ["machine learning"]), ("syntax", ["句法"])]
        row = score("AI and machine learning improve 句法。", gold, surfaced_terms={"ai", "syntax"})
        self.assertTrue(allowed_identity_retention_source("AI"))
        self.assertFalse(allowed_identity_retention_source("machine learning"))
        self.assertFalse(allowed_identity_retention_source("Neural Network"))
        self.assertTrue(allowed_identity_retention_source("PubMed"))
        self.assertTrue(allowed_identity_retention_source("KinyaBERT"))
        self.assertEqual(row["gold"], 3)
        self.assertEqual(row["term_recall"], 0.667)
        self.assertEqual(row["identity_retention_recall"], 0.333)
        self.assertEqual(row["translation_term_recall"], 0.333)
        self.assertEqual(row["term_recall_surfaced"], 1.0)
        self.assertEqual(row["term_recall_not_surfaced"], 0.0)

    def test_router_can_route_from_unassigned_state_to_domain_slice(self) -> None:
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
                fallback_preset_id="none",
            ),
        )
        state = RouterSessionState(
            active_preset_id="none",
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
                fallback_preset_id="none",
            ),
        )
        state = RouterSessionState(
            active_preset_id="none",
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
