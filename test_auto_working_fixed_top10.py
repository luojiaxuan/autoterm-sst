from __future__ import annotations

import asyncio
import os
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.streaming_sst.score_terms import allowed_identity_retention_source, score
from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.plugins.backends import get_template
from framework.agents.plugins.retrieval import MockRetrieval, RetrievalResult
from framework.agents.term_memory.slice_registry import (
    rank_references,
    slice_id_for_preset,
    slice_role_for_preset,
)
from framework.agents.term_memory.topic_router import (
    AudioNativeActiveGlossaryRouter,
    DomainProbeScore,
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


class AutoWorkingFixedTop10Tests(unittest.TestCase):
    def test_empty_translation_status_acknowledges_the_stream_cursor(self) -> None:
        events = []
        agent = OmniAgent()
        agent._emit = events.append
        session = SimpleNamespace(session_id="empty-output", segment_idx=7)

        agent._emit_cursor_status(
            session,
            text="EMPTY_TRANSLATION",
            start_sample=10,
            cursor_samples=20,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "status")
        self.assertEqual(events[0].text, "EMPTY_TRANSLATION")
        self.assertEqual(events[0].meta["start_sample"], 10)
        self.assertEqual(events[0].meta["cursor_samples"], 20)

    def test_autoterm_yaml_hysteresis_defaults_reach_omni_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = OmniConfig.from_env(get_template("qwen3_omni"))

        self.assertEqual(config.auto_glossary_min_conf, 0.60)
        self.assertEqual(config.auto_glossary_switch_margin, 0.15)
        self.assertEqual(config.auto_glossary_current_margin, 0.30)
        self.assertEqual(config.auto_glossary_min_consistent_windows, 2)
        self.assertEqual(config.auto_glossary_base_preset, "none")
        self.assertEqual(config.auto_glossary_default_preset, "nlp_core_10k")
        self.assertEqual(config.router_mode, "hybrid_window_topic")
        self.assertTrue(config.router_context_similarity_enabled)
        self.assertEqual(config.router_context_similarity_model, "BAAI/bge-m3")
        self.assertEqual(config.router_context_similarity_device, "cpu")
        self.assertEqual(config.router_context_similarity_weight, 0.60)
        self.assertEqual(len(config.auto_glossary_presets.split(",")), 10)
        self.assertEqual(config.router_domain_probe_top_k, 5)
        self.assertEqual(config.router_min_consistent_windows_with_text, 2)
        self.assertEqual(config.router_min_consistent_windows_generated_target, 2)
        self.assertEqual(config.router_min_consistent_windows_audio_only, 3)
        self.assertEqual(config.router_audio_probe_min_top_score, 0.50)
        self.assertEqual(config.router_audio_probe_min_raw_margin, 0.08)
        self.assertEqual(config.router_audio_probe_min_positive_domains, 2)
        self.assertEqual(config.router_generated_target_probe_min_top_score, 0.25)
        self.assertEqual(config.router_generated_target_probe_min_raw_margin, 0.01)
        self.assertEqual(config.router_generated_target_probe_min_positive_domains, 1)
        self.assertTrue(config.router_generated_target_enabled)
        self.assertEqual(config.router_generated_target_window_chunks, 12)
        self.assertEqual(config.router_generated_target_min_chars, 6)
        self.assertEqual(config.router_slice_selection_mode, "hard_top1")
        self.assertEqual(config.router_term_budget, 100_000)
        self.assertEqual(config.router_max_active_slices, 0)
        self.assertEqual(config.router_unknown_slice_term_count, 10_000)
        self.assertEqual(config.auto_glossary_switch_cooldown_sec, 30.0)
        self.assertEqual(config.auto_glossary_candidate_stale_sec, 120.0)
        self.assertEqual(config.autoterm_topk_per_slice, 10)
        self.assertEqual(config.autoterm_candidate_score_threshold, 0.78)

    def test_common_preset_maps_to_common_terms_slice(self) -> None:
        self.assertEqual(slice_id_for_preset("common_10k"), "common_terms")
        self.assertEqual(slice_role_for_preset("common_10k"), "base")
        self.assertEqual(slice_role_for_preset("nlp_core_10k"), "domain")
        self.assertEqual(slice_role_for_preset("open_wiki_100k"), "rescue")

    def test_rank_references_dedupes_candidates(self) -> None:
        candidates = [
            {"term": "model", "translation": "模型", "score": 0.99, "source_slice_role": "domain"},
            {"term": "BERT", "translation": "BERT", "score": 0.80, "source_slice_role": "base"},
            {"term": "BERT", "translation": "伯特", "score": 0.70, "source_slice_role": "domain"},
            {"term": "neural machine translation", "translation": "神经机器翻译", "score": 0.65, "source_slice_role": "domain"},
        ]
        ranked = rank_references(candidates, active_domain="nlp")

        self.assertEqual(len(ranked), 3)
        self.assertEqual(len({item["term"].lower() for item in ranked}), 3)
        self.assertIn("BERT", {item["term"] for item in ranked})

    def test_fixed_glossary_preset_does_not_backfill_after_filtering(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=False,
            glossary_preset="nlp_core_10k",
            active_glossary_preset="nlp_core_10k",
            active_domain="nlp",
            recent_references=deque(maxlen=16),
        )

        prompt = agent._prompt_references(session, [{"term": "BERT", "translation": "BERT", "score": 0.9}])

        self.assertEqual(len(prompt), 1)
        self.assertEqual(prompt[0]["term"], "BERT")

    def test_fixed_glossary_preset_truncates_to_prompt_top_k(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=False,
            glossary_preset="nlp_core_10k",
            active_glossary_preset="nlp_core_10k",
            active_domain="nlp",
            recent_references=deque(maxlen=16),
        )
        refs = [
            {"term": f"term {idx}", "translation": f"译文{idx}", "score": 1.0 - idx * 0.01}
            for idx in range(12)
        ]

        prompt = agent._prompt_references(session, refs)

        self.assertEqual(len(prompt), 10)
        self.assertEqual(prompt[0]["term"], "term 0")
        self.assertEqual(prompt[-1]["term"], "term 9")

    def test_auto_glossary_preset_does_not_backfill_after_filtering(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            glossary_preset="nlp_core_10k",
            active_glossary_preset="nlp_core_10k",
            active_domain="nlp",
            recent_references=deque(maxlen=16),
        )

        prompt = agent._prompt_references(session, [{"term": "BERT", "translation": "BERT", "score": 0.9}])

        self.assertEqual(len(prompt), 1)
        self.assertEqual(prompt[0]["term"], "BERT")

    def test_none_glossary_does_not_backfill_prompt_candidates(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=False,
            glossary_preset="none",
            active_glossary_preset="none",
            active_domain="general",
            recent_references=deque(maxlen=16),
        )

        prompt = agent._prompt_references(session, [])

        self.assertEqual(prompt, [])

    def test_no_glossary_alias_does_not_backfill_prompt_candidates(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=False,
            glossary_preset="no_glossary",
            active_glossary_preset="no_glossary",
            active_domain="general",
            recent_references=deque(maxlen=16),
        )

        prompt = agent._prompt_references(session, [])

        self.assertEqual(prompt, [])

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

    def test_generated_target_translation_updates_router_text_window(self) -> None:
        agent = OmniAgent()
        agent.config.router_generated_target_enabled = True
        agent.config.router_generated_target_window_chunks = 2
        agent.config.router_generated_target_min_chars = 4
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            router_text_window="",
            router_text_source="none",
            router_generated_target_history=["这是一个语言模型。"],
        )

        agent._after_translation_tick(session, text="患者接受临床治疗。", references=[])

        self.assertEqual(session.router_text_source, "generated_target")
        self.assertIn("语言模型", session.router_text_window)
        self.assertIn("临床治疗", session.router_text_window)

    def test_generated_target_translation_uses_current_text_when_history_is_empty(self) -> None:
        agent = OmniAgent()
        agent.config.router_generated_target_enabled = True
        agent.config.router_generated_target_min_chars = 4
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            router_text_window="",
            router_text_source="none",
            router_generated_target_history=[],
        )

        agent._after_translation_tick(session, text="患者接受临床治疗。", references=[])

        self.assertEqual(session.router_text_source, "generated_target")
        self.assertEqual(session.router_text_window, "患者接受临床治疗。")

    def test_generated_target_translation_does_not_override_external_router_text(self) -> None:
        agent = OmniAgent()
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            router_text_window="external source text",
            router_text_source="manifest_source",
            router_generated_target_history=["患者接受临床治疗。"],
        )

        agent._after_translation_tick(session, text="患者接受临床治疗。", references=[])

        self.assertEqual(session.router_text_source, "manifest_source")
        self.assertEqual(session.router_text_window, "external source text")

    def test_generated_target_router_history_is_independent_of_decoder_cache(self) -> None:
        agent = OmniAgent()
        agent.config.router_generated_target_enabled = True
        agent.config.router_generated_target_window_chunks = 3
        agent.config.router_generated_target_min_chars = 1
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            router_text_window="",
            router_text_source="none",
            router_generated_target_history=[],
            history=["decoder-only context"],
        )

        for text in ("oldest", "middle", "newest", "current"):
            agent._after_translation_tick(session, text=text, references=[])

        self.assertEqual(
            session.router_generated_target_history,
            ["middle", "newest", "current"],
        )
        self.assertEqual(session.router_text_window, "middle\nnewest\ncurrent")
        self.assertNotIn("decoder-only context", session.router_text_window)

    def test_generated_target_text_is_used_by_next_router_tick_only(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        agent.config.auto_glossary_warmup_sec = 0.0
        agent.config.auto_glossary_update_sec = 0.0
        agent.config.auto_glossary_switch_cooldown_sec = 0.0
        agent.config.router_min_consistent_windows_generated_target = 1
        now = time.perf_counter()
        session = SimpleNamespace(
            session_id="s-generated-target-order",
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
            active_glossary_preset="nlp_core_10k",
            active_domain="nlp",
            router_text_window="",
            router_text_source="none",
            history=[],
            recent_references=deque(maxlen=16),
            router_state=RouterSessionState("nlp_core_10k", "nlp", created_s=now - 10.0),
            created_s=now - 10.0,
            topic_confidence=0.0,
            last_topic_reason="",
            last_topic_update_s=0.0,
            last_router_decision={},
            topic_history=[],
            topic_update_task=SimpleNamespace(done=lambda: False),
        )

        asyncio.run(agent._observe_active_glossary(session, RetrievalResult(references=[])))

        self.assertFalse(session.last_router_decision["evidence"]["has_router_text"])
        self.assertEqual(session.last_router_decision["evidence"]["router_text_source"], "none")

        agent._after_translation_tick(session, text="患者接受临床治疗。", references=[])
        probe_scores = {
            "nlp": DomainProbeScore(
                domain="nlp",
                preset_id="nlp_core_10k",
                top_score=0.08,
                mean_topk_score=0.08,
                top_terms=("nlp",),
            ),
            "medicine": DomainProbeScore(
                domain="medicine",
                preset_id="medicine_core_10k",
                top_score=0.35,
                mean_topk_score=0.35,
                top_terms=("clinical",),
            ),
        }
        asyncio.run(
            agent._observe_active_glossary(
                session,
                RetrievalResult(references=[]),
                domain_probe_scores=probe_scores,
            )
        )

        self.assertTrue(session.last_router_decision["evidence"]["has_router_text"])
        self.assertEqual(session.last_router_decision["evidence"]["router_text_source"], "generated_target")
        self.assertEqual(session.last_router_decision["to_domain"], "medicine")

    def test_streaming_loop_orders_routing_before_generated_target_write(self) -> None:
        source = Path("framework/agents/omni.py").read_text(encoding="utf-8")
        retrieve_call = source.index("refs_by_session = await self._retrieve_batch")
        generate_batch_call = source.index("results = await self._generate_batch", retrieve_call)
        generate_one_call = source.index("self._generate_one(", retrieve_call)
        retrieve_def = source.index("async def _retrieve_batch")
        retrieve_end = source.index("async def _retrieve_slice_groups", retrieve_def)
        observe_call = source.index("await self._observe_active_glossary", retrieve_def, retrieve_end)

        self.assertLess(retrieve_call, generate_batch_call)
        self.assertLess(retrieve_call, generate_one_call)
        self.assertLess(observe_call, retrieve_end)

        generate_one_def = source.index("async def _generate_one")
        generate_batch_def = source.index("async def _generate_batch")
        one_history = source.index("session.history.append(text)", generate_one_def, generate_batch_def)
        one_after_tick = source.index("self._after_translation_tick(session, text=text", one_history, generate_batch_def)
        batch_history = source.index("session.history.append(text)", generate_batch_def)
        batch_after_tick = source.index("self._after_translation_tick(session, text=text", batch_history)

        self.assertLess(one_history, one_after_tick)
        self.assertLess(batch_history, batch_after_tick)

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

    def test_auto_active_retrieval_uses_domain_specific_slice_only(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
            active_retrieval_slices=[],
            glossary_preset="auto_working",
            active_glossary_preset="nlp_core_10k",
            active_slice_presets=[],
            active_slice_terms={},
            last_retrieval_plan=[],
        )

        slices = agent._active_retrieval_slices(session)

        self.assertEqual([item.preset_id for item in slices], ["nlp_core_10k"])
        self.assertEqual([item.role for item in slices], ["domain"])
        self.assertEqual(session.active_slice_presets, ["nlp_core_10k"])

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
            latency_multiplier=2,
            created_s=now - 100.0,
            router_state=RouterSessionState("nlp_core_10k", "nlp", created_s=now - 100.0),
            last_domain_probe_raw_scores={},
            last_domain_probe_scores={},
            last_domain_probe_slices=[],
            last_domain_probe_s=None,
            last_domain_probe_at_s=0.0,
            last_domain_probe_cached=False,
            active_slice_presets=["nlp_core_10k"],
        )

        before = list(session.active_slice_presets)
        scores = asyncio.run(agent._probe_domain_scores(session, end_sample=16000))

        self.assertIn("nlp", scores)
        self.assertIn("medicine", scores)
        self.assertEqual(session.active_slice_presets, before)
        self.assertTrue(session.last_domain_probe_scores)
        self.assertTrue(session.last_domain_probe_slices)
        self.assertTrue(session.last_domain_probe_raw_scores)
        self.assertGreater(session.last_domain_probe_at_s, 0.0)
        self.assertFalse(session.last_domain_probe_cached)
        self.assertTrue(all(item["role"] == "domain_probe" for item in session.last_domain_probe_slices))

    def test_zero_weight_disables_domain_probe_work(self) -> None:
        agent = OmniAgent()
        agent.config.router_domain_probe_weight = 0.0
        agent.retrieval = MockRetrieval(target_lang="zh", top_k=10)
        session = SimpleNamespace(
            last_domain_probe_raw_scores={"stale": object()},
            last_domain_probe_scores={"stale": {}},
            last_domain_probe_slices=[{"preset_id": "stale"}],
            last_domain_probe_s=1.0,
            last_domain_probe_at_s=1.0,
            last_domain_probe_cached=True,
        )

        scores = asyncio.run(agent._probe_domain_scores(session, end_sample=16000))

        self.assertEqual(scores, {})
        self.assertEqual(session.last_domain_probe_raw_scores, {})
        self.assertEqual(session.last_domain_probe_scores, {})
        self.assertEqual(session.last_domain_probe_slices, [])
        self.assertIsNone(session.last_domain_probe_s)
        self.assertEqual(session.last_domain_probe_at_s, 0.0)
        self.assertFalse(session.last_domain_probe_cached)

    def test_domain_probe_reuses_cached_scores_inside_update_gate(self) -> None:
        agent = OmniAgent()
        agent.config.mock = True
        agent.config.auto_glossary_warmup_sec = 0.0
        agent.config.auto_glossary_update_sec = 30.0
        agent.config.auto_glossary_switch_cooldown_sec = 0.0
        agent.retrieval = MockRetrieval(target_lang="zh", top_k=10)
        now = time.perf_counter()
        cached = {
            "medicine": DomainProbeScore(
                domain="medicine",
                preset_id="medicine_core_10k",
                top_score=0.95,
                mean_topk_score=0.90,
                top_terms=("oncology",),
            )
        }
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            language_pair="English -> Chinese",
            audio=[0.0] * 16000,
            last_llm_samples=0,
            router_text_window="",
            router_text_source="none",
            latency_multiplier=2,
            created_s=now - 100.0,
            router_state=RouterSessionState(
                "nlp_core_10k",
                "nlp",
                created_s=now - 100.0,
                last_decision_s=now,
            ),
            last_domain_probe_raw_scores=cached,
            last_domain_probe_scores={},
            last_domain_probe_slices=[],
            last_domain_probe_s=1.0,
            last_domain_probe_at_s=now - 0.1,
            last_domain_probe_cached=False,
        )

        scores = asyncio.run(agent._probe_domain_scores(session, end_sample=16000))

        self.assertEqual(scores, cached)
        self.assertIn("medicine", session.last_domain_probe_scores)
        self.assertTrue(session.last_domain_probe_slices)
        self.assertEqual(session.last_domain_probe_s, 0.0)
        self.assertTrue(session.last_domain_probe_cached)

    def test_domain_probe_request_uses_recent_audio_window(self) -> None:
        agent = OmniAgent()
        agent.config.rag_timeline_lookback_sec = 1.0
        session = SimpleNamespace(
            audio=[0.0] * 64000,
            last_llm_samples=32000,
            router_text_window="",
            router_text_source="none",
        )

        request = agent._domain_probe_request(session, end_sample=64000)

        self.assertEqual(len(request["audio_buffer"]), 48000)
        self.assertEqual(request["current_start_sec"], 1.0)
        self.assertEqual(request["current_end_sec"], 3.0)

    def test_domain_probe_refresh_uses_window_cadence_without_text(self) -> None:
        agent = OmniAgent()
        agent.config.base_segment_sec = 0.96
        agent.config.auto_glossary_update_sec = 30.0
        no_text = SimpleNamespace(
            router_text_window="",
            router_text_source="none",
            latency_multiplier=2,
        )
        with_text = SimpleNamespace(
            router_text_window="oncology surgery",
            router_text_source="generated_target",
            latency_multiplier=2,
        )

        self.assertEqual(agent._domain_probe_refresh_sec(no_text), 1.92)
        self.assertEqual(agent._domain_probe_refresh_sec(with_text), 30.0)

    def test_context_similarity_batches_and_reuses_cached_scores(self) -> None:
        class FakeContextSimilarity:
            enabled = True

            def __init__(self) -> None:
                self.calls = []

            async def score_batch(self, texts, *, allowed_domains):  # noqa: ANN001, ANN202
                self.calls.append((list(texts), list(allowed_domains)))
                return [{"nlp": 0.2, "medicine": 0.9} for _ in texts]

        agent = OmniAgent()
        agent.config.auto_glossary_update_sec = 30.0
        scorer = FakeContextSimilarity()
        agent.context_similarity = scorer
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            router_text_window="患者接受临床治疗。",
            router_text_source="generated_target",
            latency_multiplier=2,
            last_context_similarity_scores={},
            last_context_similarity_s=None,
            last_context_similarity_at_s=0.0,
            last_context_similarity_text="",
            last_context_similarity_cached=False,
        )

        first = asyncio.run(agent._context_similarity_scores_batch([session]))
        second = asyncio.run(agent._context_similarity_scores_batch([session]))

        self.assertEqual(first, [{"nlp": 0.2, "medicine": 0.9}])
        self.assertEqual(second, first)
        self.assertEqual(len(scorer.calls), 1)
        self.assertTrue(session.last_context_similarity_cached)
        self.assertEqual(session.last_context_similarity_s, 0.0)

    def test_cached_probe_scores_can_confirm_audio_only_switch(self) -> None:
        router = HybridWindowTopicRouter(
            [
                DomainSlice("nlp_core_10k", "nlp", centroid=[1.0, 0.0], index_path="mock://nlp"),
                DomainSlice("medicine_core_10k", "medicine", centroid=[1.0, 0.0], index_path="mock://medicine"),
            ],
            RouterConfig(
                warmup_sec=0.0,
                update_interval_sec=0.0,
                switch_cooldown_sec=0.0,
                min_confidence=0.5,
                min_margin=0.1,
                min_current_margin=0.1,
                min_consistent_windows_audio_only=2,
                text_topic_weight=0.0,
                domain_probe_weight=1.0,
                speech_centroid_weight=0.0,
                metadata_prior_weight=0.0,
                fallback_preset_id="none",
            ),
        )
        state = RouterSessionState("nlp_core_10k", "nlp", created_s=0.0)
        probe = {
            "nlp": DomainProbeScore(
                domain="nlp",
                preset_id="nlp_core_10k",
                top_score=0.4,
                mean_topk_score=0.35,
                top_terms=("language model",),
            ),
            "medicine": DomainProbeScore(
                domain="medicine",
                preset_id="medicine_core_10k",
                top_score=0.9,
                mean_topk_score=0.8,
                top_terms=("oncology",),
            )
        }

        first = router.observe(state, None, [], now_s=1.0, domain_probe_scores=probe)
        second = router.observe(state, None, [], now_s=2.0, domain_probe_scores=probe)

        self.assertEqual(first.action, "stay")
        self.assertIn("consistent_windows<2", first.reason)
        self.assertEqual(second.action, "switch")
        self.assertEqual(second.target_domain_id, "medicine")

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
