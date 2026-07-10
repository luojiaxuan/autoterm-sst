from __future__ import annotations

import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.term_memory.context_similarity import DomainDescriptionSimilarity
from framework.agents.term_memory.manifest import TermMemoryManifest
from framework.agents.term_memory.slice_registry import RetrievalSlice
from framework.agents.term_memory.topic_router import (
    DomainSlice,
    HybridWindowTopicRouter,
    RouterConfig,
    RouterSessionState,
)


def _topic_slices(count: int = 12) -> list[DomainSlice]:
    return [
        DomainSlice(
            preset_id=f"topic_{index:03d}",
            domain_id=f"domain_{index:03d}",
            index_path=f"mock://topic_{index:03d}",
            term_count=10_000,
            description=f"Topic {index} terminology.",
        )
        for index in range(count)
    ]


def _retrieval_slices(count: int) -> list[RetrievalSlice]:
    return [
        RetrievalSlice(
            preset_id=f"topic_{index:03d}",
            slice_id=f"topic_{index:03d}",
            role="domain",
            domain=f"domain_{index:03d}",
            index_path=f"mock://topic_{index:03d}",
            term_count=10_000,
        )
        for index in range(count)
    ]


class BudgetAwareRouterTests(unittest.TestCase):
    def test_context_similarity_keeps_the_most_recent_tokens(self) -> None:
        class FakeTokenizer:
            truncation_side = "right"

        tokenizer = DomainDescriptionSimilarity._configure_tokenizer(FakeTokenizer())

        self.assertEqual(tokenizer.truncation_side, "left")

    def test_shared_candidate_budget_is_equal_for_one_two_and_ten_slices(self) -> None:
        agent = OmniAgent(
            config=OmniConfig(
                mock=True,
                rag_enabled=False,
                retrieval_candidate_budget=100,
            )
        )
        auto_session = SimpleNamespace(auto_glossary_enabled=True)
        merged_session = SimpleNamespace(auto_glossary_enabled=False)

        merged = agent._retrieval_top_ks_for(merged_session, _retrieval_slices(1))
        two_slices = agent._retrieval_top_ks_for(auto_session, _retrieval_slices(2))
        ten_slices = agent._retrieval_top_ks_for(auto_session, _retrieval_slices(10))

        self.assertEqual(merged, [100])
        self.assertEqual(two_slices, [50, 50])
        self.assertEqual(ten_slices, [10] * 10)
        self.assertEqual({sum(merged), sum(two_slices), sum(ten_slices)}, {100})

    def test_shared_candidate_budget_assigns_remainder_deterministically(self) -> None:
        agent = OmniAgent(
            config=OmniConfig(
                mock=True,
                rag_enabled=False,
                retrieval_candidate_budget=103,
            )
        )

        allocation = agent._retrieval_top_ks_for(
            SimpleNamespace(auto_glossary_enabled=True),
            _retrieval_slices(10),
        )

        self.assertEqual(allocation, [11, 11, 11, 10, 10, 10, 10, 10, 10, 10])
        self.assertEqual(sum(allocation), 103)

    def test_zero_candidate_budget_preserves_legacy_per_slice_top_k(self) -> None:
        agent = OmniAgent(
            config=OmniConfig(
                mock=True,
                rag_enabled=False,
                retrieval_candidate_budget=0,
                rag_top_k=7,
                prompt_top_k=10,
                ui_top_k=8,
                autoterm_topk_per_slice=12,
            )
        )

        auto = agent._retrieval_top_ks_for(
            SimpleNamespace(auto_glossary_enabled=True),
            _retrieval_slices(2),
        )
        merged = agent._retrieval_top_ks_for(
            SimpleNamespace(auto_glossary_enabled=False),
            _retrieval_slices(1),
        )

        self.assertEqual(auto, [12, 12])
        self.assertEqual(merged, [10])

    def test_candidate_cost_metadata_reports_pre_rerank_work(self) -> None:
        agent = OmniAgent(
            config=OmniConfig(
                mock=True,
                rag_enabled=False,
                retrieval_candidate_budget=100,
            )
        )
        session = SimpleNamespace(
            last_candidate_pool_count=37,
            last_retrieval_candidate_requested=0,
            last_retrieval_candidate_returned=0,
            last_retrieval_index_queries=0,
            last_retrieval_scored_inventory_terms=0,
            last_retrieval_top_k_by_slice={},
        )
        plans = _retrieval_slices(2)

        agent._record_retrieval_query(session, plans[0], 50)
        agent._record_retrieval_query(session, plans[1], 50)
        session.last_retrieval_candidate_returned = 82
        cost = agent._retrieval_candidate_cost_meta(session)

        self.assertEqual(cost["allocation_mode"], "shared_total")
        self.assertEqual(cost["configured_budget"], 100)
        self.assertEqual(cost["requested_top_k"], 100)
        self.assertEqual(cost["returned_before_rerank"], 82)
        self.assertEqual(cost["pool_after_rerank"], 37)
        self.assertEqual(cost["index_queries"], 2)
        self.assertEqual(cost["scored_inventory_terms"], 20_000)
        self.assertEqual(cost["top_k_by_slice"], {"topic_000": 50, "topic_001": 50})
        self.assertEqual(agent._remaining_retrieval_candidate_budget(session), 0)

    def test_hard_top1_remains_the_default(self) -> None:
        router = HybridWindowTopicRouter(_topic_slices(), RouterConfig())

        selection = router.select_budgeted_slices(
            {f"domain_{index:03d}": 1.0 - index * 0.01 for index in range(12)}
        )

        self.assertFalse(router.budgeted_slice_selection_enabled())
        self.assertEqual(selection.preset_ids, [])

    def test_100k_budget_selects_top_ten_10k_slices(self) -> None:
        router = HybridWindowTopicRouter(
            _topic_slices(),
            RouterConfig(
                slice_selection_mode="budgeted_top_slices",
                term_budget=100_000,
            ),
        )
        scores = {
            f"domain_{index:03d}": 1.0 - index * 0.01
            for index in range(12)
        }

        selection = router.select_budgeted_slices(scores)

        self.assertEqual(
            selection.preset_ids,
            [f"topic_{index:03d}" for index in range(10)],
        )
        self.assertEqual(selection.total_terms, 100_000)
        self.assertEqual(selection.to_meta()["selected_slice_count"], 10)

    def test_budget_selector_skips_a_slice_that_does_not_fit(self) -> None:
        slices = _topic_slices(4)
        slices[0].term_count = 70_000
        slices[1].term_count = 40_000
        slices[2].term_count = 30_000
        router = HybridWindowTopicRouter(
            slices,
            RouterConfig(
                slice_selection_mode="budgeted_top_slices",
                term_budget=100_000,
            ),
        )

        selection = router.select_budgeted_slices(
            {
                "domain_000": 0.95,
                "domain_001": 0.90,
                "domain_002": 0.85,
                "domain_003": 0.80,
            }
        )

        self.assertEqual(selection.preset_ids, ["topic_000", "topic_002"])
        self.assertEqual(selection.total_terms, 100_000)

    def test_router_decision_records_budget_selection_from_context_similarity(self) -> None:
        router = HybridWindowTopicRouter(
            _topic_slices(),
            RouterConfig(
                warmup_sec=0.0,
                update_interval_sec=0.0,
                switch_cooldown_sec=0.0,
                slice_selection_mode="budgeted_top_slices",
                term_budget=100_000,
            ),
        )
        scores = {
            f"domain_{index:03d}": 1.0 - index * 0.01
            for index in range(12)
        }

        decision = router.observe(
            RouterSessionState("topic_011", "domain_011", created_s=0.0),
            None,
            [],
            now_s=60.0,
            router_text="Accumulated generated target context.",
            router_text_source="generated_target",
            context_similarity_scores=scores,
        )

        selection = decision.evidence["slice_selection"]
        self.assertEqual(selection["selected_slice_count"], 10)
        self.assertEqual(selection["selected_term_count"], 100_000)
        self.assertEqual(selection["selected_slice_presets"][0], "topic_000")

    def test_manifest_descriptions_define_context_similarity_prototypes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(root, count=2)
            config = OmniConfig(
                mock=True,
                rag_enabled=False,
                auto_glossary_presets="topic_000,topic_001",
                tmp_dir=str(root / "runtime"),
            )
            agent = OmniAgent(config=config, manifest=manifest)

            prototypes = agent._context_similarity_prototypes_for(
                agent._catalog("English -> Chinese")
            )
            scorer = DomainDescriptionSimilarity(prototypes=prototypes)

            self.assertEqual(
                scorer.prototypes["domain_000"],
                ("Manifest description for topic 0.",),
            )
            self.assertEqual(
                scorer.prototypes["domain_001"],
                ("Manifest description for topic 1.",),
            )

    def test_agent_activates_multiple_manifest_slices_under_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(root, count=12)
            config = OmniConfig(
                mock=True,
                rag_enabled=False,
                auto_glossary_presets=",".join(
                    f"topic_{index:03d}" for index in range(12)
                ),
                router_slice_selection_mode="budgeted_top_slices",
                router_term_budget=100_000,
                router_max_active_slices=0,
                tmp_dir=str(root / "runtime"),
            )
            agent = OmniAgent(config=config, manifest=manifest)
            router = agent._topic_router_for("English -> Chinese")
            scores = {
                f"domain_{index:03d}": 1.0 - index * 0.01
                for index in range(12)
            }
            decision = router.observe(
                RouterSessionState("topic_011", "domain_011", created_s=0.0),
                None,
                [],
                now_s=60.0,
                router_text="Accumulated generated target context.",
                router_text_source="generated_target",
                context_similarity_scores=scores,
            )
            session = SimpleNamespace(
                language_pair="English -> Chinese",
                active_retrieval_slices=[],
                active_slice_presets=[],
                active_slice_terms={},
                last_retrieval_plan=[],
            )

            activated = agent._activate_budgeted_retrieval_slices(session, decision)

            self.assertTrue(activated)
            self.assertEqual(
                session.active_slice_presets,
                [f"topic_{index:03d}" for index in range(10)],
            )
            self.assertEqual(sum(session.active_slice_terms.values()), 100_000)
            self.assertEqual(
                {item.domain for item in session.active_retrieval_slices},
                {f"domain_{index:03d}" for index in range(10)},
            )

    def test_multiple_slice_candidates_keep_one_global_prompt_top_k(self) -> None:
        agent = OmniAgent()
        agent.config.prompt_top_k = 10
        session = SimpleNamespace(
            auto_glossary_enabled=True,
            glossary_preset="auto_working",
            active_glossary_preset="topic_000",
            active_domain="general",
            recent_references=deque(maxlen=16),
        )
        references = [
            {
                "term": f"specialized term {index}",
                "translation": f"translation {index}",
                "score": 1.0 - index * 0.01,
                "source_preset": f"topic_{index % 2:03d}",
                "source_domain": f"domain_{index % 2:03d}",
                "source_slice_role": "domain",
            }
            for index in range(20)
        ]

        prompt = agent._prompt_references(session, references)

        self.assertEqual(len(prompt), 10)
        self.assertEqual({item["source_preset"] for item in prompt}, {"topic_000", "topic_001"})

    def _manifest(self, root: Path, *, count: int) -> TermMemoryManifest:
        scales = {}
        preset_meta = {}
        for index in range(count):
            preset = f"topic_{index:03d}"
            index_path = root / "indexes" / preset / "maxsim.pt"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_bytes(b"stub")
            scales[preset] = {
                "en-zh": {
                    "terms_path": f"terms/{preset}.jsonl",
                    "indexes": {"maxsim": str(index_path)},
                    "num_terms": 10_000,
                }
            }
            preset_meta[preset] = {
                "domain_id": f"domain_{index:03d}",
                "domain_description": f"Manifest description for topic {index}.",
                "term_count": 10_000,
            }
        return TermMemoryManifest.from_dict(
            {
                "snapshot_id": "budget-router-test",
                "scales": scales,
                "preset_meta": preset_meta,
            },
            base_dir=root,
            path=str(root / "manifest.json"),
        )


if __name__ == "__main__":
    unittest.main()
