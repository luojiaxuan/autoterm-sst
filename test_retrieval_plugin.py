from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from framework.agents.omni import OmniAgent, OmniConfig
from framework.agents.plugins import retrieval as retrieval_module
from framework.agents.plugins.retrieval import (
    IndexRetrievalSpec,
    MaxSimRetrievalPlugin,
    MultiIndexRetrievalResult,
    RetrievalPlugin,
    RetrievalResult,
)
from framework.agents.term_memory.slice_registry import RetrievalSlice


class DummyRetriever:
    score_threshold = None
    device = torch.device("cpu")


class MaxSimRetrievalPluginTests(unittest.IsolatedAsyncioTestCase):
    def test_add_rasst_paths_extends_cached_agents_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            repo_agents = repo_root / "agents"
            rasst_code_root = root / "rasst" / "code" / "rasst"
            eval_root = rasst_code_root / "eval"
            eval_agents = eval_root / "agents"
            repo_agents.mkdir(parents=True)
            eval_agents.mkdir(parents=True)
            (eval_agents / "__init__.py").write_text("", encoding="utf-8")

            old_code_root = retrieval_module.RASST_CODE_ROOT
            old_eval_root = retrieval_module.RASST_EVAL_ROOT
            old_sys_path = list(sys.path)
            old_agents = sys.modules.pop("agents", None)
            try:
                sys.path.insert(0, str(repo_root))
                import agents  # noqa: WPS433

                self.assertIn(str(repo_agents), list(agents.__path__))

                retrieval_module.RASST_CODE_ROOT = rasst_code_root
                retrieval_module.RASST_EVAL_ROOT = eval_root
                retrieval_module.add_rasst_paths()

                self.assertIn(str(eval_agents), list(agents.__path__))
            finally:
                retrieval_module.RASST_CODE_ROOT = old_code_root
                retrieval_module.RASST_EVAL_ROOT = old_eval_root
                sys.path[:] = old_sys_path
                sys.modules.pop("agents", None)
                if old_agents is not None:
                    sys.modules["agents"] = old_agents

    async def test_retrieve_with_metadata_restores_none_score_threshold(self) -> None:
        plugin = MaxSimRetrievalPlugin(model_path="dummy", index_path="dummy")
        plugin.retriever = DummyRetriever()
        observed = []

        def fake_retrieve(requests, top_k, lookback_sec):  # noqa: ANN001, ANN202
            del requests, top_k, lookback_sec
            observed.append(plugin.retriever.score_threshold)
            return [RetrievalResult(references=[])]

        plugin._retrieve_with_query_embeddings_sync = fake_retrieve  # type: ignore[method-assign]

        result = await plugin.retrieve_with_metadata(
            [{"audio_buffer": [0.0], "current_start_sec": 0.0, "current_end_sec": 1.0}],
            top_k=1,
            lookback_sec=0.0,
            score_threshold=0.5,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(observed, [0.5])
        self.assertIsNone(plugin.retriever.score_threshold)

    async def test_probe_domain_scores_scores_candidates_without_activating_indexes(self) -> None:
        plugin = MaxSimRetrievalPlugin(model_path="dummy", index_path="old-index")
        plugin.retriever = DummyRetriever()
        plugin._active_index_path = "old-index"
        activated = []

        def fake_activate(index_path):  # noqa: ANN001, ANN202
            activated.append(index_path)
            plugin._active_index_path = index_path

        def fake_encode(request, lookback_sec):  # noqa: ANN001, ANN202
            del request, lookback_sec
            raise AssertionError("probe should reuse query_embedding instead of re-encoding audio")

        def fake_ensure(index_path):  # noqa: ANN001, ANN202
            if index_path == "nlp-index":
                return {
                    "text_embs": torch.tensor([[0.0, 1.0], [0.4, 0.4]], dtype=torch.float32),
                    "term_list": ["nlp-window2", "weak-nlp"],
                }
            return {
                "text_embs": torch.tensor([[0.5, 0.5], [0.2, 0.2]], dtype=torch.float32),
                "term_list": ["medicine-term", "weak-medicine"],
            }

        plugin._activate_sync = fake_activate  # type: ignore[method-assign]
        plugin._encode_probe_window_sync = fake_encode  # type: ignore[method-assign]
        plugin._ensure_index = fake_ensure  # type: ignore[method-assign]

        scores = await plugin.probe_domain_scores(
            {
                "audio_buffer": [0.0],
                "current_start_sec": 0.0,
                "current_end_sec": 1.0,
                "query_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            },
            candidate_slices=[
                {"domain": "nlp", "preset_id": "nlp_core_10k", "index_path": "nlp-index"},
                {"domain": "medicine", "preset_id": "medicine_core_10k", "index_path": "medicine-index"},
            ],
            top_k=1,
            lookback_sec=0.0,
            score_threshold=0.5,
        )

        self.assertEqual(set(scores), {"nlp", "medicine"})
        self.assertEqual(scores["nlp"].top_terms[0], "nlp-window2")
        self.assertGreater(scores["nlp"].top_score, scores["medicine"].top_score)
        self.assertEqual(activated, [])
        self.assertEqual(plugin._active_index_path, "old-index")
        self.assertIsNone(plugin.retriever.score_threshold)

    async def test_multi_index_retrieval_encodes_once_for_ten_slices(self) -> None:
        encode_calls = []
        fake_module = ModuleType("agents.streaming_maxsim_retriever")
        fake_module.EXPECTED_SAMPLE_RATE = 4

        def encode(chunks, model, feat_ext, device):  # noqa: ANN001, ANN202
            del model, feat_ext, device
            encode_calls.append(len(chunks))
            projected = torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]] for _ in chunks],
                dtype=torch.float32,
            )
            mask = torch.ones((len(chunks), 2), dtype=torch.bool)
            return projected, mask

        def window_ranges(windows, stride, valid_frames):  # noqa: ANN001, ANN202
            del windows, stride
            self.assertEqual(valid_frames, 2)
            return (
                torch.tensor([0.0, 0.5], dtype=torch.float32),
                torch.tensor([0.5, 1.0], dtype=torch.float32),
            )

        fake_module._encode_audio_projected_seq_batch = encode
        fake_module._build_window_time_ranges = window_ranges

        class Pooler:
            maxsim_windows = (1,)
            maxsim_stride = 1

            def _multiscale_pool(self, sequence, mask):  # noqa: ANN001, ANN202
                del mask
                return sequence

        plugin = MaxSimRetrievalPlugin(
            model_path="dummy",
            index_path="slice-0",
            device="cpu",
            score_threshold=0.0,
        )
        plugin.retriever = SimpleNamespace(
            retriever=Pooler(),
            feat_ext=None,
            device=torch.device("cpu"),
            score_threshold=0.0,
        )
        specs = []
        for index in range(10):
            path = f"slice-{index}"
            plugin._remember_index(
                path,
                {
                    "text_embs": torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0]],
                        dtype=torch.float32,
                    ),
                    "term_list": [
                        {
                            "key": f"key-{index}-a",
                            "term": f"term-{index}-a",
                            "target_translations": {"zh": f"译文-{index}-a"},
                        },
                        {
                            "key": f"key-{index}-b",
                            "term": f"term-{index}-b",
                            "target_translations": {"zh": f"译文-{index}-b"},
                        },
                    ],
                },
            )
            specs.append(
                IndexRetrievalSpec(
                    key=f"topic-{index}",
                    index_path=path,
                    top_k=1,
                    score_threshold=1.1 if index == 0 else 0.0,
                )
            )

        with patch.dict(sys.modules, {"agents.streaming_maxsim_retriever": fake_module}):
            results = await plugin.retrieve_across_indexes_with_metadata(
                [
                    {
                        "audio_buffer": [0.0, 0.0, 0.0, 0.0],
                        "current_start_sec": 0.0,
                        "current_end_sec": 1.0,
                        "return_query_window_embeddings": True,
                    }
                ],
                index_specs_by_request=[specs],
                lookback_sec=0.0,
            )

        self.assertEqual(encode_calls, [1])
        self.assertEqual(set(results[0].references_by_key), {f"topic-{i}" for i in range(10)})
        self.assertEqual(results[0].references_by_key["topic-0"], [])
        self.assertTrue(
            all(
                len(results[0].references_by_key[f"topic-{index}"]) == 1
                for index in range(1, 10)
            )
        )
        self.assertEqual(tuple(results[0].query_window_embeddings.shape), (2, 2))

        plugin._active_index_path = "slice-1"
        with patch.dict(sys.modules, {"agents.streaming_maxsim_retriever": fake_module}):
            single_index = await plugin.retrieve_with_metadata(
                [
                    {
                        "audio_buffer": [0.0, 0.0, 0.0, 0.0],
                        "current_start_sec": 0.0,
                        "current_end_sec": 1.0,
                    }
                ],
                top_k=1,
                lookback_sec=0.0,
                score_threshold=0.0,
            )

        self.assertEqual(encode_calls, [1, 1])
        self.assertEqual(len(single_index[0].references), 1)
        self.assertEqual(single_index[0].references[0]["term"], "term-1-a")

    def test_index_cache_is_lru_bounded(self) -> None:
        plugin = MaxSimRetrievalPlugin(
            model_path="dummy",
            index_path="slice-0",
            max_cached_indexes=2,
        )
        plugin._remember_index("slice-0", {"text_embs": 0, "term_list": []})
        plugin._remember_index("slice-1", {"text_embs": 1, "term_list": []})
        plugin._remember_index("slice-2", {"text_embs": 2, "term_list": []})

        self.assertEqual(list(plugin._text_index_cache), ["slice-1", "slice-2"])
        self.assertEqual(plugin.status["cached_indexes"], 2)

    async def test_omni_globally_reranks_multi_slice_hits_with_provenance(self) -> None:
        class FakeMultiIndexRetrieval(RetrievalPlugin):
            enabled = True

            def __init__(self) -> None:
                self.calls = 0

            async def retrieve_across_indexes_with_metadata(
                self,
                requests,  # noqa: ANN001
                *,
                index_specs_by_request,  # noqa: ANN001
                lookback_sec,  # noqa: ANN001
            ):
                del requests, lookback_sec
                self.calls += 1
                outputs = []
                for specs in index_specs_by_request:
                    references = {}
                    for spec in specs:
                        index = int(spec.index_path.rsplit("-", 1)[-1])
                        references[spec.key] = [
                            {
                                "term": f"term-{index}",
                                "translation": f"translation-{index}",
                                "score": 0.80 + index * 0.01,
                            }
                        ]
                    outputs.append(
                        MultiIndexRetrievalResult(
                            references_by_key=references,
                            query_embedding=[1.0, 0.0],
                        )
                    )
                return outputs

        agent = OmniAgent(
            config=OmniConfig(mock=True, rag_enabled=False, prompt_top_k=3)
        )
        agent.retrieval = FakeMultiIndexRetrieval()
        session = SimpleNamespace(
            session_id="session-1",
            audio=[0.0] * 16_000,
            last_llm_samples=0,
            auto_glossary_enabled=True,
            glossary_preset="auto_working",
            active_glossary_preset="topic-0",
            active_domain="general",
            last_prompt_reference_count=0,
        )
        grouped = {}
        for index in range(10):
            plan = RetrievalSlice(
                preset_id=f"topic-{index}",
                slice_id=f"slice-{index}",
                role="domain",
                domain=f"domain-{index}",
                index_path=f"index-{index}",
                term_count=10_000,
            )
            grouped[(plan.index_path, 10, 0.78)] = [(0, session, plan)]
        outputs = [[]]
        query_embeddings = [None]
        query_window_embeddings = [None]

        await agent._retrieve_slice_groups(
            grouped,
            {"session-1": 16_000},
            outputs,
            query_embeddings,
            query_window_embeddings,
        )
        prompt = agent._prompt_references(session, outputs[0])

        self.assertEqual(agent.retrieval.calls, 1)
        self.assertEqual([item["term"] for item in prompt], ["term-9", "term-8", "term-7"])
        self.assertEqual(
            [item["source_preset"] for item in prompt],
            ["topic-9", "topic-8", "topic-7"],
        )
        self.assertEqual(prompt[0]["source_slice"], "slice-9")
        self.assertEqual(prompt[0]["source_domain"], "domain-9")
        self.assertEqual(prompt[0]["candidate_inventory_terms"], 10_000)


if __name__ == "__main__":
    unittest.main()
