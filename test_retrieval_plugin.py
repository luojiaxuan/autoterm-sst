from __future__ import annotations

import unittest

import torch

from framework.agents.plugins.retrieval import MaxSimRetrievalPlugin, RetrievalResult


class DummyRetriever:
    score_threshold = None
    device = torch.device("cpu")


class MaxSimRetrievalPluginTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
