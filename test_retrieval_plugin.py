from __future__ import annotations

import unittest

from framework.agents.plugins.retrieval import MaxSimRetrievalPlugin, RetrievalResult


class DummyRetriever:
    score_threshold = None


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

    async def test_probe_domain_scores_restores_active_index_and_threshold(self) -> None:
        plugin = MaxSimRetrievalPlugin(model_path="dummy", index_path="old-index")
        plugin.retriever = DummyRetriever()
        plugin._active_index_path = "old-index"
        activated = []
        observed = []

        def fake_activate(index_path):  # noqa: ANN001, ANN202
            activated.append(index_path)
            plugin._active_index_path = index_path

        def fake_retrieve(requests, top_k, lookback_sec):  # noqa: ANN001, ANN202
            del requests, top_k, lookback_sec
            observed.append((plugin._active_index_path, plugin.retriever.score_threshold))
            return [
                RetrievalResult(
                    references=[
                        {
                            "term": f"term:{plugin._active_index_path}",
                            "score": 0.7,
                        }
                    ]
                )
            ]

        plugin._activate_sync = fake_activate  # type: ignore[method-assign]
        plugin._retrieve_with_query_embeddings_sync = fake_retrieve  # type: ignore[method-assign]

        scores = await plugin.probe_domain_scores(
            {"audio_buffer": [0.0], "current_start_sec": 0.0, "current_end_sec": 1.0},
            candidate_slices=[
                {"domain": "nlp", "preset_id": "nlp_core_10k", "index_path": "nlp-index"},
                {"domain": "medicine", "preset_id": "medicine_core_10k", "index_path": "medicine-index"},
            ],
            top_k=1,
            lookback_sec=0.0,
            score_threshold=0.5,
        )

        self.assertEqual(set(scores), {"nlp", "medicine"})
        self.assertEqual(observed, [("nlp-index", 0.5), ("medicine-index", 0.5)])
        self.assertEqual(activated, ["nlp-index", "medicine-index", "old-index"])
        self.assertEqual(plugin._active_index_path, "old-index")
        self.assertIsNone(plugin.retriever.score_threshold)


if __name__ == "__main__":
    unittest.main()
