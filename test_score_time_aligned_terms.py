from __future__ import annotations

import unittest

from eval.streaming_sst.score_time_aligned_terms import TimedOccurrence, score_run


class TimeAlignedTermScorerTests(unittest.TestCase):
    def test_one_output_mention_cannot_satisfy_two_timed_occurrences(self) -> None:
        payload = {
            "block_spans": [
                {
                    "block_index": 1,
                    "start_sample": 0,
                    "end_sample": 48000,
                }
            ],
            "records": [
                {
                    "cursor_samples": 16000,
                    "text": "直肠癌",
                }
            ],
        }
        gold = [
            TimedOccurrence("medicine", 1, "rectal cancer", ["直肠癌"], 0.25, 0.5),
            TimedOccurrence("medicine", 1, "rectal cancer", ["直肠癌"], 1.25, 1.5),
        ]

        metrics = score_run(payload, gold, "zh")

        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["gold_occurrences"], 2)
        self.assertEqual(metrics["term_acc"], 0.5)
        self.assertEqual(metrics["by_domain"]["medicine"]["hits"], 1)


if __name__ == "__main__":
    unittest.main()
