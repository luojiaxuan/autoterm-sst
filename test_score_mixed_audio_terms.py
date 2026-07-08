from __future__ import annotations

import unittest

from eval.streaming_sst.score_mixed_audio_terms import GoldOccurrence, score_occurrences


class MixedAudioTermScorerTests(unittest.TestCase):
    def test_reports_occurrence_and_unique_type_accuracy(self) -> None:
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
            GoldOccurrence("medicine", "medicine_606", 1, "rectal cancer", ["直肠癌"], "medicine_oracle"),
            GoldOccurrence("medicine", "medicine_606", 1, "rectal cancer", ["直肠癌"], "medicine_oracle"),
            GoldOccurrence("medicine", "medicine_606", 1, "multi-modal approach", ["多模式治疗方法"], "medicine_oracle"),
        ]

        metrics = score_occurrences(payload, gold)
        medicine = metrics["by_domain"]["medicine"]

        self.assertEqual(metrics["hits"], 2)
        self.assertEqual(metrics["gold_occurrences"], 3)
        self.assertEqual(metrics["unique_term_types"], 2)
        self.assertEqual(metrics["type_hits_any"], 1)
        self.assertEqual(metrics["type_acc_any"], 0.5)
        self.assertEqual(medicine["unique_term_types"], 2)
        self.assertEqual(medicine["type_hits_any"], 1)


if __name__ == "__main__":
    unittest.main()
