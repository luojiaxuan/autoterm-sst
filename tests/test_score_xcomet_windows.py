import unittest

from eval.streaming_sst.score_xcomet_windows import (
    ReferenceSegment,
    exact_sign_flip_pvalue,
    group_reference_segments,
    hypothesis_for_window,
    metric_summary,
)


class ScoreXcometWindowsTest(unittest.TestCase):
    def test_groups_complete_reference_segments(self) -> None:
        segments = [
            ReferenceSegment(0.0, 12.0, "source one", "参考一"),
            ReferenceSegment(12.0, 24.0, "source two", "参考二"),
            ReferenceSegment(24.0, 37.0, "source three", "参考三"),
            ReferenceSegment(40.0, 55.0, "source four", "参考四"),
        ]
        windows = group_reference_segments(segments, block_duration_s=60.0, target_window_s=30.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["local_start_s"], 0.0)
        self.assertEqual(windows[0]["local_end_s"], 40.0)
        self.assertEqual(windows[0]["reference_segment_count"], 3)
        self.assertEqual(windows[1]["local_end_s"], 60.0)
        self.assertEqual(windows[0]["reference"], "参考一参考二参考三")

    def test_groups_latin_references_with_spaces(self) -> None:
        segments = [
            ReferenceSegment(0.0, 4.0, "source one", "erste Referenz"),
            ReferenceSegment(4.0, 8.0, "source two", "zweite Referenz"),
        ]

        windows = group_reference_segments(
            segments,
            block_duration_s=8.0,
            target_window_s=8.0,
            reference_separator=" ",
        )

        self.assertEqual(windows[0]["reference"], "erste Referenz zweite Referenz")

    def test_assigns_hypothesis_by_local_cursor(self) -> None:
        records = [
            {"cursor_samples": 16000, "text": "甲"},
            {"cursor_samples": 32000, "text": "乙"},
            {"cursor_samples": 48000, "text": "丙"},
        ]
        text, count = hypothesis_for_window(
            records,
            block_start_sample=0,
            local_start_s=1.0,
            local_end_s=3.0,
        )
        self.assertEqual(text, "乙丙")
        self.assertEqual(count, 2)

    def test_metric_summary_is_paired(self) -> None:
        rows = [
            {"block_index": 1, "reference": "甲", "auto_test": 0.8, "merged_test": 0.7},
            {"block_index": 1, "reference": "乙", "auto_test": 0.6, "merged_test": 0.7},
            {"block_index": 2, "reference": "丙", "auto_test": 0.9, "merged_test": 0.7},
        ]
        summary = metric_summary(rows, "test")
        self.assertAlmostEqual(summary["auto_mean"], 0.7666666667)
        self.assertAlmostEqual(summary["merged_mean"], 0.7)
        self.assertAlmostEqual(summary["delta_talk_macro"], 0.1)
        self.assertEqual(summary["win_tie_loss"], {"auto": 2, "tie": 0, "merged": 1})

    def test_exact_sign_flip_pvalue(self) -> None:
        self.assertEqual(exact_sign_flip_pvalue([1.0]), 1.0)
        self.assertIsNone(exact_sign_flip_pvalue([]))


if __name__ == "__main__":
    unittest.main()
