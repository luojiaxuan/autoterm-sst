from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from eval.streaming_sst.score_time_aligned_terms import (
    TimedOccurrence,
    build_timed_gold,
    score_run,
)


class TimeAlignedTermScorerTests(unittest.TestCase):
    def test_selected_acl_window_filters_and_rebases_played_times(self) -> None:
        payload = {
            "blocks": [
                {
                    "item_id": "acl-a__window_160000_320000",
                    "original_item_id": "acl-a",
                    "corpus": "acl",
                    "source_offset_samples": 160000,
                    "source_end_samples": 320000,
                }
            ],
            "block_spans": [
                {
                    "block_index": 1,
                    "start_sample": 80000,
                    "sample_count": 160000,
                }
            ],
        }
        args = SimpleNamespace(
            acl_technical_gold="technical.json",
            acl_raw_glossary="raw.json",
            target_lang="zh",
            mfa_root="mfa",
            acl_root="acl",
            medicine_oracle_dir="medicine",
        )
        with (
            patch(
                "eval.streaming_sst.score_time_aligned_terms.load_gold_entries",
                return_value=[("term", ["译文"])],
            ),
            patch(
                "eval.streaming_sst.score_time_aligned_terms.load_acl_segment_map",
                return_value={"acl-a": [(0.0, 30.0, 0.0)]},
            ),
            patch(
                "eval.streaming_sst.score_time_aligned_terms.parse_textgrid_words",
                return_value=[],
            ),
            patch(
                "eval.streaming_sst.score_time_aligned_terms.find_term_occurrences",
                return_value=[(1.0, 2.0), (12.0, 13.0), (19.0, 21.0), (22.0, 23.0)],
            ),
        ):
            gold = build_timed_gold(payload, args)

        raw_gold = gold["raw_plus_medicine"]
        self.assertEqual([occ.t_start for occ in raw_gold], [7.0, 14.0])
        self.assertEqual([occ.t_end for occ in raw_gold], [8.0, 15.0])

    def test_selected_medicine_window_filters_and_rebases_gold_times(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            acl_root.mkdir()
            (acl_root / "segments.meta.jsonl").write_text("", encoding="utf-8")
            technical = root / "technical.json"
            raw = root / "raw.json"
            technical.write_text("[]", encoding="utf-8")
            raw.write_text("[]", encoding="utf-8")
            medicine = root / "medicine"
            medicine.mkdir()
            rows = [
                {
                    "start_sec": timestamp,
                    "end_sec": timestamp + 0.5,
                    "references": [
                        {"term": f"term-{timestamp}", "translation": f"译文{timestamp}"}
                    ],
                }
                for timestamp in (9.0, 12.0, 19.0, 20.0)
            ]
            (medicine / "hard_medicine.oracle_term_map__medicine_1.json").write_text(
                json.dumps(rows),
                encoding="utf-8",
            )
            payload = {
                "blocks": [
                    {
                        "item_id": "medicine_1__window_160000_320000",
                        "original_item_id": "medicine_1",
                        "corpus": "medicine",
                        "source_offset_samples": 160000,
                        "source_end_samples": 320000,
                    }
                ],
                "block_spans": [
                    {
                        "block_index": 1,
                        "start_sample": 80000,
                        "sample_count": 160000,
                    }
                ],
            }
            args = SimpleNamespace(
                acl_technical_gold=str(technical),
                acl_raw_glossary=str(raw),
                target_lang="zh",
                mfa_root=str(root / "mfa"),
                acl_root=str(acl_root),
                medicine_oracle_dir=str(medicine),
            )

            gold = build_timed_gold(payload, args)

        raw_gold = gold["raw_plus_medicine"]
        self.assertEqual([occ.term for occ in raw_gold], ["term-12.0", "term-19.0"])
        self.assertEqual([occ.t_start for occ in raw_gold], [7.0, 14.0])
        self.assertEqual(len(gold["technical_plus_medicine"]), 2)

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
