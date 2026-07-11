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
    deduplicate_alias_occurrences,
    raw_annotation_count,
    score_run,
)


class TimeAlignedTermScorerTests(unittest.TestCase):
    def test_exact_span_same_target_aliases_are_one_headline_occurrence(self) -> None:
        annotations = [
            TimedOccurrence("nlp", 1, "model", ["MODEL"], 1.0, 1.4),
            TimedOccurrence("nlp", 1, "ｍｏｄｅｌｓ", [" ＭＯＤＥＬ "], 1.0, 1.4),
        ]

        deduplicated = deduplicate_alias_occurrences(annotations)

        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0].term, "model")
        self.assertEqual(tuple(deduplicated[0].source_aliases), ("models",))
        self.assertEqual(deduplicated[0].raw_annotation_rows, 2)
        self.assertEqual(raw_annotation_count(deduplicated), 2)

    def test_same_span_different_target_variants_remain_distinct(self) -> None:
        annotations = [
            TimedOccurrence("nlp", 1, "model", ["模型"], 1.0, 1.4),
            TimedOccurrence("nlp", 1, "models", ["模特"], 1.0, 1.4),
        ]

        deduplicated = deduplicate_alias_occurrences(annotations)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(raw_annotation_count(deduplicated), 2)

    def test_alias_dedup_requires_exact_audio_span(self) -> None:
        annotations = [
            TimedOccurrence("nlp", 1, "model", ["模型"], 1.0, 1.4),
            TimedOccurrence("nlp", 1, "models", ["模型"], 1.0, 1.400001),
        ]

        self.assertEqual(len(deduplicate_alias_occurrences(annotations)), 2)

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

    def test_score_run_defensively_deduplicates_raw_source_aliases(self) -> None:
        payload = {
            "block_spans": [
                {
                    "block_index": 1,
                    "start_sample": 0,
                    "end_sample": 32000,
                }
            ],
            "records": [{"cursor_samples": 16000, "text": "模型"}],
        }
        raw_annotations = [
            TimedOccurrence("nlp", 1, "model", ["模型"], 0.25, 0.5),
            TimedOccurrence("nlp", 1, "models", ["模型"], 0.25, 0.5),
        ]

        metrics = score_run(payload, raw_annotations, "zh")

        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["gold_occurrences"], 1)
        self.assertEqual(metrics["raw_annotation_rows"], 2)
        self.assertEqual(metrics["term_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
