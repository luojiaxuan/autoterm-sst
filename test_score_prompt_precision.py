from __future__ import annotations

import argparse
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from eval.streaming_sst.score_prompt_precision import (
    build_report,
    normalise_source_term,
    score_payload,
    timing_signature,
    timing_signature_sha256,
    validate_same_playlist,
)
from eval.streaming_sst.score_time_aligned_terms import TimedOccurrence


def occurrence(term: str, start: float, end: float) -> TimedOccurrence:
    return TimedOccurrence(
        domain="nlp",
        block_index=1,
        term=term,
        variants=["unused"],
        t_start=start,
        t_end=end,
    )


def payload(records: list[dict], *, item_id: str = "talk-1") -> dict:
    return {
        "config": {"preset": "capacity_10k"},
        "blocks": [
            {
                "item_id": item_id,
                "corpus": "acl",
                "expected_domain": "nlp",
            }
        ],
        "block_spans": [
            {
                "block_index": 1,
                "item_id": item_id,
                "start_sample": 0,
                "end_sample": 64000,
                "sample_count": 64000,
            }
        ],
        "records": records,
    }


class PromptPrecisionTests(unittest.TestCase):
    def test_normalises_reference_and_gold_terms_consistently(self) -> None:
        self.assertEqual(normalise_source_term("  DATA-set  "), "data set")
        self.assertEqual(normalise_source_term("ＮＬＰ task"), "nlp task")

    def test_scores_technical_raw_precision_and_refs_per_chunk(self) -> None:
        run = payload(
            [
                {
                    "start_sample": 0,
                    "cursor_samples": 32000,
                    "prompt_reference_count": 2,
                    "references": [{"key": "NLP"}, {"term": "distractor"}],
                },
                {
                    "start_sample": 32000,
                    "cursor_samples": 64000,
                    "prompt_reference_count": 2,
                    "references": [{"term_key": "data-set"}, {"translation": "missing source"}],
                },
            ]
        )
        gold = {
            "technical_plus_medicine": [occurrence("NLP", 0.5, 0.8)],
            "raw_plus_medicine": [
                occurrence("NLP", 0.5, 0.8),
                occurrence("data set", 2.3, 2.8),
            ],
        }

        result = score_payload(run, gold, lookback_s=0.0, tolerance_s=0.0)

        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["prompt_reference_count"], 4)
        self.assertEqual(result["retrieved_references_per_chunk"], 2.0)
        self.assertEqual(result["empty_source_reference_count"], 1)
        self.assertEqual(result["technical_plus_medicine"]["relevant_references"], 1)
        self.assertEqual(result["technical_plus_medicine"]["prompt_precision"], 0.25)
        self.assertEqual(result["raw_plus_medicine"]["relevant_references"], 2)
        self.assertEqual(result["raw_plus_medicine"]["prompt_precision"], 0.5)

    def test_lookback_includes_recent_occurrence(self) -> None:
        run = payload(
            [
                {
                    "start_sample": 32000,
                    "cursor_samples": 64000,
                    "prompt_reference_count": 1,
                    "references": [{"term": "NLP"}],
                }
            ]
        )
        gold = {"technical_plus_medicine": [occurrence("NLP", 0.7, 0.9)]}

        without_lookback = score_payload(run, gold, lookback_s=0.0, tolerance_s=0.0)
        with_lookback = score_payload(run, gold, lookback_s=1.2, tolerance_s=0.0)

        self.assertEqual(without_lookback["technical_plus_medicine"]["prompt_precision"], 0.0)
        self.assertEqual(with_lookback["technical_plus_medicine"]["prompt_precision"], 1.0)

    def test_reference_count_mismatch_is_strict_by_default(self) -> None:
        run = payload(
            [
                {
                    "start_sample": 0,
                    "cursor_samples": 32000,
                    "prompt_reference_count": 2,
                    "references": [{"term": "NLP"}],
                }
            ]
        )
        gold = {"technical_plus_medicine": [occurrence("NLP", 0.5, 0.8)]}

        with self.assertRaisesRegex(ValueError, "captured 1 references"):
            score_payload(run, gold, lookback_s=0.0, tolerance_s=0.0)

        allowed = score_payload(
            run,
            gold,
            lookback_s=0.0,
            tolerance_s=0.0,
            require_complete_reference_capture=False,
        )
        self.assertEqual(allowed["reference_count_mismatch_chunks"], 1)
        self.assertEqual(allowed["reference_capture_ratio"], 0.5)
        self.assertEqual(allowed["retrieved_references_per_chunk"], 2.0)
        self.assertEqual(allowed["technical_plus_medicine"]["prompt_precision"], 1.0)

    def test_rejects_different_playlists(self) -> None:
        first = payload([], item_id="talk-1")
        second = payload([], item_id="talk-2")

        with self.assertRaisesRegex(ValueError, "does not share"):
            validate_same_playlist([first, second])

    def test_timing_signature_detects_different_chunk_coalescing(self) -> None:
        first = payload(
            [{"start_sample": 0, "cursor_samples": 32000}]
        )
        second = payload(
            [{"start_sample": 0, "cursor_samples": 64000}]
        )

        self.assertNotEqual(timing_signature(first), timing_signature(second))
        self.assertNotEqual(timing_signature_sha256(first), timing_signature_sha256(second))

    def test_build_report_uses_medicine_timed_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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
            (medicine / "hard_medicine.oracle_term_map__medicine_1.json").write_text(
                json.dumps(
                    [
                        {
                            "start_sec": 0.5,
                            "end_sec": 0.8,
                            "references": [{"term": "rectal cancer", "translation": "直肠癌"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            run_payload = payload(
                [
                    {
                        "start_sample": 0,
                        "cursor_samples": 32000,
                        "prompt_reference_count": 1,
                        "references": [{"term": "rectal cancer", "translation": "直肠癌"}],
                    }
                ],
                item_id="medicine_1",
            )
            run_payload["blocks"][0]["corpus"] = "medicine"
            run_payload["blocks"][0]["expected_domain"] = "medicine"
            run_path = root / "run.json"
            run_path.write_text(json.dumps(run_payload), encoding="utf-8")
            args = argparse.Namespace(
                run=[f"medicine={run_path}"],
                mfa_root=str(root / "mfa"),
                acl_root=str(acl_root),
                acl_technical_gold=str(technical),
                acl_raw_glossary=str(raw),
                medicine_oracle_dir=str(medicine),
                target_lang="zh",
                lookback_s=1.92,
                alignment_tolerance_s=0.0,
                allow_reference_count_mismatch=False,
            )

            # note (luojiaxuan): Existing shared gold loaders use json.load(open(...));
            # keep their unrelated ResourceWarning out of this scorer's integration test.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                report = build_report(args)

            row = report["runs"]["medicine"]
            self.assertEqual(row["technical_plus_medicine"]["gold_occurrences"], 1)
            self.assertEqual(row["technical_plus_medicine"]["prompt_precision"], 1.0)
            self.assertEqual(row["raw_plus_medicine"]["prompt_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
