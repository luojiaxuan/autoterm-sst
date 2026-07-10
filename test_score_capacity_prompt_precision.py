from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from eval.streaming_sst.score_capacity_prompt_precision import evaluate_run


def _write_gold(path: Path, terms: list[str]) -> None:
    path.write_text(
        json.dumps([{"en": term, "zh": [f"译-{term}"]} for term in terms]),
        encoding="utf-8",
    )


def _args(
    root: Path, *, no_alignment: bool = False, lookback_s: float = 0.0
) -> SimpleNamespace:
    return SimpleNamespace(
        acl_technical_gold=str(root / "technical.json"),
        acl_raw_glossary=str(root / "raw.json"),
        mfa_root=str(root / "mfa"),
        acl_root=str(root / "acl"),
        medicine_oracle_dir=str(root / "medicine"),
        target_lang="zh",
        retrieval_lookback_s=lookback_s,
        no_source_time_alignment=no_alignment,
    )


class CapacityPromptPrecisionTests(unittest.TestCase):
    def test_prompt_volume_uses_declared_prompt_prefix_and_reports_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_gold(root / "technical.json", ["algorithm"])
            _write_gold(root / "raw.json", ["algorithm", "transformer"])
            payload = {
                "records": [
                    {
                        "prompt_reference_count": 2,
                        "references": [
                            {"term": "Algorithm"},
                            {"term": "transformer"},
                            {"term": "ui only"},
                        ],
                    },
                    {"prompt_reference_count": 1, "references": []},
                    {"references": [{"term": "distractor"}]},
                ]
            }

            result = evaluate_run(payload, _args(root, no_alignment=True))

        volume = result["reference_volume"]
        self.assertEqual(volume["observed_prompt_reference_events"], 3)
        self.assertEqual(volume["retrieved_references_per_chunk"], 1.0)
        self.assertEqual(volume["declared_prompt_reference_events"], 3)
        self.assertEqual(volume["observed_declared_prompt_reference_events"], 2)
        self.assertEqual(volume["missing_declared_prompt_reference_events"], 1)
        self.assertEqual(volume["prompt_reference_observation_coverage"], 0.666667)
        self.assertEqual(volume["prompt_reference_events_inferred_without_count"], 1)
        by_type = result["gold_type_prompt_precision"]
        self.assertEqual(by_type["acl_technical_gold"]["gold_type_matches"], 1)
        self.assertEqual(
            by_type["acl_technical_gold"]["gold_type_prompt_precision"], 0.333333
        )
        self.assertEqual(by_type["acl_raw_gold"]["gold_type_matches"], 2)
        self.assertEqual(by_type["acl_raw_gold"]["gold_type_prompt_precision"], 0.666667)

    def test_source_time_precision_reuses_textgrid_playlist_alignment(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_gold(root / "technical.json", ["algorithm"])
            _write_gold(root / "raw.json", ["algorithm", "transformer"])
            acl_root = root / "acl"
            acl_root.mkdir()
            (acl_root / "segments.meta.jsonl").write_text(
                json.dumps(
                    {
                        "talk": "talk-1",
                        "index": 0,
                        "offset": 0.0,
                        "duration": 4.0,
                        "seg_duration": 4.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            textgrid = root / "mfa" / "acl6060" / "talk-1" / "talk-1.TextGrid"
            textgrid.parent.mkdir(parents=True)
            textgrid.write_text(
                "\n".join(
                    [
                        '"IntervalTier"',
                        '"words"',
                        "0",
                        "4",
                        "4",
                        "0",
                        "0.5",
                        '"intro"',
                        "0.5",
                        "0.8",
                        '"algorithm"',
                        "0.8",
                        "2.5",
                        '"middle"',
                        "2.5",
                        "2.9",
                        '"transformer"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = {
                "blocks": [{"corpus": "acl", "item_id": "talk-1"}],
                "block_spans": [
                    {
                        "block_index": 1,
                        "start_sample": 0,
                        "end_sample": 64000,
                        "sample_count": 64000,
                    }
                ],
                "records": [
                    {
                        "start_sample": 0,
                        "cursor_samples": 32000,
                        "prompt_reference_count": 2,
                        "references": [{"term": "algorithm"}, {"term": "transformer"}],
                    },
                    {
                        "start_sample": 32000,
                        "cursor_samples": 64000,
                        "prompt_reference_count": 2,
                        "references": [{"term": "transformer"}, {"term": "distractor"}],
                    },
                    {
                        "cursor_samples": 64000,
                        "prompt_reference_count": 1,
                        "references": [{"term": "algorithm"}],
                    },
                ],
            }

            result = evaluate_run(payload, _args(root))

        aligned = result["source_time_aligned_prompt_precision"]
        self.assertTrue(aligned["available"])
        technical = aligned["acl_technical_gold"]
        raw = aligned["acl_raw_gold"]
        self.assertEqual(technical["eligible_prompt_reference_events"], 4)
        self.assertEqual(technical["observed_prompt_reference_events"], 5)
        self.assertEqual(technical["source_time_alignment_coverage"], 0.8)
        self.assertEqual(
            technical["excluded_missing_timing_or_span_prompt_reference_events"], 1
        )
        self.assertEqual(technical["source_time_aligned_matches"], 1)
        self.assertEqual(technical["source_time_aligned_prompt_precision"], 0.25)
        self.assertEqual(raw["source_time_aligned_matches"], 2)
        self.assertEqual(raw["source_time_aligned_prompt_precision"], 0.5)

    def test_missing_alignment_assets_do_not_hide_gold_type_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_gold(root / "technical.json", ["algorithm"])
            _write_gold(root / "raw.json", ["algorithm"])
            payload = {
                "blocks": [{"corpus": "acl", "item_id": "missing-talk"}],
                "block_spans": [
                    {
                        "block_index": 1,
                        "start_sample": 0,
                        "end_sample": 16000,
                        "sample_count": 16000,
                    }
                ],
                "records": [
                    {
                        "start_sample": 0,
                        "cursor_samples": 16000,
                        "prompt_reference_count": 1,
                        "references": [{"term": "algorithm"}],
                    }
                ],
            }

            result = evaluate_run(payload, _args(root))

        self.assertEqual(
            result["gold_type_prompt_precision"]["acl_technical_gold"][
                "gold_type_prompt_precision"
            ],
            1.0,
        )
        aligned = result["source_time_aligned_prompt_precision"]
        self.assertFalse(aligned["available"])
        self.assertIn("FileNotFoundError", aligned["reason"])
        self.assertEqual(aligned["observed_prompt_reference_events"], 1)


if __name__ == "__main__":
    unittest.main()
