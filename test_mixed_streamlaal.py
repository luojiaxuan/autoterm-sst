import json
import tempfile
import unittest
from pathlib import Path

from eval.streaming_sst.score_mixed_streamlaal import (
    build_character_latency_series,
    compute_laal,
    fbk_stream_elapsed,
    load_reference_text,
    main,
    score_payload,
)


def _payload(records=None):
    return {
        "config": {"feed_sleep": 1.0},
        "blocks": [
            {"item_id": "a", "corpus": "acl"},
            {"item_id": "b", "corpus": "medicine"},
        ],
        "block_spans": [
            {
                "block_index": 1,
                "item_id": "a",
                "corpus": "acl",
                "expected_domain": "nlp",
                "start_sample": 0,
                "end_sample": 500,
                "sample_count": 500,
            },
            {
                "block_index": 2,
                "item_id": "b",
                "corpus": "medicine",
                "expected_domain": "medicine",
                "start_sample": 500,
                "end_sample": 1000,
                "sample_count": 500,
            },
        ],
        "records": records
        or [
            {"cursor_samples": 400, "text": "甲乙", "emitted_wall_s": 0.5},
            {"cursor_samples": 800, "text": "丙", "emitted_wall_s": 1.1},
            {"cursor_samples": 1000, "text": "丁", "emitted_wall_s": 1.4},
        ],
    }


class MixedStreamLAALTest(unittest.TestCase):
    def test_laal_matches_simuleval_length_adaptive_formula(self) -> None:
        row = compute_laal(
            [500.0, 500.0, 1000.0, 1000.0],
            source_length_ms=1000.0,
            reference_length=6,
        )
        self.assertAlmostEqual(row["score_ms"], 500.0)
        self.assertEqual(row["tau"], 3)
        self.assertEqual(row["adaptive_target_length"], 6)
        self.assertTrue(row["reached_source_end"])

    def test_fbk_computation_aware_elapsed_transform(self) -> None:
        transformed = fbk_stream_elapsed(
            [400.0, 400.0, 800.0, 800.0],
            [500.0, 500.0, 1100.0, 1100.0],
        )
        self.assertEqual(transformed, [500.0, 500.0, 1000.0, 1000.0])

    def test_whole_playlist_standard_and_computation_aware_scores(self) -> None:
        report = score_payload(
            _payload(),
            reference_text="甲乙\n丙丁\n",
            sample_rate=1000,
            expected_block_count=2,
        )
        self.assertAlmostEqual(report["metrics"]["stream_laal_ms"], 275.0)
        self.assertAlmostEqual(
            report["metrics"]["stream_laal_ca_ms"],
            1250.0 / 3.0,
        )
        self.assertEqual(report["text"]["reference_chars"], 4)
        self.assertEqual(report["text"]["hypothesis_chars"], 4)
        self.assertEqual(report["playlist"]["tail_gap_samples"], 0)
        self.assertFalse(report["protocol"]["resegmentation"])

    def test_global_outer_whitespace_is_removed_with_timestamps(self) -> None:
        series = build_character_latency_series(
            _payload(
                [
                    {"cursor_samples": 400, "text": " 甲", "emitted_wall_s": 0.5},
                    {"cursor_samples": 800, "text": "乙 ", "emitted_wall_s": 1.1},
                ]
            ),
            source_length_samples=1000,
            sample_rate=1000,
        )
        self.assertEqual(series.prediction, "甲乙")
        self.assertEqual(series.delays_ms, (400.0, 800.0))
        self.assertEqual(series.raw_elapsed_ms, (500.0, 1100.0))

    def test_missing_wall_timestamp_fails_instead_of_silently_downgrading(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks emitted_wall_s"):
            score_payload(
                _payload([{"cursor_samples": 400, "text": "甲"}]),
                reference_text="甲",
                sample_rate=1000,
            )

    def test_standard_only_explicitly_allows_missing_wall_timestamps(self) -> None:
        report = score_payload(
            _payload(
                [
                    {"cursor_samples": 400, "text": "甲"},
                    {"cursor_samples": 1000, "text": "乙"},
                ]
            ),
            reference_text="甲乙",
            sample_rate=1000,
            standard_only=True,
        )

        self.assertIsNotNone(report["metrics"]["stream_laal_ms"])
        self.assertIsNone(report["metrics"]["stream_laal_ca_ms"])
        self.assertFalse(report["protocol"]["computation_aware_enabled"])

    def test_non_monotonic_cursor_or_wall_time_fails(self) -> None:
        bad_cursor = _payload(
            [
                {"cursor_samples": 800, "text": "甲", "emitted_wall_s": 0.8},
                {"cursor_samples": 700, "text": "乙", "emitted_wall_s": 0.9},
            ]
        )
        with self.assertRaisesRegex(ValueError, "non-monotonic cursor_samples"):
            score_payload(bad_cursor, reference_text="甲乙", sample_rate=1000)

        bad_wall = _payload(
            [
                {"cursor_samples": 400, "text": "甲", "emitted_wall_s": 0.8},
                {"cursor_samples": 800, "text": "乙", "emitted_wall_s": 0.7},
            ]
        )
        with self.assertRaisesRegex(ValueError, "non-monotonic emitted_wall_s"):
            score_payload(bad_wall, reference_text="甲乙", sample_rate=1000)

    def test_reference_loader_concatenates_lines_for_char_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.txt"
            path.write_text(" 甲乙 \n丙丁\n", encoding="utf-8")
            self.assertEqual(load_reference_text(path), "甲乙丙丁")

    def test_cli_writes_reproducible_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_path = root / "run.json"
            ref_path = root / "ref.txt"
            out_path = root / "score.json"
            md_path = root / "score.md"
            run_path.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
            ref_path.write_text("甲乙\n丙丁\n", encoding="utf-8")
            rc = main(
                [
                    "--run-json",
                    str(run_path),
                    "--reference-text-file",
                    str(ref_path),
                    "--sample-rate",
                    "1000",
                    "--expected-block-count",
                    "2",
                    "--out-json",
                    str(out_path),
                    "--out-markdown",
                    str(md_path),
                ]
            )
            self.assertEqual(rc, 0)
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "mixed_streamlaal.v1")
            self.assertIn("run_json_sha256", saved["inputs"])
            self.assertIn("StreamLAAL-CA", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
