from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from eval.streaming_sst.prepare_xcomet_capacity_windows import prepare_windows


class PrepareXcometCapacityWindowsTest(unittest.TestCase):
    def test_prepares_pair_from_chunk_cursors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            talk = "2022.acl-long.268"
            (root / "source.txt").write_text("source one\nsource two\n", encoding="utf-8")
            (root / "reference.txt").write_text("参考一\n参考二\n", encoding="utf-8")
            (root / "meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "index": 0,
                                "talk": talk,
                                "offset": 10.0,
                                "duration": 4.0,
                                "seg_duration": 4.0,
                            }
                        ),
                        json.dumps(
                            {
                                "index": 1,
                                "talk": talk,
                                "offset": 40.0,
                                "duration": 4.0,
                                "seg_duration": 4.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "runs.json").write_text(
                json.dumps(
                    [
                        {
                            "preset": "10k",
                            "streaming_chunk_samples": 32000,
                            "output_events": [
                                {"cursor_samples": 32000, "text": "甲"},
                                {"cursor_samples": 96000, "text": "乙"},
                            ],
                        },
                        {
                            "preset": "1m",
                            "streaming_chunk_samples": 32000,
                            "output_events": [
                                {"cursor_samples": 32000, "text": "丙"},
                                {"cursor_samples": 96000, "text": "丁"},
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                runs_json=root / "runs.json",
                baseline_preset="10k",
                comparison_preset="1m",
                acl_meta=root / "meta.jsonl",
                acl_source_text=root / "source.txt",
                acl_reference_text=root / "reference.txt",
                audio_dir=None,
                talks=talk,
                window_sec=3.0,
                out_jsonl=root / "unused.jsonl",
            )

            rows = prepare_windows(args)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["auto_hypothesis"], "甲")
            self.assertEqual(rows[0]["merged_hypothesis"], "丙")
            self.assertEqual(rows[1]["auto_hypothesis"], "乙")
            self.assertEqual(rows[1]["reference"], "参考二")
            self.assertEqual(rows[1]["local_start_s"], 4.0)
            self.assertEqual(rows[1]["local_end_s"], 8.0)


if __name__ == "__main__":
    unittest.main()
