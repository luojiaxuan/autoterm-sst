from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from eval.streaming_sst.score_mixed_audio_terms import GoldOccurrence, build_reference_text, score_occurrences


class MixedAudioTermScorerTests(unittest.TestCase):
    def test_build_reference_text_uses_acl_meta_and_medicine_ref(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            wav_a = acl_root / "seg" / "000.wav"
            wav_b = acl_root / "seg" / "001.wav"
            wav_a.parent.mkdir(parents=True)
            wav_a.write_bytes(b"")
            wav_b.write_bytes(b"")
            (acl_root / "segments.meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"index": 0, "seg_wav": str(wav_a), "seg_duration": 1.0}),
                        json.dumps({"index": 1, "seg_wav": str(wav_b), "seg_duration": 1.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            acl_ref = root / "acl_ref.txt"
            acl_ref.write_text("ACL参考一\nACL参考二\n", encoding="utf-8")
            medicine_dir = root / "medicine"
            medicine_dir.mkdir()
            (medicine_dir / "medicine.ref.zh__medicine_606.txt").write_text("医学参考\n", encoding="utf-8")
            payload = {
                "blocks": [
                    {"item_id": "acl_a", "corpus": "acl", "wav_paths": [str(wav_a), str(wav_b)]},
                    {"item_id": "medicine_606", "corpus": "medicine", "wav_paths": []},
                ],
                "block_spans": [
                    {"block_index": 1, "sample_count": 32000},
                    {"block_index": 2, "sample_count": 16000},
                ],
            }
            args = SimpleNamespace(
                acl_root=str(acl_root),
                acl_reference_text=str(acl_ref),
                medicine_oracle_dir=str(medicine_dir),
                target_lang="zh",
            )

            reference = build_reference_text(payload, args)

        self.assertEqual(reference, "ACL参考一\nACL参考二\n医学参考\n")

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

    def test_type_accuracy_is_block_local_for_repeated_terms(self) -> None:
        payload = {
            "block_spans": [
                {
                    "block_index": 1,
                    "start_sample": 0,
                    "end_sample": 16000,
                },
                {
                    "block_index": 2,
                    "start_sample": 16000,
                    "end_sample": 32000,
                },
            ],
            "records": [
                {
                    "cursor_samples": 16000,
                    "text": "直肠癌",
                },
                {
                    "cursor_samples": 32000,
                    "text": "没有目标术语",
                },
            ],
        }
        gold = [
            GoldOccurrence("medicine", "medicine_606", 1, "rectal cancer", ["直肠癌"], "medicine_oracle"),
            GoldOccurrence("medicine", "medicine_404", 2, "rectal cancer", ["直肠癌"], "medicine_oracle"),
        ]

        metrics = score_occurrences(payload, gold)

        self.assertEqual(metrics["unique_term_types"], 2)
        self.assertEqual(metrics["type_hits_any"], 1)
        self.assertEqual(metrics["type_acc_any"], 0.5)


if __name__ == "__main__":
    unittest.main()
