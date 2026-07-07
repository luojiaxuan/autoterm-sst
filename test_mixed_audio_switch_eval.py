from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from eval.streaming_sst.eval_mixed_audio_switch import (
    TARGET_SAMPLE_RATE,
    AudioBlock,
    build_schedule,
    build_spans,
    domain_transitions,
    extract_record,
    expected_domain_at,
    read_acl_audio_blocks,
    read_medicine_audio_blocks,
    summarize_run,
)


def _write_wav(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((frames,), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(data.tobytes())


class MixedAudioSwitchEvalTests(unittest.TestCase):
    def test_audio_block_readers_and_schedule(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            acl_wav_a = acl_root / "seg" / "000.wav"
            acl_wav_b = acl_root / "seg" / "001.wav"
            _write_wav(acl_wav_a, TARGET_SAMPLE_RATE)
            _write_wav(acl_wav_b, TARGET_SAMPLE_RATE)
            (acl_root / "segments.meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"talk": "acl_a", "seg_wav": str(acl_wav_a)}),
                        json.dumps({"talk": "acl_b", "seg_wav": str(acl_wav_b)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            med_dir = root / "medicine"
            med_wav = med_dir / "sample_404_v2" / "404_v2.wav"
            _write_wav(med_wav, TARGET_SAMPLE_RATE)

            acl = read_acl_audio_blocks(str(acl_root), limit_items=2)
            medicine = read_medicine_audio_blocks(str(med_dir), limit_items=1)
            schedule = build_schedule(acl, medicine, schedule="alternating")

        self.assertEqual([item.item_id for item in schedule], ["acl_a", "medicine_404", "acl_b"])
        self.assertEqual([item.expected_domain for item in schedule], ["nlp", "medicine", "nlp"])

    def test_acl_reader_filters_missing_wavs_before_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "acl"
            existing_a = acl_root / "seg" / "000.wav"
            existing_b = acl_root / "seg" / "002.wav"
            _write_wav(existing_a, TARGET_SAMPLE_RATE)
            _write_wav(existing_b, TARGET_SAMPLE_RATE)
            (acl_root / "segments.meta.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"talk": "missing_first", "seg_wav": str(acl_root / "seg" / "missing.wav")}),
                        json.dumps({"talk": "acl_a", "seg_wav": str(existing_a)}),
                        json.dumps({"talk": "acl_b", "seg_wav": str(existing_b)}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            blocks = read_acl_audio_blocks(str(acl_root), limit_items=2)

        self.assertEqual([block.item_id for block in blocks], ["acl_a", "acl_b"])

    def test_spans_and_expected_domain_lookup(self) -> None:
        blocks = [
            AudioBlock("acl", "nlp", "acl", ["mock-a.wav"]),
            AudioBlock("med", "medicine", "medicine", ["mock-b.wav"]),
        ]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)

        self.assertEqual(spans[0].start_sample, 0)
        self.assertEqual(spans[0].end_sample, TARGET_SAMPLE_RATE)
        self.assertEqual(spans[1].start_sample, TARGET_SAMPLE_RATE)
        self.assertEqual(expected_domain_at(spans, 1), "nlp")
        self.assertEqual(expected_domain_at(spans, TARGET_SAMPLE_RATE), "nlp")
        self.assertEqual(expected_domain_at(spans, TARGET_SAMPLE_RATE + 1), "medicine")

    def test_summary_detects_transition_latency_and_steady_state(self) -> None:
        blocks = [
            AudioBlock("acl", "nlp", "acl", ["mock-a.wav"]),
            AudioBlock("med", "medicine", "medicine", ["mock-b.wav"]),
        ]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)
        records = [
            {
                "event_idx": 1,
                "cursor_samples": TARGET_SAMPLE_RATE // 2,
                "expected_domain": "nlp",
                "active_domain": "nlp",
                "router_action": "stay",
                "router_target_domain": "nlp",
                "domain_probe_top_domain": "nlp",
            },
            {
                "event_idx": 2,
                "cursor_samples": TARGET_SAMPLE_RATE + TARGET_SAMPLE_RATE // 4,
                "expected_domain": "medicine",
                "active_domain": "nlp",
                "router_action": "stay",
                "router_target_domain": "medicine",
                "domain_probe_top_domain": "medicine",
            },
            {
                "event_idx": 3,
                "cursor_samples": TARGET_SAMPLE_RATE + TARGET_SAMPLE_RATE // 2,
                "expected_domain": "medicine",
                "active_domain": "medicine",
                "router_action": "switch",
                "router_target_domain": "medicine",
                "domain_probe_top_domain": "medicine",
                "switch_count": 1,
            },
        ]

        transitions = domain_transitions(spans, records, max_switch_events=2)
        summary = summarize_run(
            schedule_name="alternating",
            preset="auto_working",
            spans=spans,
            records=records,
            chunk_samples=TARGET_SAMPLE_RATE // 2,
            max_switch_events=2,
        )

        self.assertEqual(transitions[0]["latency_events"], 2)
        self.assertTrue(transitions[0]["passed"])
        self.assertTrue(summary["regression_pass"])
        self.assertEqual(summary["steady_state_mismatch_count"], 0)
        self.assertEqual(summary["probe_top_accuracy"], 1.0)

    def test_extract_record_requires_cursor_samples_and_runtime_schema(self) -> None:
        blocks = [AudioBlock("acl", "nlp", "acl", ["mock-a.wav"])]
        with patch("eval.streaming_sst.eval_mixed_audio_switch.wav_num_frames", return_value=TARGET_SAMPLE_RATE):
            spans = build_spans(blocks)
        event = {
            "type": "partial",
            "text": "测试",
            "meta": {
                "cursor_samples": TARGET_SAMPLE_RATE,
                "topic": {
                    "active_domain": "nlp",
                    "active_glossary_preset": "nlp_core_10k",
                    "switch_count": 0,
                },
                "topic_router": {
                    "action": "stay",
                    "to_domain": "nlp",
                    "confidence": 0.9,
                    "margin": 0.5,
                },
                "domain_probe_scores": {
                    "nlp": {"domain": "nlp", "top_score": 0.8, "mean_topk_score": 0.7}
                },
                "router_text_source": "generated_target",
                "prompt_reference_count": 10,
                "fixed_prompt_k": 10,
                "candidate_pool_count": 50,
            },
        }

        record = extract_record(event, event_idx=1, spans=spans)
        self.assertEqual(record["cursor_samples"], TARGET_SAMPLE_RATE)
        self.assertEqual(record["expected_domain"], "nlp")
        self.assertEqual(record["domain_probe_top_domain"], "nlp")

        bad = dict(event)
        bad["meta"] = dict(event["meta"])
        del bad["meta"]["cursor_samples"]
        with self.assertRaisesRegex(RuntimeError, "cursor_samples"):
            extract_record(bad, event_idx=1, spans=spans)


if __name__ == "__main__":
    unittest.main()
