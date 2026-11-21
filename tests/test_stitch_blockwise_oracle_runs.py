from __future__ import annotations

import argparse
import copy
import unittest
from pathlib import Path

from eval.streaming_sst.stitch_blockwise_oracle_runs import (
    BlockSelection,
    parse_block_selector,
    stitch_payloads,
    stitch_selected_blocks,
    validate_source_payload,
)


def _block(item_id: str, corpus: str, domain: str) -> dict:
    return {
        "item_id": item_id,
        "corpus": corpus,
        "expected_domain": domain,
        "wav_paths": [f"/{item_id}.wav"],
        "custom_block_metadata": {"item": item_id},
    }


def _record(
    *,
    start: int,
    cursor: int,
    domain: str,
    term: str,
    event_idx: int,
    **extra: object,
) -> dict:
    row = {
        "event_idx": event_idx,
        "start_sample": start,
        "cursor_samples": cursor,
        "cursor_s": round(cursor / 16000, 3),
        "expected_domain": domain,
        "active_domain": domain,
        "active_preset": f"{domain}_10k",
        "switch_count": 0,
        "router_action": "hold",
        "router_text_source": "translation_history",
        "prompt_reference_count": 1,
        "fixed_prompt_k": 1,
        "candidate_pool_count": 10,
        "references": [
            {
                "term": term,
                "translation": f"translated-{term}",
                "metadata": {"source_start_sample": 7},
            }
        ],
        "retrieve_s": 0.01,
        "text": f"translated-{term}",
        "custom_record_metadata": {"term": term},
    }
    row.update(extra)
    return row


def _payload(
    *,
    blocks: list[dict],
    spans: list[tuple[int, int]],
    records: list[dict],
    preset: str,
    chunk_samples: int = 16000,
) -> dict:
    block_spans = []
    for index, (block, (start, end)) in enumerate(zip(blocks, spans), start=1):
        block_spans.append(
            {
                "block_index": index,
                "item_id": block["item_id"],
                "corpus": block["corpus"],
                "expected_domain": block["expected_domain"],
                "start_sample": start,
                "end_sample": end,
                "sample_count": end - start,
                "wav_count": len(block["wav_paths"]),
                "custom_span_metadata": {"source_index": index},
            }
        )
    return {
        "config": {
            "schedule": "source_schedule",
            "preset": preset,
            "language_pair": "English -> Chinese",
            "latency_multiplier": 2,
            "chunk_samples": chunk_samples,
            "chunk_seconds": round(chunk_samples / 16000, 3),
            "feed_sleep": 1.0,
            "max_seconds_per_item": 0.0,
        },
        "blocks": blocks,
        "block_spans": block_spans,
        "records": records,
        "summary": {"preset": preset, "event_count": len(records), "stale": True},
        "session": {"session_id": f"session-{preset}", "metadata": {"preset": preset}},
    }


class StitchBlockwiseOracleRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.acl = _payload(
            blocks=[_block("acl-268", "acl", "nlp"), _block("acl-367", "acl", "nlp")],
            spans=[(0, 32000), (32000, 64000)],
            records=[
                _record(start=0, cursor=16000, domain="nlp", term="SRL", event_idx=1),
                _record(start=16000, cursor=32000, domain="nlp", term="NMT", event_idx=2),
                _record(
                    start=30000,
                    cursor=48000,
                    domain="nlp",
                    term="AMR",
                    event_idx=3,
                    window_start_sample=28000,
                    window_start_s=1.75,
                    window_end_sample=48000,
                    window_end_s=3.0,
                    last_llm_samples=30000,
                    last_llm_s=1.875,
                    chunk_samples=18000,
                    sample_count=18000,
                ),
                _record(start=48000, cursor=64000, domain="nlp", term="BPE", event_idx=4),
            ],
            preset="nlp_10k",
        )
        self.medicine = _payload(
            blocks=[_block("medicine-545006", "medicine", "medicine")],
            spans=[(0, 16000)],
            records=[
                _record(start=0, cursor=16000, domain="medicine", term="carcinoma", event_idx=1),
            ],
            preset="medicine_10k",
        )

    def test_reorders_blocks_clips_windows_and_preserves_metadata(self) -> None:
        original_acl = copy.deepcopy(self.acl)
        original_medicine = copy.deepcopy(self.medicine)
        result = stitch_selected_blocks(
            [
                BlockSelection(self.acl, 1, "acl.json", "acl-sha"),
                BlockSelection(self.medicine, 1, "medicine.json", "medicine-sha"),
                BlockSelection(self.acl, 2, "acl.json", "acl-sha"),
            ]
        )

        self.assertEqual([block["item_id"] for block in result["blocks"]], ["acl-268", "medicine-545006", "acl-367"])
        self.assertEqual(
            [(span["start_sample"], span["end_sample"]) for span in result["block_spans"]],
            [(0, 32000), (32000, 48000), (48000, 80000)],
        )
        self.assertEqual([record["event_idx"] for record in result["records"]], [1, 2, 3, 4, 5])
        self.assertEqual([record["block_index"] for record in result["records"]], [1, 1, 2, 3, 3])
        self.assertEqual(
            [record["cursor_samples"] for record in result["records"]],
            [16000, 32000, 48000, 64000, 80000],
        )

        clipped = result["records"][3]
        self.assertEqual(clipped["start_sample"], 48000)
        self.assertEqual(clipped["window_start_sample"], 48000)
        self.assertEqual(clipped["window_start_s"], 3.0)
        self.assertEqual(clipped["window_end_sample"], 64000)
        self.assertEqual(clipped["window_end_s"], 4.0)
        self.assertEqual(clipped["last_llm_samples"], 48000)
        self.assertEqual(clipped["last_llm_s"], 3.0)
        self.assertEqual(clipped["chunk_samples"], 18000)
        self.assertEqual(clipped["sample_count"], 18000)
        self.assertEqual(clipped["oracle_source"]["event_idx"], 3)

        self.assertEqual(
            clipped["references"],
            original_acl["records"][2]["references"],
        )
        self.assertEqual(clipped["custom_record_metadata"], {"term": "AMR"})
        self.assertEqual(result["blocks"][2]["custom_block_metadata"], {"item": "acl-367"})
        self.assertEqual(result["block_spans"][2]["custom_span_metadata"], {"source_index": 2})
        self.assertEqual(self.acl, original_acl)
        self.assertEqual(self.medicine, original_medicine)

        self.assertEqual(result["config"]["schedule"], "blockwise_oracle")
        self.assertEqual(result["config"]["preset"], "blockwise_oracle")
        self.assertEqual(result["config"]["oracle_presets"], ["nlp_10k", "medicine_10k", "nlp_10k"])
        self.assertEqual(result["config"]["medicine_ids"], ["medicine-545006"])
        self.assertEqual(result["summary"]["block_count"], 3)
        self.assertEqual(result["summary"]["source_run_count"], 2)
        self.assertEqual(result["summary"]["event_count"], 5)
        self.assertEqual(result["summary"]["audio_seconds"], 5.0)
        self.assertEqual(result["summary"]["active_domain_accuracy"], 1.0)
        self.assertNotIn("stale", result["summary"])
        self.assertEqual(
            [row["source_run_index"] for row in result["oracle_composition"]["block_sources"]],
            [1, 2, 1],
        )

    def test_whole_payload_mode_expands_every_block(self) -> None:
        result = stitch_payloads(
            [self.acl, self.medicine],
            source_labels=["acl", "medicine"],
        )

        self.assertEqual([block["item_id"] for block in result["blocks"]], ["acl-268", "acl-367", "medicine-545006"])
        self.assertEqual([span["block_index"] for span in result["block_spans"]], [1, 2, 3])
        self.assertEqual(result["records"][-1]["cursor_samples"], 80000)

    def test_rejects_source_span_overlap_and_identity_mismatch(self) -> None:
        overlapping = copy.deepcopy(self.acl)
        overlapping["block_spans"][1]["start_sample"] = 31000
        overlapping["block_spans"][1]["sample_count"] = 33000
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_source_payload(overlapping, label="overlap")

        mismatched = copy.deepcopy(self.acl)
        mismatched["block_spans"][1]["item_id"] = "wrong-talk"
        with self.assertRaisesRegex(ValueError, "does not match span identity"):
            validate_source_payload(mismatched, label="identity")

    def test_rejects_duplicate_selection_and_duplicate_output_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate block selection"):
            stitch_selected_blocks(
                [
                    BlockSelection(self.acl, 1, "acl", "same"),
                    BlockSelection(self.acl, 1, "acl", "same"),
                ]
            )

        duplicate_identity = copy.deepcopy(self.medicine)
        duplicate_identity["blocks"][0].update(
            {"item_id": "acl-268", "corpus": "acl", "expected_domain": "nlp"}
        )
        duplicate_identity["block_spans"][0].update(
            {"item_id": "acl-268", "corpus": "acl", "expected_domain": "nlp"}
        )
        duplicate_identity["records"][0].update({"active_domain": "nlp", "expected_domain": "nlp"})
        with self.assertRaisesRegex(ValueError, "duplicate output block identity"):
            stitch_selected_blocks(
                [
                    BlockSelection(self.acl, 1, "acl"),
                    BlockSelection(duplicate_identity, 1, "duplicate"),
                ]
            )

    def test_rejects_incompatible_runtime_and_non_oracle_domain(self) -> None:
        incompatible = copy.deepcopy(self.medicine)
        incompatible["config"]["chunk_samples"] = 30720
        with self.assertRaisesRegex(ValueError, "disagree on config.chunk_samples"):
            stitch_selected_blocks(
                [
                    BlockSelection(self.acl, 1, "acl"),
                    BlockSelection(incompatible, 1, "medicine"),
                ]
            )

        wrong_domain = copy.deepcopy(self.medicine)
        wrong_domain["records"][0]["active_domain"] = "nlp"
        with self.assertRaisesRegex(ValueError, "active_domain='nlp'"):
            stitch_selected_blocks([BlockSelection(wrong_domain, 1, "medicine")])

        allowed = stitch_selected_blocks(
            [BlockSelection(wrong_domain, 1, "medicine")],
            require_oracle_domain=False,
        )
        self.assertEqual(allowed["summary"]["active_domain_accuracy"], 0.0)
        self.assertFalse(allowed["summary"]["regression_pass"])

    def test_rejects_record_outside_spans_and_incomplete_reference_capture(self) -> None:
        outside = copy.deepcopy(self.medicine)
        outside["records"][0]["cursor_samples"] = 17000
        with self.assertRaisesRegex(ValueError, "falls outside all block spans"):
            validate_source_payload(outside, label="outside")

        incomplete = copy.deepcopy(self.medicine)
        incomplete["records"][0]["prompt_reference_count"] = 2
        with self.assertRaisesRegex(ValueError, "captured 1 references but reports 2"):
            validate_source_payload(incomplete, label="references")

    def test_parses_block_selector_from_the_right(self) -> None:
        path, index = parse_block_selector("/tmp/a=b/run.json=12")
        self.assertEqual(path, Path("/tmp/a=b/run.json"))
        self.assertEqual(index, 12)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_block_selector("run.json")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_block_selector("run.json=0")


if __name__ == "__main__":
    unittest.main()
