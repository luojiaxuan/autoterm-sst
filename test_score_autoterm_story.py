from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from eval.streaming_sst.score_autoterm_story import (
    RUN_ROLES,
    assemble_scorecard,
    markdown_report,
    validate_timing_compatible,
)
from eval.streaming_sst.score_time_aligned_terms import TimedOccurrence
from eval.streaming_sst.selected_window_smoke import FROZEN4_WINDOWS, PROTOCOL_ID


def make_payload(*, item_id: str = "talk-1", text: str = "术语甲") -> dict:
    return {
        "config": {"preset": "test"},
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
        "records": [
            {
                "start_sample": 0,
                "cursor_samples": 32000,
                "text": text,
                "prompt_reference_count": 1,
                "references": [{"term": "alpha", "translation": "术语甲"}],
            }
        ],
    }


def occurrence(term: str, translation: str, start: float) -> TimedOccurrence:
    return TimedOccurrence(
        domain="nlp",
        block_index=1,
        term=term,
        variants=[translation],
        t_start=start,
        t_end=start + 0.2,
    )


def selected_window_payload() -> dict:
    payload = make_payload()
    payload["config"]["selected_window_protocol"] = PROTOCOL_ID
    payload["blocks"] = []
    payload["block_spans"] = []
    cursor = 0
    for block_index, window in enumerate(FROZEN4_WINDOWS, start=1):
        common = {
            "item_id": window.item_id,
            "original_item_id": window.original_item_id,
            "corpus": window.corpus,
            "expected_domain": window.expected_domain,
            "source_offset_samples": window.source_offset_samples,
            "source_end_samples": window.source_end_samples,
        }
        payload["blocks"].append(dict(common))
        payload["block_spans"].append(
            {
                **common,
                "block_index": block_index,
                "start_sample": cursor,
                "end_sample": cursor + window.sample_count,
                "sample_count": window.sample_count,
            }
        )
        cursor += window.sample_count
    return payload


def fake_bleu_scores(*, hypothesis: str, reference: str, target_terms: list[str], sacrebleu_tokenizer: str) -> dict:
    del hypothesis, reference, sacrebleu_tokenizer
    return {
        "bleu": 10.0,
        "masked_terms_bleu": 20.0 + len(target_terms),
        "masked_terms_types": len(target_terms),
        "masked_terms_hyp_removed": 1,
        "masked_terms_ref_removed": 1,
    }


class AutoTermStoryScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payloads = {role: make_payload() for role in RUN_ROLES}
        self.inputs = {
            role: {"path": f"/{role}.json", "sha256": role * 8}
            for role in RUN_ROLES
        }
        self.gold = {
            "technical_plus_medicine": [occurrence("alpha", "术语甲", 0.5)],
            "raw_plus_medicine": [
                occurrence("alpha", "术语甲", 0.5),
                occurrence("beta", "术语乙", 2.5),
            ],
        }

    def build(
        self,
        payloads: dict | None = None,
        *,
        expected: int | None = 2,
        allow_timing_mismatch: bool = False,
        selected_window_smoke: bool = False,
    ) -> dict:
        with patch(
            "eval.streaming_sst.score_autoterm_story.compute_bleu_scores",
            side_effect=fake_bleu_scores,
        ):
            return assemble_scorecard(
                payloads or self.payloads,
                run_inputs=self.inputs,
                gold_sets=self.gold,
                reference_text="参考译文",
                target_lang="zh",
                sacrebleu_tokenizer="zh",
                post_s=30.0,
                prompt_lookback_s=1.92,
                prompt_alignment_tolerance_s=0.0,
                expected_raw_denominator=expected,
                allow_timing_mismatch=allow_timing_mismatch,
                selected_window_smoke=selected_window_smoke,
            )

    def test_headline_uses_only_fixed_raw_mfa_denominator(self) -> None:
        report = self.build()

        self.assertEqual(report["protocol"]["headline_term_accuracy"]["fixed_denominator"], 2)
        self.assertEqual(report["runs"]["oracle"]["headline_term_acc"]["hits"], 1)
        self.assertEqual(report["runs"]["oracle"]["headline_term_acc"]["term_acc"], 0.5)
        self.assertEqual(report["runs"]["oracle"]["quality"]["technical_masked_bleu"], 21.0)
        self.assertEqual(report["runs"]["oracle"]["quality"]["raw_masked_bleu"], 22.0)
        self.assertEqual(report["runs"]["oracle"]["prompt"]["strict_time_local_raw_precision"], 1.0)
        self.assertEqual(report["runs"]["oracle"]["prompt"]["retrieved_references_per_chunk"], 1.0)
        excluded = {row["name"]: row["status"] for row in report["protocol"]["excluded_metrics"]}
        self.assertEqual(excluded["legacy 419-denominator TERM_ACC"], "not computed")
        self.assertEqual(excluded["block-level count-clipping TERM_ACC"], "not computed")

    def test_rejects_denominator_drift_from_explicit_assertion(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator is 2, expected 3"):
            self.build(expected=3)

    def test_selected_window_smoke_enforces_protocol_and_fixed_179_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected-window protocol is invalid"):
            self.build(selected_window_smoke=True)

        payloads = {role: selected_window_payload() for role in RUN_ROLES}
        with self.assertRaisesRegex(ValueError, "denominator is 2, expected 179"):
            self.build(
                payloads,
                expected=None,
                selected_window_smoke=True,
            )

    def test_full_protocol_report_does_not_add_selected_window_metadata(self) -> None:
        self.assertNotIn("selected_window_smoke", self.build()["protocol"])

    def test_rejects_playlist_mismatch(self) -> None:
        payloads = copy.deepcopy(self.payloads)
        payloads["merged"]["blocks"][0]["item_id"] = "talk-2"
        payloads["merged"]["block_spans"][0]["item_id"] = "talk-2"

        with self.assertRaisesRegex(ValueError, "does not share"):
            self.build(payloads)

    def test_rejects_nonidentical_decoder_windows(self) -> None:
        payloads = copy.deepcopy(self.payloads)
        payloads["autoterm"]["records"][0]["cursor_samples"] = 48000

        with self.assertRaisesRegex(ValueError, "incompatible at event 1"):
            validate_timing_compatible(payloads)

    def test_strict_prompt_capture_rejects_truncated_references(self) -> None:
        payloads = copy.deepcopy(self.payloads)
        payloads["autoterm"]["records"][0]["prompt_reference_count"] = 2

        with self.assertRaisesRegex(ValueError, "captured 1 references"):
            self.build(payloads)

    def test_opt_in_timing_mismatch_is_unpaired_and_diagnostic(self) -> None:
        payloads = copy.deepcopy(self.payloads)
        payloads["autoterm"]["records"][0]["cursor_samples"] = 48000

        report = self.build(payloads, allow_timing_mismatch=True)
        timing = report["protocol"]["timing"]
        self.assertFalse(timing["timing_comparable"])
        self.assertTrue(timing["mismatch_explicitly_allowed"])
        self.assertFalse(timing["paired_delta_claims_allowed"])
        self.assertEqual(timing["runs"]["autoterm"]["first_mismatch"]["event_index"], 1)
        self.assertEqual(timing["runs"]["autoterm"]["event_count_delta_vs_oracle"], 0)
        self.assertIn("paired differences, deltas, and superiority claims are forbidden", markdown_report(report))

    def test_markdown_names_the_only_headline_and_exclusions(self) -> None:
        text = markdown_report(self.build())

        self.assertIn("MFA time-aligned `raw_plus_medicine`", text)
        self.assertIn("legacy 419-denominator", text)
        self.assertIn("block-level count-clipping", text)
        self.assertIn("| Oracle | 50.00 | 1 / 2 |", text)


if __name__ == "__main__":
    unittest.main()
